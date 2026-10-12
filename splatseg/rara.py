"""RaRa Clipper (Li et al., arXiv:2506.20202): smooth plane clipping of splats at render time.

Each primitive is classified against the plane n.x + d = 0 (kept side: n.x + d > 0) using its support radius
r = max scale:
    invisible  n.mu + d < -r       -> dropped
    visible    n.mu + d >  r       -> rendered unchanged
    cut        |n.mu + d| <= r     -> rendered with per-pixel alpha scaled by the visible fraction of the
                                      camera ray's chord through the primitive (CUDA, csrc/rara_rasterize.cu)

Beta-splatting kernels have compact support (zero at Mahalanobis distance 1), so the support ellipsoid is
exactly the one with semi-axes equal to the scales; the paper's 3-sigma radius for Gaussians becomes 1 here.

"Hard" clipping, the usual baseline, keeps or drops whole primitives by their centre.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from splatseg import model as M


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def classify(model, plane: np.ndarray):
    """Returns (keep_mask, cut_mask) as CUDA bool tensors over all primitives."""
    n = torch.tensor(plane[:3], dtype=torch.float32, device="cuda")
    s = model.get_xyz @ n + float(plane[3])
    r = model.get_scaling.max(dim=-1).values
    return s >= -r, s.abs() <= r


def hard_keep(model, plane: np.ndarray) -> torch.Tensor:
    n = torch.tensor(plane[:3], dtype=torch.float32, device="cuda")
    return (model.get_xyz @ n + float(plane[3])) > 0


@contextmanager
def rara_hook(model, keep: torch.Tensor, cut_all: torch.Tensor, plane: np.ndarray, K, R, t):
    """While active, gsplat's rasterizer is replaced by the RaRa kernel for this camera and plane."""
    import gsplat.rendering as gr
    from splatseg import _C

    cut = cut_all[keep]
    scales = model.get_scaling[keep]
    rot = quat_to_rotmat(model.get_rotation[keep])
    # M = R S maps the unit sphere onto the support ellipsoid; M^-1 = S^-1 R^T
    minv = (rot.transpose(1, 2) / scales[:, :, None]).reshape(-1, 9).contiguous()
    mu = model.get_xyz[keep].contiguous()
    cam_center = -R.T @ t
    c2w = R.T
    plane_cam = [*map(float, plane), *map(float, cam_center), *map(float, c2w.reshape(-1)),
                 float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    original = gr.rasterize_to_pixels

    def rasterize(means2d, conics, colors, opacities, betas, image_width, image_height, tile_size,
                  isect_offsets, flatten_ids, backgrounds=None, masks=None, packed=False, absgrad=False):
        assert not packed and means2d.shape[0] == 1, "RaRa hook supports a single camera, non-packed"
        rgb, alpha = _C.rara_rasterize_fwd(
            means2d[0].contiguous(), conics[0].contiguous(), colors[0].contiguous(),
            opacities[0].contiguous(), betas[0].contiguous(),
            None if backgrounds is None else backgrounds[0].contiguous(),
            cut.to(torch.uint8).contiguous(), minv, mu, plane_cam,
            image_width, image_height, tile_size, isect_offsets[0].contiguous(), flatten_ids.contiguous())
        return rgb[None], alpha[None, ..., None]

    gr.rasterize_to_pixels = rasterize
    try:
        yield
    finally:
        gr.rasterize_to_pixels = original


def render_clipped(model, plane: np.ndarray, K, R, t, width, height, mode: str = "rara",
                   base_keep: torch.Tensor | None = None, background=(1.0, 1.0, 1.0)) -> np.ndarray:
    """mode: 'none' (no clipping), 'hard' (centre test) or 'rara'."""
    base = base_keep if base_keep is not None else torch.ones(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    if mode == "none":
        return M.render(model, K, R, t, width, height, base, background)
    if mode == "hard":
        return M.render(model, K, R, t, width, height, base & hard_keep(model, plane), background)
    keep, cut = classify(model, plane)
    keep = keep & base
    with rara_hook(model, keep, cut, plane, K, R, t):
        return M.render(model, K, R, t, width, height, keep, background)
