"""Cut an object out of a splat by multi-view mask voting.

Every primitive centre is projected into every training view and labelled by that view's 2D object mask:
    1  inside the mask,  0  outside it,  -1  behind the camera or outside the frame (no vote).
A primitive is kept if it is labelled 1 in more than `threshold` of the views that see it, and is seen by at
least `min_views` of them. A single view is easily fooled (a background splat directly behind the car lands
inside the car's silhouette), but the same splat falls outside the mask from most other angles, so the vote
removes it.

This is the label-assignment and multi-view ensemble of SAGS (Hu et al., "Segment Anything in 3D Gaussians",
2024), re-implemented for Beta splats with masks from any 2D segmenter.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from splatseg import colmap_io as cio


def object_masks(images_dir: Path, names: list[str], cache_dir: Path, model_name: str = "u2net") -> dict:
    """Foreground masks (uint8 0/255) from rembg, cached as PNGs next to the run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = None
    masks = {}
    for name in names:
        out = cache_dir / (Path(name).stem + ".png")
        if not out.exists():
            if session is None:
                from rembg import new_session, remove
                session = new_session(model_name)
            img = cv2.imread(str(images_dir / name))
            mask = remove(img, session=session, only_mask=True)
            cv2.imwrite(str(out), mask)
        masks[name] = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    return masks


def vote(xyz: torch.Tensor, cameras: dict, images: dict, masks: dict, threshold: float = 0.8,
         min_views: int = 5, mask_level: int = 128) -> tuple[torch.Tensor, dict]:
    """Returns (keep [N] bool on xyz.device, stats)."""
    inside = torch.zeros(xyz.shape[0], device=xyz.device)
    seen = torch.zeros(xyz.shape[0], device=xyz.device)
    for im in images.values():
        if im.name not in masks:
            continue
        cam = cameras[im.camera_id]
        K = torch.tensor(cam.K(), dtype=torch.float32, device=xyz.device)
        R = torch.tensor(im.R, dtype=torch.float32, device=xyz.device)
        t = torch.tensor(im.t, dtype=torch.float32, device=xyz.device)
        pc = xyz @ R.T + t
        z = pc[:, 2]
        uv = (pc @ K.T)[:, :2] / z.clamp(min=1e-6)[:, None]
        m = torch.from_numpy(masks[im.name] >= mask_level).to(xyz.device)
        h, w = m.shape
        if (w, h) != (cam.width, cam.height):   # masks at another resolution: rescale the projections
            uv = uv * torch.tensor([w / cam.width, h / cam.height], device=xyz.device)
        u, v = uv[:, 0].long(), uv[:, 1].long()
        valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hit = torch.zeros_like(valid)
        hit[valid] = m[v[valid], u[valid]]
        seen += valid.float()
        inside += hit.float()
    ratio = inside / seen.clamp(min=1)
    keep = (seen >= min_views) & (ratio > threshold)
    stats = {"primitives": int(xyz.shape[0]), "kept": int(keep.sum()), "views": len(masks),
             "threshold": threshold, "min_views": min_views}
    return keep, stats


def ground_plane(xyz: np.ndarray, keep: np.ndarray, images: dict, percentile: float = 2.0) -> np.ndarray:
    """Plane (n, d) with n = scene up (from camera orientations) through the bottom of the kept object.

    The cameras' mean up vector gives the ground normal; the plane sits at a low percentile of the object's
    height along it, i.e. where the tyres meet the ground. Points above it have n.x + d > 0.
    """
    up = -np.mean([im.R[1] for im in images.values()], axis=0)   # OpenCV y points down
    up /= np.linalg.norm(up)
    h = xyz[keep] @ up
    return np.r_[up, -np.percentile(h, percentile)]


def load_dataset(dataset: Path):
    cams, imgs, _, _ = cio.read_model(dataset / "sparse" / "0")
    return cams, imgs
