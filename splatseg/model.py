"""Load, subset and save Deformable Beta Splatting models, and render them with gsplat."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

BETA_REPO = Path(os.environ.get("SPLATSEG_BETA_SPLATTING",
                                Path(__file__).resolve().parent.parent / "third_party" / "beta-splatting"))


def load_model(ply: Path, sb_number: int = 2):
    """A beta-splatting BetaModel loaded from a trained PLY, on the GPU."""
    sys.path.insert(0, str(BETA_REPO))
    from scene.beta_model import BetaModel  # noqa: E402

    model = BetaModel(sh_degree=0, sb_number=sb_number)
    model.load_ply(str(ply))
    model.background = torch.zeros(3, device="cuda")
    return model


def save_subset(src_ply: Path, keep: np.ndarray, dst_ply: Path) -> int:
    """Write the primitives where keep is True, preserving every PLY attribute unchanged."""
    ply = PlyData.read(str(src_ply))
    vertices = ply.elements[0].data[keep]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(dst_ply))
    return int(keep.sum())


def camera_tensors(K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """viewmat [1,4,4] (world->camera) and Ks [1,3,3] as CUDA tensors."""
    view = np.eye(4)
    view[:3, :3], view[:3, 3] = R, t
    return (torch.tensor(view, dtype=torch.float32, device="cuda")[None],
            torch.tensor(K, dtype=torch.float32, device="cuda")[None])


@torch.no_grad()
def render(model, K, R, t, width: int, height: int, keep: torch.Tensor | None = None,
           background=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Render an RGB image (H, W, 3, float in [0,1]) of the primitives in `keep`."""
    from gsplat.rendering import rasterization

    if keep is None:
        keep = torch.ones(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    viewmat, Ks = camera_tensors(K, R, t)
    rgb, _, _ = rasterization(
        means=model.get_xyz[keep], quats=model.get_rotation[keep], scales=model.get_scaling[keep],
        opacities=model.get_opacity.squeeze(-1)[keep], betas=model.get_beta.squeeze(-1)[keep],
        colors=model.get_shs[keep], viewmats=viewmat, Ks=Ks, width=width, height=height,
        backgrounds=torch.tensor([background], dtype=torch.float32, device="cuda"),
        sh_degree=model.active_sh_degree, sb_number=model.sb_number, sb_params=model.get_sb_params[keep],
        packed=False)
    return rgb[0].clamp(0, 1).cpu().numpy()
