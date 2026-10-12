"""Render cut-out objects around their capture orbit with RaRa ground clipping, as one side-by-side GIF.

    python turntable.py --scene out/car1/object.ply runs/car1/dataset --scene ... --out docs/turntable.gif

Frames come from the training cameras in capture order (evenly subsampled), so every car turns once.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from splatseg import model as M
from splatseg import rara
from splatseg import vote as V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs=2, action="append", metavar=("PLY", "DATASET"), required=True)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=420, help="width of each panel in the GIF")
    ap.add_argument("--mode", choices=["none", "hard", "rara"], default="rara")
    ap.add_argument("--ground-percentile", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    panels = []
    for ply, dataset in args.scene:
        cams, imgs = V.load_dataset(Path(dataset))
        model = M.load_model(Path(ply))
        xyz = model.get_xyz.detach().cpu().numpy()
        plane = V.ground_plane(xyz, np.ones(len(xyz), bool), imgs, args.ground_percentile)
        views = sorted(imgs.values(), key=lambda i: i.name)
        views = [views[k] for k in np.linspace(0, len(views) - 1, args.frames).round().astype(int)]
        frames = []
        for im in views:
            cam = cams[im.camera_id]
            s = args.width / cam.width
            K = cam.K() * np.array([[s], [s], [1]])
            img = rara.render_clipped(model, plane, K, im.R, im.t, args.width, int(cam.height * s), args.mode)
            frames.append((img * 255).astype(np.uint8))
        panels.append(frames)
        del model

    h = min(p[0].shape[0] for p in panels)
    gif = [np.hstack([cv2.resize(p[k], (args.width, h)) for p in panels]) for k in range(args.frames)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, gif, duration=1 / args.fps, loop=0)
    print(f"wrote {args.out} ({len(gif)} frames, {gif[0].shape[1]}x{gif[0].shape[0]})")


if __name__ == "__main__":
    main()
