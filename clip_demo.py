"""Render a splat clipped by a plane: no clip vs hard clip vs RaRa Clipper, side by side.

    python clip_demo.py --ply out/car/object.ply --dataset runs/car/dataset --plane ground --out out/car
    python clip_demo.py --ply model.ply --dataset runs/car/dataset --plane section --out out/car

--plane ground   horizontal plane at the bottom of the object (removes the ground and shadow under it)
--plane section  vertical plane through the object's centre (a cutaway)
--plane a,b,c,d  any plane a*x + b*y + c*z + d = 0 (kept side: > 0)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from splatseg import model as M
from splatseg import rara
from splatseg import vote as V


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = (img * 255).astype(np.uint8).copy()
    cv2.putText(out, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def make_plane(kind: str, model, imgs, ground_percentile: float = 1.0) -> np.ndarray:
    xyz = model.get_xyz.detach().cpu().numpy()
    if kind == "ground":
        return V.ground_plane(xyz, np.ones(len(xyz), bool), imgs, percentile=ground_percentile)
    if kind == "section":
        up = -np.mean([im.R[1] for im in imgs.values()], axis=0)
        up /= np.linalg.norm(up)
        centre = np.median(xyz, axis=0)
        # long axis of the object in the ground plane: principal direction of the points after removing "up"
        flat = (xyz - centre) - np.outer((xyz - centre) @ up, up)
        long_axis = np.linalg.svd(flat[:: max(1, len(flat) // 20000)], full_matrices=False)[2][0]
        n = np.cross(up, long_axis)   # plane contains up and the long axis: a lengthwise cut
        n /= np.linalg.norm(n)
        return np.r_[n, -n @ centre]
    return np.array([float(v) for v in kind.split(",")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--plane", default="ground")
    ap.add_argument("--view", type=int, default=0, help="training view to render from")
    ap.add_argument("--ground-percentile", type=float, default=1.0,
                    help="height percentile of the object used for --plane ground (lower = closer to the floor)")
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution relative to the training images")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cams, imgs = V.load_dataset(args.dataset)
    model = M.load_model(args.ply)
    plane = make_plane(args.plane, model, imgs, args.ground_percentile)
    im = sorted(imgs.values(), key=lambda i: i.name)[args.view]
    cam = cams[im.camera_id]
    s = args.scale
    K = cam.K() * np.array([[s], [s], [1]])
    w, h = int(cam.width * s), int(cam.height * s)

    timings, frames = {}, {}
    for mode in ("none", "hard", "rara"):
        rara.render_clipped(model, plane, K, im.R, im.t, w, h, mode)   # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            frames[mode] = rara.render_clipped(model, plane, K, im.R, im.t, w, h, mode)
        torch.cuda.synchronize()
        timings[mode] = round((time.time() - t0) / 10 * 1000, 2)

    keep, cut = rara.classify(model, plane)
    stats = {"plane": plane.round(5).tolist(), "primitives": int(keep.numel()),
             "visible": int((keep & ~cut).sum()), "cut": int(cut.sum()), "clipped": int((~keep).sum()),
             "ms_per_frame": timings, "resolution": [w, h]}
    sheet = np.hstack([label(frames["none"], "no clip"), label(frames["hard"], "hard clip"),
                       label(frames["rara"], "RaRa clip")])
    tag = args.plane if args.plane in ("ground", "section") else "custom"
    cv2.imwrite(str(args.out / f"clip_{tag}_view{args.view}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (args.out / f"clip_{tag}_view{args.view}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
