"""Cut the main object out of a trained splat by multi-view mask voting.

    python segment.py --ply model/point_cloud.ply --dataset runs/car/dataset --out out/car

Writes out/car/object.ply (only the object's primitives, every attribute preserved), the cached masks,
segment.json, and a before/after render from a training view.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from splatseg import model as M
from splatseg import vote as V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="images/ + sparse/0 used to train the model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.8, help="fraction of seeing views that must agree")
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--mask-model", default="u2net", help="rembg model used for 2D masks (u2net, isnet-general-use, birefnet-general, ...)")
    ap.add_argument("--preview-view", type=int, default=0, help="index of the training view for the preview")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cams, imgs = V.load_dataset(args.dataset)
    names = sorted(im.name for im in imgs.values())
    t0 = time.time()
    masks = V.object_masks(args.dataset / "images", names, args.out / "masks", args.mask_model)
    t_masks = time.time() - t0

    model = M.load_model(args.ply)
    t0 = time.time()
    keep, stats = V.vote(model.get_xyz, cams, imgs, masks, args.threshold, args.min_views)
    stats.update({"mask_seconds": round(t_masks, 1), "vote_seconds": round(time.time() - t0, 2)})
    M.save_subset(args.ply, keep.cpu().numpy(), args.out / "object.ply")

    im = sorted(imgs.values(), key=lambda i: i.name)[args.preview_view]
    cam = cams[im.camera_id]
    w, h = cam.width // 2, cam.height // 2
    K = cam.K() * np.array([[0.5], [0.5], [1]])
    before = M.render(model, K, im.R, im.t, w, h)
    after = M.render(model, K, im.R, im.t, w, h, keep)
    cv2.imwrite(str(args.out / "preview.jpg"),
                cv2.cvtColor((np.hstack([before, after]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    (args.out / "segment.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
