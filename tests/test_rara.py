"""RaRa kernel checks on a real trained model.

    SPLATSEG_TEST_PLY=path/to/point_cloud.ply SPLATSEG_TEST_DATASET=path/to/dataset python -m pytest tests/test_rara.py

1. With a plane that cuts nothing, the RaRa kernel must reproduce gsplat's own rasterizer exactly.
2. The per-ray visible-chord weight matches a brute-force CPU reference (dense sampling along the ray).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splatseg import colmap_io as cio  # noqa: E402
from splatseg import model as M  # noqa: E402
from splatseg import rara  # noqa: E402

PLY = os.environ.get("SPLATSEG_TEST_PLY")
DATASET = os.environ.get("SPLATSEG_TEST_DATASET")


def first_camera(dataset):
    cams, imgs, _, _ = cio.read_model(Path(dataset) / "sparse" / "0")
    cam = next(iter(cams.values()))
    im = sorted(imgs.values(), key=lambda i: i.name)[0]
    s = 0.5   # half resolution keeps the test fast
    K = cam.K() * np.array([[s], [s], [1]])
    return K, im.R, im.t, int(cam.width * s), int(cam.height * s)


@pytest.mark.skipif(not (PLY and DATASET) or not torch.cuda.is_available(), reason="needs a model, dataset and GPU")
def test_no_cut_matches_gsplat():
    model = M.load_model(Path(PLY))
    K, R, t, w, h = first_camera(DATASET)
    far = np.array([0, 0, 1, 1e6])   # every primitive is far on the visible side: nothing is cut
    ref = M.render(model, K, R, t, w, h)
    out = rara.render_clipped(model, far, K, R, t, w, h, mode="rara")
    assert np.abs(ref - out).max() < 1e-5


def chord_weight_reference(minv, mu, e, d, plane, samples=200_000):
    """Visible fraction of the chord through the unit-support ellipsoid, by dense sampling."""
    ts = np.linspace(-50, 50, samples)
    pts = e[None] + ts[:, None] * d[None]
    inside = np.linalg.norm((pts - mu) @ minv.T, axis=1) < 1
    if not inside.any():
        return 1.0
    visible = (pts @ plane[:3] + plane[3]) > 0
    return (inside & visible).sum() / inside.sum()


def chord_weight_formula(minv, mu, e, d, plane):
    o, v = minv @ (e - mu), minv @ d
    A, B, C = v @ v, 2 * o @ v, o @ o - 1
    disc = B * B - 4 * A * C
    if disc <= 0:
        return 1.0
    t1, t2 = (-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)
    s0, s1 = plane[:3] @ e + plane[3], plane[:3] @ d
    tp = -s0 / s1
    vis = max(0.0, t2 - max(t1, tp)) if s1 > 0 else max(0.0, min(t2, tp) - t1)
    return vis / (t2 - t1)


def test_chord_weight_formula():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        Rm = rara.quat_to_rotmat(torch.tensor(q)).numpy()
        s = rng.uniform(0.2, 1.5, 3)
        minv = (Rm.T / s[:, None])
        mu = rng.normal(size=3)
        e = mu + rng.normal(size=3) * 5 + np.array([0, 0, -8])
        target = mu + rng.normal(size=3) * 0.5
        d = (target - e) / np.linalg.norm(target - e)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        plane = np.r_[n, -(n @ (mu + rng.normal(size=3) * 0.3))]
        assert abs(chord_weight_formula(minv, mu, e, d, plane) -
                   chord_weight_reference(minv, mu, e, d, plane)) < 2e-3
