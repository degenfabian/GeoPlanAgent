"""Shared utilities for the LoFTR-MegaDepth probe.

Loader for both matchers (MINIMA and stock LoFTR-MegaDepth) using the
MINIMA codebase — same architecture, different weights file. The MINIMA
loader's `load_loftr` switches config based on the ckpt filename:
  - 'outdoor_ds.ckpt' → LoFTR-MegaDepth default config
  - anything else      → MINIMA's cross-modal config
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parent.parent.parent
PAIRS_DIR = REPO / "experiments" / "loftr_probe" / "pairs"
OUTPUTS_DIR = REPO / "experiments" / "loftr_probe" / "outputs"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MINIMA"))


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_matcher(ckpt_filename: str):
    """Load LoFTR-architecture matcher with the given weights.

    Returns the matcher's `.from_cv_imgs` bound method (callable as
    `matcher(map_bgr, tile_bgr)`), matching the convention used by
    `tools.matching.load_minima`. Sets the cwd to MINIMA/ during the load
    because the MINIMA codebase has hardcoded relative imports.

    Filenames:
      'outdoor_ds.ckpt'   → stock LoFTR-MegaDepth (pretrained, public).
      'minima_loftr.ckpt' → production MINIMA cross-modal fine-tune.
      anything else       → custom checkpoint (e.g. our fine-tune output).
    """
    import os
    from MINIMA.load_model import load_model  # noqa: E402

    ckpt_path = REPO / "MINIMA" / "weights" / ckpt_filename
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing {ckpt_path}. See experiments/loftr_probe/README.md "
            "for download instructions.")
    args = SimpleNamespace(
        ckpt=str(ckpt_path),
        choose_model="loftr",
        thr=0.2,
        gray=False,
    )
    minima_dir = REPO / "MINIMA"
    prev_cwd = os.getcwd()
    try:
        os.chdir(minima_dir)
        return load_model(
            "loftr", args, use_path=False,
            test_orginal_megadepth=(ckpt_filename == "outdoor_ds.ckpt"),
        )
    finally:
        os.chdir(prev_cwd)


def load_matcher_with_module(ckpt_filename: str):
    """Like `load_matcher` but also returns the underlying torch module
    (for fine-tuning — need access to `.parameters()` and `.train()`).

    Returns (callable_for_cv_imgs, underlying_loftr_module).
    """
    import os
    from MINIMA.load_model import load_loftr  # noqa: E402

    ckpt_path = REPO / "MINIMA" / "weights" / ckpt_filename
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}.")
    args = SimpleNamespace(ckpt=str(ckpt_path), choose_model="loftr",
                           thr=0.2, gray=False)
    minima_dir = REPO / "MINIMA"
    prev_cwd = os.getcwd()
    try:
        os.chdir(minima_dir)
        wrapper = load_loftr(
            args, test_orginal_megadepth=(ckpt_filename == "outdoor_ds.ckpt"))
    finally:
        os.chdir(prev_cwd)
    # DataIOWrapper exposes .from_cv_imgs as the callable; .model is the
    # underlying LoFTR module.
    return wrapper.from_cv_imgs, wrapper.model


def correspondences_from_affine(affine_H, n_pts: int, map_w: int, map_h: int,
                                  rng=None):
    """Sample `n_pts` keypoints inside the map and project them to tile
    coordinates via the ground-truth 2x3 affine.

    Returns (kp_map [N,2], kp_tile [N,2]) — both float arrays.
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(42)
    # Sample inside a margin so projected points have a chance to be inside
    # the tile canvas (matters for the loss).
    margin = 32
    xs = rng.uniform(margin, map_w - margin, size=n_pts)
    ys = rng.uniform(margin, map_h - margin, size=n_pts)
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=1)  # (N, 3)
    proj = pts @ affine_H.T                              # (N, 2)
    return np.stack([xs, ys], axis=1), proj


def load_image_bgr(p: Path):
    """OpenCV imread but raises if missing — we want hard failures."""
    import cv2
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f"Could not load {p}")
    return img
