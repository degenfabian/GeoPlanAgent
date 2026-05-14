"""Off-the-shelf comparison: LoFTR-MegaDepth (pretrained) vs MINIMA.

For each val pair, run both matchers, RANSAC-fit a 2×3 affine, count inliers,
and measure how close the predicted affine is to the ground-truth affine
(from results/benchmark_v20/). Prints a clear summary.

Interpretation:
  - LoFTR-MegaDepth wins (≥80% pairs): fine-tuning will very likely help; abort
    further MINIMA work and go full-fine-tune.
  - Roughly tied:                       fine-tuning could push LoFTR ahead;
                                        run 3_finetune_quick.py.
  - MINIMA wins (≥80% pairs):           LoFTR-MegaDepth needs more than fine-
                                        tuning to bridge the gap; skip the
                                        direction or try a different base.

Usage:
  uv run python experiments/loftr_probe/2_compare_offshelf.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.loftr_probe._shared import (
    OUTPUTS_DIR, PAIRS_DIR, _device, load_matcher, load_image_bgr,
)
from tools.matching import run_minima, estimate_affine


def _affine_error(pred_H, gt_H, w_map, h_map, n=200, rng=None):
    """Mean L2 reprojection error (in tile pixels) between predicted and GT
    affines, computed over n random keypoints inside the map."""
    if pred_H is None:
        return float("inf")
    if rng is None:
        rng = np.random.default_rng(0)
    xs = rng.uniform(0, w_map, size=n)
    ys = rng.uniform(0, h_map, size=n)
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    err = np.linalg.norm(pts @ pred_H.T - pts @ gt_H.T, axis=1)
    return float(err.mean())


def evaluate_matcher(matcher, label: str, val_pairs):
    """Run `matcher` on every val pair; return per-pair results."""
    rows = []
    for i, p in enumerate(val_pairs, 1):
        case = p.name
        map_img = load_image_bgr(p / "map.png")
        tile_img = load_image_bgr(p / "tile.png")
        gt_H = np.load(p / "affine.npy")

        t0 = time.time()
        try:
            mkpts0, mkpts1, mconf = run_minima(matcher, map_img, tile_img,
                                                 grayscale=False)
        except Exception as e:
            print(f"  [{label}] {case}: run failed ({e!s:.60})")
            rows.append({"case": case, "n_kpts": 0, "n_inliers": 0,
                         "affine_err_px": float("inf"), "wall_s": 0.0})
            continue
        n_kpts = len(mkpts0)
        affine_H, n_inliers = None, 0
        if n_kpts >= 4:
            affine_H, _, n_inliers = estimate_affine(mkpts0, mkpts1, mconf)
        err = _affine_error(affine_H, gt_H, map_img.shape[1], map_img.shape[0])
        wall = time.time() - t0
        rows.append({"case": case, "n_kpts": n_kpts, "n_inliers": int(n_inliers),
                     "affine_err_px": err, "wall_s": round(wall, 2)})
        if i % 5 == 0 or i == len(val_pairs):
            print(f"  [{label}] [{i}/{len(val_pairs)}]")
    return rows


def summarise(label: str, rows):
    if not rows:
        return None
    n_inl = [r["n_inliers"] for r in rows]
    errs = [r["affine_err_px"] for r in rows if r["affine_err_px"] != float("inf")]
    return {
        "matcher": label,
        "n_pairs": len(rows),
        "mean_inliers": float(np.mean(n_inl)),
        "median_inliers": float(np.median(n_inl)),
        "p25_inliers": float(np.percentile(n_inl, 25)),
        "p75_inliers": float(np.percentile(n_inl, 75)),
        "n_low_inlier_pairs": int(sum(1 for v in n_inl if v < 25)),
        "mean_affine_err_px": float(np.mean(errs)) if errs else float("inf"),
        "median_affine_err_px": float(np.median(errs)) if errs else float("inf"),
        "mean_wall_s": float(np.mean([r["wall_s"] for r in rows])),
    }


def main():
    if not PAIRS_DIR.exists() or not (PAIRS_DIR / "_manifest.json").exists():
        print("ERROR: pairs/ not built. Run 1_build_pairs.py first.",
              file=sys.stderr)
        return 1
    manifest = json.loads((PAIRS_DIR / "_manifest.json").read_text())
    val_cases = [m["case"] for m in manifest if m["split"] == "val"]
    val_pairs = [PAIRS_DIR / c for c in val_cases]
    print(f"Val pairs: {len(val_pairs)}")
    if not val_pairs:
        print("ERROR: no val pairs. Did fold 0 end up empty?", file=sys.stderr)
        return 1

    device = _device()
    print(f"Device: {device}")

    print("\n=== Loading LoFTR-MegaDepth (pretrained) ===")
    loftr = load_matcher("outdoor_ds.ckpt")
    loftr_rows = evaluate_matcher(loftr, "LoFTR-MegaDepth", val_pairs)
    del loftr  # free VRAM before loading the next model

    print("\n=== Loading MINIMA ===")
    minima = load_matcher("minima_loftr.ckpt")
    minima_rows = evaluate_matcher(minima, "MINIMA", val_pairs)

    # Per-pair head-to-head
    minima_by_case = {r["case"]: r for r in minima_rows}
    loftr_by_case = {r["case"]: r for r in loftr_rows}
    h2h = {"loftr_wins": 0, "minima_wins": 0, "ties": 0}
    for c in val_cases:
        a = loftr_by_case[c]["n_inliers"]
        b = minima_by_case[c]["n_inliers"]
        if a > b * 1.10:
            h2h["loftr_wins"] += 1
        elif b > a * 1.10:
            h2h["minima_wins"] += 1
        else:
            h2h["ties"] += 1

    summary = {
        "n_val": len(val_pairs),
        "loftr_megadepth": summarise("LoFTR-MegaDepth (pretrained)", loftr_rows),
        "minima": summarise("MINIMA (production)", minima_rows),
        "head_to_head_n_inliers_within_10pct_is_tie": h2h,
        "rows": {"loftr": loftr_rows, "minima": minima_rows},
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "offshelf_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Print human-readable table
    print(f"\n{'='*72}")
    print(f"Off-the-shelf comparison on {len(val_pairs)} val pairs")
    print(f"{'='*72}")
    print(f"{'Matcher':30s}  mean_inl  median_inl  catastrophic(<25)  mean_err_px")
    for s in (summary["loftr_megadepth"], summary["minima"]):
        print(f"  {s['matcher']:28s}  {s['mean_inliers']:>8.1f}  "
              f"{s['median_inliers']:>10.0f}  {s['n_low_inlier_pairs']:>17d}  "
              f"{s['mean_affine_err_px']:>11.1f}")
    print(f"\nHead-to-head (n_inliers, ±10% tolerance for tie):")
    print(f"  LoFTR-MegaDepth wins:  {h2h['loftr_wins']:>3d}")
    print(f"  MINIMA wins:           {h2h['minima_wins']:>3d}")
    print(f"  Tied:                  {h2h['ties']:>3d}")
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
