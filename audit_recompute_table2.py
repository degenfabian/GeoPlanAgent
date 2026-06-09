"""Audit recompute for paper Table 2 (locate-stage centroid error).

Recomputes median centroid error (m), %<500m, %<1km from raw per-case
artifacts, independently of any cached summaries.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from ablations._shared import gt_part_centroids, nearest_part_err_km
from tools.metrics.geojson import load_geojson, geojson_to_shape


def stats(errs_km: list[float], label: str) -> None:
    a = np.array(errs_km, dtype=float)
    n = len(a)
    med_np = float(np.median(a)) * 1000
    srt = np.sort(a)
    med_upper = float(srt[n // 2]) * 1000  # harness's "errs[len//2]" convention
    mean = float(a.mean()) * 1000
    p500 = float((a < 0.5).mean()) * 100
    p1k = float((a < 1.0).mean()) * 100
    p500le = float((a <= 0.5).mean()) * 100
    p1kle = float((a <= 1.0).mean()) * 100
    print(f"{label:<38} N={n:3d}  median={med_np:8.1f} m (upper-mid={med_upper:.1f})  "
          f"mean={mean:9.1f} m  <500m={p500:5.1f}% (<= {p500le:5.1f}%)  "
          f"<1km={p1k:5.1f}% (<= {p1kle:5.1f}%)")


def csv_errs(path: Path) -> list[float]:
    rows = list(csv.DictReader(open(path)))
    errs = [float(r["err_km"]) for r in rows if r.get("err_km")]
    missing = [r["case"] for r in rows if not r.get("err_km")]
    if missing:
        print(f"  !! {path}: {len(missing)} rows without err_km: {missing}")
    if len(rows) != 208:
        print(f"  !! {path}: {len(rows)} rows (expected 208)")
    return errs


print("=== Locate-only CSVs (nearest-GT-part centroid, haversine) ===")
base = REPO / "ablations/locate_only_eval"
for cfg, label in [
    ("min_1_tool", "Locate (place only, production)"),
    ("full", "Locate + 5 tools (all 6)"),
    ("vlm_direct_gemini-flash", "VLM-direct (Flash)"),
    ("vlm_direct_gemini-pro", "VLM-direct (Pro)"),
]:
    stats(csv_errs(base / cfg / "locate_picks.csv"), label)

print("\n=== GeoPlanAgent full pipeline (predicted.geojson centroid) ===")
res_root = REPO / "results/benchmark_std_post_fix/gemini-flash"
eval_root = REPO / "evaluation_data"
cases = sorted(d.name for d in res_root.iterdir() if d.is_dir())
print(f"result case dirs: {len(cases)}")

for pred_name in ["predicted.geojson", "predicted_worker_first.geojson"]:
    errs_whole = []      # whole predicted-shape centroid -> nearest GT part
    errs_partmin = []    # min over predicted parts x GT parts
    n_missing_pred, n_missing_gt = [], []
    for case in cases:
        pred_path = res_root / case / pred_name
        gt_files = list((eval_root / case).glob("*.geojson"))
        if not gt_files:
            n_missing_gt.append(case)
            continue
        gt = load_geojson(str(gt_files[0]))
        cents = gt_part_centroids(gt)
        if not cents:
            n_missing_gt.append(case)
            continue
        if not pred_path.exists():
            n_missing_pred.append(case)
            continue
        pred = load_geojson(str(pred_path))
        shape = geojson_to_shape(pred)
        if shape is None or shape.is_empty:
            n_missing_pred.append(case)
            continue
        c = shape.centroid
        e = nearest_part_err_km(c.y, c.x, cents)
        errs_whole.append(e)
        parts = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
        e2 = min(nearest_part_err_km(p.centroid.y, p.centroid.x, cents)
                 for p in parts)
        errs_partmin.append(e2)
    print(f"\n{pred_name}: missing/empty pred={len(n_missing_pred)} {n_missing_pred[:6]}, "
          f"missing GT={len(n_missing_gt)} {n_missing_gt[:6]}")
    stats(errs_whole, f"  whole-shape centroid")
    stats(errs_partmin, f"  nearest-part centroid")
    if n_missing_pred:
        # Penalised variant: treat missing predictions as failures (inf error)
        a = np.array(errs_whole + [np.inf] * len(n_missing_pred))
        n = len(a)
        med = float(np.median(a)) * 1000
        p500 = float((a < 0.5).mean()) * 100
        p1k = float((a < 1.0).mean()) * 100
        print(f"  incl. {len(n_missing_pred)} missing as inf:      N={n}  "
              f"median={med:.1f} m  <500m={p500:.1f}%  <1km={p1k:.1f}%")
