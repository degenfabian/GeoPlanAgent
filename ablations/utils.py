"""Shared ablation utilities: locate-stage GT-centroid scoring plus the pixel-IoU
and per-prompt summary helpers used by the segmentation ablations."""

from typing import List, Optional

import numpy as np

from geoplanagent.utils import haversine_km
from geoplanagent.metrics import geojson_to_shape
from geoplanagent.paths import TRAINING_DATASET_DIR


# Per-case CSV schema shared by the locate-stage harnesses.
LOCATE_PICKS_FIELDNAMES = [
    "case",
    "err_km",
    "picked_lat",
    "picked_lon",
    "picked_source",
    "confidence",
    "sigma_m",
    "n_gt_parts",
    "evidence",
    "error",
]


def gt_part_centroids(gt_geojson: dict) -> list[tuple[float, float]]:
    """Return one (lat, lon) per Polygon part of the GT geometry.

    Multi-area planning documents have MultiPolygon GTs; the first-pick
    scoring takes the MIN haversine distance over part centroids, so a
    multi-area case is scored by whichever component the agent landed
    nearest to.

    Returns an empty list when the geojson can't be parsed or the
    geometry can't be repaired (``geojson_to_shape`` raises).
    """
    try:
        shape = geojson_to_shape(gt_geojson)
    except ValueError:
        shape = None
    if shape is None:
        return []
    polys = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    return [(p.centroid.y, p.centroid.x) for p in polys]


def nearest_part_err_km(
    pick_lat: float,
    pick_lon: float,
    centroids: list[tuple[float, float]],
) -> Optional[float]:
    """Min haversine km from a picked (lat, lon) to any GT-part centroid.

    Returns ``None`` when ``centroids`` is empty (no parsable GT), so
    the caller can record the failure instead of silently scoring 0 km.
    """
    if not centroids:
        return None
    return min(haversine_km(pick_lat, pick_lon, c_lat, c_lon) for c_lat, c_lon in centroids)


def print_err_km_summary(out_csv) -> None:
    """Mean/median err_km aggregate printed at the end of a locate run."""
    import csv

    if not out_csv.exists():
        return
    with open(out_csv) as csv_file:
        rows = list(csv.DictReader(csv_file))
    errs = sorted(float(row["err_km"]) for row in rows if row.get("err_km"))
    if errs:
        mean = sum(errs) / len(errs)
        median = errs[len(errs) // 2]
        print(
            f"err_km: n={len(errs)}  mean={mean:.2f} km  "
            f"median={median:.2f} km  min={errs[0]:.2f}  "
            f"max={errs[-1]:.2f}",
            flush=True,
        )


def load_annotated_pages(repo_root):
    """The 211 annotated map pages to score, built in-memory from maps/*.png +
    fold_assignment.json. Each entry is {case, filename, fold}."""
    import json
    import sys

    dataset_dir = TRAINING_DATASET_DIR
    fold_assignment_path = repo_root / "models" / "fold_assignment.json"
    if not fold_assignment_path.exists():
        sys.exit(
            f"fold_assignment.json not found: {fold_assignment_path}. "
            f"Run training/build_sam3_training_set.py first."
        )
    from training.train_sam3_kfold import _build_manifest_from_disk

    fold_map = json.loads(fold_assignment_path.read_text())
    annotated_pages = _build_manifest_from_disk(dataset_dir, fold_map)
    if not annotated_pages:
        sys.exit(
            f"annotated_pages is empty — no .png files found in "
            f"{dataset_dir / 'maps'} matching fold_assignment.json"
        )
    return annotated_pages


# ----- Pixel-IoU + per-prompt summary helpers (segmentation ablations) -----


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Binary pixel IoU of two HxW masks (any nonzero pixel = foreground; accepts
    0/1 or 0/255). Returns None when the shapes differ (IoU is then undefined)."""
    if pred.shape != gt.shape:
        return None
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return float(inter / union) if union > 0 else 0.0


def summarise(name: str, xs: List[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"name": name, "n": 0}
    sorted_values = sorted(xs)
    return {
        "name": name,
        "n": n,
        "mean": sum(xs) / n,
        "median": sorted_values[n // 2],
        "ge_0.50": sum(1 for x in xs if x >= 0.50) / n,
        "ge_0.70": sum(1 for x in xs if x >= 0.70) / n,
        "ge_0.80": sum(1 for x in xs if x >= 0.80) / n,
        "ge_0.90": sum(1 for x in xs if x >= 0.90) / n,
    }


def print_summary(stats: dict) -> None:
    print(f"\n{stats['name']} (N={stats['n']})")
    if stats["n"] == 0:
        print("  (no cases)")
        return
    print(f"  mean   = {stats['mean']:.4f}")
    print(f"  median = {stats['median']:.4f}")
    print(f"  >=0.50 = {stats['ge_0.50'] * 100:.1f}%")
    print(f"  >=0.70 = {stats['ge_0.70'] * 100:.1f}%")
    print(f"  >=0.80 = {stats['ge_0.80'] * 100:.1f}%")
    print(f"  >=0.90 = {stats['ge_0.90'] * 100:.1f}%")
