"""GT-centroid extraction + nearest-part scoring; shared by locate ablations."""

from __future__ import annotations

from typing import Optional

from geoplanagent.utils import haversine_km
from geoplanagent.metrics import geojson_to_shape


# Canonical per-case scoring CSV schema, shared across harnesses.
# Note: the older ``verified_inside_admin_region`` column was dropped when
# the field was removed from LocatePick (its production value was always
# False since la_check is disabled by default, and the LLM was setting it
# to True regardless — see the ablation hallucination analysis). The
# already-saved CSVs in ablations/locate_only_eval/<config>/locate_picks.csv
# still carry the column; readers should treat it as optional.
CSV_FIELDNAMES = [
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


def add_subset_args(parser) -> None:
    """The case-subset flags every harness shares."""
    parser.add_argument(
        "--only-cases",
        default=None,
        help="Comma-separated case names; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke limit — evaluate only the first N cases.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already in the output CSV.",
    )


def print_err_km_summary(out_csv) -> None:
    """Mean/median err_km aggregate printed at the end of a locate run."""
    import csv

    if not out_csv.exists():
        return
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    errs = sorted(float(r["err_km"]) for r in rows if r.get("err_km"))
    if errs:
        mean = sum(errs) / len(errs)
        median = errs[len(errs) // 2]
        print(
            f"err_km: n={len(errs)}  mean={mean:.2f} km  "
            f"median={median:.2f} km  min={errs[0]:.2f}  "
            f"max={errs[-1]:.2f}",
            flush=True,
        )


def load_annotation_manifest(repo_root):
    """The 211-map annotation manifest, built in-memory from maps/*.png +
    fold_assignment.json (never persisted — it would only drift)."""
    import json
    import sys

    dataset_dir = repo_root / "training" / "dataset"
    fold_assignment_path = dataset_dir / "fold_assignment.json"
    if not fold_assignment_path.exists():
        sys.exit(
            f"fold_assignment.json not found: {fold_assignment_path}. "
            f"Run training/build_sam3_training_set.py first."
        )
    from training.train_sam3_kfold import _build_manifest_from_disk

    fold_map = json.loads(fold_assignment_path.read_text())
    manifest = _build_manifest_from_disk(dataset_dir, fold_map)
    if not manifest:
        sys.exit(
            f"manifest is empty — no .png files found in "
            f"{dataset_dir / 'maps'} matching fold_assignment.json"
        )
    return manifest
