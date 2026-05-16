"""Compute MHCLG-Extract-style placement metric on v21.

For each case, pass-fail is: centroid_distance_m <= 0.10 * GT_diameter.

We report the metric under three "diameter" definitions because MHCLG don't
specify which they use:
  - equiv_diameter: sqrt(4 * area / pi)  (a "circle of equivalent area")
  - bbox_diameter: diagonal of axis-aligned bbox
  - longest_axis: max pairwise vertex distance (proper Feret diameter)

Reads positioning_error_m straight from each metrics.json.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

from shapely.geometry import shape
from pyproj import Transformer

RESULTS = Path("results/benchmark_v21/gemini-flash")
EVAL = Path("evaluation_data")

TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def reproject(geom):
    from shapely.ops import transform
    return transform(lambda x, y, z=None: TO_BNG.transform(x, y), geom)


def diameters_m(gt_geojson_path: Path):
    gj = json.loads(gt_geojson_path.read_text())
    feats = gj.get("features") or [gj]
    geom = shape(feats[0]["geometry"])
    bng = reproject(geom)
    area = bng.area  # m^2
    equiv = math.sqrt(4 * area / math.pi)
    minx, miny, maxx, maxy = bng.bounds
    bbox = math.hypot(maxx - minx, maxy - miny)
    # Feret / longest axis on exterior ring
    coords = []
    if bng.geom_type == "Polygon":
        coords = list(bng.exterior.coords)
    else:
        for p in bng.geoms:
            coords += list(p.exterior.coords)
    longest = 0.0
    if len(coords) <= 200:  # exact
        for i, (x1, y1) in enumerate(coords):
            for x2, y2 in coords[i+1:]:
                d = math.hypot(x2-x1, y2-y1)
                if d > longest:
                    longest = d
    else:  # sample for speed
        step = max(1, len(coords) // 200)
        sampled = coords[::step]
        for i, (x1, y1) in enumerate(sampled):
            for x2, y2 in sampled[i+1:]:
                d = math.hypot(x2-x1, y2-y1)
                if d > longest:
                    longest = d
    return equiv, bbox, longest, area


def main():
    rows = []
    for case_dir in sorted(RESULTS.iterdir()):
        if not case_dir.is_dir():
            continue
        m_path = case_dir / "metrics.json"
        if not m_path.exists():
            rows.append((case_dir.name, None, None, None, None, None, None))
            continue
        m = json.loads(m_path.read_text())
        pe = m.get("positioning_error_m")
        iou = m.get("iou", 0.0) or 0.0

        gt_path = None
        eval_case = EVAL / case_dir.name
        if eval_case.exists():
            gts = list(eval_case.glob("*.geojson"))
            if gts:
                gt_path = gts[0]
        if gt_path is None:
            rows.append((case_dir.name, pe, iou, None, None, None, None))
            continue

        try:
            equiv, bbox, longest, area = diameters_m(gt_path)
        except Exception as e:
            rows.append((case_dir.name, pe, iou, None, None, None, f"err:{e}"))
            continue

        rows.append((case_dir.name, pe, iou, equiv, bbox, longest, area))

    # Summary
    def pct_within(rows, diam_idx):
        eligible = [r for r in rows if r[1] is not None and r[diam_idx] is not None]
        if not eligible:
            return 0.0, 0, 0
        passing = sum(1 for r in eligible if r[1] <= 0.10 * r[diam_idx])
        return 100.0 * passing / len(eligible), passing, len(eligible)

    print(f"{'Diameter def':<22} {'Pass%':>8} {'N pass':>8} {'N total':>8}")
    print("-" * 50)
    for name, idx in [("equiv (sqrt(4A/pi))", 3), ("bbox diagonal", 4),
                      ("longest axis (Feret)", 5)]:
        pct, p, n = pct_within(rows, idx)
        print(f"{name:<22} {pct:>7.1f}% {p:>8d} {n:>8d}")

    # Distribution + correlation with IoU
    print("\n=== Failure mode breakdown ===")
    eligible = [r for r in rows if r[1] is not None and r[3] is not None]
    no_pred = sum(1 for r in rows if r[1] is None)
    print(f"Cases with no centroid (no prediction): {no_pred}")
    print(f"Cases with prediction + GT diameter:    {len(eligible)}")

    # By IoU bucket
    print("\n=== Placement pass-rate by IoU bucket (equiv diameter) ===")
    buckets = [(0.0, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.001)]
    for lo, hi in buckets:
        rs = [r for r in eligible if lo <= r[2] < hi]
        if not rs:
            continue
        passing = sum(1 for r in rs if r[1] <= 0.10 * r[3])
        print(f"  IoU [{lo:.2f}, {hi:.2f}): "
              f"{100*passing/len(rs):5.1f}% pass ({passing}/{len(rs)})")

    # Median / percentiles of (centroid_distance / equiv_diameter)
    ratios = sorted([r[1] / r[3] for r in eligible if r[3] > 0])
    if ratios:
        p10 = ratios[int(0.10 * (len(ratios)-1))]
        p25 = ratios[int(0.25 * (len(ratios)-1))]
        p50 = ratios[int(0.50 * (len(ratios)-1))]
        p75 = ratios[int(0.75 * (len(ratios)-1))]
        p90 = ratios[int(0.90 * (len(ratios)-1))]
        print(f"\n=== centroid_distance / equiv_diameter percentiles ===")
        print(f"  P10={p10:.4f}  P25={p25:.4f}  P50={p50:.4f}  "
              f"P75={p75:.4f}  P90={p90:.4f}")
        print(f"  (Extract claim: 82% at <= 0.10)")

    # Optional CSV
    out_path = Path("results/benchmark_v21/placement_metric.csv")
    with out_path.open("w") as f:
        f.write("case,positioning_error_m,iou,equiv_diam_m,bbox_diam_m,"
                "longest_diam_m,area_m2,pass_equiv,pass_bbox,pass_longest\n")
        for r in rows:
            case, pe, iou, eq, bb, lg, area = r
            pass_eq = (1 if (pe is not None and eq is not None and pe <= 0.1 * eq)
                      else 0)
            pass_bb = (1 if (pe is not None and bb is not None and pe <= 0.1 * bb)
                      else 0)
            pass_lg = (1 if (pe is not None and lg is not None and pe <= 0.1 * lg)
                      else 0)
            f.write(f"{case},{pe},{iou},{eq},{bb},{lg},{area},"
                   f"{pass_eq},{pass_bb},{pass_lg}\n")
    print(f"\nWrote per-case: {out_path}")


if __name__ == "__main__":
    main()
