"""Compute MHCLG-Extract-style placement metric on v3 AND v21.

Pass-fail per case: distance(pred_centroid, gt_centroid) <= 0.10 * GT_diameter.

Computes BOTH centroids fresh from predicted.geojson (avoids the broken
positioning_error_m field in metrics.json).

Three diameter definitions reported (MHCLG doesn't specify which they use):
  - equiv_diameter: sqrt(4 * area / pi)
  - bbox_diameter: diagonal of axis-aligned bounding box
  - longest_axis: max pairwise vertex distance (Feret diameter)
"""
from __future__ import annotations
import json, math
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
V3 = REPO / "results" / "benchmark_v3" / "gemini-flash"
V21 = REPO / "results" / "benchmark_v21" / "gemini-flash"
EVAL = REPO / "evaluation_data"

TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def to_bng(geom):
    return shp_transform(lambda x, y, z=None: TO_BNG.transform(x, y), geom)


def centroid_bng(geojson_path: Path):
    """Compute the centroid of the predicted polygon in BNG (meters)."""
    if not geojson_path.exists(): return None
    try:
        gj = json.loads(geojson_path.read_text())
    except Exception:
        return None
    feats = gj.get("features") or [gj]
    if not feats: return None
    try:
        geom = shape(feats[0]["geometry"])
        bng = to_bng(geom)
        c = bng.centroid
        return (c.x, c.y)
    except Exception:
        return None


def diameters_and_centroid_bng(gt_path: Path):
    """Returns (equiv, bbox, longest, centroid_xy) all in BNG meters."""
    gj = json.loads(gt_path.read_text())
    feats = gj.get("features") or [gj]
    geom = shape(feats[0]["geometry"])
    bng = to_bng(geom)
    area = bng.area
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
    if len(coords) > 200:
        coords = coords[::max(1, len(coords) // 200)]
    for i, (x1, y1) in enumerate(coords):
        for x2, y2 in coords[i+1:]:
            d = math.hypot(x2-x1, y2-y1)
            if d > longest:
                longest = d
    c = bng.centroid
    return equiv, bbox, longest, (c.x, c.y)


def main():
    rows = []
    cases = sorted(p.name for p in V3.iterdir() if p.is_dir())
    print(f"Computing placement metric for {len(cases)} cases...")
    for i, case in enumerate(cases):
        # Pred centroids
        v3_pred = centroid_bng(V3 / case / "predicted.geojson")
        v21_pred = centroid_bng(V21 / case / "predicted.geojson")
        # GT
        eval_dir = EVAL / case
        gt_path = None
        if eval_dir.exists():
            gts = list(eval_dir.glob("*.geojson"))
            if gts: gt_path = gts[0]
        if gt_path is None:
            rows.append({"case": case}); continue
        try:
            equiv, bbox, longest, gt_c = diameters_and_centroid_bng(gt_path)
        except Exception:
            rows.append({"case": case}); continue

        def dist(p):
            if p is None: return None
            return math.hypot(p[0]-gt_c[0], p[1]-gt_c[1])

        rows.append({
            "case": case,
            "d_v3_m": dist(v3_pred),
            "d_v21_m": dist(v21_pred),
            "equiv_m": equiv,
            "bbox_m": bbox,
            "longest_m": longest,
            "area_m2": math.pi * (equiv/2)**2,
        })
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}")

    # Summary table
    def pct(rows, dist_key, diam_key, threshold=0.10):
        eligible = [r for r in rows if r.get(dist_key) is not None and r.get(diam_key)]
        if not eligible: return 0.0, 0, 0
        p = sum(1 for r in eligible if r[dist_key] <= threshold * r[diam_key])
        return 100.0*p/len(eligible), p, len(eligible)

    print()
    print("═══ MHCLG-Extract placement metric ═══")
    print("Pass = distance(pred_centroid, GT_centroid) <= 0.10 * GT_diameter")
    print()
    print(f"{'Diameter def':<26} {'v3 pass':>10} {'v21 pass':>10} {'Δ':>6}")
    print("-" * 58)
    for label, dk in [
        ("equiv (sqrt(4A/π))", "equiv_m"),
        ("bbox diagonal", "bbox_m"),
        ("longest axis (Feret)", "longest_m"),
    ]:
        p3, n3, t3 = pct(rows, "d_v3_m", dk)
        p21, n21, t21 = pct(rows, "d_v21_m", dk)
        print(f"{label:<26} {p3:>7.1f}% ({n3}/{t3})  {p21:>7.1f}% ({n21}/{t21})  {p3-p21:+5.1f}")

    print()
    print("═══ Pass-rate by tighter thresholds (equiv diameter) ═══")
    print(f"{'Threshold':<12} {'v3 pass':>10} {'v21 pass':>10} {'Δ':>6}")
    print("-" * 44)
    for thr in [0.05, 0.10, 0.20, 0.50, 1.00]:
        p3 = pct(rows, "d_v3_m", "equiv_m", thr)
        p21 = pct(rows, "d_v21_m", "equiv_m", thr)
        print(f"≤{thr:.2f} × diam  {p3[0]:>7.1f}% ({p3[1]}/{p3[2]})  {p21[0]:>7.1f}% ({p21[1]}/{p21[2]})  {p3[0]-p21[0]:+5.1f}")

    # Save
    out = REPO / "results" / "placement_v3_vs_v21.csv"
    with out.open("w") as f:
        f.write("case,d_v3_m,d_v21_m,equiv_m,bbox_m,longest_m,area_m2,"
                "v3_pass_equiv,v21_pass_equiv\n")
        for r in rows:
            d3 = r.get("d_v3_m")
            d21 = r.get("d_v21_m")
            eq = r.get("equiv_m")
            v3p = "1" if (d3 is not None and eq is not None and d3 <= 0.1*eq) else "0"
            v21p = "1" if (d21 is not None and eq is not None and d21 <= 0.1*eq) else "0"
            f.write(f"{r['case']},{d3 or ''},{d21 or ''},{eq or ''},"
                    f"{r.get('bbox_m','') or ''},{r.get('longest_m','') or ''},"
                    f"{r.get('area_m2','') or ''},{v3p},{v21p}\n")
    print(f"\nSaved → {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
