"""Ablate the SAM3 mask post-processing pipeline.

For each case in the MAX baseline (results/benchmark_v_this_is_the_
MAXIMALLYFINALVERSION/gemini-flash), this script:

  1. Loads the cached affine_H + tile_info (from the original run).
  2. Renders the same map page that the original run used.
  3. Runs SAM3 with the same fold adapter → raw mask.
  4. Projects BOTH the raw mask AND the cleanup-pipelined mask through
     the cached affine.
  5. Reports IoU vs GT for both, side by side.

If the difference is consistently small, the cleanup_mask_pipeline
(_keep_dominant_components → _expand_thin_mask → _fill_mask_holes) can
be ripped — the 5-fold fine-tuned SAM3 produces masks clean enough to
project directly. Pre-finetune, the cleanup was load-bearing; this
test answers whether it still is.

Usage:
  uv run python scripts/test_mask_postprocessing.py [N_CASES]

  N_CASES defaults to 30. Pass 0 to run all available cases.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
# Make `tools.*` importable when running the script from anywhere.
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image
from shapely.geometry import shape

BASELINE = REPO / "results/benchmark_v_this_is_the_MAXIMALLYFINALVERSION/gemini-flash"
EVAL = REPO / "evaluation_data"


def _iou(g1, g2) -> float:
    if g1 is None or g2 is None or g1.is_empty or g2.is_empty:
        return 0.0
    if not g1.is_valid: g1 = g1.buffer(0)
    if not g2.is_valid: g2 = g2.buffer(0)
    i = g1.intersection(g2).area
    u = g1.union(g2).area
    return float(i / u) if u > 0 else 0.0


def _load_geojson(p: Path):
    if not p.exists(): return None
    try:
        d = json.loads(p.read_text())
        if d.get("type") == "Feature": d = d["geometry"]
        elif d.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in d["features"]]
            if not geoms: return None
            from shapely.ops import unary_union
            return unary_union(geoms)
        return shape(d)
    except Exception:
        return None


def _gt_polygon(case_name: str):
    for g in (EVAL / case_name).glob("*.geojson"):
        return _load_geojson(g)
    return None


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    # Imports deferred so help-style invocation doesn't pay the cost.
    from tools.extraction.sam3 import (
        load_sam3_ft, set_fold_for_case, extract_boundary_sam3_semantic,
    )
    from tools.extraction.mask_ops import cleanup_mask_pipeline
    from tools.matching import mask_to_geojson_affine

    # Discover renderer for each case's map page.
    from tools.io.map_page import render_map_page

    sam = load_sam3_ft()

    # Pick the N cases with the highest cached IoU — the cases where
    # cleanup was most likely actually doing nothing useful (the bug-
    # finder is "drop in IoU on cases that previously scored high").
    all_cases = []
    for d in sorted(BASELINE.iterdir()):
        if not d.is_dir(): continue
        mf = d / "metrics.json"
        if not mf.exists(): continue
        try:
            m = json.loads(mf.read_text())
            iou = float(m.get("iou", 0.0) or 0.0)
        except Exception:
            continue
        if iou <= 0.0: continue  # skip failures
        all_cases.append((d.name, iou))
    all_cases.sort(key=lambda x: -x[1])
    if n_cases > 0:
        all_cases = all_cases[:n_cases]

    # Skip cases committed via lookup_district — they don't go through the
    # matcher and have no affine to test.
    all_cases = [(n, i) for (n, i) in all_cases
                 if (BASELINE / n / "affine_H.npy").exists()]

    print(f"Testing {len(all_cases)} cases (sorted by cached IoU; "
          f"district_lookup outcomes skipped)\n")
    print(f"{'case':40s}  {'cached':>7s}  {'cleaned':>8s}  {'raw':>7s}  {'Δ':>7s}")

    results = []
    for case_name, cached_iou in all_cases:
        case_dir = BASELINE / case_name
        try:
            affine_H = np.load(case_dir / "affine_H.npy")
            tile_info = json.loads((case_dir / "tile_info.json").read_text())
            pdf_info = json.loads((case_dir / "pdf_info.json").read_text())
            map_pages = pdf_info.get("map_pages") or [1]
            page = int(map_pages[0])
            pdf_path = next((EVAL / case_name).glob("*.pdf"), None)
            if pdf_path is None:
                print(f"{case_name:40s}  (no PDF)")
                continue

            # Render the same page the original run used. render_map_page
            # uses the keyword `page_1based`, not `page`.
            map_img, _ = render_map_page(
                str(pdf_path), page_1based=page, dpi=200,
                case_name=case_name,
            )

            # Run SAM3 with the same fold this case was evaluated under.
            set_fold_for_case(sam, case_name)

            # Save crop to a temp path because extract_boundary_sam3_semantic
            # expects a file path; reuse the cached map_crop if available.
            import tempfile, cv2
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR))
                raw_mask = extract_boundary_sam3_semantic(
                    tmp.name, sam["processor"], sam["model"], sam["device"],
                )
            if raw_mask is None:
                print(f"{case_name:40s}  (SAM3 returned None)")
                continue

            # Two projections.
            #
            # The "cleaned" branch is mask_to_geojson_affine's current
            # behaviour: it applies _keep_dominant_components +
            # _expand_thin_mask + _fill_mask_holes INTERNALLY before
            # contour extraction.
            #
            # The "raw" branch monkey-patches those three internal
            # cleanup calls to be identity functions, projecting the raw
            # SAM3-FT mask directly through the affine with no
            # morphological cleanup at all.
            from tools.matching import _core as _matching_core
            gj_cln = mask_to_geojson_affine(raw_mask.copy(), affine_H, tile_info)

            _orig_kd = _matching_core._keep_dominant_components
            _orig_ex = _matching_core._expand_thin_mask
            _orig_fh = _matching_core._fill_mask_holes
            _identity = lambda m: m
            try:
                _matching_core._keep_dominant_components = _identity
                _matching_core._expand_thin_mask = _identity
                _matching_core._fill_mask_holes = _identity
                gj_raw = mask_to_geojson_affine(raw_mask.copy(), affine_H, tile_info)
            finally:
                _matching_core._keep_dominant_components = _orig_kd
                _matching_core._expand_thin_mask = _orig_ex
                _matching_core._fill_mask_holes = _orig_fh

            gt = _gt_polygon(case_name)
            poly_raw = shape(gj_raw["geometry"]) if gj_raw else None
            poly_cln = shape(gj_cln["geometry"]) if gj_cln else None
            iou_raw = _iou(poly_raw, gt) if gt else 0.0
            iou_cln = _iou(poly_cln, gt) if gt else 0.0
            delta = iou_raw - iou_cln

            results.append((case_name, cached_iou, iou_cln, iou_raw, delta))
            print(f"{case_name:40s}  {cached_iou:>7.3f}  {iou_cln:>8.3f}  "
                  f"{iou_raw:>7.3f}  {delta:>+7.3f}")
        except Exception as e:
            print(f"{case_name:40s}  ERROR: {e!s:.80}")

    if results:
        mean_cln = sum(r[2] for r in results) / len(results)
        mean_raw = sum(r[3] for r in results) / len(results)
        worst_drop = min(r[4] for r in results)
        worst_drop_case = next(r[0] for r in results if r[4] == worst_drop)
        big_drops = [r for r in results if r[4] < -0.05]
        big_gains = [r for r in results if r[4] > 0.05]

        print()
        print(f"Summary over {len(results)} cases:")
        print(f"  Mean IoU with cleanup:    {mean_cln:.4f}")
        print(f"  Mean IoU raw (no clean):  {mean_raw:.4f}")
        print(f"  Δ mean:                   {mean_raw - mean_cln:+.4f}")
        print(f"  Worst drop:               {worst_drop:+.3f} (case {worst_drop_case})")
        print(f"  Cases dropping >0.05:     {len(big_drops)}")
        print(f"  Cases gaining  >0.05:     {len(big_gains)}")
        if big_drops:
            print()
            print("  Cases where cleanup was load-bearing:")
            for n, c, cl, r, d in sorted(big_drops, key=lambda x: x[4]):
                print(f"    {n:38s}  cleaned={cl:.3f} → raw={r:.3f}  ({d:+.3f})")


if __name__ == "__main__":
    main()
