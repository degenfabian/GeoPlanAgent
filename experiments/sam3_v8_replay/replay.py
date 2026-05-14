"""Replay SAM3 v8 on v20's cached affines to isolate SAM3 mask improvement.

For each v20 case with a cached affine + GT geojson:
  1. Re-render the planning-map page through the SAME production pipeline
     (render_pdf_page → auto_rotate → detect_title_block_crop).
  2. Set the per-case fold so SAM3 uses the adapter that didn't see this case.
  3. Run extract_boundary_sam3_semantic('planning boundary') on the map.
  4. Project the resulting mask via v20's cached affine_H → GeoJSON polygon.
  5. Compute IoU vs ground truth; compare to v20's stored IoU.

This isolates the mask-quality contribution of SAM3-v8 while holding the
agent's matching decisions (affine, tile_info) constant. It does NOT measure
gains on cases where v20's matching failed because the old SAM3 mask was
unusable — those cases have no cached affine.

Output: experiments/sam3_v8_replay/results.json + per-case CSV.

Easy to delete: rm -rf experiments/sam3_v8_replay/
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))


def render_pdf_page_v20style(pdf_path, page_index, dpi=200):
    """Match v20's exact rendering — uses the page CropBox (default PyMuPDF
    behaviour BEFORE the 2026-05-14 fix that forced MediaBox). Replay must
    use this version because v20's cached affines were computed against
    images of CropBox dimensions.
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
    finally:
        doc.close()
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    bench_dir = REPO / "results" / "benchmark_v20" / "gemini-flash"
    eval_dir = REPO / "evaluation_data"
    out_dir = HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — heavy
    from tools.io.rotation_classifier import auto_rotate
    from tools.io.map_crop import detect_title_block_crop
    from tools.extraction.sam3 import (
        load_sam3_ft, extract_boundary_sam3_semantic, set_fold_for_case,
    )
    from tools.matching import mask_to_geojson_affine
    from tools.metrics.geojson import calculate_spatial_metrics, load_geojson
    # Production post-processing: INSPIRE freehold-parcel snap with 8m
    # tolerance. Skipping this in the replay was systematically penalising
    # v8 on cases where its mask slightly over-extends into neighbouring
    # parcels — production would snap those back.
    from tools.snap.inspire import InspireSnap, la_for_admin_region
    from shapely.geometry import shape as _shape, mapping as _mapping

    # Load SAM3 v8 once (~5-30 sec depending on device)
    print("Loading SAM3 v8 (models/sam3_lora)...")
    t0 = time.time()
    sam_state = load_sam3_ft()
    print(f"  loaded in {time.time()-t0:.1f}s; kind={sam_state.get('kind')}  "
          f"folds={sorted(sam_state.get('available_folds', []))}")
    processor = sam_state["processor"]
    model = sam_state["model"]
    device = sam_state["device"]

    # Find candidate cases: must have affine_H + tile_info + GT + PDF
    cases = sorted(d.name for d in bench_dir.iterdir() if d.is_dir())
    print(f"\nScanning {len(cases)} v20 cases...")

    rows = []
    skip_no_affine = skip_no_gt = skip_no_pdf = skip_render = skip_sam = 0
    for i, case in enumerate(cases, 1):
        cd = bench_dir / case
        if not ((cd / "affine_H.npy").exists() and (cd / "tile_info.json").exists()
                and (cd / "metrics.json").exists()):
            skip_no_affine += 1
            continue

        # GT geojson
        gt_files = list((eval_dir / case).glob("*.geojson"))
        if not gt_files:
            skip_no_gt += 1
            continue
        # PDF
        pdf_files = list((eval_dir / case).glob("*.pdf"))
        if not pdf_files:
            skip_no_pdf += 1
            continue
        # Prefer 'plan' or 'map' in name (mirrors annotate_prerender heuristic)
        pdf_path = next((p for p in pdf_files
                          if any(k in p.name.lower()
                                  for k in ("map", "plan", "direction", "boundary"))),
                         max(pdf_files, key=lambda p: p.stat().st_size))

        # Cached state
        affine_H = np.load(cd / "affine_H.npy")
        tile_info = json.loads((cd / "tile_info.json").read_text())
        v20_metrics = json.loads((cd / "metrics.json").read_text())
        v20_iou = v20_metrics.get("iou")
        try:
            pdf_info = json.loads((cd / "pdf_info.json").read_text())
        except Exception:
            pdf_info = {}

        # Page index — use map_pages[0] from cached pdf_info, fallback 1
        page_idx = (pdf_info.get("map_pages") or [1])[0] - 1

        # Re-render the SAME pipeline the agent used
        try:
            map_img = render_pdf_page_v20style(str(pdf_path), page_index=page_idx, dpi=200)
            try:
                map_img, _ = auto_rotate(map_img, verbose=False)
            except Exception:
                pass
            try:
                cropped, _, _, info = detect_title_block_crop(map_img)
                if info.get("cropped"):
                    map_img = cropped
            except Exception:
                pass
        except Exception as e:
            print(f"  [{i:>3}/{len(cases)}] {case}: render failed ({e!s:.60})")
            skip_render += 1
            continue

        # Set SAM3 fold for this case (k-fold routing)
        set_fold_for_case(sam_state, case)

        # Run SAM3 v8
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, map_img)
            mask = extract_boundary_sam3_semantic(
                tmp_path, processor, model, device,
                query="planning boundary",
            )
        except Exception as e:
            print(f"  [{i:>3}/{len(cases)}] {case}: SAM3 failed ({e!s:.60})")
            skip_sam += 1
            continue
        finally:
            try: os.unlink(tmp_path)
            except: pass

        if mask is None or mask.sum() == 0:
            rows.append({"case": case, "v20_iou": v20_iou,
                          "v8_iou": 0.0, "v8_precision": 0.0,
                          "v8_recall": 0.0, "v8_f1": 0.0,
                          "delta_iou": (0.0 - (v20_iou or 0.0)),
                          "note": "empty mask"})
            print(f"  [{i:>3}/{len(cases)}] {case}: empty mask  "
                  f"v20_iou={v20_iou or 0:.3f}  v8_iou=0.000  Δ={0.0 - (v20_iou or 0):+.3f}")
            continue

        # Project mask → GeoJSON via v20's cached affine
        try:
            pred_geojson = mask_to_geojson_affine(mask, affine_H, tile_info)
        except Exception as e:
            print(f"  [{i:>3}/{len(cases)}] {case}: projection failed ({e!s:.60})")
            skip_sam += 1
            continue

        # Production post-processing: INSPIRE freehold-parcel snap (8m
        # tolerance). Mirrors tools/agent/tools/extract.py project_boundary.
        try:
            admin = (pdf_info.get("admin_region") or "").strip()
            la = la_for_admin_region(admin)
            if la and pred_geojson and "geometry" in pred_geojson:
                pred_geom = _shape(pred_geojson["geometry"])
                if pred_geom.is_valid and not pred_geom.is_empty:
                    snap_obj = InspireSnap([la])
                    snapped = snap_obj.snap_polygon(pred_geom, max_dist_m=8.0)
                    if snapped is not None and not snapped.is_empty \
                            and not snapped.equals(pred_geom):
                        pred_geojson = {
                            "type": "Feature",
                            "properties": pred_geojson.get("properties") or {},
                            "geometry": _mapping(snapped),
                        }
        except Exception:
            pass  # snap is best-effort; skip silently on failure

        # IoU vs GT
        gt = load_geojson(str(gt_files[0]))
        try:
            metrics = calculate_spatial_metrics(gt, pred_geojson)
        except Exception as e:
            print(f"  [{i:>3}/{len(cases)}] {case}: metrics failed ({e!s:.60})")
            continue
        v8_iou = float(metrics.get("iou", 0.0))
        v8_prec = float(metrics.get("precision", 0.0))
        v8_rec = float(metrics.get("recall", 0.0))
        v8_f1 = float(metrics.get("f1_score", 0.0))

        delta = v8_iou - (v20_iou if v20_iou is not None else 0.0)
        rows.append({
            "case": case, "v20_iou": v20_iou,
            "v8_iou": v8_iou, "v8_precision": v8_prec,
            "v8_recall": v8_rec, "v8_f1": v8_f1,
            "delta_iou": delta,
        })

        marker = "+" if delta > 0.02 else ("-" if delta < -0.02 else "·")
        if i % 5 == 0 or i == len(cases):
            print(f"  [{i:>3}/{len(cases)}] {case}: "
                  f"v20={v20_iou or 0:.3f}  v8={v8_iou:.3f}  Δ={delta:+.3f} {marker}")

    # Summary
    if not rows:
        print("ERROR: no cases evaluated. Check skip counts above.")
        return 1

    v20_arr = np.array([r["v20_iou"] for r in rows if r["v20_iou"] is not None])
    v8_arr = np.array([r["v8_iou"] for r in rows if r["v20_iou"] is not None])
    delta_arr = v8_arr - v20_arr

    def _stat(a):
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75)),
                "n_ge_0.8": int((a >= 0.8).sum()),
                "n_lt_0.5": int((a < 0.5).sum())}

    summary = {
        "n_cases_evaluated": len(rows),
        "n_cases_compared": len(v20_arr),
        "v20": _stat(v20_arr),
        "v8":  _stat(v8_arr),
        "delta_mean": float(delta_arr.mean()),
        "delta_median": float(np.median(delta_arr)),
        "n_improvements_>0.02": int((delta_arr > 0.02).sum()),
        "n_regressions_<-0.02": int((delta_arr < -0.02).sum()),
        "biggest_improvement": float(delta_arr.max()),
        "biggest_regression": float(delta_arr.min()),
        "skipped": {"no_affine": skip_no_affine, "no_gt": skip_no_gt,
                     "no_pdf": skip_no_pdf, "render_fail": skip_render,
                     "sam_fail": skip_sam},
    }
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))

    # CSV for paper
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case", "v20_iou", "v8_iou", "v8_precision", "v8_recall",
            "v8_f1", "delta_iou", "note"])
        w.writeheader()
        for r in rows:
            r.setdefault("note", "")
            w.writerow(r)

    # Print clean table
    print(f"\n{'='*64}")
    print(f"SAM3 v7 (cached in v20) vs SAM3 v8 (this replay)")
    print(f"on {summary['n_cases_compared']} cases with cached affines")
    print(f"{'='*64}")
    print(f"{'':10s}  {'mean':>7s}  {'median':>7s}  {'>=0.8':>6s}  {'<0.5':>5s}")
    print(f"{'v20 (v7)':10s}  {summary['v20']['mean']:7.4f}  "
          f"{summary['v20']['median']:7.4f}  {summary['v20']['n_ge_0.8']:>6d}  "
          f"{summary['v20']['n_lt_0.5']:>5d}")
    print(f"{'v8 replay':10s}  {summary['v8']['mean']:7.4f}  "
          f"{summary['v8']['median']:7.4f}  {summary['v8']['n_ge_0.8']:>6d}  "
          f"{summary['v8']['n_lt_0.5']:>5d}")
    print(f"\nΔ mean IoU:     {summary['delta_mean']:+.4f}")
    print(f"Δ median IoU:   {summary['delta_median']:+.4f}")
    print(f"Cases improved (Δ > +0.02):  {summary['n_improvements_>0.02']}")
    print(f"Cases regressed (Δ < -0.02): {summary['n_regressions_<-0.02']}")
    print(f"Best gain:      {summary['biggest_improvement']:+.4f}")
    print(f"Worst regress:  {summary['biggest_regression']:+.4f}")
    print(f"\nResults:  {out_dir / 'results.json'}")
    print(f"CSV:      {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
