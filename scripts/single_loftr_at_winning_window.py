"""ONE LoFTR forward pass per candidate page, at v_postrot's winning window.

For each multi-page case:
  - reload v_postrot's anchor + zoom + nx/ny + window (the position MINIMA's
    sliding window settled on)
  - fetch the SAME tile grid (deterministic from anchor+zoom+nx/ny)
  - crop the canvas at the winning window position
  - for each candidate map_page:
      * render + SAM3 mask
      * resize map to match the v_postrot zoom/mpp
      * ONE run_minima(resized_map, cropped_window) call — no sliding
      * estimate_affine on the resulting correspondences
      * project the SAM mask through that affine → GeoJSON
      * IoU vs GT

Outputs per case which page yields highest MINIMA inliers/score and
whether that's the page v_postrot actually used.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.extraction.sam3 import (
    extract_boundary_sam3_semantic,
    load_sam3_ft,
    set_fold_for_case,
)
from tools.io.map_page import render_map_page
from tools.io.os_tiles import fetch_os_opendata_grid
from tools.matching import (
    load_minima,
    run_minima,
    estimate_affine,
    mask_to_geojson_affine,
)
from tools.matching._core import (
    compute_map_mpp,
    resize_map_to_match_zoom,
    _build_scale_H,
)
from tools.metrics.geojson import (
    load_geojson,
    geojson_to_shape,
    calculate_iou,
)


CASES_22 = [
    "A4D4A1",
    "2CCA14C4-32BC-443D-9148-22DFE30A3DAC",
    "1D1A9561-7534-409B-9937-29C887F66069",
    "Ar4.9",
    "Ar4.6",
    "12:00126:ART4",
    "A4D-04",
    "Ar4.30",
    "12:00121:ART4",
    "CPA4(2a)",
    "A108P",
    "A4D8A_merged",
    "A4aSHP1",
    "Ar4.4",
    "23:53149:ART4",
    "A4D-24",
    "Ar4.17",
    "Ar4.2",
    "A4D9",
    "85",
    "A4D6A_merged",
    "25",
]

BENCH = Path("results/benchmark_v_postrot/gemini-flash")
EVAL = Path("evaluation_data")


def _find_pdf(case: str):
    safe = case.replace(":", "_").replace("/", "_")
    for variant in (case, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".pdf":
                    return f
    return None


def _find_gt(case: str):
    safe = case.replace(":", "_").replace("/", "_")
    for variant in (case, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".geojson":
                    return f
    return None


def _parse_scale_ratio(s):
    if not s:
        return None
    import re
    m = re.search(r"1\s*[:/]\s*([\d,]+)", str(s))
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def main():
    sam = load_sam3_ft()
    processor, model, device = sam["processor"], sam["model"], sam["device"]

    print("Loading MINIMA matcher...")
    minima = load_minima()

    rows = []
    for case in CASES_22:
        cdir = BENCH / case
        if not cdir.exists():
            print(f"[skip] {case}: no benchmark dir")
            continue
        try:
            metrics = json.loads((cdir / "metrics.json").read_text())
            pdf_info = json.loads((cdir / "pdf_info.json").read_text())
            tile_info_saved = json.loads((cdir / "tile_info.json").read_text())
        except Exception as e:
            print(f"[skip] {case}: load failed ({e})")
            continue

        mi = metrics.get("match_info") or {}
        anchor = mi.get("anchor_latlon")
        center_ll = mi.get("center_latlon")
        if not anchor:
            anchor = center_ll
        if not anchor or len(anchor) != 2:
            print(f"[skip] {case}: no anchor")
            continue
        anchor_lat, anchor_lon = float(anchor[0]), float(anchor[1])

        zoom = mi.get("zoom") or tile_info_saved.get("zoom")
        nx = tile_info_saved.get("nx", 5)
        ny = tile_info_saved.get("ny", 5)
        win_wx, win_wy = (mi.get("window") or [None, None])
        sf = mi.get("scale_factor")  # resize factor applied to map
        scale_ratio = (_parse_scale_ratio(pdf_info.get("scale"))
                       or mi.get("scale_ratio"))

        used_iou = metrics.get("iou") or metrics.get("iou_polygon")
        map_pages = pdf_info.get("map_pages") or []
        details = {d["page"]: d for d in (pdf_info.get("map_page_details") or [])}
        used_page = map_pages[0] if map_pages else None
        for msg in json.loads((cdir / "message_log.json").read_text()):
            if msg.get("kind") == "ToolCallPart" and msg.get("tool") == "render_page":
                args = msg.get("args") or {}
                if isinstance(args, dict) and "page" in args:
                    used_page = int(args["page"])

        pdf = _find_pdf(case)
        gt_path = _find_gt(case)
        if pdf is None or gt_path is None or zoom is None or win_wx is None:
            print(f"[skip] {case}: missing pdf/gt/zoom/window")
            continue
        gt = geojson_to_shape(load_geojson(str(gt_path)))
        if gt is None:
            print(f"[skip] {case}: GT not parseable")
            continue

        # Fetch the SAME tile grid v_postrot used (deterministic at same anchor+zoom+nx+ny).
        try:
            tile_info_live = fetch_os_opendata_grid(
                anchor_lat, anchor_lon, zoom, nx, ny)
        except Exception as e:
            print(f"[skip] {case}: tile fetch failed ({e})")
            continue
        canvas_rgb = tile_info_live["image"]
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        canvas_h, canvas_w = canvas_bgr.shape[:2]
        # Sanity check: tile grid should match the saved one
        if (tile_info_live["tx_min"] != tile_info_saved["tx_min"]
                or tile_info_live["ty_min"] != tile_info_saved["ty_min"]):
            print(f"  WARN {case}: tile_info drifted from saved "
                  f"(saved tx={tile_info_saved['tx_min']}, ty={tile_info_saved['ty_min']}; "
                  f"live tx={tile_info_live['tx_min']}, ty={tile_info_live['ty_min']})")

        # Compute map_mpp + winning window crop size from sf
        map_mpp = compute_map_mpp(scale_ratio, 200) if scale_ratio else None
        if map_mpp is None:
            # Fall back to deriving rh, rw from sf and the first page's shape later
            pass

        set_fold_for_case(sam, case)
        print(f"\n=== {case}  pages={map_pages}  used_p={used_page}  used_iou={used_iou}  "
              f"win=({win_wx},{win_wy}) zoom={zoom} sf={sf} ===")

        per_page = []
        for rank, p in enumerate(map_pages, 1):
            try:
                rendered = render_map_page(str(pdf), p, dpi=200, verbose=False,
                                              case_name=case)
                if rendered is None:
                    print(f"   p{p}: render failed")
                    continue
                map_img, _ = rendered

                # SAM mask
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cv2.imwrite(tmp.name, map_img)
                    mask_path = tmp.name
                mask = extract_boundary_sam3_semantic(
                    mask_path, processor, model, device, query="planning boundary")
                if mask is None:
                    print(f"   p{p}: SAM3 returned no mask")
                    continue

                # Resize map the same way sliding_window_position would
                if map_mpp is not None:
                    resized_map, sf_used = resize_map_to_match_zoom(
                        map_img, map_mpp, zoom, anchor_lat)
                    if resized_map is None:
                        print(f"   p{p}: resize failed (map > canvas at this zoom)")
                        continue
                else:
                    # Use the saved sf
                    if not sf:
                        print(f"   p{p}: no sf and no scale; skip")
                        continue
                    new_w = max(1, int(round(map_img.shape[1] * sf)))
                    new_h = max(1, int(round(map_img.shape[0] * sf)))
                    resized_map = cv2.resize(map_img, (new_w, new_h))
                    sf_used = sf

                rh, rw = resized_map.shape[:2]
                # Crop the same window v_postrot won at
                if (win_wx + rw > canvas_w) or (win_wy + rh > canvas_h):
                    # Clamp if slightly off the canvas (rare; size drift across runs)
                    rw_c = min(rw, canvas_w - win_wx)
                    rh_c = min(rh, canvas_h - win_wy)
                    if rw_c <= 32 or rh_c <= 32:
                        print(f"   p{p}: page doesn't fit at winning window "
                              f"(rh={rh},rw={rw}, canvas={canvas_h}x{canvas_w})")
                        continue
                    window = canvas_bgr[win_wy:win_wy + rh_c, win_wx:win_wx + rw_c]
                    resized_map = resized_map[:rh_c, :rw_c]
                    rh, rw = rh_c, rw_c
                else:
                    window = canvas_bgr[win_wy:win_wy + rh, win_wx:win_wx + rw]

                # ONE LoFTR forward pass
                mkpts0, mkpts1, mconf = run_minima(
                    minima, resized_map, window, grayscale=False)
                affine_H_resized, n_inliers, score, _mask_inl = estimate_affine(
                    mkpts0, mkpts1, mconf=mconf)

                affine_H = None
                geo = None
                iou = None
                if affine_H_resized is not None:
                    # Promote (resized_map → window) → (original_map → canvas)
                    affine_H = _build_scale_H(affine_H_resized, win_wx, win_wy, sf_used)
                    geo = mask_to_geojson_affine(mask, affine_H, tile_info_live)
                    if geo is not None:
                        pred = geojson_to_shape(geo)
                        if pred is not None:
                            iou = calculate_iou(gt, pred)

                row = {
                    "case": case, "page": p, "rank": rank,
                    "role": (details.get(p) or {}).get("role", ""),
                    "used": p == used_page,
                    "used_iou": used_iou,
                    "n_inliers": int(n_inliers or 0),
                    "score": float(score or 0.0),
                    "n_matches": int(len(mkpts0)),
                    "mean_conf": float(np.mean(mconf)) if len(mconf) else 0.0,
                    "iou": iou,
                    "sf_used": sf_used,
                    "resized_shape": [rh, rw],
                }
                per_page.append(row)
                mark = " USED 🏆" if p == used_page else ""
                iou_s = f"{iou:.3f}" if iou is not None else "  N/A"
                print(f"   p{p:>3d} ({rank}, {(details.get(p) or {}).get('role',''):8s}) "
                      f"matches={row['n_matches']:>4d}  inliers={row['n_inliers']:>3d}  "
                      f"score={row['score']:6.2f}  iou={iou_s}{mark}")
            except Exception as e:
                print(f"   p{p}: ERROR {e}")
                traceback.print_exc()

        rows.extend(per_page)

        if per_page:
            best_by_score = max(per_page, key=lambda r: (r["score"], r["n_inliers"]))
            used_row = next((r for r in per_page if r["used"]), None)
            print(f"   → best_by_score: p{best_by_score['page']}  "
                  f"iou={best_by_score['iou']}  (used p{used_page} "
                  f"iou={used_row['iou'] if used_row else 'N/A'})")

    out = Path("analysis/multi_map_pages/single_loftr_at_winning_window.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
