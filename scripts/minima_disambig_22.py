"""MINIMA-disambiguation upper-bound test on 22 multi-page cases.

For each multi-page case, for each candidate map_page from the reader:
  1) render the page (auto-rotated)
  2) compute SAM3 mask (k-fold adapter routed by case name)
  3) run MINIMA's sliding_window_position at the locate-anchor that
     the actual v_postrot worker committed against
  4) project the mask through MINIMA's affine → GeoJSON
  5) compute IoU vs the ground-truth polygon

Then pick the page with the best MINIMA stats (overall_score from
compute_match_reward, falling back to n_inliers) and compare its IoU
to what v_postrot actually returned.

This is an offline upper-bound — running match_at on every candidate
is too expensive for production, but the experiment tells us how much
ceiling we leave on the table by not doing it.

No API/LLM calls — purely local MINIMA + SAM3 + IoU.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.extraction.sam3 import (
    extract_boundary_sam3_semantic,
    load_sam3_ft,
    set_fold_for_case,
)
from tools.io.map_page import render_map_page
from tools.matching import (
    load_minima,
    sliding_window_position,
    mask_to_geojson_affine,
)
from tools.matching.source_priorities import sigma_from_scale
from tools.metrics.reward import compute_match_reward
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


def _find_pdf(case: str) -> Path | None:
    safe = case.replace(":", "_").replace("/", "_")
    for variant in (case, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".pdf":
                    return f
    return None


def _find_gt(case: str) -> Path | None:
    safe = case.replace(":", "_").replace("/", "_")
    for variant in (case, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".geojson":
                    return f
    return None


def _parse_scale_ratio(s) -> int | None:
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


def _used_anchor(case_dir: Path) -> tuple[float, float] | None:
    m = json.loads((case_dir / "metrics.json").read_text())
    mi = m.get("match_info") or {}
    a = mi.get("anchor_latlon")
    if a and len(a) == 2:
        return float(a[0]), float(a[1])
    a = mi.get("center_latlon")
    if a and len(a) == 2:
        return float(a[0]), float(a[1])
    return None


def _used_iou(case_dir: Path) -> float | None:
    m = json.loads((case_dir / "metrics.json").read_text())
    return m.get("iou") or m.get("iou_polygon")


def _committed_sigma(case_dir: Path) -> float | None:
    """Recover the locate-pick sigma the worker used at commit time, if
    we can match anchor_latlon to one of the propose_centers candidates.
    Else None and the caller falls back to sigma_from_scale.
    """
    m = json.loads((case_dir / "metrics.json").read_text())
    mi = m.get("match_info") or {}
    anchor = mi.get("anchor_latlon")
    if not anchor:
        return None
    log = json.loads((case_dir / "message_log.json").read_text())
    best = None
    for msg in log:
        if msg.get("kind") != "ToolReturnPart":
            continue
        if msg.get("tool") != "propose_centers":
            continue
        ret = msg.get("return") or {}
        for c in ret.get("candidates") or []:
            try:
                d2 = (c["lat"] - anchor[0]) ** 2 + (c["lon"] - anchor[1]) ** 2
            except Exception:
                continue
            if best is None or d2 < best[0]:
                best = (d2, c.get("sigma_m"))
    return float(best[1]) if best and best[1] is not None else None


def main():
    sam = load_sam3_ft()
    processor = sam["processor"]
    model = sam["model"]
    device = sam["device"]

    print("Loading MINIMA matcher...")
    minima = load_minima()

    rows = []
    for case in CASES_22:
        cdir = BENCH / case
        if not cdir.exists():
            print(f"\n[skip] {case}: no benchmark dir")
            continue
        pdf_info = json.loads((cdir / "pdf_info.json").read_text())
        map_pages = pdf_info.get("map_pages") or []
        details = {d["page"]: d for d in (pdf_info.get("map_page_details") or [])}
        if not map_pages:
            continue

        anchor = _used_anchor(cdir)
        if anchor is None:
            print(f"\n[skip] {case}: no anchor in metrics.json")
            continue
        anchor_lat, anchor_lon = anchor
        sigma = _committed_sigma(cdir)
        if sigma is None:
            sr_pi = _parse_scale_ratio(pdf_info.get("scale"))
            sigma = sigma_from_scale(sr_pi)

        scale_ratio = _parse_scale_ratio(pdf_info.get("scale"))
        road_names = pdf_info.get("road_names") or []
        directional = pdf_info.get("directional_modifier")

        pdf = _find_pdf(case)
        gt_path = _find_gt(case)
        if pdf is None or gt_path is None:
            print(f"\n[skip] {case}: PDF or GT missing")
            continue
        gt = geojson_to_shape(load_geojson(str(gt_path)))
        if gt is None:
            print(f"\n[skip] {case}: GT failed to load")
            continue

        used_iou = _used_iou(cdir)
        # Identify the worker's used page (last render_page or reader's #1)
        used_page = map_pages[0]
        for msg in json.loads((cdir / "message_log.json").read_text()):
            if msg.get("kind") == "ToolCallPart" and msg.get("tool") == "render_page":
                args = msg.get("args") or {}
                if isinstance(args, dict) and "page" in args:
                    used_page = int(args["page"])

        set_fold_for_case(sam, case)

        print(f"\n=== {case}  pages={map_pages}  anchor=({anchor_lat:.4f},{anchor_lon:.4f})  σ={sigma:.0f}m  used_p={used_page}  used_iou={used_iou} ===")

        per_page = []
        for rank, p in enumerate(map_pages, 1):
            try:
                rendered = render_map_page(str(pdf), p, dpi=200, verbose=False,
                                              case_name=case)
                if rendered is None:
                    print(f"  p{p}: render failed")
                    continue
                map_img, _ = rendered
                # SAM3 mask
                import tempfile, cv2 as _cv2
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    _cv2.imwrite(tmp.name, map_img)
                    mask_path = tmp.name
                mask = extract_boundary_sam3_semantic(
                    mask_path, processor, model, device,
                    query="planning boundary",
                )
                if mask is None:
                    print(f"  p{p}: SAM3 returned no mask")
                    continue

                # MINIMA at fixed anchor
                centers = [(f"p{p}", anchor_lat, anchor_lon, float(sigma))]
                res = sliding_window_position(
                    matcher=minima, map_img=map_img, sam3_mask=mask,
                    centers=centers, scale_ratio=scale_ratio, dpi=200,
                    rotations=None, road_names=road_names, grayscale=False,
                    directional_modifier=directional,
                )
                if not res or res.get("affine_H") is None:
                    print(f"  p{p}: MINIMA no match")
                    continue
                mi = res.get("match_info") or {}
                # Project this page's mask through the affine
                geo = res.get("geojson")
                if geo is None and mask is not None and res["affine_H"] is not None:
                    geo = mask_to_geojson_affine(
                        mask, res["affine_H"], res["tile_info"])
                # Reward score
                reward = compute_match_reward(
                    match_info=mi, pdf_info=pdf_info, inlier_pts_in_map=None,
                    map_shape_hw=tuple(map_img.shape[:2]),
                )
                # IoU vs GT
                iou = None
                if geo is not None:
                    pred_shape = geojson_to_shape(geo)
                    if pred_shape is not None:
                        iou = calculate_iou(gt, pred_shape)

                row = {
                    "case": case, "page": p, "rank": rank,
                    "role": (details.get(p) or {}).get("role", ""),
                    "used": p == used_page,
                    "used_iou": used_iou,
                    "n_inliers": int(mi.get("n_inliers", 0)),
                    "score": float(mi.get("score", 0)),
                    "overall_score": float(reward.overall_score),
                    "iou": iou,
                    "mask_pct": float(mask.sum() / mask.size / 255 * 100),
                }
                per_page.append(row)
                mark = " USED" if p == used_page else ""
                print(f"  p{p:>3d} ({rank}, {(details.get(p) or {}).get('role',''):8s}) "
                      f"inliers={row['n_inliers']:>3d}  score={row['score']:6.2f}  "
                      f"overall={row['overall_score']:.2f}  iou={iou if iou is not None else 'N/A':>6}{mark}")
            except Exception as e:
                print(f"  p{p}: ERROR {e}")
                traceback.print_exc()

        # Aggregate per-case
        if per_page:
            # Best by overall_score (primary), tie-break by n_inliers
            best = max(per_page, key=lambda r: (r["overall_score"], r["n_inliers"]))
            best_by_inliers = max(per_page, key=lambda r: r["n_inliers"])
            used_row = next((r for r in per_page if r["used"]), None)
            print(f"  → best_by_overall: p{best['page']}  iou={best['iou']}  "
                  f"(used p{used_page} iou={used_row['iou'] if used_row else 'N/A'})")
            rows.extend(per_page)

    out = Path("analysis/multi_map_pages/minima_disambig_22.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
