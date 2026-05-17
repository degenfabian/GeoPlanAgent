"""Hold the affine fixed; swap the SAM mask across candidate pages.

For each of the 22 multi-page cases:
  - load v_postrot's saved affine_H + tile_info (one MINIMA result per case)
  - for each candidate map_page from the reader, render + SAM3-mask it
  - project EACH page's mask through the same affine_H → GeoJSON
  - compute IoU vs the ground-truth polygon
  - report which page's mask, projected through the v_postrot affine,
    yields the highest IoU

This is the "is the mask the bottleneck or is the position?" test.
No MINIMA reruns — exactly one affine per case.
"""

from __future__ import annotations

import json
import sys
import tempfile
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
from tools.matching import mask_to_geojson_affine
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


def main():
    sam = load_sam3_ft()
    processor, model, device = sam["processor"], sam["model"], sam["device"]

    rows = []
    for case in CASES_22:
        cdir = BENCH / case
        if not cdir.exists():
            print(f"[skip] {case}: no benchmark dir")
            continue
        try:
            pdf_info = json.loads((cdir / "pdf_info.json").read_text())
            metrics = json.loads((cdir / "metrics.json").read_text())
            affine_H = np.load(cdir / "affine_H.npy")
            tile_info = json.loads((cdir / "tile_info.json").read_text())
        except Exception as e:
            print(f"[skip] {case}: {e}")
            continue
        map_pages = pdf_info.get("map_pages") or []
        details = {d["page"]: d for d in (pdf_info.get("map_page_details") or [])}
        used_iou = metrics.get("iou") or metrics.get("iou_polygon")

        # used page = last render_page or reader's #1
        used_page = map_pages[0]
        for msg in json.loads((cdir / "message_log.json").read_text()):
            if msg.get("kind") == "ToolCallPart" and msg.get("tool") == "render_page":
                args = msg.get("args") or {}
                if isinstance(args, dict) and "page" in args:
                    used_page = int(args["page"])

        pdf = _find_pdf(case)
        gt_path = _find_gt(case)
        if pdf is None or gt_path is None:
            print(f"[skip] {case}: PDF/GT missing")
            continue
        gt = geojson_to_shape(load_geojson(str(gt_path)))
        if gt is None:
            print(f"[skip] {case}: GT not parseable")
            continue

        set_fold_for_case(sam, case)
        print(f"\n=== {case}  pages={map_pages}  used_p={used_page}  used_iou={used_iou} ===")

        for rank, p in enumerate(map_pages, 1):
            rendered = render_map_page(str(pdf), p, dpi=200, verbose=False,
                                          case_name=case)
            if rendered is None:
                print(f"   p{p}: render failed")
                continue
            map_img, _ = rendered
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cv2.imwrite(tmp.name, map_img)
                mask_path = tmp.name
            mask = extract_boundary_sam3_semantic(
                mask_path, processor, model, device, query="planning boundary")
            if mask is None:
                print(f"   p{p}: no mask")
                continue

            geo = mask_to_geojson_affine(mask, affine_H, tile_info)
            iou = None
            if geo is not None:
                pred = geojson_to_shape(geo)
                if pred is not None:
                    iou = calculate_iou(gt, pred)

            row = {
                "case": case, "page": p, "rank": rank,
                "role": (details.get(p) or {}).get("role", ""),
                "used": p == used_page,
                "used_iou": used_iou,
                "iou_with_saved_affine": iou,
                "mask_h": int(mask.shape[0]),
                "mask_w": int(mask.shape[1]),
                "mask_pct": float((mask > 0).mean()) * 100.0,
            }
            rows.append(row)
            mark = " USED 🏆" if p == used_page else ""
            iou_s = f"{iou:.3f}" if iou is not None else "N/A"
            print(f"   p{p:>3d} ({rank}, {(details.get(p) or {}).get('role',''):8s}) "
                  f"mask={row['mask_pct']:5.2f}%  size={row['mask_w']}x{row['mask_h']}  "
                  f"iou_swapped={iou_s}{mark}")

    out = Path("analysis/multi_map_pages/mask_swap_at_fixed_affine.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
