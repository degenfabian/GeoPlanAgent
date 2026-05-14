"""Diagnose why my replay measurement says v8 IoU=0.1 when v7 cached IoU=0.98
but the masks look near-identical.

Tries A4RFa1 with: re-render → run v7 SAM3 → project through cached affine →
compute IoU vs GT. If the result is NOT close to v20's stored 0.979, my
replay has a bug somewhere in the render/project/IoU chain — NOT a v8
problem.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

CASE = "A4RFa1"


def render_pdf_page_v20style(pdf_path, page_index, dpi=200):
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
    cd = REPO / "results" / "benchmark_v20" / "gemini-flash" / CASE
    eval_d = REPO / "evaluation_data" / CASE

    affine_H = np.load(cd / "affine_H.npy")
    tile_info = json.loads((cd / "tile_info.json").read_text())
    pdf_info = json.loads((cd / "pdf_info.json").read_text())
    v20_metrics = json.loads((cd / "metrics.json").read_text())
    v20_iou = v20_metrics.get("iou")

    print(f"=== Diagnosing {CASE} ===")
    print(f"Cached affine_H:\n{affine_H}")
    print(f"Cached tile_info: zoom={tile_info.get('zoom')} "
          f"tx_min={tile_info.get('tx_min')} ty_min={tile_info.get('ty_min')} "
          f"nx={tile_info.get('nx')} ny={tile_info.get('ny')}")
    print(f"v20 stored IoU: {v20_iou:.4f}")
    print()

    # Render the page the same way replay does
    pdf_path = next((eval_d).glob("*.pdf"))
    page_idx = (pdf_info.get("map_pages") or [1])[0] - 1
    map_raw = render_pdf_page_v20style(str(pdf_path), page_idx, dpi=200)
    print(f"Raw render: {map_raw.shape}")

    # Apply auto_rotate + map_crop (replay pipeline)
    from tools.io.rotation_classifier import auto_rotate
    from tools.io.map_crop import detect_title_block_crop
    map_rot, rot_info = auto_rotate(map_raw, verbose=True)
    print(f"After auto_rotate: {map_rot.shape}  rot_info={rot_info}")
    cropped, _, _, crop_info = detect_title_block_crop(map_rot)
    if crop_info.get("cropped"):
        print(f"Crop applied: {crop_info}")
        map_img = cropped
    else:
        map_img = map_rot
    print(f"Final image to SAM: {map_img.shape}")

    # Run v7 SAM3
    from tools.extraction.sam3 import (
        load_sam3_ft, extract_boundary_sam3_semantic, set_fold_for_case,
    )
    state = load_sam3_ft(kfold_dir="models/sam3_lora_v7_both")
    set_fold_for_case(state, CASE)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        cv2.imwrite(t.name, map_img)
        v7_mask = extract_boundary_sam3_semantic(
            t.name, state["processor"], state["model"], state["device"],
            query="planning boundary",
        )
    print(f"v7 mask shape: {v7_mask.shape if v7_mask is not None else None}  "
          f"coverage: {(v7_mask>0).mean()*100:.2f}% of image")

    # Project mask
    from tools.matching import mask_to_geojson_affine
    pred = mask_to_geojson_affine(v7_mask, affine_H, tile_info)
    print(f"Projected GeoJSON: {pred['geometry']['type'] if pred else None}")
    if pred and "geometry" in pred:
        coords = pred["geometry"].get("coordinates", [])
        # First coord pair
        if pred["geometry"]["type"] == "Polygon" and coords:
            first = coords[0][0] if coords[0] else None
            print(f"  first vertex: {first}")
        elif pred["geometry"]["type"] == "MultiPolygon" and coords:
            first = coords[0][0][0] if coords[0] and coords[0][0] else None
            print(f"  first vertex: {first}  n_polys={len(coords)}")

    # GT geojson
    from tools.metrics.geojson import load_geojson, calculate_spatial_metrics
    gt_path = next(eval_d.glob("*.geojson"))
    gt = load_geojson(str(gt_path))
    if gt and "geometry" in gt:
        gcoords = gt["geometry"].get("coordinates", [])
        if gt["geometry"]["type"] == "Polygon" and gcoords:
            print(f"GT first vertex: {gcoords[0][0]}")
        elif gt["geometry"]["type"] == "MultiPolygon" and gcoords:
            print(f"GT first vertex: {gcoords[0][0][0]}  n_polys={len(gcoords)}")

    metrics = calculate_spatial_metrics(gt, pred)
    print(f"\n=== v7-rerun IoU (this script): {metrics['iou']:.4f} ===")
    print(f"=== v20 stored IoU:              {v20_iou:.4f} ===")
    print(f"=== Δ:                           {metrics['iou'] - v20_iou:+.4f} ===")

    if abs(metrics["iou"] - v20_iou) < 0.05:
        print("\n✓ Reproduction is faithful — v7 here matches v20.")
        print("  → If v8 replay shows much lower IoU, v8's mask must genuinely differ.")
    elif metrics["iou"] < 0.5 and v20_iou > 0.7:
        print("\n✗ Reproduction is BROKEN — v7 here gets much lower IoU than v20.")
        print("  → my replay setup has a bug (render mismatch? missing snap?")
        print("    different IoU computation?)")
    else:
        print("\n? Partial reproduction — investigate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
