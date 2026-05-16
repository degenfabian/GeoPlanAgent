"""Compare SAM3 semantic vs the v3 instance-mode result on the 20 cases
that agent escalated to mode='instance' in benchmark_v3.

For each case:
  1. Read the v3-saved affine_H + tile_info + GT geojson.
  2. Re-render the same map_page (auto-rotate + map_crop applied), exactly
     as agent_tools_render.render_page does.
  3. Run SAM3 semantic with query="planning boundary".
  4. Project the semantic mask through the SAVED affine_H + tile_info.
  5. Compute IoU vs GT.
  6. Print semantic IoU alongside the v3 (instance) IoU for comparison.

No API calls; local SAM3 only.
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.extraction.sam3 import (
    load_sam3_ft,
    extract_boundary_sam3_semantic,
    set_fold_for_case,
)
from tools.io.pdf import render_pdf_page
from tools.io.rotation_classifier import auto_rotate
from tools.io.map_crop import detect_title_block_crop
from tools.matching import mask_to_geojson_affine
from tools.metrics.geojson import calculate_spatial_metrics, load_geojson


INSTANCE_CASES = [
    '095AB379-F04E-473A-BC0D-8948B58E4090', '12:00130:ART4', '12_merged',
    '35046BA6-A370-41C1-8316-8797AF1524DD', '42', 'A003S', 'A005S',
    'A008S', 'A014S', 'A016S', 'A018S', 'A030S', 'A084S', 'A097S',
    'A4D10A1', 'A4DA04', 'A4Da2', 'A4_088:LL:016',
    'DE5A30DA-29A4-45BE-B60A-C201A5F11C6F', 'SSA405',
]

V3_DIR = Path('results/benchmark_v3/gemini-flash')
EVAL_DIR = Path('evaluation_data')


def find_pdf(case_name: str) -> Path | None:
    case_dir = EVAL_DIR / case_name
    if not case_dir.exists():
        return None
    pdfs = list(case_dir.glob('*.pdf'))
    return pdfs[0] if pdfs else None


def find_gt(case_name: str) -> Path | None:
    case_dir = EVAL_DIR / case_name
    if not case_dir.exists():
        return None
    gjs = list(case_dir.glob('*.geojson'))
    return gjs[0] if gjs else None


def render_case_map(pdf_path: Path, page_1based: int, dpi: int = 200):
    """Reproduce render_page's pipeline: render → auto_rotate → title-block crop."""
    img = render_pdf_page(str(pdf_path), page_1based - 1, dpi=dpi)
    try:
        img, _rot = auto_rotate(img, verbose=False)
    except Exception:
        pass
    try:
        cropped, _xo, _yo, _info = detect_title_block_crop(img)
        if _info.get('cropped'):
            img = cropped
    except Exception:
        pass
    return img


def main():
    print(f"Loading SAM3...")
    sam3 = load_sam3_ft()
    print(f"  device={sam3['device']}")

    rows = []
    for case_name in INSTANCE_CASES:
        case_dir = V3_DIR / case_name
        metrics_v3 = json.loads((case_dir / 'metrics.json').read_text())
        iou_v3 = metrics_v3.get('iou')

        aff_p = case_dir / 'affine_H.npy'
        tile_p = case_dir / 'tile_info.json'
        if not aff_p.exists() or not tile_p.exists():
            print(f"{case_name:42s}  SKIP — no saved affine/tile_info")
            rows.append((case_name, iou_v3, None, 'no_affine'))
            continue

        pdf = find_pdf(case_name)
        gt = find_gt(case_name)
        if pdf is None or gt is None:
            print(f"{case_name:42s}  SKIP — no pdf/gt in evaluation_data/")
            rows.append((case_name, iou_v3, None, 'no_pdf'))
            continue

        pi_p = case_dir / 'pdf_info.json'
        pi = json.loads(pi_p.read_text()) if pi_p.exists() else {}
        map_pages = pi.get('map_pages') or [1]
        page = map_pages[0]

        try:
            map_img = render_case_map(pdf, page)
            if map_img is None:
                raise RuntimeError("render returned None")
            # Write temp file for SAM3
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, map_img)
            set_fold_for_case(sam3, case_name)
            mask = extract_boundary_sam3_semantic(
                tmp_path, sam3['processor'], sam3['model'], sam3['device'],
                query="planning boundary",
            )
            Path(tmp_path).unlink(missing_ok=True)
            if mask is None or mask.sum() == 0:
                print(f"{case_name:42s}  IoU_v3={iou_v3:.3f}  semantic returned EMPTY mask")
                rows.append((case_name, iou_v3, 0.0, 'empty_mask'))
                continue

            affine_H = np.load(aff_p)
            tile_info = json.loads(tile_p.read_text())
            geojson = mask_to_geojson_affine(mask, affine_H, tile_info)
            if geojson is None:
                rows.append((case_name, iou_v3, None, 'project_failed'))
                continue

            metrics = calculate_spatial_metrics(load_geojson(gt), geojson)
            iou_sem = metrics.get('iou', 0.0)

            tag = 'WIN' if iou_sem > iou_v3 + 0.02 else ('LOSS' if iou_sem < iou_v3 - 0.02 else 'tie')
            delta = iou_sem - iou_v3
            print(f"{case_name:42s}  IoU_v3(inst)={iou_v3:.3f}  IoU_sem={iou_sem:.3f}  Δ={delta:+.3f}  {tag}")
            rows.append((case_name, iou_v3, iou_sem, tag))

        except Exception as e:
            print(f"{case_name:42s}  ERROR: {e!s:.120}")
            traceback.print_exc()
            rows.append((case_name, iou_v3, None, f'err:{e!s:.40}'))

    # Summary
    print()
    print("=" * 90)
    valid = [(n, i, s, t) for (n, i, s, t) in rows if isinstance(s, float)]
    if valid:
        mean_v3 = sum(i for _, i, _, _ in valid) / len(valid)
        mean_sem = sum(s for _, _, s, _ in valid) / len(valid)
        wins = sum(1 for _, _, _, t in valid if t == 'WIN')
        losses = sum(1 for _, _, _, t in valid if t == 'LOSS')
        ties = sum(1 for _, _, _, t in valid if t == 'tie')
        print(f"Comparable cases: {len(valid)}")
        print(f"  mean IoU v3 (instance): {mean_v3:.3f}")
        print(f"  mean IoU semantic:      {mean_sem:.3f}")
        print(f"  delta:                  {mean_sem - mean_v3:+.3f}")
        print(f"  wins (semantic >): {wins}   losses (semantic <): {losses}   ties: {ties}")
        # Specific high-stakes cases (v3 IoU > 0.5 with instance)
        print()
        print(f"High-IoU instance cases (v3 IoU > 0.5):")
        for n, i, s, t in valid:
            if i > 0.5:
                print(f"  {n:42s}  v3={i:.3f}  sem={s:.3f}  {t}")
    else:
        print("No comparable cases.")


if __name__ == '__main__':
    main()
