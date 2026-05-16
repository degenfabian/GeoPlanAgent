"""Replay all v3 benchmark cases with SAM3 semantic mode.

For every case in results/benchmark_v3/gemini-flash:
  1. Read the v3-saved affine_H + tile_info + GT geojson.
  2. Re-render the same map_page (auto-rotate + map_crop applied).
  3. Run SAM3 semantic with query="planning boundary".
  4. Project the semantic mask through the SAVED affine_H + tile_info.
  5. Compute IoU vs GT.

Compares the resulting mean IoU against the actual v3 mean.
No API calls; local SAM3 only.
"""
from __future__ import annotations
import json
import sys
import tempfile
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
from tools.io.map_page import render_map_page
from tools.matching import mask_to_geojson_affine
from tools.metrics.geojson import calculate_spatial_metrics, load_geojson


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
    """Thin wrapper around tools.io.map_page.render_map_page."""
    rendered = render_map_page(str(pdf_path), page_1based, dpi=dpi)
    return rendered[0] if rendered is not None else None


def main():
    print("Loading SAM3...")
    sam3 = load_sam3_ft()
    print(f"  device={sam3['device']}")

    cases = sorted(p for p in V3_DIR.iterdir() if p.is_dir())
    print(f"Replaying {len(cases)} cases...\n")

    rows = []
    for i, case_d in enumerate(cases, 1):
        case_name = case_d.name
        mf = case_d / 'metrics.json'
        if not mf.exists():
            continue
        try:
            metrics_v3 = json.loads(mf.read_text())
        except Exception:
            continue
        iou_v3 = metrics_v3.get('iou')

        aff_p = case_d / 'affine_H.npy'
        tile_p = case_d / 'tile_info.json'
        if not aff_p.exists() or not tile_p.exists():
            rows.append((case_name, iou_v3, None, 'no_affine'))
            continue

        pdf = find_pdf(case_name)
        gt = find_gt(case_name)
        if pdf is None or gt is None:
            rows.append((case_name, iou_v3, None, 'no_pdf_or_gt'))
            continue

        pi_p = case_d / 'pdf_info.json'
        pi = json.loads(pi_p.read_text()) if pi_p.exists() else {}
        map_pages = pi.get('map_pages') or [1]
        page = map_pages[0]

        try:
            map_img = render_case_map(pdf, page)
            if map_img is None:
                rows.append((case_name, iou_v3, None, 'render_failed'))
                continue
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
                rows.append((case_name, iou_v3, 0.0, 'empty_mask'))
                print(f"[{i:3d}/{len(cases)}] {case_name:42s}  v3={iou_v3:.3f}  sem=0.000 (empty mask)")
                continue

            affine_H = np.load(aff_p)
            tile_info = json.loads(tile_p.read_text())
            geojson = mask_to_geojson_affine(mask, affine_H, tile_info)
            if geojson is None:
                rows.append((case_name, iou_v3, None, 'project_failed'))
                continue

            metrics = calculate_spatial_metrics(load_geojson(gt), geojson)
            iou_sem = metrics.get('iou', 0.0)
            tag = ''
            if iou_sem > (iou_v3 or 0) + 0.02:
                tag = '↑'
            elif iou_sem < (iou_v3 or 0) - 0.02:
                tag = '↓'
            rows.append((case_name, iou_v3, iou_sem, 'ok'))
            print(f"[{i:3d}/{len(cases)}] {case_name:42s}  v3={iou_v3:.3f}  sem={iou_sem:.3f}  Δ={iou_sem-(iou_v3 or 0):+.3f} {tag}")
        except Exception as e:
            rows.append((case_name, iou_v3, None, f'err:{e!s:.40}'))
            print(f"[{i:3d}/{len(cases)}] {case_name:42s}  ERROR: {e!s:.80}")

    # Summary
    print()
    print("=" * 90)
    valid = [(n, i, s, t) for (n, i, s, t) in rows if isinstance(s, float) and isinstance(i, float)]
    if valid:
        mean_v3 = sum(i for _, i, _, _ in valid) / len(valid)
        mean_sem = sum(s for _, _, s, _ in valid) / len(valid)
        wins = sum(1 for _, i, s, _ in valid if s > i + 0.02)
        losses = sum(1 for _, i, s, _ in valid if s < i - 0.02)
        ties = len(valid) - wins - losses
        print(f"FULL BENCHMARK REPLAY")
        print(f"Comparable cases: {len(valid)} / {len(rows)}")
        print(f"  mean IoU v3 (mixed semantic+instance): {mean_v3:.4f}")
        print(f"  mean IoU semantic-only:                {mean_sem:.4f}")
        print(f"  delta:                                 {mean_sem - mean_v3:+.4f}")
        print(f"  wins (sem > v3+0.02):   {wins}")
        print(f"  losses (sem < v3-0.02): {losses}")
        print(f"  ties (|Δ|≤0.02):        {ties}")
        # Big losses
        big_losses = sorted([(n, i, s) for (n, i, s, _) in valid if s < i - 0.10], key=lambda x: x[2] - x[1])[:10]
        if big_losses:
            print(f"\nTop 10 biggest semantic LOSSES (Δ < -0.10):")
            for n, i, s in big_losses:
                print(f"  {n:42s}  v3={i:.3f}  sem={s:.3f}  Δ={s-i:+.3f}")
        big_wins = sorted([(n, i, s) for (n, i, s, _) in valid if s > i + 0.10], key=lambda x: -(x[2] - x[1]))[:10]
        if big_wins:
            print(f"\nTop 10 biggest semantic WINS (Δ > +0.10):")
            for n, i, s in big_wins:
                print(f"  {n:42s}  v3={i:.3f}  sem={s:.3f}  Δ={s-i:+.3f}")
    skipped = [(n, t) for (n, i, s, t) in rows if not isinstance(s, float)]
    if skipped:
        print(f"\nSkipped: {len(skipped)}")
        from collections import Counter
        reasons = Counter(t for _, t in skipped)
        for r, c in reasons.most_common():
            print(f"  {r}: {c}")

    # Save full results CSV for later analysis
    out_csv = Path('results/v3_semantic_replay.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w') as f:
        f.write("case,iou_v3,iou_semantic,status\n")
        for n, i, s, t in rows:
            f.write(f"{n},{i if isinstance(i,float) else ''},{s if isinstance(s,float) else ''},{t}\n")
    print(f"\nSaved per-case results to {out_csv}")


if __name__ == '__main__':
    main()
