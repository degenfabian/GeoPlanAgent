"""Render side-by-side v7 vs v8 mask comparisons for the worst regression cases.

For each of the top-K regression cases from the replay:
  1. Re-render the planning map (v20-style: CropBox, the same image v7/v20 saw)
  2. Run v7 SAM3 (models/sam3_lora_v7_both) with "planning boundary" prompt
  3. Run v8 SAM3 (models/sam3_lora) with "planning boundary" prompt
  4. Save a 4-panel PNG: map | v7 mask overlay | v8 mask overlay | both side-by-side

Output: experiments/sam3_v8_replay/regression_renders/<case>.png

Easy to delete: rm -rf experiments/sam3_v8_replay/regression_renders/
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

TOP_K = 5  # number of worst-regression cases to visualize
OUT_DIR = HERE / "regression_renders"


def render_pdf_page_v20style(pdf_path, page_index, dpi=200):
    """Same as in replay.py — matches v20's cropbox-based render."""
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


def overlay_mask(img_bgr, mask, color=(0, 0, 255), alpha=0.4):
    """Translucent mask overlay on a copy of img_bgr. mask = uint8 0/255."""
    out = img_bgr.copy()
    if mask is None or mask.sum() == 0:
        return out
    mask_bin = (mask > 0).astype(np.uint8)
    # Resize mask to img size if needed
    if mask_bin.shape[:2] != out.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (out.shape[1], out.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(out)
    layer[mask_bin > 0] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0)


def label_panel(img, text, color=(255, 255, 255), bg=(0, 0, 0)):
    """Draw a banner label at the top-left."""
    h, w = img.shape[:2]
    pad = 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.8, min(2.0, w / 1500))
    thickness = max(2, int(fscale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 2*pad, th + 2*pad), bg, -1)
    cv2.putText(img, text, (pad, th + pad // 2), font, fscale,
                color, thickness, cv2.LINE_AA)
    return img


def compose_panel(map_img, v7_mask, v8_mask, case, v20_iou, v8_iou):
    """Build a single side-by-side panel: v7 overlay | v8 overlay."""
    a = label_panel(overlay_mask(map_img, v7_mask, color=(0, 255, 0)),
                     f"{case}  v7 (cached v20)  IoU={v20_iou:.3f}")
    b = label_panel(overlay_mask(map_img, v8_mask, color=(0, 0, 255)),
                     f"{case}  v8 replay        IoU={v8_iou:.3f}")
    # Stack horizontally; ensure equal heights (they're the same image, so they are)
    return np.hstack([a, b])


def run_sam_with_dir(kfold_dir, case_name, map_path):
    """Load SAM3 from a specific kfold_dir, route to the case's fold, run
    semantic boundary extraction with the 'planning boundary' prompt.

    Loads fresh each call because we want to swap between v7 and v8 weights.
    """
    from tools.extraction.sam3 import (
        load_sam3_ft, extract_boundary_sam3_semantic, set_fold_for_case,
    )
    state = load_sam3_ft(kfold_dir=kfold_dir)
    set_fold_for_case(state, case_name)
    mask = extract_boundary_sam3_semantic(
        map_path, state["processor"], state["model"], state["device"],
        query="planning boundary",
    )
    return state, mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = json.load(open(HERE / "results.json"))
    rows = sorted(res["rows"], key=lambda x: x.get("delta_iou", 0))
    worst = [r for r in rows if r["delta_iou"] is not None][:TOP_K]
    print(f"Rendering top {TOP_K} regressions:")
    for r in worst:
        print(f"  {r['case']}  v7={r['v20_iou'] or 0:.3f} → v8={r['v8_iou']:.3f}  Δ={r['delta_iou']:+.3f}")

    bench_dir = REPO / "results" / "benchmark_v20" / "gemini-flash"
    eval_dir = REPO / "evaluation_data"

    print(f"\nLoading v7 SAM3 (models/sam3_lora_v7_both)...")
    print(f"Loading v8 SAM3 (models/sam3_lora)... done lazily per case to save memory")

    for r in worst:
        case = r["case"]
        cd = bench_dir / case
        try:
            pdf_info = json.loads((cd / "pdf_info.json").read_text())
        except Exception:
            pdf_info = {}
        page_idx = (pdf_info.get("map_pages") or [1])[0] - 1
        pdfs = list((eval_dir / case).glob("*.pdf"))
        if not pdfs:
            print(f"  SKIP {case}: no PDF on disk")
            continue
        pdf_path = next((p for p in pdfs if any(k in p.name.lower()
                          for k in ("map", "plan", "direction", "boundary"))),
                         max(pdfs, key=lambda p: p.stat().st_size))

        try:
            map_img = render_pdf_page_v20style(str(pdf_path), page_idx, dpi=200)
        except Exception as e:
            print(f"  SKIP {case}: render failed: {e!s:.60}")
            continue

        # Apply auto_rotate + crop pipeline (same as production)
        try:
            from tools.io.rotation_classifier import auto_rotate
            map_img, _ = auto_rotate(map_img, verbose=False)
        except Exception: pass
        try:
            from tools.io.map_crop import detect_title_block_crop
            cr, _, _, info = detect_title_block_crop(map_img)
            if info.get("cropped"): map_img = cr
        except Exception: pass

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            map_path = t.name
        cv2.imwrite(map_path, map_img)

        # Run v7 (legacy weights)
        try:
            print(f"\n[{case}] running v7 SAM3...")
            _, v7_mask = run_sam_with_dir("models/sam3_lora_v7_both", case, map_path)
        except Exception as e:
            print(f"  v7 failed: {e!s:.60}")
            v7_mask = None

        # Run v8 (new weights)
        try:
            print(f"[{case}] running v8 SAM3...")
            _, v8_mask = run_sam_with_dir("models/sam3_lora", case, map_path)
        except Exception as e:
            print(f"  v8 failed: {e!s:.60}")
            v8_mask = None

        try: os.unlink(map_path)
        except: pass

        v7_pct = (v7_mask > 0).mean() if v7_mask is not None else 0.0
        v8_pct = (v8_mask > 0).mean() if v8_mask is not None else 0.0
        print(f"  v7 mask: {v7_pct*100:5.2f}% of image  | v8 mask: {v8_pct*100:5.2f}% of image")

        panel = compose_panel(map_img, v7_mask, v8_mask,
                              case, r["v20_iou"] or 0.0, r["v8_iou"])
        out_path = OUT_DIR / f"{case.replace(':', '_').replace('/', '_')}.png"
        cv2.imwrite(str(out_path), panel)
        print(f"  → {out_path}")

    print(f"\nDone. Panels in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
