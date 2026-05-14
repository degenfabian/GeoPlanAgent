"""Render side-by-side v7 vs v8 mask panels for EVERY regression case
(Δ IoU < -0.02 from results.json).

Optimised: loads each model exactly once instead of per-case, so total wall
time is ~3-4 minutes for 15 cases instead of ~15-20 minutes.

Output: experiments/sam3_v8_replay/regression_renders/*.png
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

OUT_DIR = HERE / "regression_renders"
DELTA_THRESHOLD = -0.02   # all cases with Δ < this


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


def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.4):
    out = img_bgr.copy()
    if mask is None or mask.sum() == 0:
        return out
    mb = (mask > 0).astype(np.uint8)
    if mb.shape[:2] != out.shape[:2]:
        mb = cv2.resize(mb, (out.shape[1], out.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(out)
    layer[mb > 0] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0)


def label(img, text, color=(255, 255, 255), bg=(0, 0, 0)):
    pad = 14
    fscale = max(0.8, min(2.0, img.shape[1] / 1500))
    thickness = max(2, int(fscale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    fscale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 2*pad, th + 2*pad), bg, -1)
    cv2.putText(img, text, (pad, th + pad // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thickness, cv2.LINE_AA)
    return img


def render_and_prepare(case, bench_dir, eval_dir):
    """Returns (case, map_img) or None on failure."""
    cd = bench_dir / case
    try:
        pdf_info = json.loads((cd / "pdf_info.json").read_text())
    except Exception:
        pdf_info = {}
    page_idx = (pdf_info.get("map_pages") or [1])[0] - 1
    pdfs = list((eval_dir / case).glob("*.pdf"))
    if not pdfs:
        return None
    pdf_path = next((p for p in pdfs if any(k in p.name.lower()
                      for k in ("map", "plan", "direction", "boundary"))),
                     max(pdfs, key=lambda p: p.stat().st_size))
    try:
        map_img = render_pdf_page_v20style(str(pdf_path), page_idx, dpi=200)
    except Exception:
        return None
    try:
        from tools.io.rotation_classifier import auto_rotate
        map_img, _ = auto_rotate(map_img, verbose=False)
    except Exception:
        pass
    try:
        from tools.io.map_crop import detect_title_block_crop
        cr, _, _, info = detect_title_block_crop(map_img)
        if info.get("cropped"):
            map_img = cr
    except Exception:
        pass
    return map_img


def run_sam_on_all(kfold_dir, prepared, label_str):
    """Load SAM3 from kfold_dir once, run on all prepared cases."""
    from tools.extraction.sam3 import (
        load_sam3_ft, extract_boundary_sam3_semantic, set_fold_for_case,
    )
    print(f"\n=== Loading SAM3 from {kfold_dir} ===")
    state = load_sam3_ft(kfold_dir=kfold_dir)
    masks = {}
    for case, (map_img, _v20, _v8) in prepared.items():
        set_fold_for_case(state, case)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            cv2.imwrite(t.name, map_img)
            try:
                mask = extract_boundary_sam3_semantic(
                    t.name, state["processor"], state["model"], state["device"],
                    query="planning boundary",
                )
            except Exception as e:
                print(f"  {case}: {label_str} failed: {e!s:.60}")
                mask = None
            finally:
                try: os.unlink(t.name)
                except: pass
        coverage = (mask > 0).mean() * 100 if mask is not None else 0
        masks[case] = mask
        print(f"  [{label_str}] {case}: mask={coverage:.2f}%")
    del state
    import gc, torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return masks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = json.load(open(HERE / "results.json"))
    rows = sorted(res["rows"], key=lambda x: x.get("delta_iou", 0))
    regressions = [r for r in rows
                   if r.get("delta_iou") is not None and r["delta_iou"] < DELTA_THRESHOLD]
    print(f"Found {len(regressions)} regression cases (Δ < {DELTA_THRESHOLD}):")
    for r in regressions:
        print(f"  {r['case']:50s}  v7={r['v20_iou'] or 0:.3f} → v8={r['v8_iou']:.3f}  Δ={r['delta_iou']:+.3f}")

    bench_dir = REPO / "results" / "benchmark_v20" / "gemini-flash"
    eval_dir = REPO / "evaluation_data"

    print(f"\n=== Pre-rendering maps ===")
    prepared = {}
    for r in regressions:
        case = r["case"]
        map_img = render_and_prepare(case, bench_dir, eval_dir)
        if map_img is None:
            print(f"  SKIP {case}: render failed")
            continue
        prepared[case] = (map_img, r["v20_iou"], r["v8_iou"])
        print(f"  {case}: {map_img.shape[1]}x{map_img.shape[0]}")

    v7_masks = run_sam_on_all("models/sam3_lora_v7_both", prepared, "v7")
    v8_masks = run_sam_on_all("models/sam3_lora", prepared, "v8")

    print(f"\n=== Composing panels ===")
    for case, (map_img, v20_iou, v8_iou) in prepared.items():
        v7 = v7_masks.get(case)
        v8 = v8_masks.get(case)
        a = label(overlay_mask(map_img, v7, (0, 255, 0)),
                   f"{case}  v7  IoU={v20_iou or 0:.3f}")
        b = label(overlay_mask(map_img, v8, (0, 0, 255)),
                   f"{case}  v8  IoU={v8_iou:.3f}")
        panel = np.hstack([a, b])
        out_p = OUT_DIR / f"{case.replace(':', '_').replace('/', '_')}.png"
        cv2.imwrite(str(out_p), panel)
        print(f"  → {out_p}")

    print(f"\nDone. Panels in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
