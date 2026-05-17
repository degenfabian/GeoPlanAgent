"""Detect map rotation via macOS Vision OCR (Apple Neural Engine).

Idea: text on planning maps reads left-to-right when upright. OCRing the
image at all 4 rotations and picking the one that yields the most/most-
confident text identifies the upright orientation. The rotation that
maximises OCR yield is the corrective rotation needed.

Runs entirely on the Apple Neural Engine (no GPU/MPS contention with
SAM3 or rotation-classifier training).

Usage:
    uv run python scripts/vision_rotation_detect.py            (run on all 211)
    uv run python scripts/vision_rotation_detect.py --limit 20 (sample)
    uv run python scripts/vision_rotation_detect.py --max-side 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

DATASET_DIR = REPO / "boundary_annotations"
LABELS_FILE = REPO / "rotation_annotations.json"
OUTPUT_FILE = REPO / "vision_rotation_predictions.json"

CV2_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _ocr_score(img_bgr) -> tuple[float, int]:
    """Run macOS Vision OCR on a BGR image; return (mean_confidence, n_words).

    Confidence per observation × char count, summed. Higher is better.
    Returns (0, 0) if Vision is unavailable or OCR fails."""
    try:
        from Vision import (
            VNRecognizeTextRequest,
            VNImageRequestHandler,
            VNRequestTextRecognitionLevelAccurate,
        )
        from Quartz import CIImage
        from Cocoa import NSData
    except ImportError:
        return 0.0, 0
    try:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return 0.0, 0
        ns_data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf.tobytes()))
        ci = CIImage.imageWithData_(ns_data)
        if ci is None:
            return 0.0, 0
        req = VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)
        try:
            req.setRecognitionLanguages_(["en-GB", "en-US"])
        except Exception:
            pass
        handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci, None)
        success, _ = handler.performRequests_error_([req], None)
        if not success:
            return 0.0, 0
        results = req.results() or []
        # Score = sum_i (conf_i * len(text_i))   — long high-confidence
        # tokens dominate over noisy short matches.
        total_score = 0.0
        n_words = 0
        for r in results:
            top = r.topCandidates_(1)
            if not top or len(top) == 0:
                continue
            txt = str(top[0].string()).strip()
            if not txt:
                continue
            # Vision sometimes recognises random pixel noise as "I" or "l";
            # require ≥3 chars to count as a word.
            if len(txt) < 3:
                continue
            try:
                conf = float(r.confidence())
            except Exception:
                conf = 0.5
            total_score += conf * len(txt)
            n_words += 1
        return total_score, n_words
    except Exception:
        return 0.0, 0


def detect_rotation_for_image(img_bgr, max_side: int = 1600) -> dict:
    """Try all 4 rotations; return the one with highest OCR score.

    Returns dict with predicted corrective rotation, per-rotation scores,
    and the margin between best and second-best (a confidence proxy).
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
    scores = {}
    for k in (0, 90, 180, 270):
        rotated = img_bgr if k == 0 else cv2.rotate(img_bgr, CV2_ROTATE_CODES[k])
        s, n = _ocr_score(rotated)
        scores[k] = {"score": round(s, 1), "n_words": n}
    sorted_rots = sorted(scores.items(), key=lambda kv: -kv[1]["score"])
    best_rot, best = sorted_rots[0]
    second_rot, second = sorted_rots[1]
    margin = best["score"] - second["score"]
    return {
        "predicted_rotation": int(best_rot),
        "scores": scores,
        "margin": round(margin, 1),
        "low_confidence": (best["n_words"] < 3 or
                            (second["score"] > 0 and margin < 0.2 * second["score"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Cap to N cases")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="Downscale max(H,W) to this many px before OCR")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gt = {}
    if LABELS_FILE.exists():
        raw = json.loads(LABELS_FILE.read_text())
        gt = {k: v for k, v in raw.items()
              if not k.startswith("__") and v in (0, 90, 180, 270)}
    print(f"Loaded {len(gt)} GT annotations from {LABELS_FILE.name}")

    cases = sorted([p.name for p in DATASET_DIR.iterdir()
                    if p.is_dir() and (p / "map.png").exists()])
    if args.limit:
        cases = cases[:args.limit]
    print(f"Processing {len(cases)} cases (max_side={args.max_side})")

    out = {}
    correct = 0
    seen_gt = 0
    t_start = time.time()
    for i, case in enumerate(cases):
        img = cv2.imread(str(DATASET_DIR / case / "map.png"), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [{i+1}/{len(cases)}] {case}: failed to read"); continue
        t0 = time.time()
        result = detect_rotation_for_image(img, max_side=args.max_side)
        elapsed = time.time() - t0
        pred = result["predicted_rotation"]
        out[case] = result
        gt_v = gt.get(case)
        mark = ""
        if gt_v is not None:
            seen_gt += 1
            if pred == gt_v:
                correct += 1
                mark = "OK"
            else:
                mark = f"MISMATCH gt={gt_v}"
        print(f"  [{i+1}/{len(cases)}] {case[:40]:<40} pred={pred:>3} {mark} "
              f"margin={result['margin']:>6} ({elapsed:.1f}s)")
        # Flush periodically so progress survives Ctrl-C
        if (i + 1) % 25 == 0:
            OUTPUT_FILE.write_text(json.dumps(out, indent=2))

    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    total_t = time.time() - t_start
    print(f"\nDone in {total_t:.0f}s. Wrote {len(out)} predictions to {OUTPUT_FILE.name}")
    if seen_gt:
        print(f"Vision-OCR rotation agreement with GT: {correct}/{seen_gt} "
              f"({100*correct/seen_gt:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
