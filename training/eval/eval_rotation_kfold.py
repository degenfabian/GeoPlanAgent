"""Cross-fold evaluation: per-case prediction using the model that did NOT see it.

For each fold k, load fold_k/best.pt and predict on cases assigned to
fold k. Writes per-case predictions to
``training/eval/predictions/rotation_kfold.json``.

With ``--tta``, applies 4-way test-time augmentation (predict on the
image and its 90°/180°/270° rotations, re-rotate the predicted classes
back to the original frame, average the logits). Writes to
``training/eval/predictions/rotation_kfold_tta.json``. The deployed
pipeline uses TTA, so the TTA accuracy is the operationally relevant
number.

Also computes 3-way comparison (LLM / Vision OCR / ResNet kfold) vs GT
when those baseline prediction files exist on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
sys.path.insert(0, str(REPO))

from training.eval._util import write_predictions_json  # noqa: E402
from training.train_rotation import (  # noqa: E402
    CLASS_DEGREES, RotationClassifier,
    KFoldRotationDataset, load_labels, fold_for, OUTPUT_DIR, SAM3_FOLD_ASSIGNMENT,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tta", action="store_true",
        help=("Use 4-way test-time augmentation: predict on each of the "
              "image's 4 rotations and average the logits (re-rotating "
              "the class space to the original frame for each). "
              "Default is off — single-view prediction. Production "
              "inference uses TTA, so the TTA accuracy is the "
              "operationally relevant number."),
    )
    args = ap.parse_args()

    if torch.cuda.is_available(): device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Device: {device}  TTA: {args.tta}")

    labels = load_labels()
    sam3_fa = json.loads(SAM3_FOLD_ASSIGNMENT.read_text())
    case_to_fold = {c: fold_for(c, sam3_fa) for c in labels}

    predictions: dict[str, int] = {}

    for fold_k in sorted(set(case_to_fold.values())):
        ckpt_path = OUTPUT_DIR / f"fold_{fold_k}" / "best.pt"
        if not ckpt_path.exists():
            print(f"fold {fold_k}: missing {ckpt_path}, skipping"); continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        img_size = ckpt["config"]["img_size"]

        model = RotationClassifier(n_classes=4)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()

        val_cases = sorted([c for c, f in case_to_fold.items() if f == fold_k])
        ds = KFoldRotationDataset(val_cases, labels, img_size=img_size, train=False)
        # Group sample indices by case so we can do TTA over all 4
        # rotations or just the k=0 view, both off the same Dataset.
        case_idx: dict[str, dict[int, int]] = {}
        for i, (c, k) in enumerate(ds.samples):
            case_idx.setdefault(c, {})[k] = i

        n_match = 0
        t0 = time.time()
        for case in val_cases:
            views = case_idx.get(case) or {}
            if args.tta:
                # Predict on all 4 rotated views; re-rotate each view's
                # class space back to the original frame; average logits.
                accum = torch.zeros(4, device=device)
                for k in (0, 90, 180, 270):
                    if k not in views: continue
                    x, _label = ds[views[k]]
                    x = x.unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x).squeeze(0)
                    # Convert this view's class predictions back into the
                    # original frame: if we rotated by k CW before predict,
                    # a "class c (= c·90° CW corrective)" on the view is
                    # equivalent to "(c·90° + k) mod 360 CW corrective" on
                    # the original. Shift logits accordingly.
                    shift = (k // 90) % 4
                    accum += torch.roll(logits, shifts=shift)
                logits = accum
            else:
                x, _label = ds[views[0]]
                x = x.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x).squeeze(0)
            pred_class = int(logits.argmax(dim=-1).item())
            pred_deg = CLASS_DEGREES[pred_class]
            predictions[case] = pred_deg
            if pred_deg == labels[case]:
                n_match += 1
        elapsed = time.time() - t0
        print(f"fold {fold_k}: {n_match}/{len(val_cases)} correct on val "
              f"({100*n_match/max(1,len(val_cases)):.1f}%, {elapsed:.0f}s)")

    out_name = "rotation_kfold_tta.json" if args.tta else "rotation_kfold.json"
    write_predictions_json(predictions, THIS / "predictions" / out_name)

    # 3-way comparison
    print("\n========== 3-way comparison ==========")
    print(f"{'method':<22} {'overall':>10} {'on upright':>12} {'on rotated':>12}")

    # GT counts
    gt_upright = sum(1 for v in labels.values() if v == 0)
    gt_rotated = sum(1 for v in labels.values() if v != 0)
    total = len(labels)

    def score(predictor: dict, name: str):
        correct, correct_up, correct_rot = 0, 0, 0
        n, n_up, n_rot = 0, 0, 0
        for c, gt_v in labels.items():
            if c not in predictor: continue
            n += 1
            pred = predictor[c]
            if pred == gt_v: correct += 1
            if gt_v == 0:
                n_up += 1
                if pred == gt_v: correct_up += 1
            else:
                n_rot += 1
                if pred == gt_v: correct_rot += 1
        print(f"{name:<22} {correct}/{n} ({100*correct/max(1,n):4.1f}%)"
              f"  {correct_up}/{n_up} ({100*correct_up/max(1,n_up):4.1f}%)"
              f"  {correct_rot}/{n_rot} ({100*correct_rot/max(1,n_rot):4.1f}%)")

    # LLM predictions from v3 cache
    llm_preds = {}
    import glob, os
    for f in glob.glob("results/benchmark_v3/gemini-flash/*/pdf_info.json"):
        case = os.path.basename(os.path.dirname(f))
        if case in labels:
            try:
                llm_preds[case] = json.load(open(f)).get("map_rotation", 0) or 0
            except Exception: pass

    # Vision OCR
    vis = {}
    vis_path = REPO / "vision_rotation_predictions.json"
    if vis_path.exists():
        raw = json.loads(vis_path.read_text())
        vis = {c: v["predicted_rotation"] for c, v in raw.items()}

    score(llm_preds, "LLM (Gemini)")
    score(vis, "Vision OCR")
    score(predictions, "ResNet50 5-fold")
    print(f"\nGT distribution: upright={gt_upright}/{total} ({100*gt_upright/total:.0f}%), "
          f"rotated={gt_rotated}/{total} ({100*gt_rotated/total:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
