"""Cross-fold evaluation: per-case prediction using the model that did NOT see it.

For each fold k, load fold_k/best.pt and predict on cases assigned to fold k.
Output: rotation_kfold_predictions.json (case -> predicted corrective rotation in deg).

Also computes 3-way comparison (LLM / Vision OCR / ResNet kfold) vs GT.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

from training.train_rotation_classifier import (  # noqa: E402
    CLASS_DEGREES, RotationClassifier,
)
from training.train_rotation_kfold import (  # noqa: E402
    KFoldRotationDataset, load_labels, fold_for, OUTPUT_DIR, SAM3_FOLD_ASSIGNMENT,
)


def main() -> int:
    if torch.cuda.is_available(): device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Device: {device}")

    labels = load_labels()
    sam3_fa = json.loads(SAM3_FOLD_ASSIGNMENT.read_text())
    case_to_fold = {c: fold_for(c, sam3_fa) for c in labels}

    predictions: dict[str, int] = {}
    per_case_logits: dict[str, list] = {}

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
        # Only the "original" applied_rotation=0 sample per case for prediction
        # (corrective rotation is the case-level label, not augmentation-level)
        idx_orig = [i for i, (c, k) in enumerate(ds.samples) if k == 0]

        n_match = 0
        t0 = time.time()
        for i in idx_orig:
            case, _k = ds.samples[i]
            x, _label = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            pred_class = int(logits.argmax(dim=-1).item())
            pred_deg = CLASS_DEGREES[pred_class]
            predictions[case] = pred_deg
            per_case_logits[case] = [round(v, 4) for v in logits.squeeze(0).cpu().tolist()]
            if pred_deg == labels[case]:
                n_match += 1
        elapsed = time.time() - t0
        print(f"fold {fold_k}: {n_match}/{len(idx_orig)} correct on val "
              f"({100*n_match/max(1,len(idx_orig)):.1f}%, {elapsed:.0f}s)")

    out_path = REPO / "rotation_kfold_predictions.json"
    out_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))
    print(f"\nWrote {len(predictions)} predictions to {out_path.name}")

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
