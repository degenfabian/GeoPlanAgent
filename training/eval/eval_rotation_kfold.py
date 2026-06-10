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

Prints an accuracy breakdown (overall / on upright pages / on rotated
pages) so it's easy to see whether the model is doing well on the
~90% upright cases or also on the ~10% rotated ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

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
    fold_stats: list[dict] = []  # filled inside the loop below

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
        n_val_fold = len(val_cases)
        fold_acc = n_match / n_val_fold if n_val_fold > 0 else 0.0
        fold_stats.append({"fold": fold_k, "n_val": n_val_fold,
                           "n_correct": n_match, "val_acc": fold_acc})
        print(f"fold {fold_k}: {n_match}/{n_val_fold} correct on val "
              f"({100*fold_acc:.1f}%, {elapsed:.0f}s)")

    out_name = "rotation_kfold_tta.json" if args.tta else "rotation_kfold.json"
    write_predictions_json(predictions, THIS / "predictions" / out_name)

    # Accuracy breakdown: overall / upright (gt=0) / rotated (gt!=0)
    gt_upright = sum(1 for v in labels.values() if v == 0)
    gt_rotated = sum(1 for v in labels.values() if v != 0)
    total = len(labels)
    correct, correct_up, correct_rot = 0, 0, 0
    n_up, n_rot = 0, 0
    for c, gt_v in labels.items():
        if c not in predictions: continue
        pred = predictions[c]
        if pred == gt_v: correct += 1
        if gt_v == 0:
            n_up += 1
            if pred == gt_v: correct_up += 1
        else:
            n_rot += 1
            if pred == gt_v: correct_rot += 1
    print()
    print(f"  overall:     {correct}/{total} ({100*correct/max(1,total):.1f}%)")
    print(f"  on upright:  {correct_up}/{n_up} ({100*correct_up/max(1,n_up):.1f}%)")
    print(f"  on rotated:  {correct_rot}/{n_rot} ({100*correct_rot/max(1,n_rot):.1f}%)")
    print(f"  GT split:    upright={gt_upright}/{total}  rotated={gt_rotated}/{total}")

    # cv_summary{_tta}.{json,csv} — paper-table source
    # Reuses the same per-fold n_match counter that produced the printout
    # above, so the CSV and the printout can't disagree. Mean/std are over
    # the 5 folds (mean-of-fold-means, population std) — same convention
    # as models/sam3_lora/cv_summary.json.
    fold_accs = [f["val_acc"] for f in fold_stats]
    mean_acc = sum(fold_accs) / len(fold_accs) if fold_accs else 0.0
    std_acc = statistics.pstdev(fold_accs) if len(fold_accs) > 1 else 0.0
    n_total = sum(f["n_val"] for f in fold_stats)
    n_correct_total = sum(f["n_correct"] for f in fold_stats)
    overall_acc = n_correct_total / n_total if n_total > 0 else 0.0

    cv = {
        "folds": fold_stats,
        "mean": {"val_acc": mean_acc},
        "std": {"val_acc": std_acc},
        "n_total_val": n_total,
        "n_total_correct": n_correct_total,
        "overall_acc": overall_acc,
        "tta": args.tta,
        "source": "training/eval/eval_rotation_kfold.py",
        "notes": ("Per-fold val_acc is held-out accuracy of fold k's "
                  "best.pt on cases assigned to fold k. mean/std are over "
                  "the 5 folds (fold-weighted, not case-weighted)."),
    }
    suffix = "_tta" if args.tta else ""
    cv_json = OUTPUT_DIR / f"cv_summary{suffix}.json"
    cv_csv = OUTPUT_DIR / f"cv_summary{suffix}.csv"
    cv_json.write_text(json.dumps(cv, indent=2))
    with open(cv_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_val", "n_correct", "val_acc"])
        for f in fold_stats:
            w.writerow([f["fold"], f["n_val"], f["n_correct"],
                        f"{f['val_acc']:.4f}"])
    print(f"\nWrote {cv_json}")
    print(f"Wrote {cv_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
