"""Cross-fold evaluation: per-case prediction using the model that did NOT see it.

Each case is routed (via the shared models/fold_assignment.json) to the fold
whose checkpoint did not see it, and predicted with the SAME inference path the
deployed pipeline uses: geoplanagent.tools.rotation_classifier.
predict_rotation_with_confidence. So the eval accuracy is exactly the deployed
behaviour — there is no second, parallel TTA implementation to drift.

With ``--tta`` it runs 4-way test-time augmentation (predict on the image and
its 90/180/270 CW rotations, realign each to the original frame, average the
softmaxes); without it, a single view. Production uses TTA, so the TTA accuracy
is the operationally relevant number. Accuracy uses the raw (pre-abstain)
argmax — it measures the classifier, not the 0.50 abstain policy.

Writes per-page predictions to
``training/eval/predictions/rotation_kfold{_tta}.json`` (the cached predictions
scripts/compute_tables.py reads for Table 9) and a per-fold accuracy summary to
``models/rotation_classifier_kfold/cv_summary{_tta}.{json,csv}``.

Prints an accuracy breakdown (overall / on upright pages / on rotated pages) so
it's easy to see whether the model is doing well on the ~90% upright cases or
also on the ~10% rotated ones.

Run:   uv run python training/eval/eval_rotation_kfold.py [--tta]
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
sys.path.insert(0, str(REPO))

from geoplanagent.paths import EVAL_PREDICTIONS_DIR, FOLD_ASSIGNMENT  # noqa: E402
from training.eval._util import write_predictions_json  # noqa: E402
from training.train_rotation import (  # noqa: E402
    DATASET_DIR,
    OUTPUT_DIR,
    fold_for,
    load_labels,
)
from geoplanagent.tools.rotation_classifier import (  # noqa: E402
    CLASS_DEGREES,
    _load_kfold_state,
    predict_rotation_with_confidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tta",
        action="store_true",
        help=(
            "Use 4-way test-time augmentation: predict on each of the image's "
            "4 rotations and average the softmaxes (realigning the class space "
            "to the original frame for each). Default is off — single-view "
            "prediction. Production inference uses TTA, so the TTA accuracy is "
            "the operationally relevant number."
        ),
    )
    args = parser.parse_args()
    print(f"TTA: {args.tta}")

    labels = load_labels()
    fold_assignment = json.loads(FOLD_ASSIGNMENT.read_text())

    state = _load_kfold_state()
    if state is None:
        print("ERROR: no rotation k-fold checkpoints found", file=sys.stderr)
        return 1
    available_folds = state["available_folds"]

    predictions: dict[str, int] = {}
    per_fold: dict[int, dict] = defaultdict(lambda: {"n_val": 0, "n_correct": 0})
    t0 = time.time()
    for case, gt in labels.items():
        if fold_for(case, fold_assignment) not in available_folds:
            continue  # that fold wasn't trained — skip rather than predict with a leaking model.
        img = cv2.imread(str(DATASET_DIR / case / "map.png"), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  WARN: missing map.png for {case}, skipping")
            continue
        result = predict_rotation_with_confidence(img, case_name=case, tta=args.tta)
        pred_deg = CLASS_DEGREES[result["raw_class"]]  # raw (pre-abstain) prediction
        predictions[case] = pred_deg
        stats = per_fold[result["fold"]]
        stats["n_val"] += 1
        if pred_deg == gt:
            stats["n_correct"] += 1
    elapsed = time.time() - t0

    fold_stats: list[dict] = []
    for fold in sorted(per_fold):
        fold_counts = per_fold[fold]
        acc = fold_counts["n_correct"] / fold_counts["n_val"] if fold_counts["n_val"] else 0.0
        fold_stats.append(
            {
                "fold": fold,
                "n_val": fold_counts["n_val"],
                "n_correct": fold_counts["n_correct"],
                "val_acc": acc,
            }
        )
        print(
            f"fold {fold}: {fold_counts['n_correct']}/{fold_counts['n_val']} "
            f"correct ({100 * acc:.1f}%)"
        )
    print(f"({len(predictions)} pages in {elapsed:.0f}s)")

    out_name = "rotation_kfold_tta.json" if args.tta else "rotation_kfold.json"
    write_predictions_json(predictions, EVAL_PREDICTIONS_DIR / out_name)

    # Accuracy breakdown: overall / upright (gt=0) / rotated (gt!=0)
    gt_upright = sum(1 for value in labels.values() if value == 0)
    gt_rotated = sum(1 for value in labels.values() if value != 0)
    total = len(labels)
    correct, correct_up, correct_rot = 0, 0, 0
    n_up, n_rot = 0, 0
    for case, gt in labels.items():
        if case not in predictions:
            continue
        pred = predictions[case]
        if pred == gt:
            correct += 1
        if gt == 0:
            n_up += 1
            if pred == gt:
                correct_up += 1
        else:
            n_rot += 1
            if pred == gt:
                correct_rot += 1
    print()
    print(f"  overall:     {correct}/{total} ({100 * correct / max(1, total):.1f}%)")
    print(f"  on upright:  {correct_up}/{n_up} ({100 * correct_up / max(1, n_up):.1f}%)")
    print(f"  on rotated:  {correct_rot}/{n_rot} ({100 * correct_rot / max(1, n_rot):.1f}%)")
    print(f"  GT split:    upright={gt_upright}/{total}  rotated={gt_rotated}/{total}")

    # cv_summary{_tta}.{json,csv} — per-fold summary. Reuses the same per-fold
    # counters that produced the printout above, so the CSV and the printout
    # can't disagree. Mean/std are over the folds (mean-of-fold-means,
    # population std) — same convention as models/sam3_lora/cv_summary.json.
    fold_accs = [fold_stat["val_acc"] for fold_stat in fold_stats]
    mean_acc = sum(fold_accs) / len(fold_accs) if fold_accs else 0.0
    std_acc = statistics.pstdev(fold_accs) if len(fold_accs) > 1 else 0.0
    n_total = sum(fold_stat["n_val"] for fold_stat in fold_stats)
    n_correct_total = sum(fold_stat["n_correct"] for fold_stat in fold_stats)
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
        "notes": (
            "Per-fold val_acc is held-out accuracy of fold k's best.pt on "
            "cases assigned to fold k, predicted via the deployed "
            "predict_rotation_with_confidence. mean/std are over the folds "
            "(fold-weighted, not case-weighted)."
        ),
    }
    suffix = "_tta" if args.tta else ""
    cv_json = OUTPUT_DIR / f"cv_summary{suffix}.json"
    cv_csv = OUTPUT_DIR / f"cv_summary{suffix}.csv"
    cv_json.write_text(json.dumps(cv, indent=2))
    with open(cv_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "n_val", "n_correct", "val_acc"])
        for fold_stat in fold_stats:
            writer.writerow(
                [
                    fold_stat["fold"],
                    fold_stat["n_val"],
                    fold_stat["n_correct"],
                    f"{fold_stat['val_acc']:.4f}",
                ]
            )
    print(f"\nWrote {cv_json}")
    print(f"Wrote {cv_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
