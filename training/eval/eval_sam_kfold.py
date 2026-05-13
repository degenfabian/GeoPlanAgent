"""K-fold held-out pixel IoU + precision/recall/F1 — reusing training's
exact FoldDataset and val-loop logic, with autocast enabled to match
training conditions.

Writes per-page metrics to ``training/eval/predictions/sam_kfold.json``
(the cached predictions scripts/compute_tables.py reads for Table 11)
and a per-fold summary to ``models/sam3_lora/cv_summary.{json,csv}``.
The summary is computed from the saved PEFT adapters directly rather
than from training history, so it always reflects the checkpoint that
actually ships rather than whatever epoch the trainer logged last.

F1 and dice are mathematically identical for binary masks (both equal
2·TP / (2·TP + FP + FN)), so we report F1 only.

Run:   uv run python training/eval/eval_sam_kfold.py
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
sys.path.insert(0, str(REPO))

from geoplanagent.paths import FOLD_ASSIGNMENT, SAM_KFOLD_PREDICTIONS
from training.eval._util import write_predictions_json
from ablations.utils import print_summary, summarise

# Reuse the exact training-time dataset class and config constants
from training.train_sam3_kfold import (
    FoldDataset,
    collate,
    _ensure_pred_mask_on_gt,
    _autocast_ctx,
    _build_manifest_from_disk,
    MODEL_ID,
    DATASET_DIR as TRAIN_DATASET_DIR,
)
from training.seg_metrics import binary_mask_metrics as _binary_metrics

MODELS_DIR = REPO / "models" / "sam3_lora"
BF16 = True  # match training's autocast

device = (
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"device={device}  bf16={BF16}  dataset={TRAIN_DATASET_DIR}")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    # Build the per-case manifest in-place from maps/ + fold_assignment.json
    # via the shared helper in train_sam3_kfold. `case` is the original
    # name (with colons/parens) so it matches benchmark_runner output;
    # `filename` is the on-disk safe-form PNG name.
    fold_map = json.loads(FOLD_ASSIGNMENT.read_text())
    manifest = _build_manifest_from_disk(TRAIN_DATASET_DIR, fold_map)
    print(f"manifest: {len(manifest)} cases")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=hf_token)

    rows = []  # filled per-case below

    def _macro(case_rows, key):
        # Macro mean: average of the per-case values for `key`, skipping
        # cases where that metric is None (e.g. the instance head when it
        # produced no slots).
        values = [row[key] for row in case_rows if row.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    for fold in range(5):
        # Use TRAINING's FoldDataset to build val set — bit-for-bit match
        val_ds = FoldDataset(TRAIN_DATASET_DIR, manifest, fold, "valid", processor)
        if len(val_ds) == 0:
            continue
        # Need per-case names — FoldDataset stores them in self.entries
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
        )
        cases = [entry["case"] for entry in val_ds.entries]

        # Load fold's PEFT checkpoint
        # We rebuild the base + PEFT wrapper per fold rather than re-using
        # a single PeftModel: PEFT's blessed loader does NOT touch the
        # frozen base model state, so the only correct way to ensure a
        # clean fold-to-fold swap is a fresh base each iteration. Slower
        # by ~30s/fold but byte-identical to the old load_state_dict path
        # (verified at migration time: max abs diff 0.00e+00 across all
        # 1008 trainable tensors; provenance recorded in each fold's
        # training_meta.json "_source" field).
        fold_dir = MODELS_DIR / f"fold_{fold}"
        if not (fold_dir / "adapter_config.json").exists():
            print(f"fold {fold}: no PEFT adapter at {fold_dir}, skipping")
            continue
        base = Sam3Model.from_pretrained(MODEL_ID, token=hf_token)
        model = PeftModel.from_pretrained(base, str(fold_dir)).to(device)
        model.eval()

        meta_path = fold_dir / "training_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        print(
            f"\n=== fold {fold}: {len(val_ds)} cases | "
            f"epoch {meta.get('epoch')} | "
            f"reported best_val_iou {meta.get('best_val_iou', 0) or 0:.4f} ==="
        )

        fold_rows: list[dict] = []

        with torch.no_grad():
            for i, (inputs, gts, distance_maps) in enumerate(val_loader):
                inputs = {
                    key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                    for key, value in inputs.items()
                }
                with _autocast_ctx(device, BF16):
                    outputs = model(**inputs)

                # ── Semantic head metrics — autocast + CPU-bool IoU keeps
                # bit-stability under MPS fp16. _binary_metrics computes
                # IoU/P/R/F1 from one shared TP/FP/FN so they can't disagree.
                sem_pred = outputs.semantic_seg.squeeze(1).float()
                sem_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                for b in range(sem_pred.shape[0]):
                    gt_mask = gts[b].to(device)
                    pred = _ensure_pred_mask_on_gt(sem_pred[b], gt_mask)
                    pred_bin = (torch.sigmoid(pred) > 0.5).cpu().bool()
                    gt_bin = (gt_mask > 0.5).cpu().bool()
                    assert pred_bin.shape == gt_bin.shape, (
                        f"shape mismatch p={pred_bin.shape} g={gt_bin.shape}"
                    )
                    sem_metrics = _binary_metrics(pred_bin, gt_bin)

                # ── Instance head metrics — top-scoring slot's mask, same
                # IoU/P/R/F1 calc as the semantic head.
                inst_metrics = None
                inst_masks = getattr(outputs, "pred_masks", None)
                cls_logits = getattr(outputs, "pred_logits", None)
                if inst_masks is not None and cls_logits is not None:
                    inst_masks = inst_masks.float()
                    cls_logits = cls_logits.float()
                    slots = inst_masks[0]
                    if slots.dim() == 4:
                        slots = slots.view(-1, slots.shape[-2], slots.shape[-1])
                    slot_logits = cls_logits[0].view(-1)[: slots.shape[0]]
                    top_idx = int(slot_logits.argmax().item())
                    gt_mask = gts[0].to(device)
                    pred_up = (
                        F.interpolate(
                            slots[top_idx].unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    pred_bin = (torch.sigmoid(pred_up) > 0.5).cpu().bool()
                    gt_bin = (gt_mask > 0.5).cpu().bool()
                    inst_metrics = _binary_metrics(pred_bin, gt_bin)

                row = {
                    "fold": fold,
                    "case": cases[i],
                    "sem_iou": sem_metrics["iou"],
                    "sem_precision": sem_metrics["precision"],
                    "sem_recall": sem_metrics["recall"],
                    "sem_f1": sem_metrics["f1"],
                }
                if inst_metrics is not None:
                    row.update(
                        {
                            "inst_iou": inst_metrics["iou"],
                            "inst_precision": inst_metrics["precision"],
                            "inst_recall": inst_metrics["recall"],
                            "inst_f1": inst_metrics["f1"],
                        }
                    )
                else:
                    row.update(
                        {
                            "inst_iou": None,
                            "inst_precision": None,
                            "inst_recall": None,
                            "inst_f1": None,
                        }
                    )
                rows.append(row)
                fold_rows.append(row)

        # Per-fold means (macro: mean of per-case values)
        print(
            f"  fold {fold}: sem iou={_macro(fold_rows, 'sem_iou'):.4f} "
            f"p={_macro(fold_rows, 'sem_precision'):.4f} "
            f"r={_macro(fold_rows, 'sem_recall'):.4f} "
            f"f1={_macro(fold_rows, 'sem_f1'):.4f}  |  "
            f"inst iou={_macro(fold_rows, 'inst_iou'):.4f} "
            f"f1={_macro(fold_rows, 'inst_f1'):.4f}"
        )

    print("\n" + "=" * 60)
    print("AGGREGATE (bf16 autocast, exact training val-loop)")
    print("=" * 60)
    sem_all = [row["sem_iou"] for row in rows if row["sem_iou"] is not None]
    inst_all = [row["inst_iou"] for row in rows if row["inst_iou"] is not None]

    print_summary(summarise("Semantic-head IoU", sem_all))
    print_summary(summarise("Instance-head IoU", inst_all))

    # cv_summary.{json,csv} — per-fold summary shipped next to the adapters
    metric_keys = (
        "sem_iou",
        "sem_precision",
        "sem_recall",
        "sem_f1",
        "inst_iou",
        "inst_precision",
        "inst_recall",
        "inst_f1",
    )

    folds_summary = []
    for fold_key in sorted({row["fold"] for row in rows}):
        fold_rows = [row for row in rows if row["fold"] == fold_key]
        folds_summary.append(
            {
                "fold": fold_key,
                "n_val": len(fold_rows),
                **{f"val_{metric}": _macro(fold_rows, metric) for metric in metric_keys},
            }
        )

    # cv mean/std are mean-of-fold-means and pop-std-over-folds (standard
    # K-fold CV reporting — fold-weighted, not case-weighted). For
    # equal-sized folds these match macro-over-all-cases, but our split
    # is 43/42/42/42/42 so they differ in the 4th decimal. The old
    # cv_summary.json (pre-fix) used this same definition, so paper
    # numbers stay comparable across runs.
    overall_mean, overall_std = {}, {}
    for metric in metric_keys:
        values = [fold_summary[f"val_{metric}"] for fold_summary in folds_summary]
        overall_mean[f"val_{metric}"] = sum(values) / len(values) if values else 0.0
        overall_std[f"val_{metric}"] = statistics.pstdev(values) if len(values) > 1 else 0.0

    cv = {
        "folds": folds_summary,
        "mean": overall_mean,
        "std": overall_std,
        "n_total_val": len(rows),
        "source": "training/eval/eval_sam_kfold.py",
        "notes": (
            "Computed from each fold's PEFT adapter against its held-out "
            "val cases. mean/std are over the 5 folds (mean-of-fold-"
            "means, population std). F1 == dice for binary masks, "
            "so dice is not reported separately."
        ),
    }
    (MODELS_DIR / "cv_summary.json").write_text(json.dumps(cv, indent=2))
    with open(MODELS_DIR / "cv_summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "n_val", *[f"val_{metric}" for metric in metric_keys]])
        for fold_summary in folds_summary:
            writer.writerow(
                [
                    fold_summary["fold"],
                    fold_summary["n_val"],
                    *[fold_summary[f"val_{metric}"] for metric in metric_keys],
                ]
            )
    print(f"\nWrote {MODELS_DIR / 'cv_summary.json'}")
    print(f"Wrote {MODELS_DIR / 'cv_summary.csv'}")

    # per-page predictions — the cache scripts/compute_tables.py reads for Table 11
    predictions = {
        row["case"]: {
            key: (float(row[key]) if key != "fold" else int(row[key]))
            for key in ("fold", *metric_keys)
            if row.get(key) is not None
        }
        for row in rows
    }
    write_predictions_json(predictions, SAM_KFOLD_PREDICTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
