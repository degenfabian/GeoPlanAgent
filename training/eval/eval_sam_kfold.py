"""K-fold held-out pixel IoU + precision/recall/F1 — reusing training's
exact FoldDataset and val-loop logic, with autocast enabled to match
training conditions.

This is the source of truth for ``models/sam3_lora/cv_summary.{json,csv}``
(written at end of run). The trainer used to write that summary itself,
but its `best_row` lookup was keying on `"val_iou"` while the history
stores `"val_sem_iou"` — the silent fallback to `history[-1]` meant
cv_summary reported final-epoch (post-overfit) metrics, not best-epoch
ones. Computing the summary from `best.pt` directly side-steps that.

F1 and dice are mathematically identical for binary masks (both equal
2·TP / (2·TP + FP + FN)), so we report F1 only.
"""
from __future__ import annotations
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
sys.path.insert(0, str(REPO))

# Reuse the exact training-time dataset class and config constants
from training.eval._util import write_predictions_json
from training.train_sam3_kfold import (
    FoldDataset, collate, seed_everything,
    _ensure_pred_mask_on_gt, _autocast_ctx,
    _build_manifest_from_disk,
    LORA_TARGET_MODULES, MODEL_ID,
    DATASET_DIR as TRAIN_DATASET_DIR,
)

MODELS_DIR = REPO / "models" / "sam3_lora"
HEAD_MODULES = ["mask_embedder", "presence_head", "semantic_projection"]
RANK = 16
BF16 = True  # match training's autocast

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  bf16={BF16}  dataset={TRAIN_DATASET_DIR}")


def _binary_metrics(p_bin: torch.Tensor, g_bin: torch.Tensor) -> dict:
    """IoU + precision + recall + F1 from two boolean masks of equal shape.

    TP = |pred ∧ gt|; FP = |pred| − TP; FN = |gt| − TP. All four metrics
    are computed from the same TP/FP/FN triple so they're internally
    consistent (no risk of IoU and F1 disagreeing on the same case).
    """
    tp = (p_bin & g_bin).sum().item()
    union = (p_bin | g_bin).sum().item()
    p_sum = p_bin.sum().item()
    g_sum = g_bin.sum().item()
    fp = p_sum - tp
    fn = g_sum - tp
    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def main():
    # Build the per-case manifest in-place from maps/ + fold_assignment.json
    # via the shared helper in train_sam3_kfold. `case` is the original
    # name (with colons/parens) so it matches benchmark_runner output;
    # `filename` is the on-disk safe-form PNG name.
    fold_map = json.loads(
        (TRAIN_DATASET_DIR / "fold_assignment.json").read_text())
    manifest = _build_manifest_from_disk(TRAIN_DATASET_DIR, fold_map)
    print(f"manifest: {len(manifest)} cases")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=hf_token)
    base = Sam3Model.from_pretrained(MODEL_ID, token=hf_token)
    cfg = LoraConfig(r=RANK, lora_alpha=RANK * 2,
                    target_modules=LORA_TARGET_MODULES,
                    lora_dropout=0.05, bias="none",
                    modules_to_save=HEAD_MODULES)
    model = get_peft_model(base, cfg).to(device)

    rows = []  # (fold, case, sem_iou, inst_iou)

    for fold in range(5):
        # Use TRAINING's FoldDataset to build val set — bit-for-bit match
        val_ds = FoldDataset(TRAIN_DATASET_DIR, manifest, fold,
                              "valid", processor)
        if len(val_ds) == 0:
            continue
        # Need per-case names — FoldDataset stores them in self.entries
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate)
        cases = [e["case"] for e in val_ds.entries]

        # Load fold's best.pt (strict=True so any silent drop blows up)
        ckpt = torch.load(MODELS_DIR / f"fold_{fold}" / "best.pt",
                            map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        res = model.load_state_dict(state, strict=True)
        model.eval()

        print(f"\n=== fold {fold}: {len(val_ds)} cases | "
              f"epoch {ckpt.get('epoch')} | "
              f"reported best_val_iou {ckpt.get('best_val_iou',0):.4f} ===")

        fold_rows: list[dict] = []

        with torch.no_grad():
            for i, (inputs, gts, dms) in enumerate(val_loader):
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in inputs.items()}
                with _autocast_ctx(device, BF16):
                    outputs = model(**inputs)

                # ── Semantic head metrics — autocast + CPU-bool IoU keeps
                # bit-stability under MPS fp16. _binary_metrics computes
                # IoU/P/R/F1 from one shared TP/FP/FN so they can't disagree.
                sem_pred = outputs.semantic_seg.squeeze(1).float()
                sem = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                for b in range(sem_pred.shape[0]):
                    g = gts[b].to(device)
                    pred = _ensure_pred_mask_on_gt(sem_pred[b], g)
                    p_bin = (torch.sigmoid(pred) > 0.5).cpu().bool()
                    g_bin = (g > 0.5).cpu().bool()
                    assert p_bin.shape == g_bin.shape, (
                        f"shape mismatch p={p_bin.shape} g={g_bin.shape}")
                    sem = _binary_metrics(p_bin, g_bin)

                # ── Instance head metrics — top-scoring slot's mask, same
                # IoU/P/R/F1 calc as the semantic head.
                inst = None
                inst_masks = getattr(outputs, "pred_masks", None)
                cls_logits = getattr(outputs, "pred_logits", None)
                if inst_masks is not None and cls_logits is not None:
                    inst_masks = inst_masks.float()
                    cls_logits = cls_logits.float()
                    slots = inst_masks[0]
                    if slots.dim() == 4:
                        slots = slots.view(-1, slots.shape[-2], slots.shape[-1])
                    cls_b = cls_logits[0].view(-1)[:slots.shape[0]]
                    top_idx = int(cls_b.argmax().item())
                    g = gts[0].to(device)
                    pred_up = F.interpolate(
                        slots[top_idx].unsqueeze(0).unsqueeze(0),
                        size=g.shape[-2:], mode="bilinear",
                        align_corners=False).squeeze(0).squeeze(0)
                    p_bin = (torch.sigmoid(pred_up) > 0.5).cpu().bool()
                    g_bin = (g > 0.5).cpu().bool()
                    inst = _binary_metrics(p_bin, g_bin)

                row = {"fold": fold, "case": cases[i],
                       "sem_iou": sem["iou"], "sem_precision": sem["precision"],
                       "sem_recall": sem["recall"], "sem_f1": sem["f1"]}
                if inst is not None:
                    row.update({
                        "inst_iou": inst["iou"],
                        "inst_precision": inst["precision"],
                        "inst_recall": inst["recall"],
                        "inst_f1": inst["f1"],
                    })
                else:
                    row.update({"inst_iou": None, "inst_precision": None,
                                "inst_recall": None, "inst_f1": None})
                rows.append(row)
                fold_rows.append(row)

        # Per-fold means (macro: mean of per-case values)
        def _mean(key):
            xs = [r[key] for r in fold_rows if r.get(key) is not None]
            return sum(xs) / len(xs) if xs else 0.0
        print(f"  fold {fold}: sem iou={_mean('sem_iou'):.4f} "
              f"p={_mean('sem_precision'):.4f} r={_mean('sem_recall'):.4f} "
              f"f1={_mean('sem_f1'):.4f}  |  "
              f"inst iou={_mean('inst_iou'):.4f} f1={_mean('inst_f1'):.4f}")

    print("\n" + "=" * 60)
    print("AGGREGATE (bf16 autocast, exact training val-loop)")
    print("=" * 60)
    sem_all = [r["sem_iou"] for r in rows if r["sem_iou"] is not None]
    inst_all = [r["inst_iou"] for r in rows if r["inst_iou"] is not None]

    def summarise(name, xs):
        n = len(xs)
        s = sorted(xs)
        mean = sum(xs) / n
        med = s[n // 2]
        ge_80 = sum(1 for x in xs if x >= 0.8) / n
        ge_70 = sum(1 for x in xs if x >= 0.7) / n
        ge_90 = sum(1 for x in xs if x >= 0.9) / n
        ge_50 = sum(1 for x in xs if x >= 0.5) / n
        print(f"\n{name} (N={n})")
        print(f"  mean   = {mean:.4f}")
        print(f"  median = {med:.4f}")
        print(f"  >=0.50 = {ge_50*100:.1f}%")
        print(f"  >=0.70 = {ge_70*100:.1f}%")
        print(f"  >=0.80 = {ge_80*100:.1f}%   <-- vs MHCLG 90%")
        print(f"  >=0.90 = {ge_90*100:.1f}%")

    summarise("Semantic-head IoU", sem_all)
    summarise("Instance-head IoU", inst_all)

    # ── cv_summary.{json,csv} — paper-table source ────────────────────────
    metric_keys = ("sem_iou", "sem_precision", "sem_recall", "sem_f1",
                   "inst_iou", "inst_precision", "inst_recall", "inst_f1")

    def _macro(rows_, key):
        xs = [r[key] for r in rows_ if r.get(key) is not None]
        return sum(xs) / len(xs) if xs else 0.0

    folds_summary = []
    for k in sorted({r["fold"] for r in rows}):
        fr = [r for r in rows if r["fold"] == k]
        folds_summary.append({"fold": k, "n_val": len(fr),
                              **{f"val_{m}": _macro(fr, m) for m in metric_keys}})

    # cv mean/std are mean-of-fold-means and pop-std-over-folds (standard
    # K-fold CV reporting — fold-weighted, not case-weighted). For
    # equal-sized folds these match macro-over-all-cases, but our split
    # is 43/42/42/42/42 so they differ in the 4th decimal. The old
    # cv_summary.json (pre-fix) used this same definition, so paper
    # numbers stay comparable across runs.
    overall_mean, overall_std = {}, {}
    for m in metric_keys:
        xs = [f[f"val_{m}"] for f in folds_summary]
        overall_mean[f"val_{m}"] = sum(xs) / len(xs) if xs else 0.0
        overall_std[f"val_{m}"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0

    cv = {
        "folds": folds_summary,
        "mean": overall_mean,
        "std": overall_std,
        "n_total_val": len(rows),
        "source": "training/eval/eval_sam_kfold.py",
        "notes": ("Computed from each fold's best.pt against its held-out "
                  "val cases. mean/std are over the 5 folds (mean-of-fold-"
                  "means, population std). F1 == dice for binary masks, "
                  "so dice is not reported separately."),
    }
    (MODELS_DIR / "cv_summary.json").write_text(json.dumps(cv, indent=2))
    with open(MODELS_DIR / "cv_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_val", *[f"val_{m}" for m in metric_keys]])
        for f in folds_summary:
            w.writerow([f["fold"], f["n_val"],
                        *[f[f"val_{m}"] for m in metric_keys]])
    print(f"\nWrote {MODELS_DIR/'cv_summary.json'}")
    print(f"Wrote {MODELS_DIR/'cv_summary.csv'}")

    # ── per-case predictions ──────────────────────────────────────────────
    predictions = {
        r["case"]: {k: (float(r[k]) if k != "fold" else int(r[k]))
                    for k in ("fold", *metric_keys) if r.get(k) is not None}
        for r in rows
    }
    write_predictions_json(predictions, THIS / "predictions" / "sam_kfold.json")


if __name__ == "__main__":
    main()
