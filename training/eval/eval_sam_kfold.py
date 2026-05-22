"""K-fold held-out pixel IoU — reusing training's exact FoldDataset and
val-loop logic, with autocast enabled to match training conditions.

This eliminates eval-implementation drift between my eval and the val
metrics in models/sam3_lora/fold_*/history.json.
"""
from __future__ import annotations
import json
import os
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

        sem_ious_fold, inst_ious_fold = [], []

        with torch.no_grad():
            for i, (inputs, gts, dms) in enumerate(val_loader):
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in inputs.items()}
                with _autocast_ctx(device, BF16):
                    outputs = model(**inputs)

                # ── Semantic IoU — copied verbatim from train_sam3_kfold.py
                # then hardened: use bool masks + bitwise ops so any MPS
                # fp16 nondeterminism can't produce non-binary intermediates.
                sem_pred = outputs.semantic_seg.squeeze(1).float()
                for b in range(sem_pred.shape[0]):
                    g = gts[b].to(device)
                    pred = _ensure_pred_mask_on_gt(sem_pred[b], g)
                    # Force everything to CPU bool for the IoU calc — eliminates
                    # any chance of MPS-side non-determinism in the metric.
                    p_bin = (torch.sigmoid(pred) > 0.5).cpu().bool()
                    g_bin = (g > 0.5).cpu().bool()
                    assert p_bin.shape == g_bin.shape, (
                        f"shape mismatch p={p_bin.shape} g={g_bin.shape}")
                    inter = (p_bin & g_bin).sum().item()
                    union = (p_bin | g_bin).sum().item()
                    sem_iou = inter / union if union > 0 else 0.0

                # ── Instance IoU — copied verbatim from train_sam3_kfold.py ──
                inst_iou = None
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
                    inter = (p_bin & g_bin).sum().item()
                    union = (p_bin | g_bin).sum().item()
                    inst_iou = inter / union if union > 0 else 0.0

                case = cases[i]
                rows.append((fold, case, sem_iou, inst_iou))
                sem_ious_fold.append(sem_iou)
                if inst_iou is not None:
                    inst_ious_fold.append(inst_iou)

        mean_sem = sum(sem_ious_fold) / len(sem_ious_fold)
        mean_inst = (sum(inst_ious_fold) / len(inst_ious_fold)
                      if inst_ious_fold else 0.0)
        print(f"  fold {fold}: mean sem_iou={mean_sem:.4f}  "
              f"mean inst_iou={mean_inst:.4f}")

    print("\n" + "=" * 60)
    print("AGGREGATE (bf16 autocast, exact training val-loop)")
    print("=" * 60)
    sem_all = [r[2] for r in rows if r[2] is not None]
    inst_all = [r[3] for r in rows if r[3] is not None]

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

    predictions = {
        case: {"fold": int(fold),
               "sem_iou": float(si),
               "inst_iou": float(ii)}
        for fold, case, si, ii in rows
    }
    write_predictions_json(predictions, THIS / "predictions" / "sam_kfold.json")


if __name__ == "__main__":
    main()
