"""K-fold-held-out pixel-IoU on the SAM3-LoRA finetune.

For each case in the training manifest, runs the LoRA from the fold that
was VALIDATED on it (= the model that NEVER saw it during training) and
computes pixel-IoU between the predicted semantic mask and the human-
traced ground-truth boundary mask.

This is the apples-to-apples comparable to MHCLG Extract's
"90% boundary tracing accuracy at IoU > 0.8" claim — shape-only, pixel-
space, on the same kind of cropped-map input.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import Sam3Model, Sam3Processor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASET_DIR = REPO / "training" / "dataset"  # v6 — what the LoRA actually trained on
MODELS_DIR = REPO / "models" / "sam3_lora"
MANIFEST_PATH = DATASET_DIR / "manifest.json"

MODEL_ID = "facebook/sam3"
DEFAULT_QUERY = "planning boundary"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
HEAD_MODULES = ["mask_embedder", "presence_head", "semantic_projection"]
RANK = 16

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")


def build_lora_model(base):
    cfg = LoraConfig(r=RANK, lora_alpha=RANK * 2,
                    target_modules=LORA_TARGET_MODULES,
                    lora_dropout=0.05, bias="none",
                    modules_to_save=HEAD_MODULES)
    return get_peft_model(base, cfg)


def load_fold_model(fold: int, base_already_wrapped):
    """Load best.pt for fold k into the already-wrapped PEFT model."""
    ckpt_path = MODELS_DIR / f"fold_{fold}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = base_already_wrapped.load_state_dict(state, strict=False)
    # Sanity: PEFT load with strict=False can silently drop everything if
    # the key prefix differs. Insist that at least the head modules loaded.
    leftovers = [k for k in unexpected if "lora_" in k or "modules_to_save" in k]
    if leftovers:
        print(f"  WARN fold {fold}: {len(leftovers)} adapter keys unexpected")
    return ckpt.get("epoch")


def compute_iou(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> float:
    # Hard shape assertion — silent broadcasting was producing IoU > 1.
    assert pred_bin.shape == gt_bin.shape, (
        f"shape mismatch in IoU: pred={tuple(pred_bin.shape)} "
        f"gt={tuple(gt_bin.shape)}")
    assert pred_bin.dim() == 2, f"expected [H,W], got {tuple(pred_bin.shape)}"
    inter = (pred_bin * gt_bin).sum().item()
    p_sum = pred_bin.sum().item()
    g_sum = gt_bin.sum().item()
    union = p_sum + g_sum - inter
    iou = inter / union if union > 0 else 0.0
    if iou < 0 or iou > 1.0001:
        return -1.0  # sentinel: caller should drop
    return min(1.0, iou)


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"manifest cases: {len(manifest)}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print("loading base SAM3 model…")
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=hf_token)
    base = Sam3Model.from_pretrained(MODEL_ID, token=hf_token)
    model = build_lora_model(base).to(device)
    model.eval()

    results = []  # (fold, case, sem_iou, inst_iou)

    for fold in range(5):
        cases = [r for r in manifest if r["fold"] == fold]
        if not cases:
            print(f"fold {fold}: no held-out cases")
            continue
        print(f"\n=== fold {fold}: {len(cases)} held-out cases ===")
        epoch = load_fold_model(fold, model)
        print(f"  loaded fold_{fold}/best.pt (best at epoch {epoch})")
        model.eval()

        for i, entry in enumerate(cases):
            case = entry["case"]
            fname = entry["filename"]
            img_path = DATASET_DIR / "maps" / fname
            mask_path = DATASET_DIR / "boundary_masks" / fname
            if not img_path.exists() or not mask_path.exists():
                print(f"  [{i+1}/{len(cases)}] {case}: MISSING input — skip")
                continue

            img = Image.open(img_path).convert("RGB")
            gt_np = np.asarray(Image.open(mask_path).convert("L"),
                                dtype=np.float32) / 255.0
            gt = torch.from_numpy(gt_np).to(device)

            inputs = processor(images=img, text=DEFAULT_QUERY,
                               return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Semantic head IoU — keep exact 4D shape through interpolate,
            # then reduce to [H,W] only once, with an assertion that we
            # actually got a 2D mask before computing IoU.
            sem_raw = outputs.semantic_seg.float()  # could be [B,K,H,W]
            # Reduce to [1,1,Hp,Wp]: select batch 0 and query 0
            if sem_raw.dim() == 4:
                sem_4d = sem_raw[0:1, 0:1]  # [1, 1, Hp, Wp]
            elif sem_raw.dim() == 3:
                sem_4d = sem_raw[0:1].unsqueeze(1)  # [1, 1, Hp, Wp]
            elif sem_raw.dim() == 2:
                sem_4d = sem_raw.unsqueeze(0).unsqueeze(0)
            else:
                print(f"  unexpected semantic_seg shape: {tuple(sem_raw.shape)}")
                continue
            gt_4d = gt.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            if sem_4d.shape[-2:] != gt_4d.shape[-2:]:
                sem_4d = F.interpolate(sem_4d, size=gt_4d.shape[-2:],
                                       mode="bilinear", align_corners=False)
            sem_pred_2d = sem_4d.squeeze(0).squeeze(0)
            gt_2d = gt
            assert sem_pred_2d.shape == gt_2d.shape, (
                f"shape post-interp: {sem_pred_2d.shape} vs {gt_2d.shape}")
            sem_bin = (torch.sigmoid(sem_pred_2d) > 0.5).float()
            gt_bin = (gt_2d > 0.5).float()
            sem_iou = compute_iou(sem_bin, gt_bin)

            # Instance head IoU (top-1 by class logit). Mirror training loop.
            inst_iou = None
            try:
                inst_masks = outputs.pred_masks.float()[0]
                cls_logits = outputs.pred_logits.float()[0]
                if inst_masks.dim() == 4:
                    inst_masks = inst_masks.view(-1, inst_masks.shape[-2],
                                                  inst_masks.shape[-1])
                cls_b = cls_logits.view(-1)[:inst_masks.shape[0]]
                top_idx = int(cls_b.argmax().item())
                inst_pred = inst_masks[top_idx]
                # Collapse any leftover singleton dims to [H, W]
                while inst_pred.dim() > 2:
                    inst_pred = inst_pred[0]
                if inst_pred.shape != gt.shape:
                    inst_pred = F.interpolate(
                        inst_pred.unsqueeze(0).unsqueeze(0),
                        size=gt.shape[-2:], mode="bilinear",
                        align_corners=False).squeeze()
                inst_bin = (torch.sigmoid(inst_pred) > 0.5).float()
                inst_iou = compute_iou(inst_bin, gt_bin)
                if inst_iou > 1.0 or inst_iou < 0:
                    inst_iou = None
            except Exception:
                inst_iou = None

            results.append((fold, case, sem_iou, inst_iou))
            print(f"  [{i+1}/{len(cases)}] {case:<40} "
                  f"sem={sem_iou:.3f} inst={inst_iou:.3f}" if inst_iou is not None
                  else f"  [{i+1}/{len(cases)}] {case:<40} sem={sem_iou:.3f}")

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE: shape-only pixel IoU on k-fold held-out cases")
    print("=" * 60)

    sem_all = [r[2] for r in results if r[2] is not None]
    inst_all = [r[3] for r in results if r[3] is not None]

    def summarise(name, xs):
        if not xs:
            print(f"{name}: no data"); return
        n = len(xs)
        s = sorted(xs)
        mean = sum(xs) / n
        med = s[n // 2]
        ge_50 = sum(1 for x in xs if x >= 0.5) / n
        ge_70 = sum(1 for x in xs if x >= 0.7) / n
        ge_80 = sum(1 for x in xs if x >= 0.8) / n
        ge_90 = sum(1 for x in xs if x >= 0.9) / n
        print(f"\n{name} (N={n})")
        print(f"  mean   = {mean:.4f}")
        print(f"  median = {med:.4f}")
        print(f"  >=0.50 = {ge_50*100:.1f}%")
        print(f"  >=0.70 = {ge_70*100:.1f}%")
        print(f"  >=0.80 = {ge_80*100:.1f}%   <-- comparable to MHCLG Extract 90%")
        print(f"  >=0.90 = {ge_90*100:.1f}%")

    summarise("Semantic-head IoU", sem_all)
    summarise("Instance-head IoU", inst_all)

    # Per-fold
    print("\nPer-fold mean semantic IoU:")
    for k in range(5):
        xs = [r[2] for r in results if r[0] == k and r[2] is not None]
        if xs:
            ge80 = sum(1 for x in xs if x >= 0.8) / len(xs)
            print(f"  fold {k}: N={len(xs)} mean={sum(xs)/len(xs):.4f} "
                  f">=0.80={ge80*100:.1f}%")

    # CSV
    out_path = REPO / "results" / "benchmark_v21" / "sam_kfold_shape_iou.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("fold,case,sem_iou,inst_iou\n")
        for fold, case, si, ii in results:
            f.write(f"{fold},{case},{si},{ii}\n")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
