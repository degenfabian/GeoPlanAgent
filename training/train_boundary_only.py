#!/usr/bin/env python3
"""
SAM3 LoRA Fine-tuning — Boundary-Only with semantic_seg head
=============================================================

Trains a LoRA adapter on SAM3's semantic segmentation head for
"planning boundary" extraction only. No road task.

Usage:
    cd training && uv run python3 train_boundary_only.py
    cd training && uv run python3 train_boundary_only.py --epochs 40 --rank 16
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

# Advanced boundary augmentations — style transfer only
try:
    from training.boundary_augmentations import style_transfer_augment
except ImportError:
    from boundary_augmentations import style_transfer_augment
from transformers import Sam3Processor, Sam3Model
from peft import LoraConfig, get_peft_model

# ============================================================================
# Config
# ============================================================================
MODEL_ID = "facebook/sam3"
DATA_DIR = Path(__file__).parent.parent / "boundary_annotation_dataset"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "sam3_lora_v4"

# Val split — same 4 images as before
VAL_NAMES = {
    "23_53149_ART4.png",
    "35046BA6-A370-41C1-8316-8797AF1524DD.png",
    "7202D619-4C27-4DA4-857E-B89F78C9D8D5.png",
    "FDBC0FDC-D090-4778-A123-232EB71DF3C6.png",
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "fc1", "fc2",
]

LOSS_WEIGHT_FOCAL = 5.0
LOSS_WEIGHT_DICE = 5.0

# Surface loss: ramp from 0 to max over first N epochs
SURFACE_LOSS_MAX = 0.5
SURFACE_LOSS_RAMP = 15  # shorter ramp since boundary-only converges faster

# Erosion consistency loss: encourages filled predictions (not thin outlines)
# which is correct — even for outline-styled boundaries the GT is the filled interior
LOSS_WEIGHT_EROSION = 0.5
EROSION_KERNEL_SIZE = 7


# ============================================================================
# Surface loss (Kervadec et al., MIDL 2019)
# ============================================================================
def compute_signed_distance_map(mask_np):
    mask_bool = mask_np > 0.5
    if mask_bool.all() or (~mask_bool).all():
        return np.zeros_like(mask_np, dtype=np.float32)
    dist_outside = distance_transform_edt(~mask_bool)
    dist_inside = distance_transform_edt(mask_bool)
    signed_dist = dist_outside - dist_inside
    max_abs = max(np.abs(signed_dist).max(), 1e-6)
    return (signed_dist / max_abs).astype(np.float32)


def surface_loss(pred_sigmoid, dist_map):
    return (pred_sigmoid * dist_map).mean()


# ============================================================================
# Dataset — simple image + mask pairs from sam3_semantic_boundary_dataset/
# ============================================================================
class BoundaryDataset(Dataset):
    """Loads boundary images + masks from sam3_semantic_boundary_dataset/."""

    def __init__(self, data_dir: Path, split: str, processor: Sam3Processor):
        self.processor = processor
        self.split = split
        self.img_dir = data_dir / "maps"
        self.mask_dir = data_dir / "boundary_masks"

        all_files = sorted([f for f in os.listdir(self.img_dir)
                           if f.endswith(('.png', '.jpg', '.tif'))])

        if split == "train":
            self.files = [f for f in all_files if f not in VAL_NAMES]
        else:
            self.files = [f for f in all_files if f in VAL_NAMES]

        self.all_train_files = [f for f in all_files if f not in VAL_NAMES]

        print(f"  {split}: {len(self.files)} boundary images")

    # Oversample multiplier: each base sample seen N times per epoch
    OVERSAMPLE = 8

    def __len__(self):
        if self.split == "train":
            return len(self.files) * self.OVERSAMPLE
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx % len(self.files)]

        image = Image.open(self.img_dir / fname).convert("RGB")
        mask_pil = Image.open(self.mask_dir / fname).convert("L")

        # Augmentations (training only) — applied jointly to image + mask
        if self.split == "train":
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
            # Random 90-degree rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                image = image.rotate(k * 90, expand=False)
                mask_pil = mask_pil.rotate(k * 90, expand=False, resample=Image.NEAREST)
            # Random scale + crop (0.7x–1.0x zoom, then resize back)
            if random.random() > 0.3:
                w, h = image.size
                scale = random.uniform(0.7, 1.0)
                crop_w, crop_h = int(w * scale), int(h * scale)
                x0 = random.randint(0, w - crop_w)
                y0 = random.randint(0, h - crop_h)
                image = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((w, h), Image.BILINEAR)
                mask_pil = mask_pil.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((w, h), Image.NEAREST)

            # Color jitter (image only) — brightness, contrast, saturation
            if random.random() > 0.5:
                for enhancer_cls in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
                    if random.random() > 0.5:
                        factor = random.uniform(0.7, 1.3)
                        image = enhancer_cls(image).enhance(factor)

            # Light Gaussian blur (image only)
            if random.random() > 0.5:
                sigma = random.uniform(0.5, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

            # --- Advanced augmentations ---

            # Style transfer: convert filled boundary to outline/dashed/dotted
            # with random color (50% probability)
            image, mask_pil = style_transfer_augment(image, mask_pil, p=0.5)

        # Mask to float numpy
        gt_mask = np.array(mask_pil).astype(np.float32) / 255.0

        inputs = self.processor(images=image, text="planning boundary", return_tensors="pt")
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_resized = F.interpolate(gt_tensor, size=(288, 288), mode="bilinear",
                                   align_corners=False).squeeze()
        # Threshold after bilinear resize to get clean binary mask (no soft edges)
        gt_resized = (gt_resized > 0.5).float()

        gt_for_dist = gt_resized.numpy().astype(np.float32)
        dist_map = torch.from_numpy(compute_signed_distance_map(gt_for_dist))

        return inputs, gt_resized, dist_map


def collate_fn(batch):
    inputs_list, masks_list, dist_maps_list = zip(*batch)
    gt_masks = torch.stack(masks_list)
    dist_maps = torch.stack(dist_maps_list)

    collated = {}
    for key in inputs_list[0].keys():
        vals = [inp[key] for inp in inputs_list]
        if isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals)
        else:
            collated[key] = vals

    return collated, gt_masks, dist_maps


# ============================================================================
# Loss
# ============================================================================
def sigmoid_focal_loss(pred, gt, alpha=0.75, gamma=2.0):
    pred_flat = pred.flatten(1)
    gt_flat = gt.flatten(1)
    bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="none")
    p_t = torch.sigmoid(pred_flat) * gt_flat + (1 - torch.sigmoid(pred_flat)) * (1 - gt_flat)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean(1)


def dice_loss(pred, gt, smooth=1.0):
    pred_flat = torch.sigmoid(pred).flatten(1)
    gt_flat = gt.flatten(1)
    intersection = (pred_flat * gt_flat).sum(1)
    return 1 - (2 * intersection + smooth) / (pred_flat.sum(1) + gt_flat.sum(1) + smooth)


def erosion_consistency_loss(pred_sigmoid, kernel_size=EROSION_KERNEL_SIZE):
    """Penalise predictions that vanish under erosion (i.e. thin outlines).

    Differentiable erosion via min-pooling (= negative max-pool of negated
    input).  A solid filled region survives erosion; a thin outline does not.
    The loss is 1 − (eroded_mass / original_mass), so it's 0 for solid
    regions and ~1 for thin outlines.
    """
    p = pred_sigmoid.unsqueeze(0) if pred_sigmoid.dim() == 2 else pred_sigmoid
    if p.dim() == 3:
        p = p.unsqueeze(1)  # [B, 1, H, W]
    eroded = -F.max_pool2d(-p, kernel_size, stride=1, padding=kernel_size // 2)
    return 1.0 - eroded.sum() / (p.sum() + 1e-6)


def boundary_loss(pred_mask, gt_mask, dist_map=None, epoch=0):
    fl = sigmoid_focal_loss(pred_mask, gt_mask)
    dl = dice_loss(pred_mask, gt_mask)
    loss = (LOSS_WEIGHT_FOCAL * fl + LOSS_WEIGHT_DICE * dl).mean()

    pred_sigmoid = torch.sigmoid(pred_mask)

    if dist_map is not None:
        alpha = min(SURFACE_LOSS_MAX, epoch / max(1, SURFACE_LOSS_RAMP) * SURFACE_LOSS_MAX)
        if alpha > 0:
            sl = surface_loss(pred_sigmoid, dist_map)
            loss = loss + alpha * sl

    # Erosion consistency: penalise thin/unfilled predictions
    ecl = erosion_consistency_loss(pred_sigmoid)
    loss = loss + LOSS_WEIGHT_EROSION * ecl

    return loss


# ============================================================================
# LR scheduler
# ============================================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ============================================================================
# Training
# ============================================================================
def train(args):
    print("=" * 60)
    print("SAM3 LoRA — Boundary-Only Semantic Seg Training")
    print("=" * 60)

    # Reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed: {seed}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"\nLoading SAM3 from {MODEL_ID}...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID)

    print(f"\nApplying LoRA (rank={args.rank}, alpha={args.alpha})...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=None,
        # CRITICAL: Save semantic_projection weights alongside LoRA adapters.
        # Without this, save_pretrained() only saves LoRA adapters and the
        # trained semantic_projection head is lost on reload.
        modules_to_save=["semantic_projection"],
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    model = model.to(device)

    print(f"\nLoading data from {DATA_DIR}...")
    train_ds = BoundaryDataset(DATA_DIR, "train", processor)
    val_ds = BoundaryDataset(DATA_DIR, "valid", processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"\nResuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path / "training_state.pt", map_location=device)
            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt["best_val_loss"]
            # Reload model with LoRA + modules_to_save from checkpoint
            from peft import PeftModel
            base_model = Sam3Model.from_pretrained(MODEL_ID)
            model = PeftModel.from_pretrained(base_model, ckpt_path).to(device)
            # Ensure semantic_projection requires grad after reload
            for name, param in model.named_parameters():
                if "semantic_projection" in name:
                    param.requires_grad_(True)
            # Rebuild optimizer with the reloaded model's parameters
            # (can't restore optimizer state — param groups differ after reload)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                          weight_decay=args.weight_decay)
            scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)
            # Fast-forward scheduler to correct step
            for _ in range(start_epoch * len(train_loader) // args.grad_accum):
                scheduler.step()
            print(f"  Resumed from epoch {start_epoch}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train: {len(train_ds)} boundary images, batch_size={args.batch_size}")
    print(f"  Val:   {len(val_ds)} boundary images")
    print(f"  Grad accum: {args.grad_accum}, effective batch: {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}, warmup: {args.warmup_steps} steps")
    print(f"  Loss: {LOSS_WEIGHT_FOCAL}*focal + {LOSS_WEIGHT_DICE}*dice + {LOSS_WEIGHT_EROSION}*erosion(k={EROSION_KERNEL_SIZE})")
    print(f"  Surface loss: ramp 0→{SURFACE_LOSS_MAX} over {SURFACE_LOSS_RAMP} epochs")
    print(f"  LoRA rank={args.rank}, alpha={args.alpha}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    train_losses_all = []
    val_losses_all = []
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()

        for step, (inputs, gt_masks, dist_maps) in enumerate(pbar):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            gt_masks = gt_masks.to(device)
            dist_maps = dist_maps.to(device)

            outputs = model(**inputs)
            pred = outputs.semantic_seg.squeeze(1)  # [B, H, W]

            B = pred.shape[0]
            batch_loss = torch.tensor(0.0, device=device)

            for b in range(B):
                pred_b = pred[b]
                gt_b = gt_masks[b]
                dm_b = dist_maps[b]

                if pred_b.shape != gt_b.shape:
                    pred_b = F.interpolate(pred_b.unsqueeze(0).unsqueeze(0), size=gt_b.shape[-2:],
                                           mode="bilinear", align_corners=False).squeeze()
                if dm_b.shape != gt_b.shape:
                    dm_b = F.interpolate(dm_b.unsqueeze(0).unsqueeze(0), size=gt_b.shape[-2:],
                                         mode="bilinear", align_corners=False).squeeze()

                sample_loss = boundary_loss(pred_b.unsqueeze(0), gt_b.unsqueeze(0),
                                            dist_map=dm_b.unsqueeze(0), epoch=epoch)
                batch_loss = batch_loss + sample_loss

            loss = batch_loss / B / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            raw_loss = loss.item() * args.grad_accum
            epoch_losses.append(raw_loss)
            pbar.set_postfix(loss=f"{raw_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train = sum(epoch_losses) / len(epoch_losses)
        train_losses_all.append(avg_train)
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_losses = []
        val_ious = []
        with torch.no_grad():
            for inputs, gt_masks, dist_maps in val_loader:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                gt_masks = gt_masks.to(device)
                dist_maps = dist_maps.to(device)

                outputs = model(**inputs)
                pred = outputs.semantic_seg.squeeze(1)
                B = pred.shape[0]

                for b in range(B):
                    pred_b = pred[b]
                    gt_b = gt_masks[b]
                    dm_b = dist_maps[b]

                    if pred_b.shape != gt_b.shape:
                        pred_b = F.interpolate(pred_b.unsqueeze(0).unsqueeze(0), size=gt_b.shape[-2:],
                                               mode="bilinear", align_corners=False).squeeze()
                    if dm_b.shape != gt_b.shape:
                        dm_b = F.interpolate(dm_b.unsqueeze(0).unsqueeze(0), size=gt_b.shape[-2:],
                                             mode="bilinear", align_corners=False).squeeze()

                    val_loss = boundary_loss(pred_b.unsqueeze(0), gt_b.unsqueeze(0),
                                             dist_map=dm_b.unsqueeze(0), epoch=epoch).item()
                    val_losses.append(val_loss)

                    # Compute IoU
                    pred_mask = (torch.sigmoid(pred_b) > 0.5).float()
                    gt_mask_bin = (gt_b > 0.5).float()
                    inter = (pred_mask * gt_mask_bin).sum().item()
                    union = (pred_mask + gt_mask_bin).clamp(max=1).sum().item()
                    iou = inter / union if union > 0 else 0.0
                    val_ious.append(iou)

        avg_val = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_val_iou = sum(val_ious) / len(val_ious) if val_ious else 0
        val_losses_all.append(avg_val)

        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(OUTPUT_DIR / "best")
            improved = " ** BEST **"

        surf_alpha = min(SURFACE_LOSS_MAX, epoch / max(1, SURFACE_LOSS_RAMP) * SURFACE_LOSS_MAX)
        print(f"  E{epoch+1}: train={avg_train:.4f} val={avg_val:.4f} "
              f"val_iou={avg_val_iou:.4f} surf_α={surf_alpha:.3f} [{elapsed:.0f}s]{improved}")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.save_pretrained(OUTPUT_DIR / "checkpoint_latest")
            torch.save({
                "epoch": epoch + 1, "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            }, OUTPUT_DIR / "checkpoint_latest" / "training_state.pt")

        # Save log
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump({
                "epoch": epoch + 1,
                "train_losses": train_losses_all, "val_losses": val_losses_all,
                "val_iou": avg_val_iou,
                "best_val_loss": best_val_loss, "config": vars(args),
            }, f, indent=2)

    # Final save
    model.save_pretrained(OUTPUT_DIR / "final")
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 LoRA boundary-only training")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (lower for single task)")
    parser.add_argument("--alpha", type=int, default=32,
                        help="LoRA alpha (2x rank)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
