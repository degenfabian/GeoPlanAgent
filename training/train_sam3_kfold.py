"""SAM3 LoRA fine-tuning — 5-fold CV with combined semantic + instance loss.

Each of the 5 folds trains a fresh LoRA adapter on ~96 train / ~24 val
cases (deterministic md5(case_name) % 5 split). Both segmentation
heads contribute to the loss:

  semantic_seg          focal + dice + surface (ramped over first 15
                         epochs). The model's main mask output.
  instance pred_masks   best-IoU proposal across the N slots →
                         focal + dice on that one. No presence loss:
                         we want the other N-1 slots to stay diverse
                         so the agent's `mode='instance'` flow gets
                         useful alternatives at inference.

Per-fold checkpoints land in models/sam3_lora_v5/fold_<k>/:
  latest.pt   rewritten every epoch — resume target
  best.pt     rewritten when val IoU improves
  history.json  per-epoch train/val loss + IoU for analysis

fold_assignment.json (case_name → fold index) is mirrored from
training/dataset_v5/ to models/sam3_lora_v5/ so production can look up
the per-case checkpoint at inference time without colocating the
training set.

For reproducibility, every random source is seeded from --seed (default
42). With --bf16 disabled and the same seed + dataset, two runs of the
same fold produce bit-identical results on the same hardware. With bf16
on you get tiny float-rounding deltas but the trajectory is the same.

See training/README.md for the full data-pipeline reproduction recipe
(auto-label → review → build-curated-set → train).

Usage:
    cd training && uv run python train_sam3_kfold.py
    cd training && uv run python train_sam3_kfold.py --folds 0,1,2 --epochs 25
    cd training && uv run python train_sam3_kfold.py --resume

Wall: roughly 1.5-2 hours per fold on Apple MPS with bf16. Start
overnight if running all 5.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

try:
    from training.boundary_augmentations import style_transfer_augment
except ImportError:
    from boundary_augmentations import style_transfer_augment

from peft import LoraConfig, get_peft_model
from transformers import Sam3Model, Sam3Processor


# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/sam3"
DATASET_DIR = REPO / "training" / "dataset"
OUTPUT_BASE = REPO / "models" / "sam3_lora"

# Broad LoRA scope across ALL transformer subsystems (vision_encoder,
# text_encoder, geometry_encoder, detr_encoder, detr_decoder, mask_decoder).
# This is what v4 used and what made v4's instance candidates dramatically
# better than v6's restricted-scope candidates: even though v4's loss only
# supervised the semantic head, gradients flowing back through ALL layers
# made the visual features themselves more "planning-boundary aware", and
# the instance head's pretrained logic produced better candidates as a
# byproduct.
#
# The third audit's concern about "overfitting on 92 text-region mappings"
# doesn't apply here because we use ONE prompt during training ("planning
# boundary"). The text encoder just learns one good embedding; it can't
# overfit when there's only one input to it. The vision tower, by contrast,
# has 92 different images and benefits enormously from adaptation.
#
# Match count: ~490 modules (matches every q/k/v/o/fc1/fc2 in any
# subsystem), vs the restricted decoder-only regex at 88 modules.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

# Loss weights. Mask losses use focal+dice (DETR/Mask2Former defaults).
# Instance head also gets per-slot classification BCE (matched=1, unmatched=0)
# and image-level presence BCE — matches the SAM3 paper's training recipe
# (Hungarian-matched detection + presence head, BCE on object scores when
# concept is present in the image).
LOSS_WEIGHT_SEM_FOCAL = 5.0
LOSS_WEIGHT_SEM_DICE = 5.0
LOSS_WEIGHT_SURFACE_MAX = 0.5
LOSS_WEIGHT_INST_FOCAL = 5.0
LOSS_WEIGHT_INST_DICE = 5.0
LOSS_WEIGHT_INST_CLS = 2.0
LOSS_WEIGHT_INST_PRES = 1.0
SURFACE_LOSS_RAMP = 15

DEFAULT_QUERY = "planning boundary"


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's Python RNG from torch.initial_seed().

    Defined at module scope (not inside train_fold) so spawn-based
    multiprocessing can pickle it. Without per-worker seeding, fork/spawn
    workers inherit identical Python `random` and `numpy.random` state
    from the main process, and augmentation in __getitem__ produces
    correlated outputs across workers — oversample becomes wasted compute.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def seed_everything(seed: int) -> torch.Generator:
    """Seed every RNG source we touch. Returns a torch.Generator the
    DataLoader uses for its shuffle, so shuffle order is also reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Try to make the few non-deterministic CUDA kernels deterministic;
    # MPS doesn't expose this but is mostly deterministic anyway.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Loss components ────────────────────────────────────────────────────────

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


def sigmoid_focal_loss(pred, gt, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss with SAM3-author defaults (alpha=0.25, gamma=2).

    alpha=0.25 down-weights positive pixels and up-weights negatives, the
    standard imbalance handling for detection/instance-mask losses where
    positives are sparse. SAM3's `Masks` loss in
    sam3/train/loss/loss_fns.py uses these exact defaults.
    """
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
    p_t = p * gt + (1 - p) * (1 - gt)
    a_t = alpha * gt + (1 - alpha) * (1 - gt)
    return a_t * (1 - p_t) ** gamma * ce


def dice_loss(pred, gt, smooth=1.0):
    """Dice loss matching SAM3's formula: 1 - (2·inter + s) / (|A| + |B| + s).

    Note the denominator is sum(A) + sum(B) — NOT union (which would
    subtract intersection). Subtracting intersection turns this into soft-IoU
    loss, which has different gradient dynamics. SAM3's
    sam3/train/loss/loss_fns.py uses dice, so we do too.
    """
    p = torch.sigmoid(pred)
    inter = (p * gt).sum()
    denom = p.sum() + gt.sum()
    return 1 - (2 * inter + smooth) / (denom + smooth)


def semantic_loss(pred_mask, gt_mask, dist_map=None, epoch=0):
    # Semantic head uses SAM3's SemanticSegCriterion defaults: focal
    # alpha=0.6, gamma=1.6 (a milder positive down-weighting than the
    # instance/mask loss because semantic GT covers larger regions).
    fl = sigmoid_focal_loss(pred_mask, gt_mask, alpha=0.6, gamma=1.6).mean()
    dl = dice_loss(pred_mask, gt_mask)
    loss = LOSS_WEIGHT_SEM_FOCAL * fl + LOSS_WEIGHT_SEM_DICE * dl

    if dist_map is not None:
        ramp = min(1.0, epoch / max(1, SURFACE_LOSS_RAMP))
        if ramp > 0:
            pred_sig = torch.sigmoid(pred_mask)
            sl = surface_loss(pred_sig, dist_map)
            loss = loss + ramp * LOSS_WEIGHT_SURFACE_MAX * sl

    # (erosion-consistency removed: was a band-aid for outline-style
    #  hand-drawn masks; auto-labelled masks are filled-poly already and
    #  it actively hurts multi-blob cases by encouraging single-blob
    #  predictions.)
    return loss


def _downsample_gt_to(target_hw: Tuple[int, int], gt: torch.Tensor) -> torch.Tensor:
    """Resize a 2D GT mask down/up to target (H, W). Faster and MPS-safer
    than upsampling predictions to GT resolution."""
    if gt.shape[-2:] == target_hw:
        return gt
    g = F.interpolate(gt.float().unsqueeze(0).unsqueeze(0),
                       size=target_hw, mode="bilinear",
                       align_corners=False).squeeze(0).squeeze(0)
    return g


def instance_loss(pred_masks, pred_logits, presence_logits, gt_mask):
    """SAM3-style instance loss: Hungarian-matched mask + per-slot
    classification BCE + image-level presence BCE.

    Our data has exactly ONE GT polygon per training image, so the
    bipartite match degenerates to "pick the slot with lowest cost".
    Cost is dominated by the mask focal + mask dice terms, same as the
    SAM3 paper's matcher (sam3/train/matcher.py + loss/loss_fns.py:
    DETR-style cost_class + cost_mask_focal + cost_mask_dice).

    Three loss components, mirroring the official recipe:

    1. Mask loss on matched slot → focal + dice (DETR / Mask2Former /
       SAM3 standard).
    2. Per-slot classification (focal, alpha=0.25, gamma=2) → target=1
       for matched slot, target=0 for the other ~199. SAM3 has 200 query
       slots so this is a heavy 1:199 imbalance — focal is essential.
       Anchoring via cls supervision was missing in v5 and let LoRA-
       backbone updates drag the unsupervised slots into garbage.
    3. Image-level presence BCE → target=1 always (every training
       image contains the "planning boundary" concept by construction).

    Resolution handling: bipartite match runs at native res (cheap; we
    materialise [N_slots, H_p, W_p] only as sigmoid sums). The mask
    LOSS upsamples only the ONE matched slot to GT res — preserves
    thin-boundary detail that a 16× GT downsample would erase, without
    materialising the full [N, H_gt, W_gt] tensor that blows up MPS.

    pred_masks      [N, H_p, W_p] or [B, N, H_p, W_p]  per-slot mask logits
    pred_logits     [N] or [N, 1]  per-slot classification logits, may be None
    presence_logits scalar or 1-D  image-level presence logit, may be None
    gt_mask         [H, W]         target binary mask

    Returns: (total_loss, best_idx)
    """
    if pred_masks.dim() == 4:
        pred_masks = pred_masks.view(-1, pred_masks.shape[-2], pred_masks.shape[-1])
    N = pred_masks.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=gt_mask.device), 0

    # Match at native (low) resolution — cheap over 200 slots.
    # Cost includes a class-score term in addition to mask IoU. SAM3's
    # HungarianMatcher composes cost from class + bbox + GIoU; we
    # approximate with cost = -IoU - lambda * sigmoid(cls_logit) so the
    # matched slot is the one that's BOTH a good mask AND has high
    # classification confidence. Pure IoU matching lets matched-slot
    # identity oscillate epoch-to-epoch when several slots have similar
    # mask quality but different cls confidences — that oscillation
    # undoes prior cls-head learning.
    gt_small = _downsample_gt_to(pred_masks.shape[-2:], gt_mask)
    p = torch.sigmoid(pred_masks)
    inter = (p * gt_small.unsqueeze(0)).sum(dim=(-2, -1))
    p_sum = p.sum(dim=(-2, -1))
    gt_sum = gt_small.sum()
    union = p_sum + gt_sum - inter
    iou = inter / (union + 1e-6)

    cost = -iou
    if pred_logits is not None and pred_logits.numel() > 0:
        # Class-cost coefficient is small (0.05). At init sigmoid(cls)≈0.5
        # so the cls term contributes ~0.025; raw IoU on a 16× downsampled
        # mask is often <0.05 too, so a larger coefficient (originally 0.5)
        # would dominate matching with random cls signal in early epochs
        # and lock matching onto a randomly-initialised slot. With 0.05
        # the IoU term steers matching while cls only breaks ties.
        cls_for_match = pred_logits.reshape(-1)[:N]
        cost = cost - 0.05 * torch.sigmoid(cls_for_match)
    best_idx = int(cost.argmin().item())

    # 1. Mask loss on matched slot — UPSAMPLED to GT resolution. Only
    # one slot is upsampled (cheap), preserving thin-boundary detail.
    best_native = pred_masks[best_idx]
    best_up = F.interpolate(
        best_native.unsqueeze(0).unsqueeze(0),
        size=gt_mask.shape[-2:], mode="bilinear",
        align_corners=False).squeeze(0).squeeze(0)
    fl = sigmoid_focal_loss(
        best_up.unsqueeze(0), gt_mask.unsqueeze(0),
        alpha=0.25, gamma=2.0).mean()
    dl = dice_loss(best_up.unsqueeze(0), gt_mask.unsqueeze(0))
    mask_l = LOSS_WEIGHT_INST_FOCAL * fl + LOSS_WEIGHT_INST_DICE * dl

    # 2. Per-slot classification — focal (alpha=0.25, gamma=2). SAM3's
    # num_queries=200 means a 1:199 positive:negative split, so focal is
    # essential to keep easy negatives from drowning the matched slot.
    #
    # Matched-slot target uses a SOFT label = sigmoid(pred_logits) ** 0.25
    # * iou ** 0.75 (clamped at 0.01), not hard 1.0. This mirrors SAM3's
    # IABCEMdetr loss (sam3/train/loss/loss_fns.py:369-373): the cls logit
    # should track mask quality, not get pushed to infinity for an OK
    # mask. Hard 1.0 saturates the cls head and decouples it from mask
    # quality, which destabilises matched-slot identity across epochs.
    cls_l = torch.tensor(0.0, device=pred_masks.device)
    if pred_logits is not None and pred_logits.numel() > 0:
        # HF Sam3Model emits pred_logits at (B, N) so per-image is shape (N,).
        # Assert the expected rank/shape so a future HF API change surfaces
        # loud rather than silently zeroing the cls supervision.
        cls_flat = pred_logits.reshape(-1)
        assert cls_flat.shape[0] == N, (
            f"pred_logits expected shape ({N},), got {tuple(pred_logits.shape)}")
        with torch.no_grad():
            cls_target = torch.zeros_like(cls_flat)
            # Soft positive target = prob^0.25 * iou^0.75
            best_iou = float(iou[best_idx].clamp(min=0.0, max=1.0).item())
            best_prob = float(torch.sigmoid(cls_flat[best_idx]).item())
            soft_t = max(0.01, (best_prob ** 0.25) * (best_iou ** 0.75))
            cls_target[best_idx] = soft_t
        cls_focal = sigmoid_focal_loss(
            cls_flat, cls_target, alpha=0.25, gamma=2.0)
        cls_l = LOSS_WEIGHT_INST_CLS * cls_focal.mean()

    # 3. Image-level presence BCE — concept always present in our training data
    pres_l = torch.tensor(0.0, device=pred_masks.device)
    if presence_logits is not None and presence_logits.numel() > 0:
        pl = presence_logits.view(-1)
        target = torch.ones_like(pl)
        pres_l = LOSS_WEIGHT_INST_PRES * F.binary_cross_entropy_with_logits(
            pl, target, reduction="mean")

    return mask_l + cls_l + pres_l, best_idx


# ── Dataset ────────────────────────────────────────────────────────────────


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(s: str) -> str:
    return _FILENAME_SAFE_RE.sub("_", s)


def _build_manifest_from_disk(dataset_dir: Path,
                               fold_map: Dict[str, int]) -> List[Dict]:
    """Return [{case, filename, fold}, ...] from `maps/` + fold_assignment.

    ``case`` is the original case name (matching benchmark_runner's
    eval-data folder names). ``filename`` is the on-disk PNG name
    (filesystem-safe form). The reverse mapping uses fold_assignment.json,
    which records both forms (original + safe) so the file stem can be
    inverted back to the original. Falls back to the stem itself when no
    inverse mapping exists (cases whose original name was already
    filesystem-safe).
    """
    safe_to_original = {}
    for orig in fold_map:
        safe = _safe_filename(orig)
        if safe != orig:
            safe_to_original.setdefault(safe, orig)

    manifest = []
    for png in sorted((dataset_dir / "maps").glob("*.png")):
        fold = fold_map.get(png.stem)
        if fold is None:
            continue
        case = safe_to_original.get(png.stem, png.stem)
        manifest.append({"case": case,
                          "filename": png.name,
                          "fold": int(fold)})
    return manifest


class FoldDataset(Dataset):
    def __init__(self, dataset_dir: Path, manifest: List[Dict],
                  fold: int, split: str, processor: Sam3Processor,
                  oversample: int = 2):
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.split = split
        self.oversample = oversample
        if split == "train":
            self.entries = [r for r in manifest if r["fold"] != fold]
        else:
            self.entries = [r for r in manifest if r["fold"] == fold]
        print(f"  fold {fold} {split}: {len(self.entries)} cases"
              + (f" × {oversample} oversample" if split == "train" else ""))

    def __len__(self):
        return len(self.entries) * (self.oversample if self.split == "train" else 1)

    def __getitem__(self, idx):
        entry = self.entries[idx % len(self.entries)]
        fname = entry["filename"]
        img = Image.open(self.dataset_dir / "maps" / fname).convert("RGB")
        mask = Image.open(self.dataset_dir / "boundary_masks" / fname).convert("L")

        if self.split == "train":
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # Style-transfer augmentation: re-renders the boundary in random
            # styles (outline, dashed, dotted, hatched) and colors. Required
            # by the dataset design — the model has to recognise boundaries
            # under styles other than what the auto-labeller happened to draw.
            # Takes (image, mask) and returns (image, mask) — the previous
            # one-arg call was silently raising TypeError every step.
            img, mask = style_transfer_augment(img, mask)
            if random.random() > 0.5:
                img = ImageEnhance.Brightness(img).enhance(0.8 + random.random() * 0.4)
            if random.random() > 0.5:
                img = ImageEnhance.Contrast(img).enhance(0.8 + random.random() * 0.4)

        inputs = self.processor(images=img, text=DEFAULT_QUERY,
                                  return_tensors="pt")
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                   for k, v in inputs.items()}

        gt = np.asarray(mask, dtype=np.float32) / 255.0
        gt_t = torch.from_numpy(gt)
        dm = torch.from_numpy(compute_signed_distance_map(gt))
        return inputs, gt_t, dm


def collate(batch):
    inputs_list, gts, dms = zip(*batch)
    keys = inputs_list[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in inputs_list]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals
    return out, list(gts), list(dms)


# ── Train one fold ─────────────────────────────────────────────────────────

def _ensure_pred_mask_on_gt(pred, gt):
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                              size=gt.shape[-2:],
                              mode="bilinear", align_corners=False).squeeze()
    return pred


def _autocast_ctx(device: str, enabled: bool):
    """Mixed-precision autocast that works for both CUDA and MPS.

    bf16 on CUDA gives ~1.5-2x speedup with no quality loss. MPS supports
    autocast in fp16 (bf16 not yet on MPS as of recent torch). Both
    paths fall through cleanly when enabled=False.
    """
    if not enabled:
        # null context manager
        from contextlib import nullcontext
        return nullcontext()
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if device == "mps":
        return torch.amp.autocast("mps", dtype=torch.float16)
    return torch.amp.autocast("cpu", dtype=torch.bfloat16)


def train_fold(fold: int, args, manifest: List[Dict], processor: Sam3Processor,
                device: str) -> Dict:
    out_dir = OUTPUT_BASE / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n=== FOLD {fold} → {out_dir}\n{'='*70}")

    # Per-fold seeding so each fold's RNG state is deterministic, but
    # different folds explore different augmentation sequences.
    g = seed_everything(args.seed + fold)

    # Datasets
    train_ds = FoldDataset(DATASET_DIR, manifest, fold, "train", processor,
                            oversample=args.oversample)
    val_ds = FoldDataset(DATASET_DIR, manifest, fold, "valid", processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                collate_fn=collate, generator=g,
                                worker_init_fn=_worker_init_fn,
                                persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate)

    # Fresh model + LoRA per fold
    base = Sam3Model.from_pretrained(MODEL_ID)
    # Fully train the final prediction MLPs in addition to LoRA on the
    # upstream transformer blocks. v4 proved this pattern works for the
    # semantic head; we mirror it for both heads when both are trained.
    #   mask_embedder       (mask_decoder)  → produces pred_masks
    #   presence_head       (detr_decoder)  → produces pred_logits AND
    #                                          presence_logits (shared MLP)
    #   semantic_projection (mask_decoder)  → final 1×1 conv that produces
    #                                          semantic_seg
    head_modules = ["mask_embedder", "presence_head", "semantic_projection"]
    lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                            target_modules=LORA_TARGET_MODULES,
                            lora_dropout=0.05, bias="none",
                            modules_to_save=head_modules)
    model = get_peft_model(base, lora_cfg).to(device)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, total_steps), eta_min=args.lr * 0.05)

    # Resume
    start_epoch = 0
    best_val_iou = 0.0
    epochs_since_best = 0
    history: List[Dict] = []
    latest_p = out_dir / "latest.pt"
    if args.resume and latest_p.exists():
        ckpt = torch.load(latest_p, map_location="cpu", weights_only=False)
        # strict=True surfaces LoRA-target-module additions or renames
        # immediately. PEFT produces a stable key set for the same config,
        # so this should never hit unless config drifted between runs.
        model.load_state_dict(ckpt["state_dict"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        # Restore scheduler state directly when present (saved post-fix);
        # fall back to the legacy replay path for older checkpoints that
        # didn't persist sched state. The replay works for stateless
        # schedulers like CosineAnnealingLR but won't survive a switch
        # to a stateful one (OneCycleLR, anything with momentum).
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        else:
            for _ in range(ckpt["global_step"]):
                sched.step()
        start_epoch = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        epochs_since_best = ckpt.get("epochs_since_best", 0)
        history = ckpt.get("history") or []
        print(f"  Resumed at epoch {start_epoch}, best_val_iou={best_val_iou:.3f}, "
              f"epochs_since_best={epochs_since_best}")

    global_step = (start_epoch * len(train_loader)) // args.grad_accum

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        ep_losses = {"total": [], "sem": [], "inst": []}
        pbar = tqdm(train_loader, desc=f"fold{fold} ep{epoch+1}/{args.epochs}")
        optim.zero_grad()

        for step, (inputs, gts, dms) in enumerate(pbar):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in inputs.items()}
            gts_d = [g.to(device) for g in gts]
            dms_d = [d.to(device) for d in dms]

            with _autocast_ctx(device, args.bf16):
                outputs = model(**inputs)
            # Cast outputs to fp32. Sam3MaskDecoder.forward computes
            # both heads unconditionally so we always have semantic_seg
            # available. Cast is only ~0.3 MB at native res; no reason
            # to make it
            # live in fp32 if we never read from it.
            sem_pred = outputs.semantic_seg.squeeze(1).float()
            inst_masks = getattr(outputs, "pred_masks", None)
            if inst_masks is not None:
                inst_masks = inst_masks.float()
            inst_logits = getattr(outputs, "pred_logits", None)
            if inst_logits is not None:
                inst_logits = inst_logits.float()
            presence = getattr(outputs, "presence_logits", None)
            if presence is not None:
                presence = presence.float()

            B = sem_pred.shape[0]
            sem_l_total = torch.tensor(0.0, device=device)
            inst_l_total = torch.tensor(0.0, device=device)

            for b in range(B):
                pred_b = _ensure_pred_mask_on_gt(sem_pred[b], gts_d[b])
                dm_b = (dms_d[b] if dms_d[b].shape == gts_d[b].shape
                         else F.interpolate(dms_d[b].unsqueeze(0).unsqueeze(0),
                                              size=gts_d[b].shape[-2:],
                                              mode="bilinear",
                                              align_corners=False).squeeze())
                sem_l_total = sem_l_total + semantic_loss(
                    pred_b.unsqueeze(0), gts_d[b].unsqueeze(0),
                    dist_map=dm_b.unsqueeze(0), epoch=epoch)

                if inst_masks is not None and inst_masks.numel() > 0:
                    inst_b = inst_masks[b]
                    cls_b = inst_logits[b] if inst_logits is not None else None
                    pres_b = presence[b] if presence is not None else None
                    il, _ = instance_loss(inst_b, cls_b, pres_b, gts_d[b])
                    inst_l_total = inst_l_total + il

            sem_l = sem_l_total / B
            inst_l = inst_l_total / B
            loss = sem_l + inst_l
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1

            ep_losses["total"].append(loss.item())
            ep_losses["sem"].append(sem_l.item())
            ep_losses["inst"].append(inst_l.item())
            pbar.set_postfix(
                tot=f"{loss.item():.3f}",
                sem=f"{sem_l.item():.3f}",
                inst=f"{inst_l.item():.3f}",
                lr=f"{sched.get_last_lr()[0]:.2e}")

        avg_train = {k: (sum(v) / len(v) if v else 0.0)
                     for k, v in ep_losses.items()}

        # Validation: measure both heads. Gate on the SEMANTIC head (the
        # user-facing metric for the paper). Per-case precision/recall/F1
        # of the semantic head are tracked so the cross-fold summary has
        # paper-grade numbers, not just IoU.
        model.eval()
        inst_ious, sem_ious = [], []
        sem_precisions, sem_recalls, sem_f1s, sem_dices = [], [], [], []
        inst_losses = []
        with torch.no_grad():
            for inputs, gts, dms in val_loader:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()}
                with _autocast_ctx(device, args.bf16):
                    outputs = model(**inputs)

                # Instance head IoU (always measured)
                inst_masks = getattr(outputs, "pred_masks", None)
                cls_logits = getattr(outputs, "pred_logits", None)
                if inst_masks is not None and cls_logits is not None:
                    inst_masks = inst_masks.float()
                    cls_logits = cls_logits.float()
                    for b in range(inst_masks.shape[0]):
                        g = gts[b].to(device)
                        slots = inst_masks[b]
                        if slots.dim() == 4:
                            slots = slots.view(-1, slots.shape[-2], slots.shape[-1])
                        cls_b = cls_logits[b].view(-1)[:slots.shape[0]]
                        top_idx = int(cls_b.argmax().item())
                        pred_up = F.interpolate(
                            slots[top_idx].unsqueeze(0).unsqueeze(0),
                            size=g.shape[-2:], mode="bilinear",
                            align_corners=False).squeeze(0).squeeze(0)
                        fl = sigmoid_focal_loss(
                            pred_up.unsqueeze(0), g.unsqueeze(0),
                            alpha=0.25, gamma=2.0).mean()
                        dl = dice_loss(pred_up.unsqueeze(0), g.unsqueeze(0))
                        inst_losses.append((LOSS_WEIGHT_INST_FOCAL * fl
                                              + LOSS_WEIGHT_INST_DICE * dl).item())
                        p_bin = (torch.sigmoid(pred_up) > 0.5).float()
                        g_bin = (g > 0.5).float()
                        inter = (p_bin * g_bin).sum().item()
                        union = (p_bin + g_bin).clamp(max=1).sum().item()
                        inst_ious.append(inter / union if union > 0 else 0.0)

                # Semantic head IoU + paper metrics (precision / recall /
                # F1 / Dice).
                sem_pred = outputs.semantic_seg.squeeze(1).float()
                for b in range(sem_pred.shape[0]):
                    g = gts[b].to(device)
                    pred = _ensure_pred_mask_on_gt(sem_pred[b], g)
                    p_bin = (torch.sigmoid(pred) > 0.5).float()
                    g_bin = (g > 0.5).float()
                    inter = (p_bin * g_bin).sum().item()
                    p_sum = p_bin.sum().item()
                    g_sum = g_bin.sum().item()
                    union = p_sum + g_sum - inter
                    sem_ious.append(inter / union if union > 0 else 0.0)
                    # Precision = TP / (TP + FP), Recall = TP / (TP + FN)
                    prec = inter / p_sum if p_sum > 0 else 0.0
                    rec = inter / g_sum if g_sum > 0 else 0.0
                    sem_precisions.append(prec)
                    sem_recalls.append(rec)
                    sem_f1s.append(2 * prec * rec / (prec + rec)
                                   if (prec + rec) > 0 else 0.0)
                    sem_dices.append(2 * inter / (p_sum + g_sum)
                                     if (p_sum + g_sum) > 0 else 0.0)

        def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
        avg_inst_iou = _mean(inst_ious)
        avg_sem_iou = _mean(sem_ious)
        avg_sem_prec = _mean(sem_precisions)
        avg_sem_rec = _mean(sem_recalls)
        avg_sem_f1 = _mean(sem_f1s)
        avg_sem_dice = _mean(sem_dices)
        avg_inst_loss = _mean(inst_losses)
        # Early-stop / best-checkpoint key: SEMANTIC IoU (the paper-grade head).
        avg_iou = avg_sem_iou if sem_ious else avg_inst_iou
        avg_val = avg_inst_loss

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "wall_s": round(elapsed, 1),
                          **{f"train_{k}": round(v, 4)
                              for k, v in avg_train.items()},
                          "val_loss": round(avg_val, 4),
                          "val_iou": round(avg_iou, 4),
                          "val_inst_iou": round(avg_inst_iou, 4),
                          "val_sem_iou": round(avg_sem_iou, 4),
                          "val_sem_precision": round(avg_sem_prec, 4),
                          "val_sem_recall": round(avg_sem_rec, 4),
                          "val_sem_f1": round(avg_sem_f1, 4),
                          "val_sem_dice": round(avg_sem_dice, 4)})
        sem_str = f"  sem_iou={avg_sem_iou:.3f}  sem_f1={avg_sem_f1:.3f}"
        print(f"  ep{epoch+1}: train={avg_train['total']:.3f} "
              f"(sem={avg_train['sem']:.3f} inst={avg_train['inst']:.3f})  "
              f"inst_iou={avg_inst_iou:.3f}{sem_str}  "
              f"val_loss={avg_val:.3f}  wall={elapsed:.0f}s")

        # Update epochs_since_best BEFORE the checkpoint save so resumed
        # runs see the correct counter (otherwise the saved value lags by
        # one epoch and patience-based early stopping is off-by-one across
        # a resume boundary).
        new_best = avg_iou > best_val_iou
        if new_best:
            best_val_iou = avg_iou
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Save checkpoint. Includes scheduler state (so resume doesn't have
        # to replay sched.step() count-by-count, which only worked for
        # stateless schedulers).
        ckpt = {
            "state_dict": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_iou": best_val_iou,
            "epochs_since_best": epochs_since_best,
            "history": history,
            "fold": fold,
            "config": {"rank": args.rank, "lr": args.lr,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "oversample": args.oversample,
                        "num_workers": args.num_workers,
                        "seed": args.seed,
                        "bf16": bool(args.bf16),
                        "patience": args.patience},
        }
        torch.save(ckpt, latest_p)
        if new_best:
            torch.save(ckpt, out_dir / "best.pt")
            print(f"    new best val_iou={best_val_iou:.3f}, saved best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Early stopping: if val IoU hasn't improved for `patience` epochs,
        # stop this fold. Saves wall on the back end of training where
        # the model has converged but we'd otherwise keep going.
        if args.patience > 0 and epochs_since_best >= args.patience:
            print(f"    early stopping: no val_iou improvement for "
                  f"{args.patience} epochs (best={best_val_iou:.3f})")
            break

    # Final summary — pull the best epoch's row from history (the row whose
    # val_sem_iou matches best_val_iou, taking the earliest match if there
    # are ties to be deterministic). Paper metrics come from that row.
    # NB: key is `val_sem_iou` not `val_iou` — the gate-metric IoU is stored
    # under the semantic-head-specific name. A prior version used `val_iou`
    # here, which silently fell back to history[-1] (the final epoch) — making
    # cv_summary.json under-report by reading post-overfitting numbers.
    best_row = next((r for r in history if r.get("val_sem_iou") == best_val_iou),
                    history[-1] if history else {})
    print(f"\n=== fold {fold} done. best val_iou={best_val_iou:.3f}. "
          f"checkpoints in {out_dir}")
    return {
        "fold": fold,
        "best_val_iou": best_val_iou,
        "best_epoch": best_row.get("epoch"),
        "n_val": len(val_loader.dataset) if val_loader is not None else None,
        "val_inst_iou": best_row.get("val_inst_iou"),
        "val_sem_iou": best_row.get("val_sem_iou"),
        "val_sem_precision": best_row.get("val_sem_precision"),
        "val_sem_recall": best_row.get("val_sem_recall"),
        "val_sem_f1": best_row.get("val_sem_f1"),
        "val_sem_dice": best_row.get("val_sem_dice"),
        "history": history,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR))
    ap.add_argument("--folds", default="0,1,2,3,4",
                    help="Comma-separated fold indices to train")
    ap.add_argument("--epochs", type=int, default=20,
                    help="Max epochs per fold. Lowered from 30 because "
                         "early stopping (patience 6) usually fires sooner.")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=0.1,
                    help="Gradient clip norm. SAM3 authors clip at 0.1 in "
                         "their eval_base config; 1.0 (the previous default) "
                         "rarely clipped given our high loss-weight magnitudes.")
    ap.add_argument("--oversample", type=int, default=2,
                    help="Each train sample is seen this many times per "
                         "epoch (with fresh augmentation each time). 2 is "
                         "right for a 92-case training pool; the legacy "
                         "23-case setup used 8.")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader worker count for image decode + "
                         "augmentation. 0 = main-thread only (slow). "
                         "2-4 typical.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume each fold from its latest.pt if present")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                    help="Mixed precision (bf16 on CUDA, fp16 on MPS). "
                         "Default on. Use --no-bf16 to disable.")
    ap.add_argument("--patience", type=int, default=6,
                    help="Early-stop fold if val IoU doesn't improve for "
                         "this many epochs. 0 = disabled.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Master seed. Per-fold seed = seed + fold_idx, so "
                         "two runs of the same fold are reproducible (modulo "
                         "bf16 float-rounding).")
    args = ap.parse_args()

    DATASET_DIR_ = Path(args.dataset_dir)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    fold_assignment_path = DATASET_DIR_ / "fold_assignment.json"
    if not fold_assignment_path.exists():
        print(f"ERROR: missing {fold_assignment_path}. Run "
              f"training/build_sam3_training_set.py first.", file=sys.stderr)
        return 1
    fold_map = json.loads(fold_assignment_path.read_text())
    # Build the per-case manifest in-place from the maps/ + fold assignment.
    # Each map file's stem is the filesystem-safe form of the case name
    # (e.g. "12_00114_ART4" for original "12:00114:ART4"). Recover the
    # original case name by reverse-mapping through fold_assignment.json,
    # which records all key variants (original + canonical + safe-form)
    # per case. This matters because downstream consumers cross-reference
    # the predictions JSON against benchmark_runner output (which uses
    # the original folder name).
    manifest = _build_manifest_from_disk(DATASET_DIR_, fold_map)

    # Mirror fold_assignment.json into the training-output dir so production
    # can find it next to the checkpoints.
    (OUTPUT_BASE / "fold_assignment.json").write_text(
        json.dumps(fold_map, indent=2, sort_keys=True))

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_DIR_} ({len(manifest)} cases)")

    processor = Sam3Processor.from_pretrained(MODEL_ID)

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    summary = []
    for fold in folds:
        r = train_fold(fold, args, manifest, processor, device)
        summary.append(r)

    # Cross-fold summary — paper-grade. Writes cv_summary.{json,csv} so the
    # numbers below can be cited verbatim without re-deriving them.
    if summary:
        import csv as _csv

        def _agg(key):
            vals = [s.get(key) for s in summary if s.get(key) is not None]
            if not vals: return None, None
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return m, sd

        metric_keys = ["val_sem_iou", "val_sem_precision", "val_sem_recall",
                        "val_sem_f1", "val_sem_dice", "val_inst_iou"]
        means = {k: _agg(k)[0] for k in metric_keys}
        stds = {k: _agg(k)[1] for k in metric_keys}
        n_total_val = sum((s.get("n_val") or 0) for s in summary)

        cv = {
            "folds": [
                {k: s.get(k) for k in
                 ["fold", "best_epoch", "n_val",
                  "val_sem_iou", "val_sem_precision", "val_sem_recall",
                  "val_sem_f1", "val_sem_dice", "val_inst_iou"]}
                for s in summary
            ],
            "mean": means,
            "std": stds,
            "n_total_val": n_total_val,
            "gate_metric": "val_sem_iou",
            "dataset_dir": str(DATASET_DIR_),
            "model_dir": str(OUTPUT_BASE),
            "config": {"rank": args.rank, "lr": args.lr,
                       "epochs": args.epochs, "batch_size": args.batch_size,
                       "grad_accum": args.grad_accum,
                       "oversample": args.oversample,
                       "seed": args.seed, "bf16": bool(args.bf16),
                       "patience": args.patience},
        }
        (OUTPUT_BASE / "cv_summary.json").write_text(json.dumps(cv, indent=2))
        with open(OUTPUT_BASE / "cv_summary.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=[
                "fold", "best_epoch", "n_val",
                "val_sem_iou", "val_sem_precision", "val_sem_recall",
                "val_sem_f1", "val_sem_dice", "val_inst_iou"])
            w.writeheader()
            for row in cv["folds"]: w.writerow(row)

        print(f"\n=== 5-fold summary (sem-gated) ===")
        for s in summary:
            print(f"  fold {s['fold']:>1d} (n_val={s.get('n_val','?'):>3}, "
                  f"best_ep={s.get('best_epoch')}): "
                  f"sem_iou={s.get('val_sem_iou', 0) or 0:.3f}  "
                  f"prec={s.get('val_sem_precision', 0) or 0:.3f}  "
                  f"rec={s.get('val_sem_recall', 0) or 0:.3f}  "
                  f"f1={s.get('val_sem_f1', 0) or 0:.3f}  "
                  f"dice={s.get('val_sem_dice', 0) or 0:.3f}  "
                  f"inst_iou={s.get('val_inst_iou', 0) or 0:.3f}")
        print(f"\n  Paper-grade aggregates (n_total_val={n_total_val}):")
        for k in metric_keys:
            label = k.replace("val_", "").replace("_", " ")
            if means[k] is None: continue
            print(f"    {label:22s}  {means[k]:.4f} ± {stds[k]:.4f}")
        print(f"\n  Wrote {OUTPUT_BASE/'cv_summary.json'}")
        print(f"  Wrote {OUTPUT_BASE/'cv_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
