"""5-fold rotation classifier training (ResNet50, ImageNet pre-trained).

Inputs:
  - boundary_annotations/<case>/map.png  (the rendered map images)
  - training/dataset/rotation_annotations.json  (corrective-rotation
    labels in degrees, hand-annotated)

Fold routing is shared with SAM3 via tools.core.fold_routing + the
existing models/sam3_lora/fold_assignment.json. A case routed to fold
K at SAM3 inference is the same case routed to fold K here, so the
rotation classifier's hold-out partition matches SAM3's.

Each labelled case generates 4 training samples by applying all 4 CW
rotations to the image — the corrective rotation of the rotated image
is (R - k) mod 360 where R is the annotation and k is the applied
rotation. This makes the training set class-balanced regardless of the
heavy 0° skew in the raw annotations.

Output: models/rotation_classifier_kfold/fold_K/best.pt + history.json
plus a top-level fold_assignment.json copied from SAM3 for the 211
annotated cases (so this training output is self-contained at
inference time).

This module also exposes the shared model + utilities used by
``training/eval/eval_rotation_kfold.py``: RotationClassifier,
KFoldRotationDataset, evaluate, seed_everything, _make_transform,
CLASS_DEGREES, etc.

Run:   uv run python training/train_rotation.py
       uv run python training/train_rotation.py --folds 0,1      (subset)
       uv run python training/train_rotation.py --epochs 15
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models
import torchvision.transforms as T

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

# Reuse SAM3's fold routing verbatim.
from tools.core.fold_routing import (  # noqa: E402
    N_FOLDS,
    fold_for_case as _fold_for_case,
    normalise_case_name as _normalise_case_name,
)


# ── Shared constants ────────────────────────────────────────────────────────

CLASS_DEGREES = [0, 90, 180, 270]  # class_idx → degrees CW to make upright

# These are the rotations APPLIED to upright training images. Their inverse
# (the rotation needed to UNDO them, i.e. the model's target class) is the
# label.
APPLIED_ROTATIONS_CW = [0, 90, 180, 270]

CV2_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Shared model + utilities ────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class RotationClassifier(nn.Module):
    """ResNet50 (ImageNet-pretrained) full fine-tune.

    Tried DINOv2 frozen first; it hit val_acc≈0.51. The DINOv2 SSL
    objective makes features rotation-INVARIANT (good for "what is this"
    classification, bad for "what rotation is this"). ImageNet
    classification training preserves rotation-sensitive features
    (upside-down cat ≠ upright cat) — exactly what we want here.
    Full fine-tune adapts the late layers to planning-map appearance
    while keeping the rotation-discriminative early features.
    """

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)


def _make_transform(img_size: int, train: bool):
    """Image preprocessing pipeline.

    Train: RandomResizedCrop + ColorJitter + RandomErasing kill the case-
    identity shortcut so the model has to use rotation-discriminative
    features (text orientation, north arrows, scale bar position) rather
    than memorising layouts. No horizontal flip — that would change the
    rotation label.

    Eval: plain resize + normalise.
    """
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0),
                                ratio=(0.85, 1.18), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.15, hue=0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.4, scale=(0.02, 0.18),
                            ratio=(0.3, 3.3), value=0),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    n_correct, n_total = 0, 0
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total += y.numel()
    return sum(losses) / max(1, len(losses)), n_correct / max(1, n_total)


# Inputs
DATASET_DIR = REPO / "boundary_annotations"
LABELS_FILE = REPO / "training" / "dataset" / "rotation_annotations.json"
SAM3_FOLD_ASSIGNMENT = REPO / "models" / "sam3_lora" / "fold_assignment.json"

# Output
OUTPUT_DIR = REPO / "models" / "rotation_classifier_kfold"


# Class index <-> degrees CW corrective rotation
DEG_TO_CLASS = {d: i for i, d in enumerate(CLASS_DEGREES)}  # {0:0, 90:1, 180:2, 270:3}


def load_labels() -> dict[str, int]:
    """Read rotation_annotations.json, drop skips and timestamp keys.

    Returns dict[case_name -> corrective_rotation_degrees].
    """
    raw = json.loads(LABELS_FILE.read_text())
    out: dict[str, int] = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if v == "skip":
            continue
        if isinstance(v, int) and v in DEG_TO_CLASS:
            out[k] = v
    return out


def fold_for(case: str, sam3_fa: dict) -> int:
    """Return the fold index a case belongs to, using SAM3's exact routing:
    direct lookup → canonical lookup → md5 hash fallback."""
    f = sam3_fa.get(case)
    if f is None:
        f = sam3_fa.get(_normalise_case_name(case))
    if f is None:
        f = _fold_for_case(case)
    return int(f)


class KFoldRotationDataset(Dataset):
    """One sample per (case, applied_rotation_k) pair.

    Each labelled case M with corrective rotation R generates 4 samples:
      applied k=0   → label = R                     (image as-is)
      applied k=90  → label = (R - 90) mod 360      (image rotated 90 CW)
      applied k=180 → label = (R - 180) mod 360
      applied k=270 → label = (R - 270) mod 360

    The image augmentations (RandomResizedCrop, ColorJitter, RandomErasing)
    are reused unchanged from RotationDataset so the trained classifier
    sees the same augmentation regime the existing one was trained with.
    """

    def __init__(self, cases: list[str], labels: dict[str, int],
                 img_size: int = 768, train: bool = True):
        self.cases = cases
        self.labels = labels
        self.train = train
        self.samples = [(c, k) for c in cases for k in APPLIED_ROTATIONS_CW]
        self.transform = _make_transform(img_size, train=train)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        case, applied_k = self.samples[idx]
        img_path = DATASET_DIR / case / "map.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if applied_k != 0:
            img = cv2.rotate(img, CV2_ROTATE_CODES[applied_k])
        tensor = self.transform(img)
        base_r = self.labels[case]
        new_r = (base_r - applied_k) % 360
        return tensor, DEG_TO_CLASS[new_r]


def train_one_fold(fold_idx: int, train_cases: list[str], val_cases: list[str],
                    labels: dict[str, int], args, device: str) -> dict:
    fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx}: train={len(train_cases)} cases "
          f"({len(train_cases)*4} samples), "
          f"val={len(val_cases)} cases ({len(val_cases)*4} samples)")
    print(f"  output: {fold_dir}")

    g = seed_everything(args.seed + fold_idx)

    train_ds = KFoldRotationDataset(train_cases, labels, img_size=args.img_size, train=True)
    val_ds = KFoldRotationDataset(val_cases, labels, img_size=args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = RotationClassifier(n_classes=4).to(device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    best_val_acc = 0.0
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, n_correct, n_total = [], 0, 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
            n_correct += (logits.argmax(-1) == y).sum().item()
            n_total += y.numel()
        sched.step()
        train_loss = sum(losses) / max(1, len(losses))
        train_acc = n_correct / max(1, n_total)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "wall_s": round(elapsed, 1),
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            "lr": sched.get_last_lr()[0],
        })
        print(f"  fold{fold_idx} ep{epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  "
              f"wall={elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "img_size": args.img_size,
                    "n_classes": 4,
                    "class_degrees": CLASS_DEGREES,
                    "imagenet_mean": IMAGENET_MEAN,
                    "imagenet_std": IMAGENET_STD,
                },
                "epoch": epoch, "best_val_acc": best_val_acc,
                "fold_idx": fold_idx, "history": history,
            }, fold_dir / "best.pt")
            print(f"    fold{fold_idx} new best val_acc={best_val_acc:.3f}, saved.")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"    fold{fold_idx} early stop: no improvement for "
                      f"{args.patience} epochs (best={best_val_acc:.3f})")
                break

        (fold_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Fold {fold_idx} done. best_val_acc={best_val_acc:.3f}")
    return {"fold": fold_idx, "best_val_acc": best_val_acc, "history": history,
            "n_train": len(train_cases), "n_val": len(val_cases)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4",
                    help="Comma-separated fold indices to train (default all 5).")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    if not DATASET_DIR.exists():
        print(f"ERROR: dataset dir missing: {DATASET_DIR}", file=sys.stderr); return 1
    if not LABELS_FILE.exists():
        print(f"ERROR: labels missing: {LABELS_FILE}", file=sys.stderr); return 1
    if not SAM3_FOLD_ASSIGNMENT.exists():
        print(f"ERROR: SAM3 fold_assignment.json missing: {SAM3_FOLD_ASSIGNMENT}",
              file=sys.stderr); return 1

    labels = load_labels()
    print(f"Loaded {len(labels)} labels from {LABELS_FILE.name} "
          f"(skipping skip/timestamp entries)")

    # Class distribution before augmentation
    from collections import Counter
    raw_dist = Counter(labels.values())
    print(f"Raw label distribution: {dict(sorted(raw_dist.items()))}")
    print(f"After 4-rotation augmentation per case: each class gets "
          f"{len(labels)} samples ({len(labels)*4} total).")

    sam3_fa = json.loads(SAM3_FOLD_ASSIGNMENT.read_text())
    case_to_fold = {c: fold_for(c, sam3_fa) for c in labels}

    # Write our own fold_assignment.json (subset of SAM3's, restricted to
    # cases we have labels for). Inference can load this independently of
    # the SAM3 file, but it's guaranteed identical for shared cases.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "fold_assignment.json").write_text(
        json.dumps(case_to_fold, indent=2, sort_keys=True))
    print(f"Wrote fold_assignment.json ({len(case_to_fold)} entries) to {OUTPUT_DIR}")

    requested_folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if not all(0 <= f < N_FOLDS for f in requested_folds):
        print(f"ERROR: --folds must be in [0,{N_FOLDS-1}]", file=sys.stderr); return 1

    summary = []
    for fold in requested_folds:
        train_cases = sorted([c for c, f in case_to_fold.items() if f != fold])
        val_cases = sorted([c for c, f in case_to_fold.items() if f == fold])
        if not val_cases:
            print(f"FOLD {fold}: no val cases (skip)"); continue
        result = train_one_fold(fold, train_cases, val_cases, labels, args, device)
        summary.append(result)

    (OUTPUT_DIR / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        mean_acc = sum(r["best_val_acc"] for r in summary) / len(summary)
        print(f"\n{'='*70}\nAll requested folds done. "
              f"mean best_val_acc = {mean_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
