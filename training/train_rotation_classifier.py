"""Train a 4-class rotation classifier on planning maps.

Source: boundary_annotation_v2_auto/maps/ (202 maps, all confirmed
upright by the user). Synthetic data:

  For each upright map M, generate 4 training samples:
    M rotated 0°   CW  → class 0  (apply 0° to fix)
    M rotated 90°  CW  → class 3  (apply 270° = 90° CCW to fix)
    M rotated 180° CW  → class 2  (apply 180° to fix)
    M rotated 270° CW  → class 1  (apply 90° CW to fix)

  Class index ↔ degrees: 0→0°, 1→90°, 2→180°, 3→270° (CW to fix).

At inference: pass any planning-map render through the model, get the
class index back, rotate by that many CW degrees to produce an upright
view. Pre-process before SAM3/MINIMA see the map.

Architecture: DINOv2-Base (frozen, ImageNet-pretrained but the SSL
training data includes scientific/document/aerial imagery → much
better OOD transfer to planning maps than ResNet's ImageNet
classification training). Just a Linear(768, 4) head trained on top.
Frozen backbone means: minimal trainable params (~3K), no overfitting
on 808 samples, fast training.

Input size: must be a multiple of 14 (DINOv2 patch size). Default 448
gives 32×32 = 1024 spatial tokens + 1 CLS, plenty of resolution for
text/north-arrow detection.

Split: 80/20 by CASE (all 4 rotations of one case go to the same split
so the val metric reflects generalisation, not memorisation).

Output: models/rotation_classifier/best.pt + history.json
Usage:    cd training && uv run python train_rotation_classifier.py
"""

from __future__ import annotations

# Silence the pynvml FutureWarning that torch.cuda fires on every subprocess
# (we're on Mac/MPS — no CUDA — but each DataLoader worker re-imports torch
# and re-fires the warning). Setting PYTHONWARNINGS in the env before any
# import means the workers inherit the silencing.
import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

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


SOURCE_DIR = REPO / "boundary_annotation_v2_auto" / "maps"
OUTPUT_DIR = REPO / "models" / "rotation_classifier"

CLASS_DEGREES = [0, 90, 180, 270]  # class_idx → degrees CW to make upright

# These are the rotations APPLIED to upright training images. Their inverse
# (the rotation needed to UNDO them, i.e. the model's target class) is the
# label.
APPLIED_ROTATIONS_CW = [0, 90, 180, 270]
# Map: applied CW → class_idx (rotation needed to UNDO it)
#   applied 0   → fix with 0   → class 0
#   applied 90  → fix with 270 → class 3
#   applied 180 → fix with 180 → class 2
#   applied 270 → fix with 90  → class 1
APPLIED_TO_LABEL = {0: 0, 90: 3, 180: 2, 270: 1}

CV2_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = 42) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class RotationDataset(Dataset):
    """Each item: (img_tensor, class_label).
    Each underlying map produces 4 items (one per rotation)."""

    def __init__(self, cases: list[str], img_size: int = 224,
                 train: bool = True):
        self.cases = cases
        self.img_size = img_size
        self.train = train
        # Pre-compute (case, applied_rotation) tuples
        self.samples = [(c, r) for c in cases for r in APPLIED_ROTATIONS_CW]

        if train:
            # No horizontal flip — that changes the rotation label.
            # RandomResizedCrop + RandomErasing kill the case-identity
            # shortcut: with only 162 unique training cases the model
            # was memorising overall layout and looking up rotation;
            # stronger crops + erased patches force it to actually use
            # rotation-discriminative features (text orientation,
            # north arrows, scale bar position).
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0),
                                     ratio=(0.85, 1.18),
                                     antialias=True),
                T.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.15, hue=0.05),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.RandomErasing(p=0.4, scale=(0.02, 0.18),
                                 ratio=(0.3, 3.3), value=0),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size), antialias=True),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        case, applied_rot = self.samples[idx]
        img_path = SOURCE_DIR / f"{case}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {img_path}")
        # cv2 returns BGR; convert to RGB for ImageNet-pretrained backbones
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if applied_rot != 0:
            img = cv2.rotate(img, CV2_ROTATE_CODES[applied_rot])
        tensor = self.transform(img)
        label = APPLIED_TO_LABEL[applied_rot]
        return tensor, label


def split_cases(all_cases: list[str], val_frac: float = 0.2,
                seed: int = 42) -> tuple[list[str], list[str]]:
    """80/20 split BY CASE — all rotations of a case go to one split."""
    rng = random.Random(seed)
    cases = sorted(all_cases)
    rng.shuffle(cases)
    n_val = max(1, int(round(len(cases) * val_frac)))
    return cases[n_val:], cases[:n_val]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Standard for full-finetune. The whole ResNet50 "
                         "trains, not just a head.")
    ap.add_argument("--img-size", type=int, default=768,
                    help="Input resolution. 768 keeps text labels, scale "
                         "bars, and north arrows clearly readable for "
                         "the model.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4,
                    help="Early-stop after N epochs of no val acc improvement")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    g = seed_everything(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        print(f"ERROR: source dir missing: {SOURCE_DIR}", file=sys.stderr)
        return 1

    all_cases = [p.stem for p in SOURCE_DIR.glob("*.png")]
    if not all_cases:
        print(f"ERROR: no PNG files in {SOURCE_DIR}", file=sys.stderr)
        return 1
    print(f"Found {len(all_cases)} maps. With 4 rotations each: "
          f"{len(all_cases) * 4} training samples total.")

    train_cases, val_cases = split_cases(all_cases, val_frac=0.2,
                                          seed=args.seed)
    print(f"Train: {len(train_cases)} cases ({len(train_cases) * 4} samples)")
    print(f"Val:   {len(val_cases)} cases ({len(val_cases) * 4} samples)")

    train_ds = RotationDataset(train_cases, img_size=args.img_size, train=True)
    val_ds = RotationDataset(val_cases, img_size=args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = RotationClassifier(n_classes=4).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.4f}%)")
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs))

    best_val_acc = 0.0
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, n_correct, n_total = [], 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
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
            "epoch": epoch,
            "wall_s": round(elapsed, 1),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": sched.get_last_lr()[0],
        })
        print(f"  ep{epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  "
              f"wall={elapsed:.0f}s")

        improved = val_acc > best_val_acc
        if improved:
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
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "history": history,
            }, OUTPUT_DIR / "best.pt")
            print(f"    new best val_acc={best_val_acc:.3f}, saved best.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"    early stop: no improvement for "
                      f"{args.patience} epochs (best={best_val_acc:.3f})")
                break

        (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nDone. Best val_acc={best_val_acc:.3f}.  "
          f"Checkpoint: {OUTPUT_DIR / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
