"""5-fold rotation classifier training (ResNet50, ImageNet pre-trained).

Inputs:
  - boundary_annotations/<case>/map.png  (the rendered map images;
    not distributed with the release)
  - training/dataset/rotation_annotations.json  (corrective-rotation
    labels in degrees, hand-annotated)

Fold routing is shared with SAM3 via geoplanagent.utils + the
canonical models/fold_assignment.json. A case routed to fold
K at SAM3 inference is the same case routed to fold K here, so the
rotation classifier's hold-out partition matches SAM3's.

Each labelled case generates 4 training samples by applying all 4 CW
rotations to the image — the corrective rotation of the rotated image
is (R - k) mod 360 where R is the annotation and k is the applied
rotation. This makes the training set class-balanced regardless of the
heavy 0° skew in the raw annotations.

Output: models/rotation_classifier_kfold/fold_K/best.pt + history.json,
plus a kfold_summary.json over the trained folds. Fold routing reads the
shared models/fold_assignment.json — no per-model copy is written.

The shared model, transform and constants (RotationClassifier, make_transform,
CLASS_DEGREES, IMAGENET_*) live in geoplanagent.tools.rotation_classifier; this
module imports them and adds the training-only pieces. The cross-fold evaluator
``training/eval/eval_rotation_kfold.py`` reuses the labels/fold helpers
(load_labels, fold_for, DATASET_DIR, OUTPUT_DIR) from here and the shared
inference pieces from geoplanagent.tools.rotation_classifier.

Run:   uv run python training/train_rotation.py
       uv run python training/train_rotation.py --folds 0,1      (subset)
       uv run python training/train_rotation.py --epochs 15
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

from geoplanagent.paths import (  # noqa: E402
    FOLD_ASSIGNMENT,
    ROTATION_KFOLD_DIR,
    TRAINING_DATASET_DIR,
)

# Reuse SAM3's fold routing verbatim.
from geoplanagent.utils import (  # noqa: E402
    N_FOLDS,
    device as _device,
    resolve_fold as _resolve_fold,
)


# These are the rotations APPLIED to upright training images. Their inverse
# (the rotation needed to UNDO them, i.e. the model's target class) is the
# label.
APPLIED_ROTATIONS_CW = [0, 90, 180, 270]

# Shared model/transform/constants from the production classifier.
from geoplanagent.tools.rotation_classifier import (  # noqa: E402
    _CV2_ROTATE_CODES as CV2_ROTATE_CODES,
    CLASS_DEGREES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RotationClassifier,
    make_transform,
)


# Training-local helpers


def seed_everything(seed: int = 42) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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
DATASET_DIR = REPO / "boundary_annotations"  # no paths.py constant for this dir
LABELS_FILE = TRAINING_DATASET_DIR / "rotation_annotations.json"

# Output
OUTPUT_DIR = ROTATION_KFOLD_DIR


# Class index <-> degrees CW corrective rotation
DEG_TO_CLASS = {d: i for i, d in enumerate(CLASS_DEGREES)}  # {0:0, 90:1, 180:2, 270:3}


def load_labels() -> dict[str, int]:
    """Read rotation_annotations.json, drop skips and timestamp keys.

    Returns dict[case_name -> corrective_rotation_degrees].
    """
    raw_labels = json.loads(LABELS_FILE.read_text())
    labels: dict[str, int] = {}
    for case, value in raw_labels.items():
        if case.startswith("__"):
            continue
        if value == "skip":
            continue
        if isinstance(value, int) and value in DEG_TO_CLASS:
            labels[case] = value
    return labels


def fold_for(case: str, fold_assignment: dict) -> int:
    """Return the fold index a case belongs to, using the same routing as
    SAM3 (direct lookup → canonical lookup → ``min(folds)`` fallback for
    cases the training pool didn't contain)."""
    return _resolve_fold(case, fold_assignment, set(range(N_FOLDS)))


class KFoldRotationDataset(Dataset):
    """One sample per (case, applied_rotation_k) pair.

    Each labelled case M with corrective rotation R generates 4 samples:
      applied k=0   → label = R                     (image as-is)
      applied k=90  → label = (R - 90) mod 360      (image rotated 90 CW)
      applied k=180 → label = (R - 180) mod 360
      applied k=270 → label = (R - 270) mod 360

    The image augmentations (RandomResizedCrop, ColorJitter, RandomErasing)
    come from the shared geoplanagent.tools.rotation_classifier.make_transform,
    so the trained classifier sees the same augmentation regime production
    inference assumes.
    """

    def __init__(
        self, cases: list[str], labels: dict[str, int], img_size: int = 768, train: bool = True
    ):
        self.cases = cases
        self.labels = labels
        self.train = train
        self.samples = [
            (case, applied_k) for case in cases for applied_k in APPLIED_ROTATIONS_CW
        ]
        self.transform = make_transform(img_size, train=train)

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


def train_one_fold(
    fold_idx: int,
    train_cases: list[str],
    val_cases: list[str],
    labels: dict[str, int],
    args,
    device: torch.device,
) -> dict:
    fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(
        f"FOLD {fold_idx}: train={len(train_cases)} cases "
        f"({len(train_cases) * 4} samples), "
        f"val={len(val_cases)} cases ({len(val_cases) * 4} samples)"
    )
    print(f"  output: {fold_dir}")

    generator = seed_everything(args.seed + fold_idx)

    train_ds = KFoldRotationDataset(train_cases, labels, img_size=args.img_size, train=True)
    val_ds = KFoldRotationDataset(val_cases, labels, img_size=args.img_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RotationClassifier(n_classes=4, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_acc = 0.0
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        losses, n_correct, n_total = [], 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            n_correct += (logits.argmax(-1) == y).sum().item()
            n_total += y.numel()
        scheduler.step()
        train_loss = sum(losses) / max(1, len(losses))
        train_acc = n_correct / max(1, n_total)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time
        history.append(
            {
                "epoch": epoch,
                "wall_s": round(elapsed, 1),
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"  fold{fold_idx} ep{epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  "
            f"wall={elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_best = 0
            torch.save(
                {
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
                    "fold_idx": fold_idx,
                    "history": history,
                },
                fold_dir / "best.pt",
            )
            print(f"    fold{fold_idx} new best val_acc={best_val_acc:.3f}, saved.")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(
                    f"    fold{fold_idx} early stop: no improvement for "
                    f"{args.patience} epochs (best={best_val_acc:.3f})"
                )
                break

        (fold_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Fold {fold_idx} done. best_val_acc={best_val_acc:.3f}")
    return {
        "fold": fold_idx,
        "best_val_acc": best_val_acc,
        "history": history,
        "n_train": len(train_cases),
        "n_val": len(val_cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Max epochs per fold; early stopping (--patience) usually fires sooner.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--img-size",
        type=int,
        default=768,
        help="Square input resolution (px); recorded in the checkpoint config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed. Per-fold seed = seed + fold_idx, so each fold is reproducible.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early-stop a fold if val accuracy doesn't improve for this many epochs.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader worker count for image decode."
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated fold indices to train (default all 5).",
    )
    args = parser.parse_args()

    device = _device()
    print(f"Device: {device}")

    if not DATASET_DIR.exists():
        print(f"ERROR: dataset dir missing: {DATASET_DIR}", file=sys.stderr)
        return 1
    if not LABELS_FILE.exists():
        print(f"ERROR: labels missing: {LABELS_FILE}", file=sys.stderr)
        return 1
    if not FOLD_ASSIGNMENT.exists():
        print(f"ERROR: fold_assignment.json missing: {FOLD_ASSIGNMENT}", file=sys.stderr)
        return 1

    labels = load_labels()
    print(f"Loaded {len(labels)} labels from {LABELS_FILE.name} (skipping skip/timestamp entries)")

    # Class distribution before augmentation
    from collections import Counter

    raw_dist = Counter(labels.values())
    print(f"Raw label distribution: {dict(sorted(raw_dist.items()))}")
    print(
        f"After 4-rotation augmentation per case: each class gets "
        f"{len(labels)} samples ({len(labels) * 4} total)."
    )

    fold_assignment = json.loads(FOLD_ASSIGNMENT.read_text())
    case_to_fold = {case: fold_for(case, fold_assignment) for case in labels}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requested_folds = [int(token) for token in args.folds.split(",") if token.strip()]
    if not all(0 <= f < N_FOLDS for f in requested_folds):
        print(f"ERROR: --folds must be in [0,{N_FOLDS - 1}]", file=sys.stderr)
        return 1

    summary = []
    for fold in requested_folds:
        train_cases = sorted([case for case, assigned_fold in case_to_fold.items() if assigned_fold != fold])
        val_cases = sorted([case for case, assigned_fold in case_to_fold.items() if assigned_fold == fold])
        if not val_cases:
            print(f"FOLD {fold}: no val cases (skip)")
            continue
        result = train_one_fold(fold, train_cases, val_cases, labels, args, device)
        summary.append(result)

    (OUTPUT_DIR / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        mean_acc = sum(result["best_val_acc"] for result in summary) / len(summary)
        print(f"\n{'=' * 70}\nAll requested folds done. mean best_val_acc = {mean_acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
