"""Visualise the training-time augmentations on real training samples.

Renders, for 5 random training cases, a row of:
  [ original map | mask overlay | aug1 | aug2 | aug3 | aug4 ]

Saves to experiments/sam3_v8_replay/augmentation_samples/<case>.png

Each `augN` is a fresh roll of `style_transfer_augment` (random style + colour +
fade amount), to show the *distribution* of what the model sees during training.

If the augmentations look pathological — over-fading, wrong-coloured boundaries
on regions that don't look like real planning maps, masks misaligned with the
drawn boundary — that could explain v8's regression cases.

Easy to delete: rm -rf experiments/sam3_v8_replay/augmentation_samples/
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

DATASET_DIR = REPO / "training" / "dataset"
OUT_DIR = HERE / "augmentation_samples"
N_CASES = 5      # number of cases to render
N_AUGS = 4       # number of augmentation rolls per case

# Reproducibility — same seed → same set of samples + same augmentation rolls
SEED = 42


def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.4):
    """Translucent mask overlay (no resize — assume same dims)."""
    out = img_bgr.copy()
    if mask is None or mask.sum() == 0:
        return out
    mb = (mask > 0).astype(np.uint8)
    layer = np.zeros_like(out)
    layer[mb > 0] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0)


def label(img, text, color=(255, 255, 255), bg=(0, 0, 0)):
    """Banner label at top-left."""
    pad = 12
    fscale = max(0.7, min(1.6, img.shape[1] / 1600))
    thickness = max(2, int(fscale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 2*pad, th + 2*pad), bg, -1)
    cv2.putText(img, text, (pad, th + pad // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thickness, cv2.LINE_AA)
    return img


def downsize(img, max_side=800):
    """Downsample for tiling — keeps the panel viewable."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    new_w, new_h = int(w * s), int(h * s)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from training.boundary_augmentations import style_transfer_augment

    if not (DATASET_DIR / "manifest.json").exists():
        print(f"ERROR: {DATASET_DIR}/manifest.json missing", file=sys.stderr)
        return 1

    import json
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    random.seed(SEED)
    chosen = random.sample(manifest, N_CASES)

    for entry in chosen:
        fname = entry["filename"]
        case = entry["case"]
        print(f"\n=== {case} ===")

        map_p = DATASET_DIR / "maps" / fname
        mask_p = DATASET_DIR / "boundary_masks" / fname
        img_bgr = cv2.imread(str(map_p))
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask is None:
            print(f"  SKIP: missing {map_p} or {mask_p}")
            continue

        # Downsize for tiling
        img_small = downsize(img_bgr)
        mask_small = cv2.resize(mask, (img_small.shape[1], img_small.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # Panel 0: original
        p0 = label(img_small.copy(), f"{case}  ORIG  {img_bgr.shape[1]}x{img_bgr.shape[0]}")
        # Panel 1: mask overlay
        p1 = label(overlay_mask(img_small, mask_small, color=(0, 255, 0)),
                    f"GT mask  ({(mask>0).mean()*100:.1f}% of image)")

        # Panels 2..N+1: N augmentation rolls (force p=1.0 so we always apply)
        aug_panels = []
        # PIL inputs (style_transfer_augment expects PIL)
        img_pil_full = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mask_pil_full = Image.fromarray(mask)
        for i in range(N_AUGS):
            # Apply with p=1.0 so every roll is augmented
            aug_img, _ = style_transfer_augment(img_pil_full, mask_pil_full, p=1.0)
            aug_bgr = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)
            aug_small = downsize(aug_bgr)
            aug_panels.append(label(aug_small.copy(), f"aug #{i+1}"))

        # Tile into a 2 x (1+N_AUGS//2 + ?) layout; simpler: 1 row of (2 + N_AUGS)
        # equal-sized panels. Resize all panels to the same shape (img_small's).
        target_h, target_w = p0.shape[:2]
        def fit(panel):
            return cv2.resize(panel, (target_w, target_h),
                               interpolation=cv2.INTER_AREA)
        row = np.hstack([fit(p0), fit(p1)] + [fit(a) for a in aug_panels])
        out_path = OUT_DIR / f"{case.replace(':', '_').replace('/', '_')}.png"
        cv2.imwrite(str(out_path), row)
        print(f"  → {out_path}  ({row.shape[1]}x{row.shape[0]})")

    print(f"\nDone. Panels in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
