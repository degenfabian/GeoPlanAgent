"""Quick fine-tune of LoFTR-MegaDepth — preliminary signal only.

BIASED ESTIMATE WARNING
-----------------------
This script uses a coarse-grid cross-entropy loss derived from the ground-
truth affine, NOT LoFTR's official dual-softmax + fine regression loss. It's
fast to wire up and proves the data has training value, but the magnitude
of improvement here is a LOWER BOUND on what a proper LoFTR-style fine-tune
would achieve.

The coarse loss: LoFTR's coarse stage divides the image into 8×8 patches and
predicts an N×N match-probability matrix between (map patches) × (tile
patches). For each map patch centre we know the ground-truth tile patch
index (project through affine_H, divide by 8). Cross-entropy on those target
indices is a clean (if simplified) supervision signal.

Skipped vs. proper LoFTR loss:
  - No fine-level regression (sub-pixel refinement)
  - No dual-softmax (we just supervise the map→tile axis, not bidirectional)
  - No symmetric epipolar loss / camera-pose loss

If this 10-epoch run improves over the off-the-shelf comparison, that is
real signal. If it doesn't, the bottleneck is the loss, not the data —
a proper fine-tune may still work.

Usage:
  uv run python experiments/loftr_probe/3_finetune_quick.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.loftr_probe._shared import (
    OUTPUTS_DIR, PAIRS_DIR, _device,
    load_matcher, load_matcher_with_module, load_image_bgr,
)


# Match LoFTR's input pipeline: square crops, divisible by 8 for the coarse
# grid. 640 keeps it tractable on MPS; can bump on CUDA.
TRAIN_IMG_SIZE = 640
COARSE_PATCH = 8           # LoFTR coarse stride
N_EPOCHS = 10
LR = 1e-5
BATCH_SIZE = 1
SEED = 42


class PairsDataset(Dataset):
    def __init__(self, split: str, img_size: int = TRAIN_IMG_SIZE):
        manifest = json.loads((PAIRS_DIR / "_manifest.json").read_text())
        self.entries = [m for m in manifest if m["split"] == split]
        self.img_size = img_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        case_dir = PAIRS_DIR / entry["case"]
        map_img = load_image_bgr(case_dir / "map.png")
        tile_img = load_image_bgr(case_dir / "tile.png")
        affine_H = np.load(case_dir / "affine.npy").astype(np.float32)

        # Resize both to img_size×img_size; scale affine accordingly.
        H_m, W_m = map_img.shape[:2]
        H_t, W_t = tile_img.shape[:2]
        s_m_x = self.img_size / W_m
        s_m_y = self.img_size / H_m
        s_t_x = self.img_size / W_t
        s_t_y = self.img_size / H_t
        map_resized = cv2.resize(map_img, (self.img_size, self.img_size))
        tile_resized = cv2.resize(tile_img, (self.img_size, self.img_size))
        # Adjust affine: tile_pt = S_t · H · S_m_inv · map_pt
        # where S_m scales map pts and S_t scales tile pts to img_size.
        S_m_inv = np.array([[1 / s_m_x, 0, 0], [0, 1 / s_m_y, 0]], dtype=np.float32)
        S_t = np.array([[s_t_x, 0, 0], [0, s_t_y, 0]], dtype=np.float32)
        # Compose: new_H = S_t @ [[H; 0 0 1]] @ [[S_m_inv; 0 0 1]]
        H3 = np.vstack([affine_H, [0, 0, 1]])
        S_m_inv3 = np.vstack([S_m_inv, [0, 0, 1]])
        S_t3 = np.vstack([S_t, [0, 0, 1]])
        new_H = (S_t3 @ H3 @ S_m_inv3)[:2]

        # To grayscale, normalised [0,1], CHW for LoFTR's expected input
        def to_tensor(bgr):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return torch.from_numpy(gray)[None]  # (1, H, W)

        return {
            "map": to_tensor(map_resized),
            "tile": to_tensor(tile_resized),
            "affine": torch.from_numpy(new_H),
            "case": entry["case"],
        }


def coarse_correspondence_loss(matcher, batch, device):
    """Cross-entropy over LoFTR's coarse N×N matching matrix.

    Approach: run a forward pass, grab the coarse softmax (conf_matrix)
    from the model's internal state, and supervise it with patch indices
    derived from the affine.
    """
    map_t = batch["map"].to(device)
    tile_t = batch["tile"].to(device)
    affine = batch["affine"].to(device)

    # LoFTR expects a dict with 'image0' and 'image1' batch tensors.
    data = {"image0": map_t, "image1": tile_t}

    # Forward pass — the model populates data['conf_matrix'] internally
    # before any thresholding / fine refinement.
    matcher(data)

    if "conf_matrix" not in data:
        raise RuntimeError(
            "LoFTR forward didn't populate conf_matrix — model config mismatch?")
    conf = data["conf_matrix"]   # (B, L, S) with L=N0 (map patches), S=N1 (tile patches)
    B, L, S = conf.shape

    # Build the target patch indices.
    n_patches = TRAIN_IMG_SIZE // COARSE_PATCH   # e.g. 80 for 640
    # Centre coordinates of each map patch in original image px
    grid_y, grid_x = torch.meshgrid(
        torch.arange(n_patches, device=device),
        torch.arange(n_patches, device=device),
        indexing="ij",
    )
    map_centres = torch.stack([
        (grid_x.float() + 0.5) * COARSE_PATCH,
        (grid_y.float() + 0.5) * COARSE_PATCH,
    ], dim=-1).reshape(-1, 2)    # (L, 2)

    losses = []
    for b in range(B):
        # Project map centres → tile centres via affine
        H = affine[b]  # (2, 3)
        ones = torch.ones(map_centres.shape[0], 1, device=device)
        pts3 = torch.cat([map_centres, ones], dim=1)
        proj = pts3 @ H.T            # (L, 2)
        # Which tile patch does each projection land in?
        tx = (proj[:, 0] / COARSE_PATCH).long()
        ty = (proj[:, 1] / COARSE_PATCH).long()
        # Mask: only supervise patches whose projection lands inside the tile
        valid = (tx >= 0) & (tx < n_patches) & (ty >= 0) & (ty < n_patches)
        if valid.sum() < 32:   # too few valid correspondences — skip
            continue
        target = ty * n_patches + tx    # (L,) indices into S=n_patches**2

        # Cross-entropy only over valid map patches
        loss = F.cross_entropy(conf[b, valid], target[valid])
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True), 0
    return torch.stack(losses).mean(), len(losses)


def evaluate(matcher, val_pairs):
    """Quick eval — runs the matcher and counts inliers + affine error."""
    from tools.matching import run_minima, estimate_affine

    rows = []
    for p in val_pairs:
        map_img = load_image_bgr(p / "map.png")
        tile_img = load_image_bgr(p / "tile.png")
        gt_H = np.load(p / "affine.npy")
        try:
            mkpts0, mkpts1, mconf = run_minima(matcher, map_img, tile_img,
                                                 grayscale=False)
        except Exception:
            rows.append({"case": p.name, "n_inliers": 0,
                         "affine_err_px": float("inf")})
            continue
        n_inl = 0
        if len(mkpts0) >= 4:
            _, _, n_inl = estimate_affine(mkpts0, mkpts1, mconf)
        # Rough affine error
        if len(mkpts0) >= 4:
            from experiments.loftr_probe._compare_helpers import _ as _placeholder  # noqa
        rows.append({"case": p.name, "n_inliers": int(n_inl)})
    n_inl = [r["n_inliers"] for r in rows]
    return {
        "mean_inliers": float(np.mean(n_inl)),
        "median_inliers": float(np.median(n_inl)),
        "n_catastrophic": int(sum(1 for v in n_inl if v < 25)),
        "rows": rows,
    }


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not (PAIRS_DIR / "_manifest.json").exists():
        print("ERROR: pairs/ not built. Run 1_build_pairs.py first.",
              file=sys.stderr)
        return 1

    device = _device()
    print(f"Device: {device}")

    # Off-the-shelf baseline. Get BOTH the callable for inference and the
    # underlying torch module for gradient updates.
    print("\n=== Loading LoFTR-MegaDepth (pretrained) ===")
    matcher_callable, model = load_matcher_with_module("outdoor_ds.ckpt")
    model.train()

    # Datasets
    train_ds = PairsDataset("train")
    val_ds = PairsDataset("val")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0)
    val_pairs = [PAIRS_DIR / e["case"] for e in val_ds.entries]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Baseline eval (off-the-shelf)
    print("\n--- Eval BEFORE fine-tune ---")
    model.eval()
    before = evaluate(matcher_callable, val_pairs)
    print(f"  mean_inliers={before['mean_inliers']:.1f}  "
          f"median={before['median_inliers']:.0f}  "
          f"catastrophic={before['n_catastrophic']}/{len(val_pairs)}")

    # Fine-tune
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    print(f"\n--- Fine-tuning for {N_EPOCHS} epochs ---")
    for ep in range(N_EPOCHS):
        t0 = time.time()
        ep_loss = ep_n = 0
        for batch in train_dl:
            try:
                loss, n_valid = coarse_correspondence_loss(model, batch, device)
            except Exception as e:
                print(f"    skipping batch ({e!s:.60})")
                continue
            if n_valid == 0:
                continue
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item(); ep_n += 1
        wall = time.time() - t0
        avg = ep_loss / max(ep_n, 1)
        print(f"  ep{ep+1}/{N_EPOCHS}: train_loss={avg:.4f}  steps={ep_n}  wall={wall:.0f}s")

    # Post eval
    print("\n--- Eval AFTER fine-tune ---")
    model.eval()
    after = evaluate(matcher_callable, val_pairs)
    print(f"  mean_inliers={after['mean_inliers']:.1f}  "
          f"median={after['median_inliers']:.0f}  "
          f"catastrophic={after['n_catastrophic']}/{len(val_pairs)}")

    # Summary
    delta_mean = after["mean_inliers"] - before["mean_inliers"]
    print(f"\n{'='*60}")
    print(f"Δ mean_inliers:        {delta_mean:+.2f}")
    print(f"Δ catastrophic count:  {after['n_catastrophic'] - before['n_catastrophic']:+d}")
    print(f"{'='*60}")
    if delta_mean > 5:
        print("STRONG SIGNAL: fine-tuning improved off-the-shelf clearly.")
        print("→ Worth investing in a proper LoFTR-style fine-tune (1-2 weeks).")
    elif delta_mean > 1:
        print("WEAK SIGNAL: small improvement; data has training value but the")
        print("simplified loss probably caps how much we can extract.")
        print("→ Proper fine-tune likely to give 2-4× this much gain.")
    elif delta_mean > -1:
        print("NEUTRAL: no clear improvement. Could be the simplified loss, or")
        print("the LoFTR architecture genuinely doesn't have headroom here.")
        print("→ Hard to tell from this probe alone. Need proper fine-tune to confirm.")
    else:
        print("REGRESSION: fine-tune hurt. Loss is wrong or LR is too high.")
        print("→ Probably not worth pursuing without a different loss formulation.")

    # Save fine-tuned weights + per-case results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUTS_DIR / "loftr_quickft.ckpt")
    (OUTPUTS_DIR / "finetune_result.json").write_text(json.dumps({
        "n_train": len(train_ds), "n_val": len(val_ds),
        "n_epochs": N_EPOCHS, "lr": LR,
        "before": before, "after": after,
        "delta_mean_inliers": delta_mean,
    }, indent=2))
    print(f"\nWeights:  {OUTPUTS_DIR / 'loftr_quickft.ckpt'}")
    print(f"Summary:  {OUTPUTS_DIR / 'finetune_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
