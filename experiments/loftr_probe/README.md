# LoFTR-MegaDepth fine-tune probe

**Quick experiment to test whether fine-tuning LoFTR-MegaDepth (the standard
public LoFTR weights, not the cross-modal MINIMA fine-tune) on UK planning
maps could outperform stock MINIMA.**

Self-contained. To delete: `rm -rf experiments/loftr_probe/`. Touches no
production code.

## Why this exists

The team has tried RoMa, OmniGlue, MatchAnything-ELoFTR, pix2pix-turbo
preprocessing — none beat MINIMA on planning maps off-the-shelf. **Nobody
has tried fine-tuning LoFTR-MegaDepth on our specific data.** MINIMA was
trained on a general cross-modal mix (RGB↔IR, RGB↔thermal, etc.); LoFTR-
MegaDepth is outdoor stereo. Neither was trained on planning-map↔OS-tile
pairs, but LoFTR's public training recipe is easier to extend.

## Three scripts, run in order

| Script | What it does | Wall (MPS) |
|---|---|---|
| `1_build_pairs.py` | Reads `results/benchmark_v20/` cases with IoU > 0.7, renders the matched OS tile canvas, saves (map, tile, affine) triples | ~15 min |
| `2_compare_offshelf.py` | Runs LoFTR-MegaDepth (pretrained, no fine-tune) AND MINIMA on the val split; prints n_inliers comparison | ~10 min |
| `3_finetune_quick.py` | Fine-tunes LoFTR-MegaDepth for 10 epochs with a coarse-correspondence loss; re-evaluates | ~60-90 min |

After (2) you have the most important data point: **is off-the-shelf
LoFTR-MegaDepth competitive with MINIMA on planning maps?**
- If yes → fine-tuning is very likely to help; commit to a real LoFTR-style
  training loop (1-2 weeks of work).
- If LoFTR-MegaDepth is at parity → fine-tuning could push it ahead; worth
  doing (3) to get a noisy estimate.
- If LoFTR-MegaDepth is way worse → fine-tuning probably can't bridge the
  gap; abandon the direction.

(3) is a **biased** estimate. It uses correspondence MSE loss, not LoFTR's
actual dual-softmax + fine regression loss. If (3) shows improvement, that's
a *real* signal the data has training value — but the magnitude of
improvement underestimates what a proper fine-tune would achieve.

## Setup (one-time)

Download LoFTR-MegaDepth weights (~96 MB) into `MINIMA/weights/`:

```bash
# The official LoFTR drop. ~96 MB.
mkdir -p MINIMA/weights
cd MINIMA/weights
wget -O outdoor_ds.ckpt https://huggingface.co/Frosty-aurora/loftr_outdoor_ds/resolve/main/outdoor_ds.ckpt
cd -
```

If the HF mirror isn't available, the original is on Google Drive (LoFTR
authors host them). Use `gdown` or download manually.

## Run

```bash
uv run python experiments/loftr_probe/1_build_pairs.py
uv run python experiments/loftr_probe/2_compare_offshelf.py
# (optional) uv run python experiments/loftr_probe/3_finetune_quick.py
```

Each script prints a clear summary at the end. Results live in
`experiments/loftr_probe/{pairs,outputs}/` — also gitignored / safe to delete.
