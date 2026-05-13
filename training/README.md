# training/

SAM3 LoRA fine-tuning for the "planning boundary" segmentation task. The output adapter is what `tools.sam3_boundary.load_sam3_ft` loads at inference time.

Not required for inference — a trained adapter ships in `models/sam3_lora_v4/` and `models/sam3_lora_v5/` (k-fold CV). Run this only to reproduce or extend the fine-tune.

## Files

| File | Purpose |
|---|---|
| `train_sam3_kfold.py` | **Current trainer.** 5-fold CV with combined semantic + instance-head losses. Reproducible (seeded). |
| `train_boundary_only.py` | Legacy single-fold trainer (semantic head only, hand-drawn 27-case dataset). Kept for archival reference. |
| `boundary_augmentations.py` | Style-transfer + copy-paste augmentations used at train time. |

## Prerequisites

```bash
uv sync --extra training
```

`pyproject.toml` pins `transformers`, `peft`, `torch`, etc. so a fresh `uv sync` reproduces the dependency set. SAM3 base weights download from HuggingFace on first run (~3 GB; needs `HF_TOKEN`).

## Reproducibility — the full data pipeline

The training set in `training/dataset_v5/` is **derived from a multi-step pipeline**, not a single annotation file. To reproduce from scratch, run these in order:

### 1. Auto-label boundary masks from cached benchmark affines

```bash
uv run python scripts/auto_label_boundary_dataset.py
```

Inputs:
- `evaluation_data/<case>/<gt>.geojson` — ground-truth polygons
- `results/benchmark_v7_v2_full/gemini-flash/<case>/affine_H.npy` — cached agent-produced affine transforms
- `boundary_annotation_dataset/{maps,boundary_masks}/` — pre-existing hand-drawn cases (copied as-is)

Output: `boundary_annotation_v2_auto/{maps,boundary_masks,overlays,metadata.csv}` with ~204 candidate (map, mask) pairs.

For each case, the script does ONE of three things:
- **Path A — hand-annotated:** copy from `boundary_annotation_dataset/`. Used for the 27 cases that pre-existed.
- **Path B — backprojected GT:** invert the cached affine, rasterise the GT polygon back into map-image coordinates. Used for cases where v7 produced a usable affine.
- **Path C — default placement:** plop the GT polygon at the map centre at 25% width. Used when no affine exists OR when the backprojected polygon lands outside the map. The user manually aligns these in the next step.

### 2. Manual review + alignment correction

```bash
uv run python scripts/alignment_review_server.py
# → http://localhost:8765/
```

Browser-based review UI. For each of the 204 cases you click `A` (accept), `F` (fix and accept), or `R` (reject). Path C cases need a small translate/scale/rotate/mirror to align the GT shape to the actual ink on the map. State persists to `boundary_annotation_curated/<case>/state.json` so you can quit and resume.

For the published v5 dataset, this produced 119 accepted + 1 fixed + 83 rejected (= 120 in the training pool).

### 3. Assemble the final training set

```bash
uv run python scripts/build_curated_training_set.py
```

Reads `boundary_annotation_curated/`, filters to `accepted` + `fixed`, copies into `training/dataset_v5/{maps,boundary_masks}/`, and writes:
- `manifest.json` — per-case metadata (filename, fold, status, …)
- `fold_assignment.json` — case_name → fold (deterministic via `md5(case_name) % 5`)
- `manifest.csv` — same as JSON, for inspection

Re-running this is idempotent — same input → bit-identical output.

### 4. Train

```bash
cd training
uv run python train_sam3_kfold.py
```

Iterates folds 0→4 sequentially. Each fold:
- Trains on cases NOT in fold k (≈ 95-100 cases)
- Validates on cases IN fold k (≈ 20-30 cases)
- Writes per-epoch checkpoint to `models/sam3_lora_v5/fold_<k>/`

Wall: ~1.5–2 hr per fold on Apple MPS with bf16; ~1 hr per fold on a CUDA GPU. ~10 hr total for all 5.

## Trainer flags

```bash
uv run python train_sam3_kfold.py --help
```

| Flag | Default | Purpose |
|---|---|---|
| `--folds` | `0,1,2,3,4` | Comma-separated fold indices |
| `--epochs` | 30 | Max epochs per fold |
| `--rank` | 16 | LoRA rank |
| `--lr` | 2e-4 | Initial learning rate (cosine decay to 0.05× by end) |
| `--batch-size` | 1 | Effective batch = batch_size × grad_accum |
| `--grad-accum` | 4 | |
| `--bf16 / --no-bf16` | `--bf16` | Mixed precision (bf16 on CUDA, fp16 on MPS) |
| `--patience` | 6 | Early stop fold if val IoU hasn't improved in N epochs (0 = disabled) |
| `--seed` | 42 | Master seed; per-fold seed = seed + fold |
| `--resume` | off | Resume each fold from `latest.pt` if present |

## Reproducibility guarantees

With a fixed `--seed` and the same dataset, two runs produce identical training trajectories on the same hardware:
- ✅ `random` (Python), `numpy`, `torch`, CUDA RNG all seeded
- ✅ `DataLoader` shuffle uses a `torch.Generator` seeded per fold
- ✅ LoRA initialisation seeded (PEFT honours `torch.manual_seed`)
- ✅ Per-fold seed = `seed + fold_idx` so different folds explore different sequences
- ⚠️ With `--bf16` on, you get tiny float-rounding deltas across runs (the trajectory is the same, the bits aren't). Disable for bit-exact reproduction.
- ⚠️ MPS has a few non-deterministic ops; CUDA with `torch.use_deterministic_algorithms(True)` is the gold path for bit-exact repro.

Each saved checkpoint embeds its `config` dict (rank, lr, epochs, seed, bf16, patience) so a 3rd party can read a checkpoint and know exactly which flags produced it.

## Loss formulation

```
SEMANTIC HEAD (focal + dice + ramped surface)
  total_sem = 5.0·focal(sem_pred, gt) + 5.0·dice(sem_pred, gt)
            + min(1, epoch/15) · 0.5 · surface(σ(sem_pred), signed_dist(gt))

INSTANCE HEAD (best-IoU proposal only — preserves diversity of others)
  best = argmax_i  IoU(σ(pred_masks[i]), gt)
  total_inst = 3.0·focal(pred_masks[best], gt) + 3.0·dice(pred_masks[best], gt)

total_loss = total_sem + total_inst
```

Notes:
- **No erosion-consistency loss.** It was a band-aid for hand-drawn outline-style masks; auto-labelled masks are filled `cv2.fillPoly` outputs, so erosion adds no signal and harms multi-blob predictions.
- **No presence loss on the instance head.** Training "best slot fires, others zero" collapses the instance head into a single-mask predictor — architecturally redundant with the semantic head. Without presence loss, the unused slots stay diverse via SAM3's pretrained behaviour, so the agent's `mode='instance'` gets useful alternatives at inference.
- Instance loss runs at the head's native resolution (~256×256), not the full map resolution. Necessary on MPS to avoid materialising a `[100, 2300, 1654]` intermediate that exceeds MPSGraph's INT_MAX.

## Output layout per fold

```
models/sam3_lora_v5/
├── fold_0/
│   ├── latest.pt       # rewritten every epoch (resume target)
│   ├── best.pt         # rewritten when val IoU improves
│   └── history.json    # per-epoch train/val loss + val IoU
├── fold_1/   …
├── fold_2/   …
├── fold_3/   …
├── fold_4/   …
└── fold_assignment.json   # mirror of training/dataset_v5/fold_assignment.json
                            # for production lookup at inference time
```

At inference time, `tools.sam3_boundary` loads `models/sam3_lora_v5/fold_<k>/best.pt` where `k = md5(case_name) % 5`, so each case is evaluated by the model that was held out from training on that case (no leakage).

## Cross-fold summary

After all 5 folds finish, the script prints:

```
=== 5-fold summary ===
  fold 0: best val_iou = 0.78
  fold 1: best val_iou = 0.81
  fold 2: best val_iou = 0.76
  fold 3: best val_iou = 0.79
  fold 4: best val_iou = 0.80
  mean ± std:  0.788 ± 0.018
```

Interpreting `std`:
- `< 0.02` — folds agree, model is converged. More data unlikely to help much.
- `> 0.05` — folds disagree, more training data (or stronger augmentation) would close the gap.

## Legacy single-fold trainer

`train_boundary_only.py` is the previous setup that produced `models/sam3_lora_v4/`. Differences:
- Trains on the 27 hand-drawn cases only (no auto-labelled data, no review pipeline)
- Hardcoded train/val split (4 specific cases as val)
- Semantic head only — no instance-head loss
- Includes erosion-consistency loss (kept as a band-aid for outline-style hand-drawn masks)

Use `train_sam3_kfold.py` for new work; the legacy script is retained for ablation against earlier results.
