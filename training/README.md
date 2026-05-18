# training/

Fine-tuning code for the two learned components in the pipeline:

1. **SAM3 LoRA** — the boundary segmentation adapter loaded by
   `tools.extraction.sam3.load_sam3_ft`.
2. **Rotation classifier** — the 4-way ResNet50 used by
   `tools.io.rotation_classifier.predict_rotation_cw` (called via
   `tools.io.map_page.render_map_page`).

Both are 5-fold cross-validated so each benchmark case is evaluated by
the fold whose val set it belonged to (no leakage). Not required for
inference if you have the shipped weights in `models/`.

## Files

| File | Purpose |
|---|---|
| `train_sam3_kfold.py` | SAM3 LoRA 5-fold trainer. Combined semantic + instance-head loss, focal + dice + ramped surface. |
| `train_rotation_kfold.py` | Rotation classifier 5-fold trainer. ResNet50, 4-way head, 4-rotation TTA at eval. |
| `train_rotation_classifier.py` | Legacy single-split rotation trainer (kept for parity / ablation). |
| `boundary_augmentations.py` | Style-transfer + copy-paste augmentations used at train time. |
| `dataset/` | Assembled SAM3 training data: `maps/`, `boundary_masks/`, `manifest.json`, `fold_assignment.json`. |

## Prerequisites

```bash
uv sync --extra training
```

`pyproject.toml` pins `transformers`, `peft`, `torch`, etc. SAM3 base
weights (~3 GB) download from HuggingFace on first run; needs
`HF_TOKEN` in `.env`.

## Reproducing the dataset

The dataset is assembled from a hand-annotation pipeline rather than
auto-labelled. Three stages, in order:

### 1. Pre-render every eval case

```bash
uv run python scripts/annotate_prerender.py
```

Renders each `evaluation_data/<case>/document.pdf` at DPI 200 (the
production DPI), **without** auto-rotation — we want the annotator to
work in the same frame as the raw PDF render so the annotation isn't
coupled to the rotation classifier's confidence. Output:
`boundary_annotations/<case>/map.png` plus per-case initial polygon
coords seeded from cached affines.

### 2. Annotate in the browser

```bash
uv run python scripts/annotate_server.py
# → http://localhost:5000/
```

Flask + canvas UI. For each case, the annotator traces / corrects the
boundary polygon over the rendered map. Saves to
`boundary_annotations/<case>/edited.json` (image-pixel rings) plus a
rasterised `edited_mask.png`. State persists per-case so quitting and
resuming is fine.

### 3. Assemble the training set

```bash
uv run python scripts/build_sam3_training_set.py
```

Copies `boundary_annotations/<case>/{map.png, edited_mask.png}` into
`training/dataset/{maps, boundary_masks}/<case>.png` and writes:

- `manifest.json` — per-case `{case, filename, fold, group_key, …}`
- `fold_assignment.json` — `{case_name: fold}` for production lookup
  (mirrored into `models/sam3_lora/fold_assignment.json` after training)
- `manifest.csv` — same as JSON for inspection

Fold assignment uses LPT (longest-processing-time-first) bin-packing
for balanced fold sizes while keeping "stay-together" groups intact
(multi-page renders from one source, twin sets sharing a planning
site). Idempotent: same input → bit-identical output.

## Train SAM3 LoRA k-fold

```bash
uv run python training/train_sam3_kfold.py
```

Iterates folds 0→4 sequentially. Each fold trains on cases NOT in fold
k (~95-100 cases) and validates on cases IN fold k (~40 cases). Per-
fold checkpoints land in `models/sam3_lora/fold_<k>/`:

```
models/sam3_lora/fold_<k>/
├── best.pt         # rewritten when val IoU improves
├── latest.pt       # rewritten every epoch (resume target)
└── history.json    # per-epoch train/val loss + val IoU
```

Wall time: ~1.5–2 hr per fold on Apple MPS with bf16; ~1 hr per fold
on CUDA. ~8-10 hr for all five.

### Trainer flags

```bash
uv run python training/train_sam3_kfold.py --help
```

| Flag | Default | Purpose |
|---|---|---|
| `--folds` | `0,1,2,3,4` | Comma-separated fold indices to run |
| `--epochs` | 30 | Max epochs per fold |
| `--rank` | 16 | LoRA rank |
| `--lr` | 2e-4 | Initial learning rate (cosine decay to 0.05× by end) |
| `--batch-size` | 1 | Effective batch = `batch_size × grad_accum` |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--bf16 / --no-bf16` | `--bf16` | Mixed precision (bf16 on CUDA, fp16 on MPS) |
| `--patience` | 6 | Early-stop fold if val IoU stalls (0 = disabled) |
| `--seed` | 42 | Master seed; per-fold seed = `seed + fold_idx` |
| `--resume` | off | Resume each fold from `latest.pt` if present |
| `--dataset-dir` | `training/dataset` | Override dataset location |

## Train rotation classifier k-fold

```bash
uv run python training/train_rotation_kfold.py
```

Same 5-fold split as SAM3 (`fold_assignment.json` is shared via case
name). Each fold trains a ResNet50 (ImageNet pretrained, frozen first
N layers) with 4-way head on rotation-augmented map pages. Output:

```
models/rotation_classifier_kfold/
├── fold_<k>/best.pt + history.json
├── fold_assignment.json    # mirror for inference lookup
└── kfold_summary.json      # per-fold best_val_acc + history
```

## Reproducibility guarantees

With a fixed `--seed` and the same dataset, two runs on the same
hardware produce matching trajectories:

- ✅ `random` (Python), `numpy`, `torch`, CUDA RNG all seeded.
- ✅ `DataLoader` shuffle uses a `torch.Generator` seeded per fold.
- ✅ LoRA init seeded (PEFT honours `torch.manual_seed`).
- ✅ Per-fold seed = `master_seed + fold_idx` so folds explore
  independent sequences.
- ⚠️ With `--bf16` on, you get tiny float-rounding deltas across runs
  (same trajectory, different last-bits). Disable for bit-exact repro.
- ⚠️ MPS has a few non-deterministic ops. CUDA with
  `torch.use_deterministic_algorithms(True)` is the gold path for
  bit-exact reproduction.

Each saved checkpoint embeds its `config` dict (rank, lr, epochs,
seed, bf16, patience) so a third party can read a checkpoint and know
exactly which flags produced it.

## Loss formulation (SAM3 LoRA)

Weights live at the top of `train_sam3_kfold.py` (`LOSS_WEIGHT_*`):

```
SEMANTIC HEAD (focal + dice + ramped surface)
  L_sem = 5.0·focal(α=0.6, γ=1.6)(sem_pred, gt)
        + 5.0·dice(sem_pred, gt)
        + min(1, epoch/15) · 0.5 · surface(σ(sem_pred), signed_dist(gt))

INSTANCE HEAD (mask losses on the Hungarian-matched slot, classification
and presence on all slots — matches the SAM3 author training recipe)
  best  = argmin_i  cost_match(slot_i, gt)
        where cost = -IoU - 0.05·σ(cls_i)
  L_inst = 5.0·focal(α=0.25, γ=2)(pred_masks[best], gt)
         + 5.0·dice(pred_masks[best], gt)
         + 2.0·focal_cls(cls_logits, soft_target)
         + 1.0·BCE(presence_logits, target=1)

total_loss = L_sem + L_inst
```

Notes on the per-term design:

- **Classification target is soft, not 1-hot.** The matched slot
  receives `σ(cls_best)^0.25 · IoU_best^0.75` as its positive target;
  unmatched slots get 0. This prevents the cls head from saturating
  to infinity on partially-correct masks while still anchoring slot
  identity across epochs.
- **Presence BCE target is always 1** because every training image
  contains a planning boundary. At inference the worker uses the
  presence head as a confidence gate on the instance flow.
- **No erosion-consistency loss.** Was a band-aid for hand-drawn
  outline-style masks; the curated training set uses
  `cv2.fillPoly`-style filled masks, so erosion adds no signal and
  hurts multi-blob predictions.
- **Instance loss runs at the head's native resolution (~256×256)**,
  not the full map resolution. Necessary on MPS to avoid materialising
  a `[100, 2300, 1654]` intermediate that exceeds MPSGraph's INT_MAX.

## Cross-fold reporting

After all 5 folds finish, the trainer prints:

```
=== 5-fold summary ===
  fold 0: best val_sem_iou = 0.877  val_inst_iou = 0.867
  fold 1: best val_sem_iou = 0.922  val_inst_iou = 0.922
  fold 2: best val_sem_iou = 0.827  val_inst_iou = 0.827
  fold 3: best val_sem_iou = 0.879  …
  fold 4: …
  mean ± std:  …
```

Reading the variance: `std < 0.02` → folds agree, model is converged.
`std > 0.05` → folds disagree, more training data or stronger
augmentation could close the gap.

## Inference-time fold routing

`tools.extraction.sam3.set_fold_for_case(state, case_name)`:

1. Canonicalise the case_name (`replace(":", "_").replace("/", "_")`).
2. Look up `fold_assignment.json[case_name]` if present.
3. Fall back to `md5(canonical) % 5` if the case wasn't in the
   training pool.
4. Load `models/sam3_lora/fold_<k>/best.pt` (LoRA weights + saved
   head modules).

Same mechanism for the rotation classifier:
`tools.io.rotation_classifier` looks up the fold via case_name and
loads `models/rotation_classifier_kfold/fold_<k>/best.pt`.
