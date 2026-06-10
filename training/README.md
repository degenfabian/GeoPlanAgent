# training/

Fine-tuning code for the two learned components in the pipeline:

1. **SAM3 LoRA** — the boundary segmentation adapter loaded by
   `tools.extraction.sam3.load_sam3_ft`.
2. **Rotation classifier** — the 4-way ResNet50 used by
   `tools.io.rotation_classifier.predict_rotation_cw` (called via
   `tools.io.map_page.render_map_page`).

Both are 5-fold cross-validated so each benchmark case is evaluated by
the fold whose val set it belonged to (no leakage). Routing is via
[`tools.core.fold_routing.resolve_fold`](../tools/core/fold_routing.py)
(`fold_assignment.json[case_name]` → `min(available_folds)` fallback
for cases outside the training pool). Not required for inference if
you have the shipped weights in `models/`.

## Layout

```
training/
├── README.md                       ← you are here
├── boundary_augmentations.py       style-transfer + copy-paste augmentations
├── build_sam3_training_set.py      assemble training set from boundary_annotations/
├── train_rotation.py               ResNet50 5-fold rotation classifier (shared
│                                   model class + KFoldRotationDataset live here too,
│                                   used by training/eval/)
├── train_sam3_kfold.py             SAM3 LoRA 5-fold trainer (combined semantic
│                                   + instance head, focal + dice + ramped surface)
├── annotation/                     interactive UIs for building the training set
│   ├── README.md
│   ├── boundary_annotator.py       Flask backend for the boundary-polygon UI
│   ├── boundary_annotator_ui.html  the HTML/JS canvas UI
│   ├── boundary_prerender.py       pre-render every eval case at production DPI
│   └── rotation_annotator.py       Flask UI for hand-labelling corrective rotations
├── dataset/                        assembled training inputs
│   ├── maps/                       211 PNG renders (one per case)
│   ├── boundary_masks/             211 binary masks (annotator output)
│   ├── fold_assignment.json        single source of truth: {case_name: fold}
│   └── rotation_annotations.json   hand-labelled corrective rotations per case
└── eval/                           held-out k-fold validators
    ├── README.md
    ├── _util.py                    write_predictions_json helper
    ├── eval_rotation_kfold.py
    ├── eval_sam_kfold.py
    └── predictions/                JSON outputs of the two eval scripts
```

## Prerequisites

```bash
uv sync --extra training
```

`pyproject.toml` pins `transformers`, `peft`, `torch`, etc. SAM3 base
weights (~3 GB) download from HuggingFace on first run; needs
`HF_TOKEN` in `.env`.

## Reproducing the training set

The dataset is assembled from a hand-annotation pipeline rather than
auto-labelled. Three stages, in order — see
[`annotation/README.md`](annotation/README.md) for the UI details:

### 1. Pre-render every eval case

```bash
uv run python training/annotation/boundary_prerender.py
```

Renders each `evaluation_data/<case>/document.pdf` at DPI 200 (the
production DPI), **without** auto-rotation — annotation happens in the
raw PDF frame so it isn't coupled to the rotation classifier's
confidence. Output: `boundary_annotations/<case>/map.png` plus a
per-case `initial.json` seeded with polygon coords from cached
affines.

### 2. Annotate boundaries in the browser

```bash
uv run python training/annotation/boundary_annotator.py
# → http://localhost:5000/
```

Flask + canvas UI. The annotator traces / corrects each case's
boundary polygon over the rendered map. Saves
`boundary_annotations/<case>/{edited.json, edited_mask.png}` (image
pixel coords + raster mask). State persists per-case; resumable.

### 3. Annotate corrective rotations

```bash
uv run python training/annotation/rotation_annotator.py
# → http://localhost:5000/
```

For each `map.png`, click `0` / `90` / `180` / `270` for the
corrective rotation needed to make the map upright. Writes to
`training/dataset/rotation_annotations.json`.

### 4. Assemble the SAM3 training set

```bash
uv run python training/build_sam3_training_set.py
```

Copies `boundary_annotations/<case>/{map.png, edited_mask.png}` into
`training/dataset/{maps, boundary_masks}/<case>.png` and writes
`fold_assignment.json` (`{case_name: fold}`).

Fold assignment uses LPT (longest-processing-time-first) bin-packing
for balanced fold sizes (43/42/42/42/42 for our 211-case pool) while
keeping "stay-together" groups intact: multi-page renders from one
source PDF and explicit twin sets sharing a planning site never
straddle the train/val split. **Idempotent: same input → bit-identical
output.** No separate manifest file — every case's filename stem IS
its canonical name, and the fold-assignment JSON contains its fold.

`fold_assignment.json` records three keys per case (original case
name, canonical underscore form, and filesystem-safe form) so lookups
by any of those three forms resolve. Production inference looks up by
the case name straight from the eval-data folder; training looks up
by the map filename's stem.

## Train SAM3 LoRA k-fold

```bash
uv run python training/train_sam3_kfold.py
```

Iterates folds 0→4 sequentially. Each fold trains on cases NOT in
fold k (~170 cases) and validates on cases IN fold k (~42 cases).
Per-fold checkpoints land in `models/sam3_lora/fold_<k>/` in PEFT
format:

```
models/sam3_lora/fold_<k>/
├── adapter_config.json         # PEFT config (best-val checkpoint)
├── adapter_model.safetensors   # LoRA + saved head weights (~76 MB)
├── training_meta.json          # epoch, best_val_iou, config
└── history.json                # per-epoch train/val loss + val IoU
```

Only the best-val checkpoint ships. **During training**, the trainer
also writes a transient `<fold_k>/latest/` PEFT subdir each epoch
(same `adapter_*` files plus a `trainer_state.pt` sidecar carrying
the optimizer / scheduler / epoch / history). It's the `--resume`
target only — delete it after training completes (or skip it
entirely if you don't intend to resume).

The top-level PEFT files are rewritten whenever val IoU improves;
they're what production loads via `tools.extraction.sam3.load_sam3_ft`.

Wall time: ~1.5–2 hr per fold on Apple MPS with bf16; ~1 hr per fold
on CUDA. ~8–10 hr for all five.

### Trainer flags

```bash
uv run python training/train_sam3_kfold.py --help
```

| Flag | Default | Purpose |
|---|---|---|
| `--folds` | `0,1,2,3,4` | Comma-separated fold indices to run |
| `--epochs` | `20` | Max epochs per fold |
| `--rank` | `16` | LoRA rank |
| `--lr` | `2e-4` | Initial LR (cosine decay to 5% of base) |
| `--batch-size` | `1` | Effective batch = `batch_size × grad_accum` |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--grad-clip` | `0.1` | Max grad norm |
| `--oversample` | `2` | Train-set oversample factor |
| `--bf16 / --no-bf16` | `--bf16` | Mixed precision (bf16 on CUDA, fp16 on MPS) |
| `--patience` | `6` | Early-stop fold if val IoU stalls (0 = disabled) |
| `--seed` | `42` | Master seed; per-fold seed = `seed + fold_idx` |
| `--num-workers` | `2` | DataLoader workers |
| `--resume` | off | Resume each fold from `latest.pt` if present |
| `--dataset-dir` | `training/dataset` | Override dataset location |

## Train rotation classifier k-fold

```bash
uv run python training/train_rotation.py
```

Same 5-fold split as SAM3 (the same `fold_assignment.json` keys are
read). Each fold trains a ResNet50 (ImageNet pretrained, full
fine-tune) with a 4-way head on rotation-augmented map pages. Output:

```
models/rotation_classifier_kfold/
├── fold_<k>/best.pt + history.json
├── fold_assignment.json    # mirror for inference lookup
└── kfold_summary.json      # per-fold best_val_acc
```

`train_rotation.py` also defines the shared `RotationClassifier`,
`KFoldRotationDataset`, and helper utilities; the eval script
in `training/eval/eval_rotation_kfold.py` imports from it.

## Held-out evaluation

See [`eval/README.md`](eval/README.md) for the per-case k-fold
validators. They write `training/eval/predictions/<model>.json`.

## Reproducibility guarantees

With a fixed `--seed` and the same dataset, two runs on the same
hardware produce matching trajectories:

- `random` (Python), `numpy`, `torch`, CUDA RNG all seeded.
- `DataLoader` shuffle uses a `torch.Generator` seeded per fold.
- LoRA init seeded (PEFT honours `torch.manual_seed`).
- Per-fold seed = `master_seed + fold_idx` so folds explore
  independent sequences.
- Caveat: with `--bf16` on, tiny float-rounding deltas across runs (same
  trajectory, different last-bits). Disable for bit-exact repro.
- Caveat: MPS has a few non-deterministic ops. CUDA with
  `torch.use_deterministic_algorithms(True)` is the gold path for
  bit-exact reproduction.

Each saved checkpoint embeds its `config` dict (rank, lr, epochs,
seed, bf16, patience) so a third party can read a checkpoint and
know exactly which flags produced it.

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
  outline-style masks; the curated training set uses filled masks,
  so erosion adds no signal and hurts multi-blob predictions.
- **Instance loss runs at the head's native resolution (~256×256)**,
  not the full map resolution. Necessary on MPS to avoid materialising
  a `[100, 2300, 1654]` intermediate that exceeds MPSGraph's INT_MAX.

## Cross-fold reporting

After all 5 folds finish, the trainer prints per-fold metrics + an
aggregate, and writes `models/sam3_lora/cv_summary.{json,csv}`. Held-out
re-eval (`training/eval/eval_sam_kfold.py`) overwrites the same files
with bit-for-bit reproducible numbers; the paper table sources from
those. Reference values from the current checkpoints:

```
=== 5-fold summary (sem-gated) ===
  fold 0 (n_val=43, best_ep=8):  sem_iou=0.911  f1=0.943  inst_iou=0.909
  fold 1 (n_val=42, best_ep=5):  sem_iou=0.932  f1=0.961  inst_iou=0.894
  fold 2 (n_val=42, best_ep=6):  sem_iou=0.879  f1=0.909  inst_iou=0.874
  fold 3 (n_val=42, best_ep=3):  sem_iou=0.886  f1=0.920  inst_iou=0.883
  fold 4 (n_val=42, best_ep=19): sem_iou=0.952  f1=0.973  inst_iou=0.953

  Paper-grade aggregates (n_total_val=211):
    sem iou     0.9121 ± 0.0276
    sem f1      0.9414 ± 0.0241
    inst iou    0.9027 ± 0.0279
```

Reading the variance: `std < 0.02` → folds agree, model is converged.
`std > 0.05` → folds disagree, more training data or stronger
augmentation could close the gap.

## Inference-time fold routing

The shared routing helper is
[`tools.core.fold_routing.resolve_fold`](../tools/core/fold_routing.py),
used both by `tools.extraction.sam3.set_fold_for_case` and by
`tools.io.rotation_classifier`:

1. Look up `fold_assignment.json[case_name]` (raw eval-data folder name).
2. If missing, retry the canonical underscore form
   (`replace(":", "_").replace("/", "_")`).
3. **If the case still isn't in `fold_assignment.json`** (only possible
   for an external deployment on a fresh case not in the training
   pool), fall back to `min(available_folds)`. The hash-based fallback
   that lived here previously was redundant: an unseen case has no
   preferred fold, so any deterministic constant is equivalent.
4. The returned fold index is clamped to `available_folds`, so a
   half-populated `models/<model>/` (only some folds on disk) routes
   safely to one of the loaded adapters rather than crashing.
