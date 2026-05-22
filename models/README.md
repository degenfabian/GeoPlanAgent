# Model Weights

Two trained model assets, both 5-fold cross-validated so every
benchmark case is evaluated by the fold that did **not** see it during
training (no leakage).

```
models/
├── sam3_lora/                    # SAM3 boundary segmentation
│   ├── fold_<k>/                 # one per fold (0..4)
│   │   ├── best.pt               #   peak-val checkpoint (~3.3 GB)
│   │   └── history.json          #   per-epoch train/val metrics
│   ├── fold_assignment.json      # {case_name: fold} — sync'd from
│   │                             # training/dataset/fold_assignment.json
│   └── cv_summary.{csv,json}     # 5-fold val IoU + F1 + precision/recall
│
└── rotation_classifier_kfold/    # 4-way page-rotation classifier
    ├── fold_<k>/
    │   ├── best.pt
    │   └── history.json
    ├── fold_assignment.json      # same content as the SAM3 one
    └── kfold_summary.json
```

`models/` is gitignored — the .pt files are ~17 GB total and live
locally only.

## SAM3 LoRA adapter (`sam3_lora/`)

- **Base model**: `facebook/sam3` from HuggingFace (~3 GB,
  auto-downloaded on first run via `HF_TOKEN`).
- **Fine-tune**: LoRA r=16 on both heads (semantic + instance) via
  PEFT. Trained with focal + dice + (ramped) surface loss; instance
  head trained on the Hungarian-matched best-IoU slot.
- **Text query** at inference is locked to `"planning boundary"` (the
  literal phrase the LoRA was trained against).
- **Checkpoint size**: ~3.3 GB per fold (full PyTorch state_dict —
  base model + LoRA + saved head modules `mask_embedder`,
  `presence_head`, `semantic_projection`).

### Fold routing at inference

`tools.extraction.sam3.set_fold_for_case(state, case_name)` delegates
to the shared helper `tools.core.fold_routing.resolve_fold`:

1. Try `fold_assignment[case_name]` (raw eval-data folder name).
2. Fall back to the canonical underscore form
   (`:` and `/` → `_`).
3. **If the case isn't in `fold_assignment.json`** (only possible for
   an external case not in our 211-case training pool), fall back to
   `min(available_folds)`. The case wasn't in any fold's training
   set, so every adapter is equally valid; we pick deterministically
   rather than via hash (the previous md5 fallback carried no real
   signal).

5-fold cross-val (from `cv_summary.json`, matches `tab:sam3-cv` in
the paper):

| Fold | n_val | val_sem_iou | val_sem_f1 | val_inst_iou |
|---|---|---|---|---|
| 0 | 43 | 0.877 | 0.908 | 0.867 |
| 1 | 42 | 0.922 | 0.946 | 0.922 |
| 2 | 42 | 0.827 | 0.860 | 0.827 |
| 3 | 42 | 0.879 | 0.914 | … |
| 4 | 42 | 0.953 | 0.974 | … |
| Mean ± std | — | **0.892 ± 0.043** | 0.920 | — |

To regenerate exactly:

```bash
uv run python training/eval/eval_sam_kfold.py
# → training/eval/predictions/sam_kfold.json (one entry per case:
#   {fold, sem_iou, inst_iou})
```

### Loading at inference

```python
from tools.extraction.sam3 import load_sam3_ft, set_fold_for_case

state = load_sam3_ft()                # base SAM3 + PEFT wrapper
set_fold_for_case(state, case_name)   # swap in fold k's adapter
# → state["processor"], state["model"], state["device"] ready
```

If `models/sam3_lora/` is missing, `load_sam3_ft` falls through to
base SAM3 (no LoRA). Accuracy drops materially — the fine-tune is
where the boundary-specific knowledge lives.

## Rotation classifier (`rotation_classifier_kfold/`)

- **Architecture**: ResNet50 (ImageNet pretrained), 4-way output
  (0°/90°/180°/270° CW corrective). Trained on `boundary_annotations/
  <case>/map.png` with 4× rotation augmentation per case.
- **Inference**: 4-way test-time augmentation by default. Predicts
  on the input AND its 90°/180°/270° CW rotations, cyclically shifts
  each rotated prediction back to the original frame, ensembles the
  logits, abstains (returns 0°) when the top-class softmax is below
  the configured confidence threshold.
- **Fold routing**: same `resolve_fold` helper as SAM3 — production
  reads from `fold_assignment.json` in this dir.

5-fold accuracy (per-fold top-1, ResNet50 backbone):

| Mode | Accuracy |
|---|---|
| No TTA | 0.953 |
| 4-way TTA (deployed) | **0.986** |

The paper currently cites the no-TTA number (`0.960`), but the
deployed pipeline uses TTA — the TTA accuracy is the operationally
relevant one. To regenerate either:

```bash
uv run python training/eval/eval_rotation_kfold.py        # single-view → rotation_kfold.json
uv run python training/eval/eval_rotation_kfold.py --tta  # 4-way TTA → rotation_kfold_tta.json
```

### Loading at inference

Handled internally by `tools.io.map_page.render_map_page`. No need to
call directly:

```python
from tools.io.map_page import render_map_page
img, rot_info = render_map_page(pdf_path, page_1based=3, dpi=200,
                                  case_name="12:00116:ART4")
# rot_info["rotation_cw_degrees"]: 90  (CW degrees the classifier applied)
# rot_info["confidence"]: top-class softmax probability
# rot_info["fold"]: which fold's checkpoint ran
```

## Why two copies of `fold_assignment.json`?

The same content lives at:

- `training/dataset/fold_assignment.json` — source of truth, written
  by `training/build_sam3_training_set.py`
- `models/sam3_lora/fold_assignment.json` — deployment copy alongside
  the checkpoints, so production inference doesn't need `training/`
- `models/rotation_classifier_kfold/fold_assignment.json` — same
  reason

After `train_sam3_kfold.py` finishes, it copies the training/dataset
version into `models/sam3_lora/`. The three files should always have
**identical content**. If they ever drift, treat the training/dataset
version as authoritative and copy it forward.

Each file has 270 entries for 211 unique cases — three lookup-key
variants per case (original folder name, canonical underscore form
after `:`/`/` → `_`, and the filesystem-safe form used as the map
filename stem). All variants map to the same fold; this is purely
for robust lookup, not a leakage source.

## Notes on what isn't here

- **No MapSAM weights.** MapSAM was tested as an alternative
  segmentation backbone and did not outperform SAM3-LoRA on this
  dataset.
- **No critic weights.** The critic uses an LLM at inference; it
  doesn't have its own learned model.
- **No `latest.pt` files.** Those were the trainer's resume target,
  rewritten every epoch. Removed once training was done (~17 GB
  saved). Re-running training with `--resume` regenerates them.
- **No legacy `models/rotation_classifier/best.pt` single-checkpoint
  fallback path.** The k-fold layout is the only supported one.
