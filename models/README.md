# Model Weights

Two trained model assets, both 5-fold cross-validated so every
benchmark case is evaluated by the fold that did **not** see it during
training (no leakage).

```
models/
├── sam3_lora/                    # SAM3 boundary segmentation
│   ├── fold_<k>/                 # one per fold (0..4)
│   │   ├── adapter_config.json       # PEFT config (best-val checkpoint)
│   │   ├── adapter_model.safetensors # LoRA + heads weights (~76 MB)
│   │   ├── training_meta.json        # epoch, best_val_iou, training config
│   │   ├── history.json              # per-epoch train/val metrics
│   │   └── latest/                   # resume target (PEFT + sidecar);
│   │                                 # deletable after training completes
│   ├── fold_assignment.json      # {case_name: fold} — sync'd from
│   │                             # training/dataset/fold_assignment.json
│   └── cv_summary.{csv,json}     # held-out 5-fold sem/inst IoU + P/R/F1
│                                 # (written by training/eval/eval_sam_kfold.py)
│
└── rotation_classifier_kfold/    # 4-way page-rotation classifier
    ├── fold_<k>/
    │   ├── best.pt                   # ~90 MB (full ResNet50 state_dict)
    │   └── history.json
    ├── fold_assignment.json      # same content as the SAM3 one
    ├── kfold_summary.json        # training-time per-epoch + best_val_acc
    └── cv_summary{,_tta}.{csv,json}   # held-out 5-fold accuracy, written
                                 # by training/eval/eval_rotation_kfold.py;
                                 # _tta suffix when run with --tta
```

`models/` is gitignored except for the small JSON/MD metadata that's
useful to version. The PEFT safetensors (SAM3 LoRA, ~76 MB / fold) and
the ResNet50 checkpoints (~90 MB / fold) live locally only —
~830 MB total across the two model families and ten folds.

## SAM3 LoRA adapter (`sam3_lora/`)

- **Base model**: `facebook/sam3` from HuggingFace (~3 GB,
  auto-downloaded on first run via `HF_TOKEN`).
- **Fine-tune**: LoRA r=16 on both heads (semantic + instance) via
  PEFT. Trained with focal + dice + (ramped) surface loss; instance
  head trained on the Hungarian-matched best-IoU slot.
- **Text query** at inference is locked to `"planning boundary"` (the
  literal phrase the LoRA was trained against).
- **Checkpoint size**: ~76 MB per fold in PEFT format. Only the LoRA
  matrices + the saved head modules (`mask_embedder`, `presence_head`,
  `semantic_projection`) are persisted; the base SAM3 weights load
  from HuggingFace at inference.

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
the paper). Each fold's `best.pt` evaluated on its held-out val set;
mean/std are over the 5 folds (mean-of-fold-means, population std):

| Fold | n_val | val_sem_iou | val_sem_f1 | val_inst_iou | val_inst_f1 |
|---|---|---|---|---|---|
| 0 | 43 | 0.911 | 0.943 | 0.909 | 0.942 |
| 1 | 42 | 0.932 | 0.961 | 0.894 | 0.919 |
| 2 | 42 | 0.879 | 0.909 | 0.874 | 0.907 |
| 3 | 42 | 0.886 | 0.920 | 0.883 | 0.916 |
| 4 | 42 | 0.952 | 0.973 | 0.953 | 0.974 |
| Mean ± std | — | **0.912 ± 0.028** | 0.941 ± 0.024 | 0.903 ± 0.028 | 0.931 ± 0.024 |

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

5-fold held-out accuracy (each fold's `best.pt` evaluated on its
held-out val set; mean ± std over the 5 folds, from
`cv_summary{,_tta}.json`):

| Mode | Accuracy |
|---|---|
| No TTA | 0.953 ± 0.061 |
| 4-way TTA (deployed) | **0.981 ± 0.010** |

TTA recovers 6 cases the single-view classifier misses (most of them in
fold 0); on the rotated subset both modes hit 20-21/22. The deployed
pipeline uses TTA, so the TTA accuracy is the operationally relevant
number. With MPS bf16 inference there's ~1-case run-to-run jitter at
the softmax decision boundary, so this number is reproducible to
roughly ±0.005. To regenerate either:

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
- **No `latest/` PEFT dirs in the shipped checkpoints.** Those are
  the trainer's resume target, rewritten every epoch. Safe to
  delete once training completes. Re-running training with
  `--resume` regenerates them.
- **No legacy `models/rotation_classifier/best.pt` single-checkpoint
  fallback path.** The k-fold layout is the only supported one.
