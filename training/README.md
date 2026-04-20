# training/

SAM3 LoRA fine-tuning for the "planning boundary" segmentation task. This is the single pipeline that produces the production adapter loaded by `tools.sam3_boundary.load_sam3_ft` at inference time.

Not required for inference — a trained adapter ships in `models/sam3_lora_v4/`. Only run this if you want to reproduce or extend the fine-tune.

## Prerequisites

```bash
uv sync --extra training
```

Training data lives in `boundary_annotation_dataset/` at the repo root (gitignored; not shipped). Expected layout:

```
boundary_annotation_dataset/
├── maps/                 # PNG planning map images
└── boundary_masks/       # Matching binary PNG masks
```

## Files

| File | Purpose |
|---|---|
| `train_boundary_only.py` | Main trainer — SAM3 LoRA on the semantic-segmentation head |
| `boundary_augmentations.py` | Style-transfer and copy-paste augmentations used during training |
| `outputs/` | Checkpoints (gitignored) |

## Reproducing the production checkpoint

```bash
cd training
uv run python train_boundary_only.py --epochs 40 --rank 16
```

Outputs land in `training/outputs/boundary_only_semantic/`. The production path copies the best checkpoint to `models/sam3_lora_v4/checkpoint_latest`, which is what `load_sam3_ft` reads.

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | (script default) | Total epochs |
| `--rank` | 16 | LoRA rank |
| `--batch-size` | (script default) | Batch size |
| `--lr` | (script default) | Learning rate |

See `uv run python train_boundary_only.py --help` for the full list.

## Notes

- `boundary_augmentations.style_transfer_augment` is the workhorse — it converts filled boundary regions into outline / dashed / dotted variants and recolours them, teaching the model to recognise the boundary outlines common in real UK planning maps. Without this, the 27-sample training set does not generalise.
- The trainer reads directly from `boundary_annotation_dataset/`; there is no separate synthetic-data pipeline.
- To point inference at a different adapter, edit the default `lora_path` in `tools.sam3_boundary.load_sam3_ft`.
