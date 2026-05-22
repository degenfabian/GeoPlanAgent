# training/eval/

Held-out cross-fold validators for the two trained models. Each one
loads every fold's checkpoint, runs it on the cases assigned to that
fold (i.e. cases the checkpoint did NOT see at training time), and
writes per-case predictions to `predictions/<model>.json`.

These are the scripts that produce the paper's per-fold numbers
(`tab:sam3-cv`, rotation classifier `0.960 ± 0.022`).

| Script | Output | Paper table |
|---|---|---|
| `eval_sam_kfold.py` | `predictions/sam_kfold.json` ← `{case: {fold, sem_iou, inst_iou}}` | `tab:sam3-cv` |
| `eval_rotation_kfold.py` | `predictions/rotation_kfold.json` ← `{case: predicted_rotation_deg}` (no TTA) | (mentioned in §5.x) |
| `eval_rotation_kfold.py --tta` | `predictions/rotation_kfold_tta.json` ← same shape, 4-way TTA | this is what the deployed pipeline actually uses |

## How fold routing works

Each case is looked up in `training/dataset/fold_assignment.json` to
find its held-out fold, then the script loads
`models/<model>/fold_<k>/best.pt` and runs that case through it. Same
logic as production inference — `tools.core.fold_routing.resolve_fold`
is the shared helper. No leakage by construction.

## Run

```bash
uv run python training/eval/eval_sam_kfold.py
uv run python training/eval/eval_rotation_kfold.py        # single-view
uv run python training/eval/eval_rotation_kfold.py --tta  # 4-way TTA (matches deployed)
```

Both scripts pick device automatically: CUDA if available, else MPS,
else CPU. SAM3 eval uses the same bf16 autocast as training so the
numbers match the trainer's saved `val_sem_iou`.

## Reading the outputs

```python
import json
sam = json.load(open("training/eval/predictions/sam_kfold.json"))
# {"095AB379-...": {"fold": 0, "sem_iou": 0.95, "inst_iou": 0.93}, ...}
mean_iou = sum(v["sem_iou"] for v in sam.values()) / len(sam)
```

```python
rot = json.load(open("training/eval/predictions/rotation_kfold_tta.json"))
# {"095AB379-...": 0, "12_00114_ART4": 90, ...}   (degrees CW corrective)
```

## TTA vs single-view (rotation)

The 4-way TTA mode predicts on the image + its 90° / 180° / 270°
rotated views, re-rotates each view's class-space predictions back
into the original frame, and averages the logits. This exactly
mirrors the inference-time TTA in `tools.io.rotation_classifier`
(`predict_rotation_with_confidence` with `tta=True`).

In our run TTA improves overall accuracy from ~0.953 to ~0.986. The
deployed pipeline uses TTA, so the TTA accuracy is the operationally
relevant number; the no-TTA file is kept for historical comparability.

## Shared helper

`training/eval/_util.py` exposes `write_predictions_json(predictions,
output_path)` — sorts keys, creates parent dirs, prints a one-line
summary. Both eval scripts use it so the output files always have
the same shape (sorted JSON, two-space indent).
