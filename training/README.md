# training — SAM3-LoRA and the rotation classifier

Both learned components are trained as 5-fold cross-validation over the same
case→fold split (`models/fold_assignment.json`), so at inference every case is
handled by the fold that never saw it. The trained weights are distributed on
HuggingFace and the hand-annotated boundary masks are released together with the main dataset (see the main
README's Setup); this module contains the training and evaluation code.

## Reproducing Tables 9 and 11 from the paper

This requires the downloaded weights and the
boundary annotations (main README, Setup steps 4 and 7):

```bash
uv run training/build_sam3_training_set.py        # annotations -> training/dataset/
uv run training/eval/eval_sam_kfold.py            # SAM3-LoRA out-of-fold predictions
uv run training/eval/eval_rotation_kfold.py       # rotation, single-view
uv run training/eval/eval_rotation_kfold.py --tta # rotation, with test time augmentation
uv run scripts/compute_tables.py table9 table11
```

The eval scripts write their predictions to `eval/predictions/`, which is
where `scripts/compute_tables.py` reads them from.

## Retraining from scratch

Retraining needs the boundary annotations (main README, Setup step 7) and, for
SAM3, the assembled training set from `build_sam3_training_set.py`. Both
scripts train all five folds by default (seed 42) and write their checkpoints
into `models/`, replacing any downloaded weights there:

```bash
uv run training/train_sam3_kfold.py
uv run training/train_rotation.py
```

We trained SAM3 on an H200 and the rotation classifier on a Macbook M3 Max with MPS.
The hyperparameters can be found in the Appendix of the paper.

## Files

| File | Role |
|---|---|
| `build_sam3_training_set.py` | Assembles the SAM3 training set (map crops + binary masks) from the hand annotations into `training/dataset/` |
| `boundary_augmentations.py` | Style-transfer and geometric augmentations that widen the visual diversity of the small annotated pool |
| `train_sam3_kfold.py` | 5-fold LoRA fine-tune of SAM3 on the boundary masks (trained on an H200; paper Appendix has the hyper-parameters) |
| `train_rotation.py` | 5-fold rotation classifier (ResNet50) for scanned-map orientation; inference uses 4-way test-time augmentation (see paper Appendix for more details) |
| `seg_metrics.py` | `binary_mask_metrics` — the one IoU/precision/recall/F1/Dice implementation shared by training and eval |
| `eval/eval_sam_kfold.py` | Out-of-fold SAM3-LoRA evaluation → `eval/predictions/sam_kfold.json` |
| `eval/eval_rotation_kfold.py` | Out-of-fold rotation evaluation (single-view and `--tta`) → `eval/predictions/rotation_kfold*.json` |
| `dataset/rotation_annotations.json` | Ground-truth page orientations |
