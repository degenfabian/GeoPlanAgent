"""Held-out k-fold validators for the trained models.

Each ``eval_*_kfold.py`` script loads every fold's checkpoint, runs it
against its held-out cases, and writes per-page results to
``training/eval/predictions/<model>_kfold{_tta}.json``. The paper's
Table 9 (rotation) and Table 11 (SAM3-LoRA segmentation) are computed
from these outputs by scripts/compute_tables.py.
"""
