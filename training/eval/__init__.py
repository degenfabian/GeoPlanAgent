"""Held-out k-fold validators for the trained models.

Each ``eval_*_kfold.py`` script loads every fold's checkpoint, runs it
against its held-out cases, and writes per-case results to
``training/eval/predictions/<model>.json``. The aggregate per-fold
numbers reported in the paper come from these outputs.
"""

from training.eval._util import write_predictions_json

__all__ = ["write_predictions_json"]
