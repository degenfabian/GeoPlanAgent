"""Shared helpers for the held-out k-fold validators."""

import json
from pathlib import Path
from typing import Any, Dict


def write_predictions_json(predictions: Dict[str, Any], output_path: Path) -> None:
    """Write per-page predictions to ``output_path`` as a sorted JSON file.

    Creates parent directories if needed. Used by both
    ``eval_rotation_kfold.py`` (predicted rotation degrees per page) and
    ``eval_sam_kfold.py`` (per-page dicts of fold plus the semantic/instance
    IoU/precision/recall/F1 metrics).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))
    print(
        f"\nWrote {len(predictions)} predictions to "
        f"{output_path.relative_to(output_path.parent.parent.parent)}"
    )
