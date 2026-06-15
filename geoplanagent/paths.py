"""Canonical dataset + repo paths.

Single source of truth so the eval-data location and the dataset spreadsheet
are set once here, not hardcoded across benchmark_runner.py, scripts/, and
ablations/.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The 208 evaluation cases: one folder per case (PDF + ground-truth GeoJSON).
DATA_DIR = REPO_ROOT / "data"

# The dataset spreadsheet and the cleaned 208-case sheet within it.
DATASET_XLSX = DATA_DIR / "0_planning_dataset_list.xlsx"
DATASET_SHEET = "Cleaned_up_208_planning_dataset"
