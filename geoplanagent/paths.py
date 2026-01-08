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

# OS OpenData root + the OML road artifacts derived from open_map_local/.
# Both are built by scripts/setup_os_opendata.py and read by the locate
# road()/intersect() tools.
OS_OPENDATA_DIR = REPO_ROOT / "os_opendata"
OML_ROAD_INDEX = OS_OPENDATA_DIR / "open_map_local" / "oml_road_index.json"
OML_ROAD_GEOM = OS_OPENDATA_DIR / "open_map_local" / "oml_road_geom.json"

# OS Open Zoomstack basemap: rendered to tiles (tiles.py) and queried for
# nearby road names during commit verification (matching.py).
OS_ZOOMSTACK_GPKG = OS_OPENDATA_DIR / "OS_Open_Zoomstack.gpkg"

# Auto-rotation classifier (tools/rotation_classifier.py): the k-fold dir of
# per-fold checkpoints (fold_*/best.pt) for no-leakage routing.
ROTATION_KFOLD_DIR = REPO_ROOT / "models" / "rotation_classifier_kfold"

# One canonical case->fold map for inference, read by tools/rotation_classifier.py
# (rotation) and segment.py (SAM3). Written by the training scripts from the split;
# resolve_fold normalises keys so every eval folder name resolves against it.
FOLD_ASSIGNMENT = REPO_ROOT / "models" / "fold_assignment.json"

# The SAM3 training set (maps/ + boundary_masks/), assembled by
# training/build_sam3_training_set.py from boundary_annotations/. Read by the
# training scripts and the SAM segmentation ablations.
TRAINING_DATASET_DIR = REPO_ROOT / "training" / "dataset"

# Cached k-fold eval predictions (training/eval/predictions/), read by the
# paper-figure scripts: SAM3-LoRA per-page semantic-seg sem_iou/sem_f1.
EVAL_PREDICTIONS_DIR = REPO_ROOT / "training" / "eval" / "predictions"
SAM_KFOLD_PREDICTIONS = EVAL_PREDICTIONS_DIR / "sam_kfold.json"

# ----- Run outputs (regenerable, gitignored). Ablation code/config live in ablations/. -----
RESULTS_DIR = REPO_ROOT / "results"

# The benchmark run whose cached per-case metrics reproduce the paper's main
# numbers — the default --run-dir of compute_tables.py and compute_figures.py.
MAIN_RUN_DIR = RESULTS_DIR / "main_pipeline" / "gemini-flash"

# Ablation run outputs: one subdir per ablation under results/ablations/. The
# ablation driver scripts (ablations/*.py) write here and scripts/compute_tables.py
# reads the same constants, so writer and reader can't drift.
ABLATIONS_RESULTS_DIR = RESULTS_DIR / "ablations"
ABL_LOCATE_ONLY = ABLATIONS_RESULTS_DIR / "locate_only_eval"
ABL_VLM_E2E = ABLATIONS_RESULTS_DIR / "vlm_e2e"
ABL_SAM_BASE = ABLATIONS_RESULTS_DIR / "sam_base"
ABL_VLM_SEG = ABLATIONS_RESULTS_DIR / "vlm_seg"
ABL_NO_READER = ABLATIONS_RESULTS_DIR / "no_reader"

# Ablation INPUTS the drivers consume live WITH the ablation code under ablations/
# (not results/): the VLM-subset definition and the frozen reader output that the
# locate ablations share (holds the reader output constant across configs).
VLM_E2E_SUBSET = REPO_ROOT / "ablations" / "subset_40.json"
ABL_PDF_INFO_CACHE = REPO_ROOT / "ablations" / "cached_pdf_info_for_locate_ablations.json"
