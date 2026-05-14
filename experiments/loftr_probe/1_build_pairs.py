"""Build (map, tile, affine) training pairs from cached benchmark v20 results.

For each case where IoU > 0.7 (reliable affine), produce:
  pairs/<case>/map.png       — the planning-map image (copied from boundary_annotations/)
  pairs/<case>/tile.png      — the OS tile canvas at the matched location (re-rendered)
  pairs/<case>/affine.npy    — the 2x3 affine_H mapping map_px → tile_canvas_px
  pairs/<case>/meta.json     — case id, iou, tile_info, fold

Splits cases into train/val using the existing fold_assignment.json from
training/dataset/ — fold 0 → val, folds 1-4 → train. Respects stay-together
groups so leakage is prevented.

Usage:
  uv run python experiments/loftr_probe/1_build_pairs.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Repo + path bootstrap
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.loftr_probe._shared import PAIRS_DIR

ANNOT_DIR = REPO / "boundary_annotations"
BENCH_DIR = REPO / "results" / "benchmark_v20" / "gemini-flash"
DATASET_DIR = REPO / "training" / "dataset"

# Tunable: a 7x7 grid of 256-px tiles ~= 1792x1792 canvas covers most cases.
# For pairs where the matched window was near the canvas edge, we centre
# the canvas on the matched centre so the projected polygon usually lands
# inside.
TILE_GRID = (7, 7)
IOU_THRESHOLD = 0.7


def _load_case_payload(case_dir: Path):
    """Return (affine_H, tile_info, iou) or None if the case is unusable."""
    aff_p = case_dir / "affine_H.npy"
    ti_p = case_dir / "tile_info.json"
    m_p = case_dir / "metrics.json"
    if not (aff_p.exists() and ti_p.exists() and m_p.exists()):
        return None
    try:
        metrics = json.loads(m_p.read_text())
    except Exception:
        return None
    iou = metrics.get("iou")
    if iou is None or float(iou) < IOU_THRESHOLD:
        return None
    try:
        affine_H = np.load(aff_p)
        tile_info = json.loads(ti_p.read_text())
    except Exception:
        return None
    if affine_H.shape != (2, 3):
        return None
    return affine_H, tile_info, float(iou)


def _render_tile_canvas(tile_info):
    """Re-render the OS Zoomstack tile canvas the matcher saw at match time.

    Uses tools.io.os_tiles.render_tile to assemble the same tile grid that
    fetch_os_opendata_grid produced during benchmark_v20.
    """
    from tools.io.os_tiles import render_tile  # local import — heavy deps

    zoom = tile_info["zoom"]
    tx_min = tile_info["tx_min"]
    ty_min = tile_info["ty_min"]
    nx = tile_info.get("nx", TILE_GRID[0])
    ny = tile_info.get("ny", TILE_GRID[1])
    tile_size = tile_info.get("tile_size", 256)

    canvas = np.zeros((ny * tile_size, nx * tile_size, 3), dtype=np.uint8)
    for j in range(ny):
        for i in range(nx):
            try:
                t = render_tile(zoom, tx_min + i, ty_min + j)
            except Exception:
                continue
            if t is None:
                continue
            canvas[j * tile_size:(j + 1) * tile_size,
                   i * tile_size:(i + 1) * tile_size] = t
    return canvas


def _fold_of_case(case_name: str, fold_map: dict) -> int | None:
    """Look up the fold via fold_assignment.json (handles colon→underscore)."""
    if case_name in fold_map:
        return int(fold_map[case_name])
    canonical = case_name.replace(":", "_").replace("/", "_")
    if canonical in fold_map:
        return int(fold_map[canonical])
    return None


def main():
    if not BENCH_DIR.exists():
        print(f"ERROR: {BENCH_DIR} doesn't exist — need a finished v20 run",
              file=sys.stderr)
        return 1

    fa_p = DATASET_DIR / "fold_assignment.json"
    if not fa_p.exists():
        print(f"ERROR: {fa_p} missing — run scripts/build_dataset.py first",
              file=sys.stderr)
        return 1
    fold_map = json.loads(fa_p.read_text())

    # Skip-existing: don't nuke pairs already built. A previous incarnation
    # of this script wiped the directory on every launch, which made a
    # crashed/killed run lose all progress. Now we incrementally add new pairs.
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in BENCH_DIR.iterdir() if d.is_dir())
    print(f"Scanning {len(case_dirs)} cases for IoU > {IOU_THRESHOLD}...")
    kept = []
    skipped_low_iou = skipped_no_affine = skipped_no_map = skipped_render = 0
    skipped_no_fold = 0
    for i, cd in enumerate(case_dirs, 1):
        payload = _load_case_payload(cd)
        if payload is None:
            # Distinguish reasons for the summary
            m_p = cd / "metrics.json"
            if m_p.exists():
                try:
                    iou = json.loads(m_p.read_text()).get("iou")
                    if iou is not None and float(iou) < IOU_THRESHOLD:
                        skipped_low_iou += 1
                        continue
                except Exception:
                    pass
            skipped_no_affine += 1
            continue
        affine_H, tile_info, iou = payload

        map_p = ANNOT_DIR / cd.name / "map.png"
        if not map_p.exists():
            skipped_no_map += 1
            continue

        fold = _fold_of_case(cd.name, fold_map)
        if fold is None:
            skipped_no_fold += 1
            continue

        # Skip if this case is already fully built — protects against
        # restarting after a crash without losing earlier progress.
        out = PAIRS_DIR / cd.name
        if (out / "map.png").exists() and (out / "tile.png").exists() \
                and (out / "affine.npy").exists() and (out / "meta.json").exists():
            # Still add to the kept list so the manifest gets emitted correctly.
            kept.append({"case": cd.name, "iou": iou, "fold": fold,
                          "split": "val" if fold == 0 else "train"})
            continue

        try:
            tile_canvas = _render_tile_canvas(tile_info)
        except Exception as e:
            print(f"  !! render failed for {cd.name}: {e!s:.80}")
            skipped_render += 1
            continue

        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(map_p, out / "map.png")
        cv2.imwrite(str(out / "tile.png"), tile_canvas)
        np.save(out / "affine.npy", affine_H)
        (out / "meta.json").write_text(json.dumps({
            "case": cd.name,
            "iou": iou,
            "fold": fold,
            "tile_info": tile_info,
            "split": "val" if fold == 0 else "train",
        }, indent=2))
        kept.append({"case": cd.name, "iou": iou, "fold": fold,
                     "split": "val" if fold == 0 else "train"})

        if i % 25 == 0 or i == len(case_dirs):
            print(f"  [{i}/{len(case_dirs)}] kept={len(kept)} "
                  f"low_iou={skipped_low_iou} no_affine={skipped_no_affine} "
                  f"no_map={skipped_no_map} no_fold={skipped_no_fold} "
                  f"render_fail={skipped_render}")

    # Summary
    n_train = sum(1 for k in kept if k["split"] == "train")
    n_val = sum(1 for k in kept if k["split"] == "val")
    print(f"\nDone. Kept {len(kept)} pairs (train={n_train}, val={n_val})")
    print(f"Skipped: {skipped_low_iou} low_iou, "
          f"{skipped_no_affine} no_affine, {skipped_no_map} no_map, "
          f"{skipped_no_fold} no_fold, {skipped_render} render_failed")

    (PAIRS_DIR / "_manifest.json").write_text(json.dumps(kept, indent=2))
    print(f"Manifest: {PAIRS_DIR / '_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
