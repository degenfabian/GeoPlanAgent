"""Live monitor for the lucky-shot ablation run.

Watches results/ablation_lucky_shot/gemini-flash for new metrics.json
files. For each new case:
  - prints IoU + delta vs MAX (the with-all-features cached run)
  - if delta <= -0.10, runs an investigation (which removal likely
    caused it; comparison to MAX's affine/pdf_info to diagnose).

Usage:
  uv run python scripts/monitor_lucky_shot.py
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

# Make `tools.*` importable when running this script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


NEW_DIR = Path("results/ablation_lucky_shot/gemini-flash")
MAX_DIR = Path("results/benchmark_v_this_is_the_MAXIMALLYFINALVERSION/gemini-flash")
EVAL_DIR = Path("evaluation_data")

REGRESSION_THRESHOLD = 0.10   # |Δ| ≥ this triggers investigation
POLL_INTERVAL = 25            # seconds


def iou_of(case_dir: Path) -> Optional[float]:
    mf = case_dir / "metrics.json"
    if not mf.exists():
        return None
    try:
        m = json.loads(mf.read_text())
        if "error" in m and "iou" not in m:
            return None
        return float(m.get("iou", 0.0) or 0.0)
    except Exception:
        return None


def load_centroid_latlon(geojson_path: Path) -> Optional[tuple]:
    from tools.geo.geojson import centroid_latlon
    return centroid_latlon(geojson_path)


def haversine_km(p1, p2):
    if not p1 or not p2:
        return None
    from tools.geo.coords import haversine_km as _hk
    return _hk(p1[0], p1[1], p2[0], p2[1])


def load_pdf_info(case_dir: Path) -> dict:
    p = case_dir / "pdf_info.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def investigate(case_name: str, new_iou: float, max_iou: float):
    """Diagnose a regression. Print hypothesis based on geometry +
    pdf_info comparison."""
    print(f"\n  === INVESTIGATING {case_name} (Δ = {new_iou - max_iou:+.3f}) ===")
    new_case = NEW_DIR / case_name
    max_case = MAX_DIR / case_name
    eval_case = EVAL_DIR / case_name

    # Centroid distances vs GT
    gt_files = list(eval_case.glob("*.geojson")) if eval_case.exists() else []
    gt_c = load_centroid_latlon(gt_files[0]) if gt_files else None
    new_c = load_centroid_latlon(new_case / "predicted.geojson")
    max_c = load_centroid_latlon(max_case / "predicted.geojson")
    if gt_c:
        new_d = haversine_km(gt_c, new_c) if new_c else None
        max_d = haversine_km(gt_c, max_c) if max_c else None
        nd = f"{new_d:.2f} km" if new_d is not None else "no pred"
        md = f"{max_d:.2f} km" if max_d is not None else "no pred"
        print(f"  GT centroid distance:  new={nd}  vs MAX={md}")

        # Categorise the failure
        if new_d is not None:
            if new_d > 50:
                category = "WRONG REGION (>50 km off — likely letterhead postcode without OCR, OR catastrophic locate)"
            elif new_d > 5:
                category = "WRONG TOWN (>5 km — likely homonym road / wrong-LA pick that road-name verifier would have caught)"
            elif new_d > 0.5:
                category = "WRONG NEIGHBOURHOOD (>500 m — could be quadrant-coverage or distance penalty)"
            else:
                category = "LOCAL ERROR (≤500 m — likely 6-DOF or Delaunay loss on shape/scale)"
            print(f"  → {category}")

    # Affine comparison
    new_H_path = new_case / "affine_H.npy"
    max_H_path = max_case / "affine_H.npy"
    if new_H_path.exists() and max_H_path.exists():
        try:
            Hn = np.load(new_H_path); Hm = np.load(max_H_path)
            def aspect_shear(H):
                a, b = H[0,0], H[0,1]; c, d = H[1,0], H[1,1]
                sx = math.sqrt(a*a + c*c); sy = math.sqrt(b*b + d*d)
                asp = min(sx,sy)/max(sx,sy) if min(sx,sy) > 0 else 0
                shear = abs(b + c)
                return sx, sy, asp, shear
            sxn, syn, aspn, shn = aspect_shear(Hn)
            sxm, sym, aspm, shm = aspect_shear(Hm)
            print(f"  Affine new:  sx={sxn:.3f} sy={syn:.3f} aspect={aspn:.3f} shear={shn:.3f}")
            print(f"  Affine MAX:  sx={sxm:.3f} sy={sym:.3f} aspect={aspm:.3f} shear={shm:.3f}")
            if (aspm < 0.98 or shm > 0.02) and (aspn >= 0.98 and shn <= 0.02):
                print(f"  → MAX committed a 6-DOF affine; new is forced 4-DOF. 6-DOF RIP IS THE CAUSE.")
        except Exception as e:
            print(f"  affine compare failed: {e}")

    # match_info comparison
    try:
        new_m = json.loads((new_case / "metrics.json").read_text())
        max_m = json.loads((max_case / "metrics.json").read_text())
        new_mi = new_m.get("match_info") or {}
        max_mi = max_m.get("match_info") or {}
        print(f"  n_inliers:  new={new_mi.get('n_inliers','?')}  MAX={max_mi.get('n_inliers','?')}")
        print(f"  score:      new={new_mi.get('score','?')}  MAX={max_mi.get('score','?')}")
    except Exception:
        pass

    # pdf_info comparison — did the reader get the same signals?
    new_pi = load_pdf_info(new_case)
    max_pi = load_pdf_info(max_case)
    new_pcs = set(p.upper().replace(' ','') for p in (new_pi.get("postcodes") or []) if p)
    max_pcs = set(p.upper().replace(' ','') for p in (max_pi.get("postcodes") or []) if p)
    pc_diff = max_pcs - new_pcs
    if pc_diff:
        print(f"  Postcodes MAX had but new DIDN'T: {sorted(list(pc_diff))[:5]}"
              f"{'  ← OCR RIP likely the cause' if pc_diff else ''}")
    elif new_pcs != max_pcs:
        diff = new_pcs - max_pcs
        if diff:
            print(f"  Postcodes new found but MAX didn't: {sorted(list(diff))[:5]}")

    new_grs = (new_pi.get("grid_refs") or [])
    max_grs = (max_pi.get("grid_refs") or [])
    if len(new_grs) != len(max_grs):
        print(f"  grid_refs: new={len(new_grs)}  MAX={len(max_grs)}")

    print()


def main():
    print(f"Lucky-shot monitor starting")
    print(f"  new run : {NEW_DIR}")
    print(f"  baseline: {MAX_DIR}")
    print(f"  regression threshold |Δ| ≥ {REGRESSION_THRESHOLD}")
    print()

    seen: dict[str, float] = {}
    n_done = 0; n_regressions = 0; n_improvements = 0
    sum_delta = 0.0; n_compared = 0

    while True:
        if NEW_DIR.exists():
            for case_dir in sorted(NEW_DIR.iterdir()):
                if not case_dir.is_dir():
                    continue
                name = case_dir.name
                if name in seen:
                    continue
                new_iou = iou_of(case_dir)
                if new_iou is None:
                    continue
                max_iou = iou_of(MAX_DIR / name)
                seen[name] = new_iou
                n_done += 1
                if max_iou is None:
                    print(f"[{n_done}] {name}: iou={new_iou:.3f}  (no MAX baseline)")
                    continue
                d = new_iou - max_iou
                sum_delta += d
                n_compared += 1
                tag = ""
                if d <= -REGRESSION_THRESHOLD:
                    tag = "  ⚠ REGRESSION"
                    n_regressions += 1
                elif d >= REGRESSION_THRESHOLD:
                    tag = "  ✓ improvement"
                    n_improvements += 1
                print(f"[{n_done}] {name}: new={new_iou:.3f}  MAX={max_iou:.3f}  Δ={d:+.3f}{tag}")
                if tag.startswith("  ⚠"):
                    investigate(name, new_iou, max_iou)

        if n_compared > 0 and n_done % 5 == 0 and n_done > 0:
            mean_d = sum_delta / n_compared
            print(f"  -- running over {n_compared} matched cases: "
                  f"mean Δ={mean_d:+.4f}  regr={n_regressions}  impr={n_improvements}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
