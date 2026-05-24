"""IoU regression check: full 6-tool kit vs min_1_tool (place only).

Reads metrics.json from both runs for the 11-case cross-1km subset, prints
a side-by-side comparison, and computes Δmean / case-level flips.

Usage:
    uv run python ablations/locate_iou_subset/compare_iou.py \\
        --full-dir   results/benchmark_v_post_refactor/gemini-flash \\
        --min1-dir   results/benchmark_min1_subset/gemini-flash \\
        --cases-file ablations/locate_iou_subset/cases.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_iou(case_dir: Path) -> float | None:
    """Read iou from <case_dir>/metrics.json, or None if missing/unreadable."""
    mp = case_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        with mp.open() as f:
            m = json.load(f)
        v = m.get("iou", None)
        return float(v) if v is not None else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", default="results/benchmark_v_post_refactor/gemini-flash",
                    help="Directory containing the 6-tool-kit metrics.json files.")
    ap.add_argument("--min1-dir", default="results/benchmark_min1_subset/gemini-flash",
                    help="Directory containing the min_1_tool metrics.json files.")
    ap.add_argument("--cases-file", default="ablations/locate_iou_subset/cases.txt",
                    help="Newline-separated list of case folder names.")
    args = ap.parse_args()

    full_root = Path(args.full_dir)
    min1_root = Path(args.min1_dir)
    cases = [
        line.strip() for line in Path(args.cases_file).read_text().splitlines()
        if line.strip()
    ]

    print(f"Comparing {len(cases)} cases")
    print(f"  full kit:   {full_root}")
    print(f"  min_1_tool: {min1_root}")
    print()
    print(f"{'case':<30} {'full_IoU':>10} {'min1_IoU':>10} {'Δ':>8}  status")
    print("-" * 80)

    full_ious = []
    min1_ious = []
    n_lost   = 0   # full had IoU≥0.5, min_1_tool dropped to <0.5
    n_gained = 0   # the other direction (shouldn't happen often, but check)
    n_held   = 0   # both ≥0.5 within ±0.05
    n_both0  = 0   # both already at 0
    for c in cases:
        f_iou = load_iou(full_root / c)
        m_iou = load_iou(min1_root / c)
        delta = (m_iou - f_iou) if (f_iou is not None and m_iou is not None) else None

        # Bucket the change
        if f_iou is None or m_iou is None:
            status = "MISSING"
        elif f_iou < 0.05 and m_iou < 0.05:
            status = "both~0"
            n_both0 += 1
        elif f_iou >= 0.5 and m_iou < 0.5:
            status = "LOST (≥0.5 → <0.5)"
            n_lost += 1
        elif f_iou < 0.5 and m_iou >= 0.5:
            status = "GAINED (<0.5 → ≥0.5)"
            n_gained += 1
        elif abs(delta) <= 0.05:
            status = "held"
            n_held += 1
        elif delta < 0:
            status = f"down {abs(delta):.2f}"
        else:
            status = f"up {delta:.2f}"

        fstr = f"{f_iou:>10.3f}" if f_iou is not None else f"{'NA':>10}"
        mstr = f"{m_iou:>10.3f}" if m_iou is not None else f"{'NA':>10}"
        dstr = f"{delta:>+8.3f}" if delta is not None else f"{'NA':>8}"
        print(f"{c:<30} {fstr} {mstr} {dstr}  {status}")

        if f_iou is not None: full_ious.append(f_iou)
        if m_iou is not None: min1_ious.append(m_iou)

    print()
    n_f = len(full_ious)
    n_m = len(min1_ious)
    print(f"n (full):     {n_f}")
    print(f"n (min1):     {n_m}")
    if n_f and n_m:
        print(f"mean full:    {sum(full_ious)/n_f:.4f}")
        print(f"mean min1:    {sum(min1_ious)/n_m:.4f}")
        # Per-case Δmean is only meaningful over the intersection of cases
        # that have IoU in both runs:
        paired = [
            (load_iou(full_root / c), load_iou(min1_root / c))
            for c in cases
        ]
        paired = [(f, m) for f, m in paired if f is not None and m is not None]
        if paired:
            dm = sum(m - f for f, m in paired) / len(paired)
            print(f"Δmean (paired, n={len(paired)}):  {dm:+.4f}")
    print()
    print(f"LOST   (≥0.5 → <0.5):  {n_lost}")
    print(f"GAINED (<0.5 → ≥0.5):  {n_gained}")
    print(f"held   (|Δ| ≤ 0.05):   {n_held}")
    print(f"both~0:                {n_both0}")


if __name__ == "__main__":
    main()
