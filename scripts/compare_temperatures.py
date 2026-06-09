"""Paired temperature ablation: compare IoU at T=0 vs T=1 on the same cases.

Reads metrics.json from two run directories, joins by case name, prints a
paired summary + Wilcoxon p-value suitable for an appendix temperature table.

Usage:
    uv run python scripts/compare_temperatures.py \\
        results/cost_audit_v1/gemini-flash \\
        results/temperature_ablation_t1/gemini-flash
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def load_ious(d: Path) -> dict[str, float]:
    out = {}
    for c in sorted(d.iterdir()):
        if not c.is_dir():
            continue
        m = c / "metrics.json"
        if not m.exists():
            continue
        iou = json.loads(m.read_text()).get("iou")
        if iou is not None:
            out[c.name] = float(iou)
    return out


def wilcoxon(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value (uses scipy if available)."""
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 6:
        return None
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(nonzero).pvalue)
    except ImportError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_t0", type=Path, help="e.g. results/cost_audit_v1/gemini-flash")
    ap.add_argument("dir_t1", type=Path, help="e.g. results/temperature_ablation_t1/gemini-flash")
    args = ap.parse_args()

    t0 = load_ious(args.dir_t0)
    t1 = load_ious(args.dir_t1)
    paired = sorted(set(t0) & set(t1))
    if not paired:
        print("No overlapping cases between the two runs.", file=sys.stderr)
        return 1

    print(f"Paired cases: {len(paired)}  (T=0 dir has {len(t0)}, T=1 has {len(t1)})")
    if len(t0) > len(paired):
        print(f"  T=0 only: {sorted(set(t0) - set(paired))[:5]}…")
    if len(t1) > len(paired):
        print(f"  T=1 only: {sorted(set(t1) - set(paired))[:5]}…")

    v0 = [t0[c] for c in paired]
    v1 = [t1[c] for c in paired]
    diffs = [b - a for a, b in zip(v0, v1)]

    print(f"\n┌─ Temperature ablation, paired by case ─────────────────────┐")
    print(f"│ {'Stat':<22s} {'T=0':>11s} {'T=1':>11s} {'Δ (T1-T0)':>11s} │")
    print(f"│ {'mean IoU':<22s} {st.mean(v0):>11.4f} {st.mean(v1):>11.4f} {st.mean(diffs):>+11.4f} │")
    print(f"│ {'median IoU':<22s} {st.median(v0):>11.4f} {st.median(v1):>11.4f} {st.median(diffs):>+11.4f} │")
    print(f"│ {'≥0.5 fraction':<22s} "
          f"{sum(1 for x in v0 if x >= 0.5)/len(v0)*100:>10.1f}% "
          f"{sum(1 for x in v1 if x >= 0.5)/len(v1)*100:>10.1f}% "
          f"{(sum(1 for x in v1 if x >= 0.5) - sum(1 for x in v0 if x >= 0.5))/len(v0)*100:>+10.1f}% │")
    print(f"│ {'≥0.8 fraction':<22s} "
          f"{sum(1 for x in v0 if x >= 0.8)/len(v0)*100:>10.1f}% "
          f"{sum(1 for x in v1 if x >= 0.8)/len(v1)*100:>10.1f}% "
          f"{(sum(1 for x in v1 if x >= 0.8) - sum(1 for x in v0 if x >= 0.8))/len(v0)*100:>+10.1f}% │")
    print(f"│ {'max |Δ| per case':<22s}      {'-':>5s} {'-':>11s} {max(abs(d) for d in diffs):>+11.4f} │")
    print(f"│ {'mean |Δ|':<22s}      {'-':>5s} {'-':>11s} {st.mean(abs(d) for d in diffs):>+11.4f} │")

    p = wilcoxon(diffs)
    if p is not None:
        print(f"│ {'Wilcoxon p (two-sided)':<22s}      {'-':>5s} {'-':>11s} {p:>+11.4f} │")
    else:
        print(f"│ {'Wilcoxon p':<22s}      {'-':>5s} {'-':>11s} {'(scipy needed)':>11s} │")
    print("└────────────────────────────────────────────────────────────┘")

    # Print top per-case shifts so a glance tells you whether change is
    # spread or concentrated on one or two cases
    shifts = sorted(zip(paired, v0, v1, diffs), key=lambda r: abs(r[3]), reverse=True)
    print(f"\nLargest per-case shifts (case, T=0, T=1, Δ):")
    for c, a, b, d in shifts[:8]:
        print(f"  {c:<40s} {a:.4f} → {b:.4f}   Δ={d:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
