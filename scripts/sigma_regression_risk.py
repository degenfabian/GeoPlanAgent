"""Regression-risk analysis for the σ contract fix.

If we respect the locate sub-agent's σ in sliding_window_position
(tightening MINIMA's search window from the current 5000m default),
which v3 cases could regress?

Specifically: cases where pick_err > pick.sigma_m (the sub-agent was
OVER-confident — picked a tight σ but landed further from GT than σ
allows). Under current behavior these get a 5km MINIMA search that
might rescue them; under the fix they'd only get a σ-sized search.

We classify each over-confident case by:
  - Did MINIMA's 5km search actually help? (final match landed close
    to GT despite tight σ)
  - Final IoU > 0.5 today?
If both YES, the case is at risk of regression under the σ fix.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CSV_PATH = REPO / "results" / "v3_sigma_signal.csv"


def main():
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            row = {k: v for k, v in r.items()}
            # cast numerics
            for k in ("iou", "sigma_m", "pick_err_km", "match_err_km", "pred_err_km"):
                row[k] = float(row[k]) if row[k] not in ("", None) else None
            row["specificity"] = int(row["specificity"]) if row["specificity"] else None
            rows.append(row)

    print(f"Loaded {len(rows)} cases.\n")

    # 1. Cases where the sub-agent was over-confident: pick_err > σ
    over_conf = []
    under_conf = []
    well_calib = []
    for r in rows:
        if r["pick_err_km"] is None or r["sigma_m"] is None:
            continue
        err_m = r["pick_err_km"] * 1000
        sig = r["sigma_m"]
        if err_m > 1.5 * sig:
            over_conf.append(r)
        elif err_m < 0.5 * sig:
            under_conf.append(r)
        else:
            well_calib.append(r)

    print(f"σ calibration buckets (relative to pick_err):")
    print(f"  over-confident (pick_err > 1.5 × σ): {len(over_conf)}")
    print(f"  well-calibrated (0.5σ ≤ pick_err ≤ 1.5σ): {len(well_calib)}")
    print(f"  under-confident (pick_err < 0.5 × σ): {len(under_conf)}\n")

    # 2. Among over-confident cases, where pick_err exceeds the current
    # 5km MINIMA search radius — those are losing today already, σ fix
    # can't make them worse.
    too_far_for_5km = [r for r in over_conf if r["pick_err_km"] * 1000 > 5000]
    in_5km_band = [r for r in over_conf if r["pick_err_km"] * 1000 <= 5000]
    print(f"Over-confident cases breakdown:")
    print(f"  pick_err > 5km (already lost under current 5km default): {len(too_far_for_5km)}")
    print(f"    of which IoU > 0.5: {sum(1 for r in too_far_for_5km if r['iou'] > 0.5)}")
    print(f"  pick_err ≤ 5km (rescuable by 5km default):                {len(in_5km_band)}")
    print(f"    of which IoU > 0.5: {sum(1 for r in in_5km_band if r['iou'] > 0.5)}")
    print(f"    of which IoU > 0.5 AND pick_err > σ: {sum(1 for r in in_5km_band if r['iou'] > 0.5)}\n")

    # 3. The actual regression-risk subset:
    # pick_err is between σ and 5km — currently rescuable, would no
    # longer be after tightening.
    at_risk = [r for r in in_5km_band if r["iou"] > 0.5
                and r["pick_err_km"] * 1000 > r["sigma_m"]]
    print(f"=" * 78)
    print(f"REGRESSION RISK: cases that win today (IoU>0.5) but pick_err > σ")
    print(f"and pick_err ≤ 5km (so the 5km default is what saved them).")
    print(f"")
    print(f"Count: {len(at_risk)} / {len(rows)} ({100*len(at_risk)/len(rows):.1f}%)")
    print(f"=" * 78)
    if at_risk:
        # Show top by pick_err / σ ratio
        ranked = sorted(at_risk, key=lambda r: -(r["pick_err_km"] * 1000 / r["sigma_m"]))
        print(f"\n{'case':42s} {'σ_m':>6s}  {'pick_err':>9s}  {'ratio':>6s}  {'IoU':>5s}  source")
        for r in ranked[:15]:
            ratio = r["pick_err_km"] * 1000 / r["sigma_m"]
            print(f"{r['case']:42s} {r['sigma_m']:>5.0f}m  "
                  f"{r['pick_err_km']:>7.2f}km  {ratio:>6.1f}×  "
                  f"{r['iou']:>5.3f}  {r['source'][:50]}")
        # Summary
        med_iou = sorted(r["iou"] for r in at_risk)[len(at_risk)//2]
        mean_iou = sum(r["iou"] for r in at_risk) / len(at_risk)
        total_iou = sum(r["iou"] for r in at_risk)
        print(f"\nAt-risk subset stats:")
        print(f"  mean IoU = {mean_iou:.3f}")
        print(f"  median IoU = {med_iou:.3f}")
        print(f"  total IoU contribution to benchmark = {total_iou:.2f}")
        # Worst case: assume ALL these regress to IoU=0
        full_mean_iou_now = sum(r["iou"] for r in rows) / len(rows)
        new_mean = (sum(r["iou"] for r in rows) - total_iou) / len(rows)
        delta = new_mean - full_mean_iou_now
        print(f"  if ALL at-risk regress to IoU=0: mean benchmark IoU "
              f"{full_mean_iou_now:.3f} → {new_mean:.3f} "
              f"(Δ {delta:+.3f})")

    # 4. Sources/source-types in at-risk vs all rows
    print(f"\nSource-prefix counts:")
    print(f"  ALL cases: {Counter([r['source'].split(':',2)[1] if ':' in r['source'] else '?' for r in rows]).most_common()}")
    if at_risk:
        print(f"  AT-RISK:   {Counter([r['source'].split(':',2)[1] if ':' in r['source'] else '?' for r in at_risk]).most_common()}")


if __name__ == "__main__":
    main()
