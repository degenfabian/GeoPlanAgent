#!/usr/bin/env python3
"""Monitor an in-flight benchmark vs. the reference run.

Prints, on each tick:
  - count of completed cases
  - per-overlap mean IoU vs reference
  - any case >= 0.05 worse than reference (flagged as REGRESSION)

Designed to be run from cron / a wake loop. Exits 0 every time so a
caller can poll.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

CUR = Path("results/benchmark_v_post_refactor/gemini-flash")
REF = Path("results/benchmark_v_this_is_the_MAXIMALLYFINALVERSION/gemini-flash")
THRESHOLD = 0.05
STATE_FILE = Path("results/benchmark_v_post_refactor/_monitor_state.json")


def load_metrics(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not d.exists():
        return out
    for case_dir in d.iterdir():
        m = case_dir / "metrics.json"
        if not m.exists():
            continue
        try:
            out[case_dir.name] = json.loads(m.read_text())
        except Exception:
            pass
    return out


def main() -> int:
    cur = load_metrics(CUR)
    ref = load_metrics(REF)

    n_cur = len(cur)
    if n_cur == 0:
        print(f"[{time.strftime('%H:%M:%S')}] no cases done yet")
        return 0

    # Compare overlap
    shared = sorted(set(cur) & set(ref))
    regressions = []
    improvements = []
    for case in shared:
        ri, ci = ref[case].get("iou"), cur[case].get("iou")
        if ri is None or ci is None:
            continue
        d = ci - ri
        if d <= -THRESHOLD:
            regressions.append((case, ri, ci, d))
        elif d >= THRESHOLD:
            improvements.append((case, ri, ci, d))

    # Headline
    cur_ious = [c.get("iou") for c in cur.values() if c.get("iou") is not None]
    cur_mean = sum(cur_ious) / len(cur_ious) if cur_ious else 0.0
    if shared:
        ref_sub = [ref[c].get("iou") for c in shared if ref[c].get("iou") is not None]
        cur_sub = [cur[c].get("iou") for c in shared if cur[c].get("iou") is not None]
        ref_sub_mean = sum(ref_sub) / len(ref_sub) if ref_sub else 0.0
        cur_sub_mean = sum(cur_sub) / len(cur_sub) if cur_sub else 0.0
        delta = cur_sub_mean - ref_sub_mean
    else:
        ref_sub_mean = cur_sub_mean = delta = 0.0

    # Diff vs prior tick — only print new regressions/improvements
    prior_regr_cases = set()
    prior_impr_cases = set()
    prior_n_cur = 0
    if STATE_FILE.exists():
        try:
            st = json.loads(STATE_FILE.read_text())
            prior_regr_cases = set(st.get("regressed_cases") or [])
            prior_impr_cases = set(st.get("improved_cases") or [])
            prior_n_cur = int(st.get("n_cur") or 0)
        except Exception:
            pass

    # Milestone bands: 25, 50, 75, 100, 125, 150, 175, 200
    # Fires when n_cur crosses one of these for the first time this run.
    milestone = None
    for m in (25, 50, 75, 100, 125, 150, 175, 200):
        if prior_n_cur < m <= n_cur:
            milestone = m
            break

    current_regr_cases = {r[0] for r in regressions}
    current_impr_cases = {r[0] for r in improvements}
    new_regr = [r for r in regressions if r[0] not in prior_regr_cases]
    new_impr = [r for r in improvements if r[0] not in prior_impr_cases]

    # Output
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] cases={n_cur}/209  cur_mean={cur_mean:.3f}  "
          f"overlap={len(shared)}  ref_subset={ref_sub_mean:.3f}  "
          f"delta={delta:+.3f}  "
          f"regressions={len(regressions)}  improvements={len(improvements)}")

    if new_regr:
        print(f"  🚨 NEW REGRESSIONS ({len(new_regr)}):")
        for case, ri, ci, d in sorted(new_regr, key=lambda r: r[3]):
            print(f"    {case:<40} {ri:.3f} -> {ci:.3f}  ({d:+.3f})")

    if new_impr:
        print(f"  ✓ new improvements ({len(new_impr)}):")
        for case, ri, ci, d in sorted(new_impr, key=lambda r: -r[3])[:5]:
            print(f"    {case:<40} {ri:.3f} -> {ci:.3f}  ({d:+.3f})")

    if milestone is not None:
        print(f"  📍 MILESTONE {milestone}/209 — cur_mean={cur_mean:.3f}, "
              f"overlap delta={delta:+.3f}, regressions={len(regressions)}, "
              f"improvements={len(improvements)}")

    # Save state for next tick
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "ts": ts,
        "n_cur": n_cur,
        "regressed_cases": sorted(current_regr_cases),
        "improved_cases": sorted(current_impr_cases),
        "cur_mean": cur_mean,
        "delta_vs_ref_on_overlap": delta,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
