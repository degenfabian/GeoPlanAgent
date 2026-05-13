#!/usr/bin/env python3
"""Snapshot a live benchmark run vs the v17 baseline.

Walks the newest results/benchmark*/<model>/<case>/metrics.json files,
compares each completed case's IoU to the cached v17 IoU for the same
case, and prints:
  - n completed / total cases
  - rolling mean IoU + ≥0.8 count
  - per-case regressions (v17 ≥0.8 → run <0.5)
  - per-case rescues (v17 <0.3 → run ≥0.7)
  - slow cases (>5 min wall-clock)
  - any error cases

Pure read-only, no API calls, safe to run mid-benchmark.
"""
from __future__ import annotations
import json, os, sys, glob
from pathlib import Path
from collections import OrderedDict

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "results" / "benchmark_v17" / "gemini-flash"


def _load_baseline():
    base = {}
    for d in BASELINE.iterdir():
        m = d / "metrics.json"
        if m.exists():
            try:
                base[d.name] = json.loads(m.read_text()).get("iou") or 0.0
            except Exception:
                pass
    return base


def _find_active_run():
    # Newest dir in results/ matching benchmark* that's NOT the v17 baseline
    cands = []
    for p in (REPO / "results").iterdir():
        if not p.is_dir(): continue
        if p.name == "benchmark_v17": continue
        if not p.name.startswith("benchmark"): continue
        # Find newest metrics.json inside
        latest = 0
        for m in p.rglob("metrics.json"):
            latest = max(latest, m.stat().st_mtime)
        cands.append((latest, p))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


def main():
    run_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = Path(run_dir_arg) if run_dir_arg else _find_active_run()
    if not run_dir or not run_dir.exists():
        print("No active run found. Pass the output dir as arg.")
        return

    baseline = _load_baseline()

    # Pick the model subdir (usually one)
    models = [d for d in run_dir.iterdir() if d.is_dir()]
    if not models:
        print(f"{run_dir}: no model subdirs yet")
        return

    for model_dir in models:
        cases = []
        for case_dir in model_dir.iterdir():
            if not case_dir.is_dir(): continue
            m = case_dir / "metrics.json"
            if not m.exists(): continue
            try:
                d = json.loads(m.read_text())
                cases.append((case_dir.name, d))
            except Exception:
                cases.append((case_dir.name, None))

        if not cases:
            print(f"{model_dir.name}: 0 cases complete")
            continue

        cases.sort(key=lambda x: x[0])
        ious = [(d.get("iou") or 0) for _, d in cases if d]
        ge80 = sum(1 for i in ious if i >= 0.8)
        ge50 = sum(1 for i in ious if i >= 0.5)
        mean = sum(ious) / max(1, len(ious))
        errors = [c for c, d in cases if d and not d.get("agent_accepted") and d.get("validation_error")]

        # v17 baseline on the same subset
        run_caseids = set(c for c, _ in cases)
        v17_subset = [baseline[c] for c in run_caseids if c in baseline]
        v17_mean = sum(v17_subset)/max(1,len(v17_subset)) if v17_subset else 0
        v17_ge80 = sum(1 for i in v17_subset if i >= 0.8)

        print(f"\n=== {model_dir.name} ({len(cases)} cases done) ===")
        print(f"Run:   mean={mean:.3f}  ≥0.5={ge50}/{len(cases)}  ≥0.8={ge80}/{len(cases)}")
        print(f"v17 on same subset: mean={v17_mean:.3f}  ≥0.8={v17_ge80}/{len(v17_subset)}")
        print(f"Δ mean: {mean - v17_mean:+.3f}  Δ ≥0.8: {ge80 - v17_ge80:+d}")

        regressions = []
        rescues = []
        slow = []
        for cid, d in cases:
            if d is None: continue
            iou = d.get("iou") or 0
            t = d.get("processing_time") or 0
            v17_iou = baseline.get(cid)
            if v17_iou is None: continue
            if v17_iou >= 0.8 and iou < 0.5:
                regressions.append((cid, v17_iou, iou))
            if v17_iou < 0.3 and iou >= 0.7:
                rescues.append((cid, v17_iou, iou))
            if t > 300:
                slow.append((cid, t, iou))

        if regressions:
            print(f"\nREGRESSIONS ({len(regressions)}): v17≥0.8 → run<0.5")
            for cid, v17, iou in sorted(regressions, key=lambda x: x[2]):
                print(f"  {cid:30s} v17={v17:.2f} → run={iou:.2f}")
        if rescues:
            print(f"\nRESCUES ({len(rescues)}): v17<0.3 → run≥0.7")
            for cid, v17, iou in sorted(rescues, key=lambda x: -x[2]):
                print(f"  {cid:30s} v17={v17:.2f} → run={iou:.2f}")
        if slow:
            print(f"\nSLOW (>5min, top 5):")
            for cid, t, iou in sorted(slow, key=lambda x: -x[1])[:5]:
                print(f"  {cid:30s} t={t/60:.1f}min  iou={iou:.2f}")
        if errors:
            print(f"\nVALIDATION ERRORS: {len(errors)}")


if __name__ == "__main__":
    main()
