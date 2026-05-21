"""Live monitor for an in-progress benchmark run.

Watches the new run's per-case metrics.json files as they appear, compares
each one to the baseline (most recent full run), and reports:
- per-case IoU delta (HIGH-severity regressions are flagged)
- aggregate running mean and how it tracks the baseline's
- cost-per-case via check_credits (must never call paid APIs ourselves)

Usage:
  uv run python scripts/monitor_run.py \\
    --new   results/benchmark_v_this_is_the_MAXIMALLYFINALVERSION/gemini-flash \\
    --base  results/benchmark_critic_new/gemini-flash \\
    --target-cost 0.05 \\
    --regression-threshold 0.10
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def load_summary_iou(summary_path: Path) -> Optional[float]:
    """Read mean IoU from a summary.json. Returns None if not present."""
    if not summary_path.exists():
        return None
    try:
        s = json.loads(summary_path.read_text())
        m = (s.get("metrics") or {}).get("iou") or {}
        v = m.get("mean")
        return float(v) if v is not None else None
    except Exception:
        return None


def load_baseline_ious(base_dir: Path) -> Dict[str, float]:
    """Map case_folder -> baseline IoU (or 0.0 for no-polygon)."""
    out: Dict[str, float] = {}
    for case_dir in base_dir.iterdir():
        if not case_dir.is_dir():
            continue
        mf = case_dir / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
            # honest-mean convention: missing iou == 0.0 (no polygon)
            out[case_dir.name] = float(m.get("iou", 0.0) or 0.0)
        except Exception:
            continue
    return out


def case_iou(metrics_path: Path) -> Optional[float]:
    try:
        m = json.loads(metrics_path.read_text())
    except Exception:
        return None
    if "iou" not in m:
        # Crashed case or no polygon
        return 0.0 if "error" not in m else None
    v = m.get("iou")
    return float(v) if v is not None else 0.0


def run_check_credits(repo_root: Path) -> Dict[str, float]:
    """Run check_credits.py and parse its output."""
    try:
        result = subprocess.run(
            ["python3", str(repo_root / "check_credits.py")],
            capture_output=True, text=True, timeout=15,
            cwd=str(repo_root),
        )
    except Exception as e:
        return {"error": str(e)}
    out = result.stdout
    parsed: Dict[str, float] = {}
    for line in out.splitlines():
        line = line.strip()
        if "Used today:" in line:
            parsed["used_today"] = _to_float(line.split("$")[-1])
        elif "Used this month:" in line:
            parsed["used_month"] = _to_float(line.split("$")[-1])
        elif "Total used:" in line:
            parsed["used_total"] = _to_float(line.split("$")[-1])
        elif "Remaining:" in line and "unlimited" not in line:
            parsed["remaining"] = _to_float(line.split("$")[-1])
    return parsed


def _to_float(s: str) -> float:
    try:
        return float(s.strip().replace(",", ""))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True,
                    help="dir of in-progress run (e.g. results/.../gemini-flash)")
    ap.add_argument("--base", required=True,
                    help="dir of baseline run for comparison")
    ap.add_argument("--target-cost", type=float, default=0.05,
                    help="target $/case (default 0.05)")
    ap.add_argument("--regression-threshold", type=float, default=0.10,
                    help="flag when new_iou is below baseline_iou by >= this (default 0.10)")
    ap.add_argument("--poll-interval", type=int, default=20,
                    help="seconds between filesystem polls (default 20)")
    ap.add_argument("--credit-interval", type=int, default=5,
                    help="poll check_credits every Nth tick (default 5 ticks = ~100s)")
    args = ap.parse_args()

    new_dir = Path(args.new).resolve()
    base_dir = Path(args.base).resolve()
    repo_root = Path(__file__).resolve().parent.parent

    if not base_dir.exists():
        print(f"ERROR: baseline dir not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Monitor starting", flush=True)
    print(f"  new run : {new_dir}", flush=True)
    print(f"  baseline: {base_dir}", flush=True)
    print(f"  regression threshold: |Δ| >= {args.regression_threshold}", flush=True)
    print(f"  target cost: ${args.target_cost:.4f}/case", flush=True)
    print(flush=True)

    baseline = load_baseline_ious(base_dir)
    base_summary = load_summary_iou(base_dir / "summary.json")
    print(f"Baseline: {len(baseline)} cases loaded, "
          f"summary mean IoU = {base_summary:.4f}" if base_summary
          else f"Baseline: {len(baseline)} cases loaded, summary IoU not found",
          flush=True)
    print(flush=True)

    seen: Dict[str, float] = {}
    n_regress = 0
    n_improve = 0
    sum_delta = 0.0
    sum_new_iou = 0.0
    n_baseline_match = 0  # cases also in baseline (for fair running mean)

    # Cost tracking
    initial_credit = run_check_credits(repo_root)
    initial_total = initial_credit.get("used_total")
    if initial_total is not None:
        print(f"Initial credit usage: ${initial_total:.4f} total, "
              f"${initial_credit.get('used_today', 0):.4f} today",
              flush=True)
    print(flush=True)

    tick = 0
    last_credit_total: Optional[float] = initial_total

    while True:
        tick += 1
        # Scan for new metrics.json files
        if new_dir.exists():
            for case_dir in sorted(new_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                name = case_dir.name
                if name in seen:
                    continue
                mf = case_dir / "metrics.json"
                if not mf.exists():
                    continue
                new_iou = case_iou(mf)
                if new_iou is None:
                    print(f"[case {name}] CRASH (no iou in metrics.json)", flush=True)
                    seen[name] = -1.0
                    continue

                base_iou = baseline.get(name)
                if base_iou is None:
                    msg = f"[case {name}] NEW (not in baseline) iou={new_iou:.3f}"
                else:
                    delta = new_iou - base_iou
                    sum_delta += delta
                    sum_new_iou += new_iou
                    n_baseline_match += 1
                    tag = ""
                    if delta <= -args.regression_threshold:
                        tag = " ⚠ REGRESSION"
                        n_regress += 1
                    elif delta >= args.regression_threshold:
                        tag = " ✓ improvement"
                        n_improve += 1
                    msg = (f"[case {name}] iou={new_iou:.3f} "
                           f"(base={base_iou:.3f}, Δ={delta:+.3f}){tag}")
                print(msg, flush=True)
                seen[name] = new_iou

        # Periodic running aggregate
        if n_baseline_match > 0 and tick % 3 == 0:
            mean_new = sum_new_iou / n_baseline_match
            mean_delta = sum_delta / n_baseline_match
            print(f"  -- running over {n_baseline_match} matched cases: "
                  f"mean new IoU={mean_new:.4f}, mean Δ vs baseline={mean_delta:+.4f}, "
                  f"regressions={n_regress}, improvements={n_improve}",
                  flush=True)

        # Periodic cost check
        if tick % args.credit_interval == 0:
            credit = run_check_credits(repo_root)
            total = credit.get("used_total")
            if total is not None and last_credit_total is not None:
                spent = total - (initial_total or total)
                n_done = len(seen)
                per_case = spent / max(1, n_done)
                status = "OK"
                if per_case > args.target_cost * 1.5:
                    status = "⚠ HIGH"
                elif per_case > args.target_cost * 1.1:
                    status = "elevated"
                print(f"  -- cost: ${spent:.4f} spent on {n_done} cases = "
                      f"${per_case:.4f}/case [target ${args.target_cost:.4f}] {status}",
                      flush=True)
                last_credit_total = total

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
