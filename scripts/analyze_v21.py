"""Deep analysis of benchmark_v21 — produces tables/numbers for the paper.

Outputs to stdout (paste into paper) + a JSON dump at
results/benchmark_v21/analysis.json.

Sections:
  1. Aggregate stats (n, mean, median, thresholds)
  2. Comparison to benchmark_v20 (per-case, big wins/losses)
  3. Critic ablation (pre-critic IoU vs final IoU; per-action effectiveness)
  4. Critic action distribution
  5. Failure mode breakdown
  6. Per-case CSV for paper appendix
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict
import csv

import numpy as np

V21 = Path("results/benchmark_v21/gemini-flash")
V20 = Path("results/benchmark_v20/gemini-flash")
OUT_JSON = Path("results/benchmark_v21/analysis.json")
OUT_CSV  = Path("results/benchmark_v21/analysis_per_case.csv")


def safe_load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def iou(m):
    if not m or m.get("error"):
        return None
    v = m.get("iou")
    return float(v) if isinstance(v, (int, float)) else None


def load_v21_cases():
    """Per-case dict: case_name -> {metrics, critic_log, trace, error}."""
    cases = {}
    for d in sorted(V21.iterdir()):
        if not d.is_dir():
            continue
        m = safe_load(d / "metrics.json")
        cl = safe_load(d / "critic_log.json")
        tr = safe_load(d / "critic_debug" / "trace.json")
        cases[d.name] = {
            "metrics": m,
            "critic_log": cl,
            "trace": tr,
            "is_error": bool(m and m.get("error")),
            "error_msg": m.get("error") if m else None,
            "iou": iou(m),
            "acc": (m or {}).get("agent_accepted"),
            "pre_critic_iou": (tr or {}).get("pre_critic_iou_vs_gt"),
            "iou_delta": (tr or {}).get("iou_delta_from_critic"),
            "critic_decisions": [it.get("decision") for it in (cl or {}).get("iterations", [])],
            "critic_diagnoses": [it.get("diagnosis", "")[:200] for it in (cl or {}).get("iterations", [])],
        }
    return cases


def load_v20_cases():
    cases = {}
    for d in sorted(V20.iterdir()):
        if not d.is_dir():
            continue
        m = safe_load(d / "metrics.json")
        cases[d.name] = iou(m)
    return cases


def aggregate_stats(cases):
    ious = [c["iou"] for c in cases.values() if c["iou"] is not None]
    errs = sum(1 for c in cases.values() if c["is_error"])
    a = np.array(ious)
    return {
        "n_total_cases": len(cases),
        "n_with_iou": len(ious),
        "n_errors": errs,
        "mean_iou": float(a.mean()),
        "median_iou": float(np.median(a)),
        "ge_0.9_count": int((a >= 0.9).sum()),
        "ge_0.7_count": int((a >= 0.7).sum()),
        "ge_0.5_count": int((a >= 0.5).sum()),
        "ge_0.3_count": int((a >= 0.3).sum()),
        "lt_0.05_count": int((a < 0.05).sum()),
        "ge_0.9_pct": float((a >= 0.9).mean()),
        "ge_0.7_pct": float((a >= 0.7).mean()),
        "ge_0.5_pct": float((a >= 0.5).mean()),
        "lt_0.05_pct": float((a < 0.05).mean()),
    }


def comparison_to_v20(v21, v20):
    """Compare v21 vs v20 on the cases that exist in both."""
    common = sorted(set(v21) & set(v20))
    v21_iou = []
    v20_iou = []
    pairs = []
    for c in common:
        i21 = v21[c]["iou"]
        i20 = v20[c]
        if i21 is None or i20 is None:
            continue
        pairs.append((c, i21, i20))
        v21_iou.append(i21)
        v20_iou.append(i20)
    a21 = np.array(v21_iou); a20 = np.array(v20_iou)
    delta = a21 - a20
    big_wins = sorted(
        [(c, i21, i20) for c, i21, i20 in pairs if i21 - i20 > 0.10],
        key=lambda x: -(x[1] - x[2])
    )
    big_losses = sorted(
        [(c, i21, i20) for c, i21, i20 in pairs if i21 - i20 < -0.10],
        key=lambda x: (x[1] - x[2])
    )
    catastrophic_regr = [(c, i21, i20) for c, i21, i20 in pairs
                         if i20 >= 0.7 and i21 < 0.1]
    rescues = [(c, i21, i20) for c, i21, i20 in pairs
                if i20 < 0.1 and i21 >= 0.5]
    return {
        "n_common": len(pairs),
        "v20_mean": float(a20.mean()),
        "v21_mean": float(a21.mean()),
        "mean_delta": float(delta.mean()),
        "n_big_wins": len(big_wins),
        "n_big_losses": len(big_losses),
        "n_catastrophic_regressions": len(catastrophic_regr),
        "n_rescues": len(rescues),
        "top_8_wins": [(c, round(i21, 4), round(i20, 4)) for c, i21, i20 in big_wins[:8]],
        "top_8_losses": [(c, round(i21, 4), round(i20, 4)) for c, i21, i20 in big_losses[:8]],
        "catastrophic_regressions": [(c, round(i21, 4), round(i20, 4)) for c, i21, i20 in catastrophic_regr],
        "rescues_list": [(c, round(i21, 4), round(i20, 4)) for c, i21, i20 in rescues],
    }


def critic_ablation(cases):
    """Compare pre-critic IoU vs final IoU.

    The pre-critic IoU is what the worker would have submitted without critic.
    Comparing this to the final IoU tells us if the critic helped overall.
    """
    pairs = []
    for c, info in cases.items():
        if info["pre_critic_iou"] is None or info["iou"] is None:
            continue
        pairs.append((c, info["pre_critic_iou"], info["iou"]))
    if not pairs:
        return {"n": 0}
    pre = np.array([p[1] for p in pairs])
    final = np.array([p[2] for p in pairs])
    delta = final - pre

    # Count where critic changed something
    changed = (np.abs(delta) > 0.001).sum()
    critic_wins = (delta > 0.05).sum()        # final much better than pre
    critic_losses = (delta < -0.05).sum()      # critic ruined a good output
    big_critic_wins = sorted(
        [(c, pre_i, fin_i) for c, pre_i, fin_i in pairs if fin_i - pre_i > 0.10],
        key=lambda x: -(x[2] - x[1])
    )
    big_critic_losses = sorted(
        [(c, pre_i, fin_i) for c, pre_i, fin_i in pairs if fin_i - pre_i < -0.10],
        key=lambda x: (x[2] - x[1])
    )
    return {
        "n": len(pairs),
        "pre_critic_mean_iou": float(pre.mean()),
        "final_mean_iou": float(final.mean()),
        "mean_delta": float(delta.mean()),
        "n_critic_changed": int(changed),
        "n_critic_wins (delta>0.05)": int(critic_wins),
        "n_critic_losses (delta<-0.05)": int(critic_losses),
        "top_5_critic_wins": [(c, round(p, 4), round(f, 4)) for c, p, f in big_critic_wins[:5]],
        "top_5_critic_losses": [(c, round(p, 4), round(f, 4)) for c, p, f in big_critic_losses[:5]],
    }


def critic_action_breakdown(cases):
    """Distribution of critic actions + their effectiveness."""
    action_counts = Counter()
    action_deltas = defaultdict(list)
    n_iters_distribution = Counter()
    for c, info in cases.items():
        decisions = info["critic_decisions"]
        n_iters_distribution[len(decisions)] += 1
        if not decisions:
            continue
        for d in decisions:
            action_counts[d] += 1
        if info["iou_delta"] is not None:
            for d in decisions:
                action_deltas[d].append(info["iou_delta"])
    by_action = {}
    for action, deltas in action_deltas.items():
        a = np.array(deltas)
        by_action[action] = {
            "count": len(a),
            "mean_delta": float(a.mean()),
            "n_helped (delta>0.05)": int((a > 0.05).sum()),
            "n_hurt (delta<-0.05)": int((a < -0.05).sum()),
            "n_no_change (|delta|<=0.05)": int((np.abs(a) <= 0.05).sum()),
        }
    return {
        "action_total_counts": dict(action_counts),
        "n_iterations_distribution": dict(n_iters_distribution),
        "by_action": by_action,
    }


def failure_modes(cases):
    """Categorise the hard-fail cases."""
    hard_fails = [(c, info) for c, info in cases.items()
                   if info["iou"] is not None and info["iou"] < 0.05]
    district_lookup_hard_fails = sum(1 for _, info in hard_fails
                                       if info["acc"] is False)
    api_errors = sum(1 for c, info in cases.items() if info["is_error"])
    moderate_fails = [(c, info) for c, info in cases.items()
                       if info["iou"] is not None and 0.05 <= info["iou"] < 0.5]
    return {
        "n_hard_fails (<0.05)": len(hard_fails),
        "n_hard_fail_district_path": district_lookup_hard_fails,
        "n_api_errors": api_errors,
        "n_moderate_fails (0.05-0.5)": len(moderate_fails),
        "hard_fail_examples": [
            {"case": c, "iou": round(info["iou"] or 0, 4), "acc": info["acc"],
             "critic_decisions": info["critic_decisions"],
             "error": info["error_msg"]}
            for c, info in hard_fails[:15]
        ],
    }


def write_per_case_csv(cases, v20, out_path):
    rows = []
    for c, info in cases.items():
        rows.append({
            "case": c,
            "v20_iou": v20.get(c),
            "v21_iou": info["iou"],
            "pre_critic_iou": info["pre_critic_iou"],
            "critic_actions": "|".join(info["critic_decisions"]),
            "n_critic_iters": len(info["critic_decisions"]),
            "agent_accepted": info["acc"],
            "error": info["error_msg"],
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print("Loading v21 cases...")
    v21 = load_v21_cases()
    print(f"  {len(v21)} v21 cases loaded")
    print("Loading v20 cases...")
    v20 = load_v20_cases()
    print(f"  {len(v20)} v20 cases loaded")
    print()

    agg = aggregate_stats(v21)
    cmp = comparison_to_v20(v21, v20)
    crit_abl = critic_ablation(v21)
    crit_act = critic_action_breakdown(v21)
    fail_modes = failure_modes(v21)

    write_per_case_csv(v21, v20, OUT_CSV)

    result = {
        "v21_aggregate": agg,
        "comparison_to_v20": cmp,
        "critic_ablation": crit_abl,
        "critic_action_breakdown": crit_act,
        "failure_modes": fail_modes,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    # Print summary
    def section(title):
        print(f"\n{'='*72}\n{title}\n{'='*72}")

    section("1. v21 AGGREGATE STATS")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:24s}  {v:.4f}")
        else:
            print(f"  {k:24s}  {v}")

    section("2. v21 vs v20 (common cases)")
    for k in ["n_common", "v20_mean", "v21_mean", "mean_delta",
              "n_big_wins", "n_big_losses",
              "n_catastrophic_regressions", "n_rescues"]:
        v = cmp[k]
        print(f"  {k:32s}  {v}")
    print("\n  Top wins (v20 → v21):")
    for c, i21, i20 in cmp["top_8_wins"]:
        print(f"    {c:40s}  {i20:.3f} → {i21:.3f}  ({i21-i20:+.3f})")
    print("\n  Top losses (v20 → v21):")
    for c, i21, i20 in cmp["top_8_losses"]:
        print(f"    {c:40s}  {i20:.3f} → {i21:.3f}  ({i21-i20:+.3f})")
    if cmp["catastrophic_regressions"]:
        print(f"\n  Catastrophic regressions (v20 ≥ 0.7, v21 < 0.1):")
        for c, i21, i20 in cmp["catastrophic_regressions"]:
            print(f"    {c:40s}  {i20:.3f} → {i21:.3f}")

    section("3. CRITIC ABLATION (pre-critic IoU vs final IoU)")
    for k, v in crit_abl.items():
        if k.startswith("top_"): continue
        if isinstance(v, float):
            print(f"  {k:30s}  {v:.4f}")
        else:
            print(f"  {k:30s}  {v}")
    if crit_abl.get("top_5_critic_wins"):
        print("\n  Top critic wins:")
        for c, p, f in crit_abl["top_5_critic_wins"]:
            print(f"    {c:40s}  pre={p:.3f} → final={f:.3f}  ({f-p:+.3f})")
    if crit_abl.get("top_5_critic_losses"):
        print("\n  Top critic losses:")
        for c, p, f in crit_abl["top_5_critic_losses"]:
            print(f"    {c:40s}  pre={p:.3f} → final={f:.3f}  ({f-p:+.3f})")

    section("4. CRITIC ACTION DISTRIBUTION")
    print("  Action counts:")
    for a, c in sorted(crit_act["action_total_counts"].items(),
                        key=lambda x: -x[1]):
        print(f"    {a:30s}  {c}")
    print(f"\n  N iterations per case:")
    for n, c in sorted(crit_act["n_iterations_distribution"].items()):
        print(f"    {n} iters: {c} cases")
    print("\n  Effectiveness per action (delta = final - pre-critic IoU):")
    for action, stats in crit_act["by_action"].items():
        print(f"    {action}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"      {k:30s}  {v:.4f}")
            else:
                print(f"      {k:30s}  {v}")

    section("5. FAILURE MODES")
    for k, v in fail_modes.items():
        if k == "hard_fail_examples": continue
        print(f"  {k:35s}  {v}")
    print("\n  Sample hard-fail cases (iou < 0.05):")
    for ex in fail_modes["hard_fail_examples"][:10]:
        print(f"    {ex['case']:40s}  iou={ex['iou']:.4f}  acc={ex['acc']}  actions={ex['critic_decisions']}")

    print(f"\n\nFull JSON: {OUT_JSON}")
    print(f"Per-case CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
