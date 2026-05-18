"""Live monitor for results/benchmark_v_R16/. Prints compact status every
60s, full summary every 10 polls, and immediate alerts on failures/anomalies.
Baseline for comparison: benchmark_v_postrot (0.682 production-honest mean
IoU, 132/205 = 64.4% at IoU≥0.8).
"""
from __future__ import annotations
import json, glob, time, os, sys, statistics, collections

RUN = "results/benchmark_v_R16/gemini-flash"
TARGET = 208
BASELINE_MEAN = 0.682
BASELINE_PCT_GE_08 = 64.4
POLL_S = 60
FULL_EVERY = 10  # full summary every Nth poll (so every 10 min)

seen_cases: set = set()
seen_failures: set = set()
seen_weak_retries = {"fired": 0, "kept": 0, "deltas": []}

def scan() -> dict:
    ious, failures, weak = [], [], {"fired": 0, "kept": 0, "deltas": []}
    new_cases, new_failures = [], []
    cases = sorted(glob.glob(f"{RUN}/*/metrics.json"))
    for m in cases:
        case = m.split("/")[-2]
        try:
            d = json.loads(open(m).read())
        except Exception:
            continue
        iou = d.get("iou")
        reason = (d.get("agent_reason") or "")[:160]
        is_new = case not in seen_cases
        if is_new:
            seen_cases.add(case)
            new_cases.append((case, iou, reason))
        if iou is not None:
            ious.append(iou)
        else:
            tag = "unknown"
            if "UsageLimitExceeded" in reason: tag = "worker_usage_limit"
            elif "token limit" in reason.lower(): tag = "reader_token_limit"
            elif "validation" in reason.lower(): tag = "reader_validation"
            elif "no_polygon" in reason.lower() or not d.get("valid_prediction"): tag = "no_polygon"
            failures.append((case, tag, reason))
            if case not in seen_failures:
                seen_failures.add(case)
                new_failures.append((case, tag, reason))
        # weak-retry telemetry (from message_log if present)
        ml = m.replace("metrics.json", "message_log.json")
        if os.path.exists(ml):
            try:
                msgs = json.loads(open(ml).read())
            except Exception:
                msgs = []
            import ast as _ast
            for msg in msgs:
                if msg.get("kind") != "ToolReturnPart" or msg.get("tool") != "match_at":
                    continue
                ret = msg.get("return")
                if isinstance(ret, str):
                    try: ret = _ast.literal_eval(ret)
                    except Exception: continue
                if not isinstance(ret, dict): continue
                for g in (ret.get("per_group") or []):
                    wr = g.get("weak_retry") or {}
                    if wr.get("fired"):
                        weak["fired"] += 1
                        if wr.get("kept"):
                            weak["kept"] += 1
                            d_score = (wr.get("retry_overall_score") or 0) - (wr.get("original_overall_score") or 0)
                            weak["deltas"].append(d_score)
    return {
        "n_done": len(cases),
        "ious": ious,
        "failures": failures,
        "weak": weak,
        "new_cases": new_cases,
        "new_failures": new_failures,
    }

def summary_line(s):
    n = s["n_done"]
    ious = s["ious"]
    if ious:
        mean_honest = sum(ious) / n if n else 0
        mean_succ = statistics.mean(ious) if ious else 0
        n_ge_08 = sum(1 for x in ious if x >= 0.8)
        pct_ge_08 = 100 * n_ge_08 / n if n else 0
    else:
        mean_honest = mean_succ = pct_ge_08 = 0
    return (f"[{time.strftime('%H:%M:%S')}] {n}/{TARGET} done | "
            f"mean_honest={mean_honest:.3f} (base={BASELINE_MEAN:.3f}) | "
            f"≥0.8: {pct_ge_08:.1f}% (base={BASELINE_PCT_GE_08:.1f}%) | "
            f"fails={len(s['failures'])} | weak={s['weak']['fired']}fired/{s['weak']['kept']}kept")

def full_summary(s):
    n = s["n_done"]
    ious = s["ious"]
    print("=" * 78)
    print(f"FULL SUMMARY @ {time.strftime('%H:%M:%S')}  ({n}/{TARGET} cases)")
    print("=" * 78)
    if ious:
        print(f"  mean_honest IoU (fails=0): {sum(ious)/n:.3f}  (baseline {BASELINE_MEAN:.3f})")
        print(f"  mean_successful   IoU:      {statistics.mean(ious):.3f}  (n={len(ious)})")
        print(f"  median IoU:                 {statistics.median(ious):.3f}")
        for thr in [0.5, 0.7, 0.8, 0.9]:
            c = sum(1 for x in ious if x >= thr)
            print(f"  IoU ≥ {thr}: {c}/{n} ({100*c/n:.1f}%)")
    fail_tags = collections.Counter(t for _, t, _ in s["failures"])
    if fail_tags:
        print("  failures by tag:")
        for tag, c in fail_tags.most_common():
            print(f"    {tag:24s} {c}")
    w = s["weak"]
    if w["fired"]:
        kept_pct = 100 * w["kept"] / w["fired"]
        print(f"  weak-retry: fired={w['fired']}, kept={w['kept']} ({kept_pct:.0f}%)")
        if w["deltas"]:
            med_d = statistics.median(w["deltas"])
            print(f"              median Δoverall_score on kept: +{med_d:.2f}")
    print("=" * 78, flush=True)

def main():
    os.chdir("/Users/fabiandegen/Documents/VSCODE/GeoMapAgent_autonomous")
    print(f"Monitor started — watching {RUN}/", flush=True)
    print(f"Baseline: {BASELINE_MEAN:.3f} mean_honest, {BASELINE_PCT_GE_08:.1f}% ≥0.8 (benchmark_v_postrot)", flush=True)
    poll_count = 0
    while True:
        try:
            s = scan()
            poll_count += 1
            # Always report new failures immediately
            for case, tag, reason in s["new_failures"]:
                print(f"  ⚠  FAIL: {case[:35]:35s}  [{tag}]  {reason[:80]}", flush=True)
            # Compact one-liner on each poll if anything moved
            if s["new_cases"] or poll_count == 1:
                print(summary_line(s), flush=True)
                for case, iou, _ in s["new_cases"][-5:]:
                    if iou is not None:
                        emoji = "✓" if iou >= 0.5 else "○"
                        print(f"    {emoji} {case[:40]:40s}  IoU={iou:.3f}", flush=True)
            # Full summary every 10 polls
            if poll_count % FULL_EVERY == 0:
                full_summary(s)
            # Stop when done
            if s["n_done"] >= TARGET:
                print(f"Reached {TARGET} cases — final summary:", flush=True)
                full_summary(s)
                return
        except Exception as e:
            print(f"[monitor error] {e}", flush=True)
        time.sleep(POLL_S)

if __name__ == "__main__":
    main()
