"""One stdout line per actionable event. Each line becomes a Monitor notification.
Events: new completion (with IoU + Δ vs baseline), new failure (tagged),
milestone every 25 cases, final summary at 208/208 (then exit).

Emits a case event when its metrics.json's mtime is *newer* than the monitor's
start time. This handles --force restarts correctly: pre-existing metrics.json
files from a previous run have old mtimes and are filtered out, but as --force
re-runs them and writes new metrics.json files, the new mtime triggers an emit.
"""
import json, glob, os, sys, time, statistics

RUN = "results/benchmark_v_R20/gemini-flash"
BASE_RUN = "results/benchmark_v_postrot/gemini-flash"
TARGET = 208

baseline = {}
for m in glob.glob(f"{BASE_RUN}/*/metrics.json"):
    try:
        baseline[m.split("/")[-2]] = json.loads(open(m).read()).get("iou")
    except Exception:
        pass

# Monitor's own start time. A metrics.json file is emitted only if its
# mtime is newer than this — handles --force restart cleanly.
MONITOR_START = time.time()
seen: set = set()
n_existing = len(glob.glob(f"{RUN}/*/metrics.json"))
print(f"Monitor: {n_existing} pre-existing case dirs at startup; will emit only on metrics.json files written AFTER {time.strftime('%H:%M:%S', time.localtime(MONITOR_START))}.", flush=True)

last_milestone = 0

def fail_tag(reason: str) -> str:
    r = (reason or "").lower()
    if "token limit" in r: return "READER_TOKEN"
    if "usagelimitexceeded" in r: return "WORKER_USAGE"
    if "validation" in r: return "READER_VALIDATION"
    return "OTHER"

while True:
    new_events = 0
    all_cases = sorted(glob.glob(f"{RUN}/*/metrics.json"))
    # Only count cases written by THIS run (mtime > monitor start)
    fresh_cases = [m for m in all_cases if os.path.getmtime(m) >= MONITOR_START]
    n_done = len(fresh_cases)  # progress of the current run
    for m in fresh_cases:
        case = m.split("/")[-2]
        if case in seen:
            continue
        seen.add(case)
        new_events += 1
        try:
            d = json.loads(open(m).read())
        except Exception:
            continue
        iou = d.get("iou")
        reason = (d.get("agent_reason") or "")[:80]
        base = baseline.get(case)
        if iou is None:
            print(f"FAIL [{fail_tag(reason)}] ({n_done}/{TARGET}) {case[:35]}  {reason!r}", flush=True)
        else:
            mark = "PASS" if iou >= 0.8 else "OK  " if iou >= 0.5 else "WEAK"
            delta = ""
            if base is not None and abs(iou - base) >= 0.05:
                delta = f"  Δ={iou - base:+.3f} vs base"
            base_str = f"{base:.3f}" if base is not None else "?"
            print(f"{mark} ({n_done}/{TARGET}) {case[:35]:35s} IoU={iou:.3f} (base={base_str}){delta}", flush=True)

    # Milestone every 25 cases
    milestone = (n_done // 25) * 25
    if milestone > last_milestone and milestone >= 25:
        last_milestone = milestone
        ious = []
        fails = 0
        for m in fresh_cases:
            try: d = json.loads(open(m).read())
            except: continue
            iou = d.get("iou")
            if iou is None: fails += 1
            else: ious.append(iou)
        if ious:
            mh = sum(ious) / n_done
            pct_08 = 100 * sum(1 for x in ious if x >= 0.8) / n_done
            print(f"--- {milestone}/{TARGET} milestone: mean_honest={mh:.3f} (base 0.682)  "
                  f">=0.8: {pct_08:.0f}% (base 64%)  fails={fails} ---", flush=True)

    # Final summary
    if n_done >= TARGET:
        ious = []
        fails = 0
        for m in fresh_cases:
            try: d = json.loads(open(m).read())
            except: continue
            iou = d.get("iou")
            if iou is None: fails += 1
            else: ious.append(iou)
        mh = sum(ious) / n_done if n_done else 0
        ms = statistics.mean(ious) if ious else 0
        pct_08 = 100 * sum(1 for x in ious if x >= 0.8) / n_done if n_done else 0
        print(f"DONE {n_done}/{TARGET}  mean_honest={mh:.3f} (base 0.682)  "
              f"mean_success={ms:.3f} (base 0.692)  >=0.8: {pct_08:.0f}% (base 64%)  fails={fails}",
              flush=True)
        sys.exit(0)

    time.sleep(30)
