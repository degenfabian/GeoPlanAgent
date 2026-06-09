"""Audit: recompute Table 1 VLM-e2e rows + GeoPlanAgent 40-subset row."""
import json, csv, math
from pathlib import Path
import numpy as np

BASE = Path("ablations/vlm_e2e_pdf_to_geojson")
PRICES = {"gemini-flash": (0.55, 2.20), "gemini-pro": (1.25, 12.50),
          "claude-opus": (5.00, 25.00), "gpt-5.5-pro": (30.0, 180.0)}

subset40 = {c["folder"] for c in json.load(open(BASE/"subset_40.json"))["cases"]}
print("subset40 size:", len(subset40))

# feret + per-case pipeline data from my earlier table-1 audit
t1 = {r["case"]: r for r in json.load(open("audit_table1_rows.json"))}

def report(name, rows):
    iou = np.array([float(r["iou"]) for r in rows])
    err = np.array([float(r["positioning_error_m"]) if r["positioning_error_m"] else np.nan for r in rows])
    fer = np.array([t1[r["case"]]["feret"] for r in rows])
    t = np.array([float(r["call_seconds"]) for r in rows])
    pin, pout = PRICES[name.split()[0]]
    cost = np.array([(int(r["vlm_request_tokens"]) * pin + int(r["vlm_response_tokens"]) * pout) / 1e6 for r in rows])
    acc = np.mean(np.where(np.isnan(err), np.inf, err) <= 0.1 * fer)
    print(f"{name:22s} n={len(rows):3d}  %>0 {100*np.mean(iou>0):5.1f}  mean {np.mean(iou):.3f}  "
          f"med {np.median(iou):.3f}  %>=.8 {100*np.mean(iou>=0.8):4.1f}  "
          f"medErr {np.nanmedian(err):6.0f}  Acc {100*acc:4.1f}  $ {np.mean(cost):.3f}  t {np.mean(t):.0f}s")

for model in ["gemini-flash", "gemini-pro", "claude-opus", "gpt-5.5-pro"]:
    rows = list(csv.DictReader(open(BASE/model/"results.csv")))
    rows40 = [r for r in rows if r["case"] in subset40]
    report(f"{model} 40", rows40)
    if len(rows) == 208:
        report(f"{model} 208", rows)

# GeoPlanAgent on the 40 subset, with the paper's worker-first fallback convention
run = Path("results/benchmark_std_post_fix/gemini-flash")
ious, errs, fers, ts, costs = [], [], [], [], []
for c in subset40:
    m = json.load(open(run/c/"metrics.json"))
    wf = m.get("worker_first_iou")
    r = t1[c]
    if wf is None:
        ious.append(m["iou"]); errs.append(r["final_err"])
    else:
        ious.append(wf)
        wfm = m.get("worker_first_metrics") or {}
        errs.append(wfm.get("positioning_error_m", r["worker_err"]))
    fers.append(r["feret"]); ts.append(m["processing_time"])
    s = m["agent_stats"]
    costs.append(((s.get("request_tokens",0))*0.55 + (s.get("response_tokens",0))*2.20)/1e6)
iou, err, fer = np.array(ious), np.array(errs), np.array(fers)
print(f"{'GeoPlanAgent 40(wf)':22s} n={len(iou):3d}  %>0 {100*np.mean(iou>0):5.1f}  mean {np.mean(iou):.3f}  "
      f"med {np.median(iou):.3f}  %>=.8 {100*np.mean(iou>=0.8):4.1f}  "
      f"medErr {np.median(err):6.1f}  Acc {100*np.mean(err<=0.1*fer):4.1f}  $ {np.mean(costs):.3f}  t {np.mean(ts):.0f}s")
