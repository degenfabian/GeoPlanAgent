"""Audit recompute: Table 1 VLM rows, gemini-pro 208 row, GeoPlanAgent 40-subset row."""
import csv, json, math, statistics as st
from pathlib import Path
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
ABL = ROOT / "ablations" / "vlm_e2e_pdf_to_geojson"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_shape(path):
    gj = json.loads(Path(path).read_text())
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"] if f.get("geometry")]
        return unary_union(geoms)
    if gj.get("type") == "Feature":
        return shape(gj["geometry"])
    return shape(gj)

def feret_m(geom):
    hull = geom.convex_hull
    if hull.geom_type == "Point":
        return 0.0
    coords = list(hull.exterior.coords) if hull.geom_type == "Polygon" else list(hull.coords)
    best = 0.0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            d = haversine_m(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
            best = max(best, d)
    return best

subset40 = json.loads((ABL / "subset_40.json").read_text())
cases40 = {c["folder"]: c for c in subset40["cases"]}
subset208 = json.loads((ABL / "subset_208.json").read_text())
cases208 = {c["folder"]: c for c in subset208["cases"]}

# cache GT centroid + feret per case
gt_cache = {}
def gt_info(folder, meta):
    if folder in gt_cache:
        return gt_cache[folder]
    rel = meta.get("gt_geojson_relpath")
    g = load_shape(ROOT / rel)
    c = g.centroid
    gt_cache[folder] = (c.y, c.x, feret_m(g))
    return gt_cache[folder]

def sanitize(case):
    return case.replace("/", "_").replace(":", "_")

PRICES = {
    "gemini-flash": (0.55, 2.20),
    "gemini-pro": (1.25, 12.50),
    "claude-opus": (5.00, 25.00),
    "gpt-5.5-pro": (30.0, 180.0),
}

def analyze(model, case_filter, label, meta_lookup):
    d = ABL / model
    rows = list(csv.DictReader(open(d / "results.csv")))
    rows = [r for r in rows if r["case"] in case_filter]
    n = len(rows)
    seen = [r["case"] for r in rows]
    dup = {c for c in seen if seen.count(c) > 1}
    missing = set(case_filter) - set(seen)
    ious = [float(r["iou"]) if r.get("iou") else 0.0 for r in rows]
    pct_gt0 = 100*sum(1 for x in ious if x > 0)/n
    pct_ge08 = 100*sum(1 for x in ious if x >= 0.8)/n
    pin, pout = PRICES[model]
    costs, times = [], []
    errs = {}
    acc_hits = 0
    n_pred = 0
    for r in rows:
        if r.get("call_seconds"):
            times.append(float(r["call_seconds"]))
        if r.get("vlm_request_tokens") and r.get("vlm_response_tokens"):
            costs.append((int(r["vlm_request_tokens"])*pin + int(r["vlm_response_tokens"])*pout)/1e6)
        pred_path = d / "pred_geojsons" / f"{sanitize(r['case'])}.geojson"
        if pred_path.exists():
            try:
                pg = load_shape(pred_path)
                if pg.is_empty:
                    raise ValueError("empty")
                glat, glon, fer = gt_info(r["case"], meta_lookup[r["case"]])
                pc = pg.centroid
                e = haversine_m(glat, glon, pc.y, pc.x)
                errs[r["case"]] = e
                n_pred += 1
                if fer > 0 and e <= 0.1*fer:
                    acc_hits += 1
            except Exception as ex:
                pass
    ev = sorted(errs.values())
    med_err_valid = st.median(ev) if ev else None
    print(f"\n== {label} (n={n}, dups={dup or 'none'}, missing={missing or 'none'}) ==")
    print(f"  IoU>0: {pct_gt0:.1f}%   mean IoU: {st.mean(ious):.4f}   median IoU: {st.median(ious):.4f}   IoU>=0.8: {pct_ge08:.1f}%")
    print(f"  n_with_pred_geojson={n_pred}")
    print(f"  Err(m) median over {len(ev)} preds: {med_err_valid:.1f}" if ev else "  no preds")
    print(f"  Acc@0.1D (hits/{n}): {acc_hits}/{n} = {100*acc_hits/n:.1f}%   (hits/preds = {100*acc_hits/max(n_pred,1):.1f}%)")
    print(f"  $/doc mean over {len(costs)}: {st.mean(costs):.4f}   median: {st.median(costs):.4f}" if costs else "  no token data")
    print(f"  time mean over {len(times)}: {st.mean(times):.1f}s   median: {st.median(times):.1f}s" if times else "  no time data")
    # row positioning_error_m comparison (harness values)
    perrs = sorted(float(r["positioning_error_m"]) for r in rows if r.get("positioning_error_m"))
    if perrs:
        print(f"  [harness positioning_error_m] median over {len(perrs)}: {st.median(perrs):.1f}")

for model in ["gemini-flash", "gemini-pro", "claude-opus", "gpt-5.5-pro"]:
    analyze(model, set(cases40), f"{model} 40-subset", cases40)

analyze("gemini-pro", set(cases208), "gemini-pro FULL 208", cases208)

# ── GeoPlanAgent on the 40 subset, from benchmark_std_post_fix ──
BENCH = ROOT / "results" / "benchmark_std_post_fix" / "gemini-flash"
ious, errs, times, costs_rw = [], [], [], []
acc_hits = 0; n_found = 0; miss = []
pin, pout = PRICES["gemini-flash"]
for folder, meta in cases40.items():
    cd = BENCH / folder
    mp = cd / "metrics.json"
    if not mp.exists():
        miss.append(folder); continue
    n_found += 1
    m = json.loads(mp.read_text())
    iou = m.get("iou") or 0.0
    ious.append(float(iou))
    times.append(float(m.get("processing_time") or 0))
    s = m.get("agent_stats", {}) or {}
    ti = int(s.get("reader_request_tokens",0) or 0) + int(s.get("worker_request_tokens",0) or 0) + int(s.get("locate_request_tokens",0) or 0)
    to = int(s.get("reader_response_tokens",0) or 0) + int(s.get("worker_response_tokens",0) or 0) + int(s.get("locate_response_tokens",0) or 0)
    costs_rw.append((ti*pin + to*pout)/1e6)
    pred = cd / "predicted.geojson"
    if pred.exists():
        try:
            pg = load_shape(pred)
            glat, glon, fer = gt_info(folder, meta)
            pc = pg.centroid
            e = haversine_m(glat, glon, pc.y, pc.x)
            errs.append(e)
            if fer > 0 and e <= 0.1*fer:
                acc_hits += 1
        except Exception:
            pass
n = len(ious)
print(f"\n== GeoPlanAgent (benchmark_std_post_fix) on 40-subset: found {n_found}/40, missing={miss} ==")
print(f"  IoU>0: {100*sum(1 for x in ious if x>0)/n:.1f}%  mean: {st.mean(ious):.4f}  median: {st.median(ious):.4f}  >=0.8: {100*sum(1 for x in ious if x>=0.8)/n:.1f}%")
ev = sorted(errs)
print(f"  Err(m) median over {len(ev)} preds: {st.median(ev):.2f}")
print(f"  Acc@0.1D: {acc_hits}/{n} = {100*acc_hits/n:.1f}%")
print(f"  time mean: {st.mean(times):.1f}s  median: {st.median(times):.1f}s")
print(f"  $/doc (reader+worker+locate tokens in metrics.json, flash prices): mean {st.mean(costs_rw):.4f}")
