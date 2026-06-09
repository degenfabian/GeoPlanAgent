"""Audit: recompute Collapsed Reader row (Table 1) from ablations/no_reader."""
import json, math
from pathlib import Path
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from itertools import combinations

RUN = Path("ablations/no_reader/gemini-flash")
EVAL = Path("evaluation_data")

def load_shape(p):
    gj = json.load(open(p))
    if gj.get("type") == "FeatureCollection":
        s = unary_union([shape(f["geometry"]).buffer(0) for f in gj["features"]])
    else:
        s = shape(gj["geometry"] if gj.get("type") == "Feature" else gj)
    return s if s.is_valid else s.buffer(0)

def hav(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return 2*R*math.asin(math.sqrt(a))

def feret(g):
    h = g.convex_hull
    cs = list(h.exterior.coords)[:-1] if h.geom_type == "Polygon" else list(h.coords)
    return max((hav(y1,x1,y2,x2) for (x1,y1),(x2,y2) in combinations(cs,2)), default=0.0)

ious, errs, fers, times, toks = [], [], [], [], []
miss = []
for d in sorted(p for p in RUN.iterdir() if p.is_dir()):
    mp = d/"metrics.json"
    if not mp.exists():
        miss.append(d.name); continue
    m = json.load(open(mp))
    pred_p = d/"predicted.geojson"
    gtl = list((EVAL/d.name).glob("*.geojson"))
    if not pred_p.exists() or not gtl:
        ious.append(0.0); errs.append(float("inf")); fers.append(1.0)
        miss.append(d.name + " (no pred/gt)")
        continue
    gt, pred = load_shape(gtl[0]), load_shape(pred_p)
    u = gt.union(pred).area
    ious.append(gt.intersection(pred).area/u if u > 0 else 0.0)
    pc, gc = pred.centroid, gt.centroid
    errs.append(hav(gc.y, gc.x, pc.y, pc.x))
    fers.append(feret(gt))
    times.append(m.get("processing_time"))
    toks.append(m.get("agent_stats", {}).get("total_tokens"))

iou, err, fer = np.array(ious), np.array(errs), np.array(fers)
print(f"n={len(iou)} missing={miss}")
print(f"%IoU>0 {100*np.mean(iou>0):.1f}  mean {np.mean(iou):.4f}  med {np.median(iou):.4f}  %>=0.8 {100*np.mean(iou>=0.8):.1f}")
print(f"medErr {np.median(err):.2f}  Acc@0.1D {100*np.mean(err<=0.1*fer):.1f}")
print(f"mean time {np.mean([t for t in times if t]):.1f}s  mean tokens {np.mean([t for t in toks if t]):.0f}")

# compare tokens vs main run
main_toks = []
for d in Path("results/benchmark_std_post_fix/gemini-flash").iterdir():
    if d.is_dir():
        m = json.load(open(d/"metrics.json"))
        t = m.get("agent_stats", {}).get("total_tokens")
        if t: main_toks.append(t)
print(f"main run mean tokens {np.mean(main_toks):.0f}  -> ratio {np.mean([t for t in toks if t])/np.mean(main_toks):.2f}")
