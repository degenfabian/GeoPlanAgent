"""Audit: independently recompute Table 1 full-dataset rows from per-case artifacts.

Recomputes IoU stats from the *stored geojsons* (not the cached metrics.json
numbers) so we verify the metric code too. Also computes Err(m) median and
Acc@0.1D with the Feret-diameter definition stated in the paper.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from shapely.geometry import shape
from itertools import combinations

RUN = Path("results/benchmark_std_post_fix/gemini-flash")
EVAL = Path("evaluation_data")

def load_shape(p):
    gj = json.load(open(p))
    if gj.get("type") == "FeatureCollection":
        feats = gj["features"]
        from shapely.ops import unary_union
        s = unary_union([shape(f["geometry"]).buffer(0) for f in feats])
    else:
        geom = gj["geometry"] if gj.get("type") == "Feature" else gj
        s = shape(geom)
    if not s.is_valid:
        s = s.buffer(0)
    return s

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def feret_diameter_m(geom):
    hull = geom.convex_hull
    if hull.geom_type == "Point":
        return 0.0
    coords = list(hull.exterior.coords) if hull.geom_type == "Polygon" else list(hull.coords)
    best = 0.0
    for (x1,y1),(x2,y2) in combinations(coords[:-1] if hull.geom_type=="Polygon" else coords, 2):
        d = haversine_m(y1,x1,y2,x2)
        if d > best: best = d
    return best

cases = sorted([d for d in RUN.iterdir() if d.is_dir()])
print(f"n case dirs: {len(cases)}")

rows = []
for d in cases:
    name = d.name
    gt_candidates = list((EVAL/name).glob("*.geojson"))
    assert len(gt_candidates) == 1, (name, gt_candidates)
    gt = load_shape(gt_candidates[0])
    m = json.load(open(d/"metrics.json"))
    row = {"case": name, "time": m.get("processing_time")}
    for tag, fname, stored_iou in [
        ("final", "predicted.geojson", m.get("iou")),
        ("worker", "predicted_worker_first.geojson", m.get("worker_first_iou")),
    ]:
        p = d/fname
        if not p.exists():
            if tag == "worker":
                # no critic intervention -> same file
                p = d/"predicted.geojson"
            else:
                row[tag+"_iou"] = None
                continue
        pred = load_shape(p)
        inter = gt.intersection(pred).area
        union = gt.union(pred).area
        iou = inter/union if union > 0 else 0.0
        pc, gc = pred.centroid, gt.centroid
        err = haversine_m(gc.y, gc.x, pc.y, pc.x)
        row[tag+"_iou"] = iou
        row[tag+"_err"] = err
        row[tag+"_stored_iou"] = stored_iou
    row["feret"] = feret_diameter_m(gt)
    rows.append(row)

def agg(tag):
    ious = np.array([r[tag+"_iou"] for r in rows], float)
    errs = np.array([r[tag+"_err"] for r in rows], float)
    fer  = np.array([r["feret"] for r in rows], float)
    acc = np.mean(errs <= 0.1*fer)
    print(f"\n== {tag} (n={len(ious)}) ==")
    print(f"%IoU>0   : {100*np.mean(ious>0):.1f}%")
    print(f"mean IoU : {np.mean(ious):.4f}")
    print(f"med IoU  : {np.median(ious):.4f}")
    print(f"%IoU>=0.8: {100*np.mean(ious>=0.8):.1f}%")
    print(f"med err  : {np.median(errs):.2f} m")
    print(f"Acc@0.1D : {100*acc:.1f}%")

agg("worker")
agg("final")

times = np.array([r["time"] for r in rows], float)
print(f"\nmean time: {np.mean(times):.1f}s, median: {np.median(times):.1f}s")

# how far do stored metrics.json IoUs deviate from recomputed?
dev = [abs(r["final_iou"]-r["final_stored_iou"]) for r in rows if r["final_stored_iou"] is not None]
print(f"max |recomputed - stored| final IoU: {max(dev):.2e}")
dev_w = [abs(r["worker_iou"]-r["worker_stored_iou"]) for r in rows if r.get("worker_stored_iou") is not None]
print(f"max |recomputed - stored| worker IoU: {max(dev_w):.2e}")

json.dump(rows, open("audit_table1_rows.json","w"), indent=1)
