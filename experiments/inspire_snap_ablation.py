"""Does INSPIRE freehold-parcel snap actually help IoU?

Pure replay ablation — no SAM, no LLM, no affine recomputation. For each
v20 case where the agent produced a predicted polygon:
  1. Load the stored predicted.geojson (raw projection — snap was silently
     broken in v20 due to the path bug, so this is pre-snap output)
  2. Apply InspireSnap with 8 m tolerance (the production setting)
  3. Compute IoU vs GT
  4. Compare to v20's stored IoU (also raw)

If snap helps: IoUs go up across cases. If neutral / harmful: useful data
to say in the paper "INSPIRE snap didn't move the needle".

Easy to delete: rm experiments/inspire_snap_ablation.py + results json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


def main():
    bench_dir = REPO / "results" / "benchmark_v20" / "gemini-flash"
    eval_dir = REPO / "evaluation_data"
    out_path = HERE / "inspire_snap_results.json"

    from tools.snap.inspire import InspireSnap, la_for_admin_region
    from tools.metrics.geojson import load_geojson, calculate_spatial_metrics
    from shapely.geometry import shape as _shape, mapping as _mapping

    # Cache InspireSnap instances by LA — only loaded once each
    snap_cache: dict[str, InspireSnap] = {}

    def _get_snap(la_name: str) -> InspireSnap | None:
        if la_name in snap_cache:
            return snap_cache[la_name]
        try:
            snap_cache[la_name] = InspireSnap([la_name])
            return snap_cache[la_name]
        except Exception as e:
            print(f"  !! snap-load failed for {la_name}: {e!s:.80}")
            snap_cache[la_name] = None
            return None

    cases = sorted(d.name for d in bench_dir.iterdir() if d.is_dir())
    print(f"Scanning {len(cases)} v20 cases for cached predicted.geojson...")

    rows = []
    skip_no_pred = skip_no_gt = skip_no_admin = skip_no_la = 0
    snap_changed = 0
    for i, case in enumerate(cases, 1):
        cd = bench_dir / case
        pred_p = cd / "predicted.geojson"
        if not pred_p.exists():
            skip_no_pred += 1; continue
        gt_files = list((eval_dir / case).glob("*.geojson"))
        if not gt_files: skip_no_gt += 1; continue
        try:
            pi = json.loads((cd / "pdf_info.json").read_text())
        except Exception:
            pi = {}
        admin = (pi.get("admin_region") or "").strip()
        if not admin:
            skip_no_admin += 1; continue
        la = la_for_admin_region(admin)
        if la is None:
            skip_no_la += 1; continue

        snap_obj = _get_snap(la)
        if snap_obj is None:
            skip_no_la += 1; continue

        try:
            pred_geojson = json.loads(pred_p.read_text())
        except Exception:
            continue
        if "geometry" not in pred_geojson:
            continue

        gt = load_geojson(str(gt_files[0]))
        if gt is None: continue

        # Pre-snap IoU
        try:
            pre = calculate_spatial_metrics(gt, pred_geojson)
            pre_iou = float(pre.get("iou", 0.0))
        except Exception:
            pre_iou = 0.0

        # Apply snap
        try:
            pred_geom = _shape(pred_geojson["geometry"])
            if not pred_geom.is_valid or pred_geom.is_empty:
                continue
            snapped = snap_obj.snap_polygon(pred_geom, max_dist_m=8.0)
            if snapped is None or snapped.is_empty:
                post_geojson = pred_geojson
            else:
                if not snapped.equals(pred_geom):
                    snap_changed += 1
                post_geojson = {
                    "type": "Feature",
                    "properties": pred_geojson.get("properties") or {},
                    "geometry": _mapping(snapped),
                }
        except Exception as e:
            print(f"  {case}: snap failed: {e!s:.80}")
            continue

        try:
            post = calculate_spatial_metrics(gt, post_geojson)
            post_iou = float(post.get("iou", 0.0))
        except Exception:
            post_iou = pre_iou

        rows.append({
            "case": case, "admin_region": admin, "la": la,
            "pre_snap_iou": pre_iou,
            "post_snap_iou": post_iou,
            "delta": post_iou - pre_iou,
        })
        if i % 25 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] kept={len(rows)} changed={snap_changed} "
                  f"skipped(no_pred={skip_no_pred} no_gt={skip_no_gt} "
                  f"no_admin={skip_no_admin} no_la={skip_no_la})")

    if not rows:
        print("ERROR: no cases evaluated"); return 1
    pre = np.array([r["pre_snap_iou"] for r in rows])
    post = np.array([r["post_snap_iou"] for r in rows])
    delta = post - pre

    summary = {
        "n_cases": len(rows),
        "n_snap_changed_polygon": snap_changed,
        "mean_pre":  float(pre.mean()),
        "mean_post": float(post.mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "n_helped_>+0.02": int((delta > 0.02).sum()),
        "n_hurt_<-0.02":   int((delta < -0.02).sum()),
        "n_neutral":       int(((-0.02 <= delta) & (delta <= 0.02)).sum()),
        "biggest_gain":  float(delta.max()),
        "biggest_loss":  float(delta.min()),
    }
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"\n{'='*68}")
    print(f"INSPIRE snap ablation on {len(rows)} cases")
    print(f"{'='*68}")
    print(f"  mean IoU pre-snap:  {summary['mean_pre']:.4f}")
    print(f"  mean IoU post-snap: {summary['mean_post']:.4f}")
    print(f"  Δ mean IoU:         {summary['delta_mean']:+.4f}")
    print(f"  Δ median IoU:       {summary['delta_median']:+.4f}")
    print(f"  Cases where snap changed the polygon at all: {summary['n_snap_changed_polygon']}")
    print(f"  Cases helped (Δ > +0.02): {summary['n_helped_>+0.02']}")
    print(f"  Cases hurt   (Δ < -0.02): {summary['n_hurt_<-0.02']}")
    print(f"  Cases neutral:            {summary['n_neutral']}")
    print(f"  Best gain:  {summary['biggest_gain']:+.4f}")
    print(f"  Worst loss: {summary['biggest_loss']:+.4f}")
    print(f"\nResults: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
