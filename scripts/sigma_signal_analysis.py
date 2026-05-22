"""σ-signal analysis: does the locate sub-agent's pick.sigma_m correlate
with positioning quality on benchmark_v3?

We want to answer: when the sub-agent says σ=200m (tight consensus), does
MINIMA actually land closer to GT than when it says σ=2500m (LA-only
fallback)? If YES, the σ is a real signal worth respecting in
sliding_window_position (fixing the contract bug). If NO, the σ is noise
and respecting it could harm the pipeline.

For each case in results/benchmark_v3/gemini-flash/:
  - Extract pick.sigma_m + confidence + source from message_log.json
  - Extract IoU + positioning error from metrics.json + predicted.geojson
  - Group by σ-band and confidence

Outputs results/v3_sigma_signal.csv + a console summary.

No API calls; reads cached results only.
"""
from __future__ import annotations
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

V3 = REPO / "results" / "benchmark_v3" / "gemini-flash"
EVAL = REPO / "evaluation_data"
OUT_CSV = REPO / "results" / "v3_sigma_signal.csv"


from tools.geo.coords import haversine_m  # noqa: E402 (kept as local name)


def polygon_centroid(geom: dict):
    """Approximate centroid of a (Multi)Polygon GeoJSON geometry."""
    coords = geom.get("coordinates")
    if not coords:
        return None
    if geom.get("type") == "MultiPolygon":
        pts = [pt for poly in coords for pt in poly[0]]
    elif geom.get("type") == "Polygon":
        pts = list(coords[0])
    else:
        return None
    if not pts:
        return None
    lon = mean(p[0] for p in pts)
    lat = mean(p[1] for p in pts)
    return lat, lon


def first_pick(case_dir: Path):
    """Extract the FIRST propose_centers return (the initial locate pick)."""
    ml = case_dir / "message_log.json"
    if not ml.exists():
        return None
    try:
        d = json.loads(ml.read_text())
    except Exception:
        return None
    for entry in d:
        if entry.get("tool") == "propose_centers" and "return" in entry:
            ret = entry["return"]
            if not isinstance(ret, dict):
                continue
            if ret.get("engine") != "live_llm_locate":
                continue  # skip cascade fallbacks
            cands = ret.get("candidates") or []
            if not cands:
                continue
            c = cands[0]
            return {
                "sigma_m": c.get("sigma_m"),
                "specificity": c.get("specificity"),
                "source": c.get("source", ""),
                "la_check_passed": ret.get("la_check_passed"),
                "lat": c.get("lat"),
                "lon": c.get("lon"),
            }
    return None


def gt_centroid(case_name: str):
    case_dir = EVAL / case_name
    if not case_dir.exists():
        return None
    gjs = list(case_dir.glob("*.geojson"))
    if not gjs:
        return None
    try:
        g = json.loads(gjs[0].read_text())
    except Exception:
        return None
    if g.get("type") == "FeatureCollection":
        feats = g.get("features", [])
        if not feats:
            return None
        geom = feats[0].get("geometry", {})
    elif g.get("type") == "Feature":
        geom = g.get("geometry", {})
    else:
        geom = g
    return polygon_centroid(geom)


def confidence_from_specificity(spec):
    return {5: "high", 3: "med", 1: "low"}.get(spec, "?")


def sigma_band(sigma):
    if sigma is None:
        return "unknown"
    if sigma <= 300:
        return "tight (≤300m)"
    if sigma <= 1500:
        return "narrow (≤1.5km)"
    if sigma <= 5000:
        return "medium (≤5km)"
    return "wide (>5km)"


def main():
    rows = []
    skipped = defaultdict(int)
    for case in sorted(V3.iterdir()):
        if not case.is_dir():
            continue
        mf = case / "metrics.json"
        if not mf.exists():
            skipped["no_metrics"] += 1
            continue
        try:
            m = json.loads(mf.read_text())
        except Exception:
            skipped["bad_metrics"] += 1
            continue
        iou = m.get("iou")
        if iou is None:
            skipped["no_iou"] += 1
            continue

        pick = first_pick(case)
        if pick is None:
            skipped["no_live_locate"] += 1
            continue

        # Positioning error: locate pick centroid vs GT centroid
        gt = gt_centroid(case.name)
        pick_err_km = None
        if gt is not None and pick.get("lat") is not None:
            pick_err_km = haversine_m(pick["lat"], pick["lon"], gt[0], gt[1]) / 1000

        # MINIMA matched centre vs GT centre (post-match positioning)
        mi = m.get("match_info") or {}
        ll = mi.get("center_latlon") or mi.get("chosen_center_latlon")
        match_err_km = None
        if gt is not None and ll:
            match_err_km = haversine_m(ll[0], ll[1], gt[0], gt[1]) / 1000

        # Final predicted polygon centroid vs GT
        pred_err_km = None
        pj = case / "predicted.geojson"
        if pj.exists() and gt is not None:
            try:
                pred = json.loads(pj.read_text())
                gm = pred.get("geometry") or pred
                pc = polygon_centroid(gm)
                if pc:
                    pred_err_km = haversine_m(pc[0], pc[1], gt[0], gt[1]) / 1000
            except Exception:
                pass

        rows.append({
            "case": case.name,
            "iou": iou,
            "sigma_m": pick["sigma_m"],
            "specificity": pick["specificity"],
            "confidence": confidence_from_specificity(pick["specificity"]),
            "source": pick["source"][:80],
            "la_check_passed": pick["la_check_passed"],
            "pick_err_km": pick_err_km,
            "match_err_km": match_err_km,
            "pred_err_km": pred_err_km,
        })

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}\nSkipped: {dict(skipped)}\n")

    # ── Analysis: does σ correlate with positioning error? ─────────────────
    by_band = defaultdict(list)
    for r in rows:
        b = sigma_band(r["sigma_m"])
        by_band[b].append(r)

    print("=" * 82)
    print(f"{'σ band':22s}  {'n':>3s}  {'mean σ':>8s}  "
          f"{'mean IoU':>9s}  {'med IoU':>8s}  "
          f"{'pick_err':>9s}  {'match_err':>10s}  {'pred_err':>9s}")
    print("-" * 82)
    for band in ["tight (≤300m)", "narrow (≤1.5km)", "medium (≤5km)", "wide (>5km)"]:
        bucket = by_band.get(band, [])
        if not bucket:
            continue
        n = len(bucket)
        msig = mean(r["sigma_m"] for r in bucket)
        miou = mean(r["iou"] for r in bucket)
        med_iou = median(r["iou"] for r in bucket)
        pe = [r["pick_err_km"] for r in bucket if r["pick_err_km"] is not None]
        me = [r["match_err_km"] for r in bucket if r["match_err_km"] is not None]
        pred_e = [r["pred_err_km"] for r in bucket if r["pred_err_km"] is not None]
        pick_e_str = f"{median(pe):>6.2f}km" if pe else "  n/a"
        match_e_str = f"{median(me):>7.2f}km" if me else "    n/a"
        pred_e_str = f"{median(pred_e):>6.2f}km" if pred_e else "  n/a"
        print(f"{band:22s}  {n:>3d}  {msig:>7.0f}m  {miou:>9.3f}  {med_iou:>8.3f}  "
              f"{pick_e_str}  {match_e_str}  {pred_e_str}")

    # ── By confidence (specificity-derived label) ─────────────────────────
    by_conf = defaultdict(list)
    for r in rows:
        by_conf[r["confidence"]].append(r)
    print()
    print(f"{'confidence':12s}  {'n':>3s}  {'mean σ':>8s}  {'mean IoU':>9s}  "
          f"{'IoU≥0.5':>8s}  {'median pick_err':>16s}")
    print("-" * 70)
    for conf in ["high", "med", "low"]:
        b = by_conf.get(conf, [])
        if not b: continue
        n = len(b)
        msig = mean(r["sigma_m"] for r in b)
        miou = mean(r["iou"] for r in b)
        good = sum(1 for r in b if r["iou"] >= 0.5)
        pe = [r["pick_err_km"] for r in b if r["pick_err_km"] is not None]
        pe_str = f"{median(pe):>13.2f}km" if pe else "          n/a"
        print(f"{conf:12s}  {n:>3d}  {msig:>7.0f}m  {miou:>9.3f}  "
              f"{good:>3d}/{n:<4d} {pe_str}")

    # ── σ vs pick error scatter quality: Spearman correlation ──────────────
    print()
    # Pick err: σ should correlate POSITIVELY (higher σ = sub-agent says
    # "I'm less sure" = larger actual error). If correlation is strong, the
    # σ is a good signal we should respect.
    pairs = [(r["sigma_m"], r["pick_err_km"])
             for r in rows
             if r["sigma_m"] is not None and r["pick_err_km"] is not None]
    if pairs:
        sigmas = [p[0] for p in pairs]
        errs = [p[1] for p in pairs]
        # Spearman by ranking
        def rank(xs):
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ranks = [0.0] * len(xs)
            for r_, idx in enumerate(order):
                ranks[idx] = r_
            return ranks
        rs = rank(sigmas)
        re = rank(errs)
        n = len(pairs)
        mean_rs = sum(rs) / n
        mean_re = sum(re) / n
        num = sum((rs[i] - mean_rs) * (re[i] - mean_re) for i in range(n))
        den_a = math.sqrt(sum((rs[i] - mean_rs) ** 2 for i in range(n)))
        den_b = math.sqrt(sum((re[i] - mean_re) ** 2 for i in range(n)))
        spearman = num / (den_a * den_b) if den_a * den_b > 0 else 0
        print(f"Spearman(σ, pick_err_km) over {n} cases: ρ = {spearman:+.3f}")
        print("  Positive ρ → larger σ correlates with larger actual error "
              "(σ is a useful confidence signal)")
        print("  Near zero ρ → σ is noise; respecting it would be misleading")
        print("  Negative ρ → σ is inverted; sub-agent's confidence is anti-correlated with truth")

    # ── Outliers: high-confidence picks that landed BADLY ──────────────────
    bad_high_conf = sorted(
        [r for r in rows if r["confidence"] == "high"
         and r["pick_err_km"] is not None and r["pick_err_km"] > 5],
        key=lambda r: -r["pick_err_km"]
    )[:8]
    if bad_high_conf:
        print(f"\nTop high-confidence picks with pick_err > 5km "
              f"({len(bad_high_conf)} shown):")
        for r in bad_high_conf:
            print(f"  {r['case']:42s} σ={r['sigma_m']:.0f}m  "
                  f"pick_err={r['pick_err_km']:.1f}km  IoU={r['iou']:.2f}  "
                  f"src={r['source'][:40]}")


if __name__ == "__main__":
    main()
