"""Reproduce every number in the paper from the cached run artifacts.

No API calls, no model inference -- everything is recomputed from the
per-case outputs checked into results/, ablations/ and training/eval/.
Each section prints the recomputed values next to the value reported in
the paper so drift is obvious at a glance.

Usage:
    uv run scripts/reproduce_paper.py all
    uv run scripts/reproduce_paper.py table1 table2
    uv run scripts/reproduce_paper.py fig3 --run-dir results/benchmark_std_post_fix/gemini-flash

Sections: table1 table2 table4 table9 table11 table12 fig3 fig4 costs dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "evaluation_data"
DEFAULT_RUN = REPO / "results/benchmark_std_post_fix/gemini-flash"

from _pricing import PRICES  # noqa: E402 (scripts/ on sys.path when run as a file)


# ---------------------------------------------------------------- geometry


def load_shape(path: Path):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    gj = json.loads(path.read_text())
    if gj.get("type") == "FeatureCollection":
        geom = unary_union([shape(f["geometry"]).buffer(0) for f in gj["features"]])
    else:
        geom = shape(gj["geometry"] if gj.get("type") == "Feature" else gj)
    return geom if geom.is_valid else geom.buffer(0)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def feret_m(geom):
    """Max pairwise haversine distance over the convex hull (GT diameter)."""
    hull = geom.convex_hull
    if hull.geom_type == "Point":
        return 0.0
    pts = list(hull.exterior.coords)[:-1] if hull.geom_type == "Polygon" else list(hull.coords)
    return max(
        (haversine_m(y1, x1, y2, x2) for (x1, y1), (x2, y2) in combinations(pts, 2)), default=0.0
    )


_feret_cache: dict[str, float] = {}


def gt_feret(case: str) -> float:
    if case not in _feret_cache:
        gt_file = next((EVAL / case).glob("*.geojson"))
        _feret_cache[case] = feret_m(load_shape(gt_file))
    return _feret_cache[case]


# ------------------------------------------------------------- run loading


def case_dirs(run_dir: Path) -> list[Path]:
    return sorted(d for d in run_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists())


def read_metrics(run_dir: Path) -> dict[str, dict]:
    return {d.name: json.loads((d / "metrics.json").read_text()) for d in case_dirs(run_dir)}


def worker_first(m: dict) -> tuple[float, float | None]:
    """Pre-critic (iou, err_m) for one case.

    The benchmark ran with the critic enabled; metrics.json keeps the
    pre-critic result in worker_first_*. Where worker_first_iou is null
    the critic never changed anything, so the final value is already the
    pre-critic value.
    """
    if m.get("worker_first_iou") is None:
        return m["iou"], m.get("centroid_distance_m")
    wf = m.get("worker_first_metrics") or {}
    return m["worker_first_iou"], wf.get("centroid_distance_m")


def summarise(ious, errs, ferets):
    iou = np.asarray(ious, float)
    err = np.asarray([e if e is not None else np.inf for e in errs], float)
    fer = np.asarray(ferets, float)
    return {
        "n": len(iou),
        "pct_pos": 100 * np.mean(iou > 0),
        "mean": float(np.mean(iou)),
        "median": float(np.median(iou)),
        "pct_08": 100 * np.mean(iou >= 0.8),
        "med_err": float(np.median(err)),
        "acc_01d": 100 * np.mean(err <= 0.1 * fer),
    }


def print_row(label, s, paper=None, cost=None, secs=None):
    line = (
        f"  {label:<28} n={s['n']:<4} %IoU>0 {s['pct_pos']:5.1f}  "
        f"mean {s['mean']:.3f}  med {s['median']:.3f}  "
        f"%>=0.8 {s['pct_08']:5.1f}  medErr {s['med_err']:7.1f} m  "
        f"Acc@0.1D {s['acc_01d']:5.1f}"
    )
    if cost is not None:
        line += f"  ${cost:.3f}/doc"
    if secs is not None:
        line += f"  {secs:.0f} s"
    print(line)
    if paper:
        print(f"  {'':<28} paper: {paper}")


# ------------------------------------------------------------------ tables


def table1(run_dir: Path):
    print("\n=== Table 1: main results ===")
    metrics = read_metrics(run_dir)
    if len(metrics) != 208:
        print(f"  warning: {len(metrics)} cases under {run_dir}, expected 208")

    fer = [gt_feret(c) for c in metrics]
    wf = [worker_first(m) for m in metrics.values()]
    print_row(
        "GeoPlanAgent",
        summarise([x[0] for x in wf], [x[1] for x in wf], fer),
        paper="89.4 / 0.736 / 0.904 / 67.8 / 4.6 m / 78.8",
    )
    print_row(
        "+ Critic",
        summarise(
            [m["iou"] for m in metrics.values()],
            [m.get("centroid_distance_m") for m in metrics.values()],
            fer,
        ),
        paper="89.9 / 0.740 / 0.906 / 67.8 / 4.6 m / 78.8",
        secs=float(np.mean([m["processing_time"] for m in metrics.values()])),
    )

    # Critic delta quoted in the text (+0.003 mean IoU, 2 cases changed)
    changed = [
        (c, x[0], m["iou"]) for (c, m), x in zip(metrics.items(), wf) if abs(m["iou"] - x[0]) > 1e-9
    ]
    delta = np.mean([m["iou"] for m in metrics.values()]) - np.mean([x[0] for x in wf])
    print(
        f"  critic interventions: {len(changed)} cases "
        f"{[(c, round(a, 3), round(b, 3)) for c, a, b in changed]}, "
        f"mean IoU delta {delta:+.4f} (paper: +0.003)"
    )

    # Collapsed Reader ablation
    nr = read_metrics(REPO / "ablations/no_reader/gemini-flash")
    fer_nr = [gt_feret(c) for c in nr]
    print_row(
        "Collapsed Reader",
        summarise(
            [m["iou"] for m in nr.values()],
            [m.get("centroid_distance_m") for m in nr.values()],
            fer_nr,
        ),
        paper="88.9 / 0.733 / 0.904 / 67.3 / 4.7 m / 78.4",
        secs=float(np.mean([m["processing_time"] for m in nr.values()])),
    )
    tok = np.mean([m["agent_stats"]["total_tokens"] for m in nr.values()])
    tok_main = np.mean([m["agent_stats"]["total_tokens"] for m in metrics.values()])
    print(
        f"  collapsed-reader tokens/case: {tok:.0f} vs {tok_main:.0f} "
        f"= {100 * (tok / tok_main - 1):+.0f}% (paper: +63%)"
    )

    # IoU distribution claims from the text
    iou_wf = np.asarray([x[0] for x in wf])
    print(
        f"  worker-first IoU distribution: "
        f">=0.5 {100 * np.mean(iou_wf >= 0.5):.1f}% (paper 81.3)  "
        f">=0.9 {100 * np.mean(iou_wf >= 0.9):.1f}% (paper 50.5)  "
        f"<0.05 {100 * np.mean(iou_wf < 0.05):.1f}% (paper 12.0)  "
        f"[0.3,0.7] {100 * np.mean((iou_wf >= 0.3) & (iou_wf <= 0.7)):.1f}% (paper 12.5)"
    )

    # VLM end-to-end baselines + GeoPlanAgent on the 40-case subset
    base = REPO / "ablations/vlm_e2e_pdf_to_geojson"
    subset40 = {c["folder"] for c in json.loads((base / "subset_40.json").read_text())["cases"]}

    paper_vlm = {
        ("gemini-flash", 40): "30.0 / 0.053 / 0.000 / 0.0 / 920 m / 2.5 / $0.003 / 5 s",
        ("gemini-pro", 40): "42.5 / 0.112 / 0.000 / 0.0 / 490 m / 7.5 / $0.106* / 75 s",
        ("claude-opus", 40): "22.5 / 0.044 / 0.000 / 0.0 / 1131 m / 0.0 / $0.059 / 7 s",
        ("gpt-5.5-pro", 40): "50.0 / 0.106 / 0.005 / 0.0 / 386 m / 10.0 / $2.855 / 650 s",
        ("gemini-pro", 208): "40.4 / 0.108 / 0.000 / 1.4 / 480 m / 9.6 / $0.106 / 75 s",
    }
    print("\n  VLM end-to-end baselines (PDF -> GeoJSON, single call):")
    for model in ["gemini-flash", "gemini-pro", "claude-opus", "gpt-5.5-pro"]:
        rows = list(csv.DictReader(open(base / model / "results.csv")))
        for n, sel in [
            (40, [r for r in rows if r["case"] in subset40]),
            (208, rows if len(rows) == 208 else None),
        ]:
            if not sel:
                continue
            pin, pout = PRICES[model]
            ious = [float(r["iou"]) for r in sel]
            errs = [
                float(r["centroid_distance_m"]) if r["centroid_distance_m"] else None for r in sel
            ]
            cost = np.mean(
                [
                    (int(r["vlm_request_tokens"]) * pin + int(r["vlm_response_tokens"]) * pout)
                    / 1e6
                    for r in sel
                ]
            )
            secs = np.mean([float(r["call_seconds"]) for r in sel])
            print_row(
                f"{model} ({n})",
                summarise(ious, errs, [gt_feret(r["case"]) for r in sel]),
                paper=paper_vlm.get((model, n)),
                cost=cost,
                secs=secs,
            )
    print("  * paper reuses the 208-run $/doc for the gemini-pro 40 row (subset-exact: $0.108)")

    sub = {c: metrics[c] for c in subset40}
    wf_sub = [worker_first(m) for m in sub.values()]
    print_row(
        "GeoPlanAgent (40 subset)",
        summarise([x[0] for x in wf_sub], [x[1] for x in wf_sub], [gt_feret(c) for c in sub]),
        paper="85.0 / 0.721 / 0.901 / 67.5 / 6.7 m / 80.0",
        secs=float(np.mean([m["processing_time"] for m in sub.values()])),
    )


def table2(run_dir: Path):
    print("\n=== Table 2: locate-stage centroid error ===")

    def stats(errs_m, label, paper):
        a = np.asarray(errs_m, float)
        print(
            f"  {label:<32} n={len(a):<4} median {np.median(a):7.1f} m  "
            f"<500m {100 * np.mean(a < 500):5.1f}%  <1km {100 * np.mean(a < 1000):5.1f}%"
            f"   paper: {paper}"
        )

    rows = [
        ("min_1_tool", "Place only (production)", "176 m / 78.8 / 91.3"),
        ("full", "All 6 geocoder tools", "181 m / 82.2 / 92.8"),
        ("vlm_direct_gemini-flash", "VLM-direct (Flash)", "522 m / 47.1 / 73.6"),
    ]
    for cfg, label, paper in rows:
        path = REPO / "ablations/locate_only_eval" / cfg / "locate_picks.csv"
        errs = [float(r["err_km"]) * 1000 for r in csv.DictReader(open(path)) if r.get("err_km")]
        stats(errs, label, paper)

    # Full-pipeline row: per-case centroid_distance_m under the same
    # pre-critic convention as Table 1.
    metrics = read_metrics(run_dir)
    errs = [worker_first(m)[1] for m in metrics.values()]
    errs = [e if e is not None else float("inf") for e in errs]
    stats(errs, "Full pipeline (+ match_at)", "5 m / 88.9 / 92.3")


def table4():
    print("\n=== Table 4: stratified 40-case subset ===")
    sub = json.loads((REPO / "ablations/vlm_e2e_pdf_to_geojson/subset_40.json").read_text())
    counts: dict[str, int] = {}
    for c in sub["cases"]:
        counts[c["stratum"]] = counts.get(c["stratum"], 0) + 1
    for stratum in sorted(counts):
        print(f"  {stratum:<18} {counts[stratum]}")
    print(
        f"  total {sum(counts.values())}   paper: good_x_easy 15, good_x_medium 12, "
        f"good_x_hard 3, bad_x_easy 4, bad_x_medium 3, bad_x_hard 3"
    )


def _page_to_case(page: str) -> str:
    return re.sub(r"_p\d+$", "", page)


def _fold_table(per_page: dict[str, dict], value_keys: list[str], label: str, paper: str):
    """Collapse page-level k-fold predictions to cases, aggregate per fold."""
    by_case: dict[str, dict] = {}
    for page, rec in per_page.items():
        case = _page_to_case(page)
        slot = by_case.setdefault(case, {"fold": rec["fold"], "vals": []})
        slot["vals"].append([rec[k] for k in value_keys])
    folds: dict[int, list] = {}
    for case, slot in by_case.items():
        folds.setdefault(slot["fold"], []).append(np.mean(slot["vals"], axis=0))

    print(f"  {label} (case-level, pages of one case averaged first):")
    means = []
    for k in sorted(folds):
        arr = np.asarray(folds[k])
        means.append(arr.mean(axis=0))
        cells = "  ".join(f"{v:.4f}" for v in arr.mean(axis=0))
        print(f"    fold {k}: |V|={len(arr):<3} {cells}")
    means = np.asarray(means)
    agg = "  ".join(f"{m:.4f} +/- {s:.4f}" for m, s in zip(means.mean(axis=0), means.std(axis=0)))
    print(f"    mean over folds: {agg}")
    print(f"    paper: {paper}")


def table9():
    print("\n=== Table 9: rotation classifier (5-fold, case-level) ===")
    labels = json.loads((REPO / "training/dataset/rotation_annotations.json").read_text())
    fa = json.loads((REPO / "training/dataset/fold_assignment.json").read_text())
    for fname, label, paper in [
        ("rotation_kfold.json", "single view", "0.9527 +/- 0.0605"),
        ("rotation_kfold_tta.json", "4-way TTA (deployed)", "0.9806 +/- 0.0097"),
    ]:
        preds = json.loads((REPO / "training/eval/predictions" / fname).read_text())
        per_page = {p: {"fold": fa[p], "acc": float(preds[p] == labels[p])} for p in preds}
        _fold_table(per_page, ["acc"], label, paper)


def table11():
    print("\n=== Table 11: SAM3-LoRA out-of-fold segmentation ===")
    per_page = json.loads((REPO / "training/eval/predictions/sam_kfold.json").read_text())
    _fold_table(
        per_page,
        ["sem_iou", "sem_f1"],
        "pixel IoU / F1",
        "IoU 0.9117 +/- 0.0276, F1 0.9410 +/- 0.0241, |V|=43/40/42/41/42",
    )


def table12():
    print("\n=== Table 12: vanilla SAM3 prompt sweep (N=211 pages) ===")
    root = REPO / "results/ablation_sam_base"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        ious = np.asarray([float(r["iou"]) for r in csv.DictReader(open(d / "results.csv"))])
        print(
            f"  {d.name:<28} n={len(ious)}  mean {ious.mean():.3f}  "
            f"median {np.median(ious):.3f}  >=0.5 {100 * np.mean(ious >= 0.5):.1f}%  "
            f">=0.8 {100 * np.mean(ious >= 0.8):.1f}%"
        )
    print("  paper best row (highlighted marked area): 0.611 / 0.808 / 64.0% / 51.2%")


def fig3():
    print("\n=== Figure 3: segmentation method comparison (case-level) ===")

    def case_means(csv_path: Path) -> np.ndarray:
        per_case: dict[str, list[float]] = {}
        for r in csv.DictReader(open(csv_path)):
            if r.get("iou") in (None, ""):
                continue
            per_case.setdefault(_page_to_case(r["case"]), []).append(float(r["iou"]))
        return np.asarray([np.mean(v) for v in per_case.values()])

    sam_lora = json.loads((REPO / "training/eval/predictions/sam_kfold.json").read_text())
    per_case: dict[str, list[float]] = {}
    for page, rec in sam_lora.items():
        per_case.setdefault(_page_to_case(page), []).append(rec["sem_iou"])
    lora = np.asarray([np.mean(v) for v in per_case.values()])

    bars = [
        (
            "VLM-direct (Flash)",
            case_means(REPO / "results/ablation_vlm_seg/gemini-flash/results.csv"),
            "0.52 / 28%",
        ),
        (
            "VLM-direct (Pro)",
            case_means(REPO / "results/ablation_vlm_seg/gemini-pro/results.csv"),
            "0.63 / 41%",
        ),
        (
            "Vanilla SAM3 (best prompt)",
            case_means(REPO / "results/ablation_sam_base/highlighted_marked_area/results.csv"),
            "0.62 / 52%",
        ),
        ("SAM3-LoRA (ours)", lora, "0.91 / 91%"),
    ]
    for label, vals, paper in bars:
        print(
            f"  {label:<28} n={len(vals)}  mean IoU {vals.mean():.4f}  "
            f">=0.8 {100 * np.mean(vals >= 0.8):.1f}%   paper: {paper}"
        )


def fig4(run_dir: Path):
    print("\n=== Figure 4: IoU by document attribute (final, with critic) ===")
    import pandas as pd

    df = pd.read_excel(EVAL / "new_updated.xlsx", sheet_name="Cleaned_up_208_planning_dataset")
    merged = pd.read_excel(EVAL / "new_updated.xlsx", sheet_name="Merged cases")
    bridge = dict(zip(merged["Unnamed: 5"].astype(str), merged["Merged folder"].astype(str)))
    df["folder"] = df["Unique ID (Folder_Name)"].astype(str).map(lambda x: bridge.get(x, x))

    metrics = read_metrics(run_dir)
    df["iou"] = df["folder"].map(lambda f: metrics.get(f, {}).get("iou"))
    missing = df[df["iou"].isna()]["folder"].tolist()
    if missing:
        print(f"  warning: no metrics for {missing}")

    norm = lambda col: df[col].astype(str).str.strip().str.lower()
    df["colour"] = norm("Document Colour").map(lambda x: "yellow" if x == "yellow" else "white")
    df["quality"] = norm("Document Quality")
    df["complexity"] = norm("Shape Complexity")

    total = df["iou"].dropna()
    print(
        f"  total           n={len(total):<4} mean {total.mean():.3f}  "
        f">=0.8 {100 * (total >= 0.8).mean():.1f}%   paper: 0.74 / 68%"
    )
    paper = {
        ("colour", "white"): "0.73 / 66%",
        ("colour", "yellow"): "0.76 / 74%",
        ("quality", "good"): "0.76 / 71%",
        ("quality", "bad"): "0.61 / 50%",
        ("complexity", "easy"): "0.70 / 62%",
        ("complexity", "medium"): "0.85 / 84%",
        ("complexity", "hard"): "0.56 / 39%",
    }
    for col, order in [
        ("colour", ["white", "yellow"]),
        ("quality", ["good", "bad"]),
        ("complexity", ["easy", "medium", "hard"]),
    ]:
        for bucket in order:
            v = df.loc[df[col] == bucket, "iou"].dropna()
            print(
                f"  {col}={bucket:<8} n={len(v):<4} mean {v.mean():.3f}  "
                f">=0.8 {100 * (v >= 0.8).mean():.1f}%   "
                f"paper: {paper[(col, bucket)]}"
            )


def costs(run_dir: Path):
    print("\n=== Costs ($/doc, gemini-flash prices) ===")
    pin, pout = PRICES["gemini-flash"]

    def stage_cost(m, stage):
        s = m.get("agent_stats", {}) or {}
        return (
            int(s.get(f"{stage}_request_tokens", 0) or 0) * pin
            + int(s.get(f"{stage}_response_tokens", 0) or 0) * pout
        ) / 1e6

    metrics = read_metrics(run_dir)
    reader = np.mean([stage_cost(m, "reader") for m in metrics.values()])
    worker = np.mean([stage_cost(m, "worker") for m in metrics.values()])
    print(
        f"  main run ({len(metrics)} cases): reader ${reader:.4f} + "
        f"worker ${worker:.4f} = ${reader + worker:.4f}  (reader+worker only)"
    )

    audit_dir = REPO / "results/cost_audit_v1/gemini-flash"
    if not audit_dir.exists():
        audit_dir = next((REPO / "results/cost_audit_v1").glob("*"), None)
    if audit_dir and audit_dir.exists():
        am = read_metrics(audit_dir)
        parts = {
            s: np.mean([stage_cost(m, s) for m in am.values()])
            for s in ("reader", "worker", "locate")
        }
        total = sum(parts.values())
        print(
            f"  cost audit ({len(am)} cases, locate telemetry on): "
            f"reader ${parts['reader']:.4f} + worker ${parts['worker']:.4f} "
            f"+ locate ${parts['locate']:.4f} = ${total:.4f}"
        )
        print(f"  locate share: {100 * parts['locate'] / total:.0f}%   paper: $0.043/doc total")
    else:
        print("  results/cost_audit_v1 not found -- locate-inclusive cost needs the telemetry run")


def dataset():
    print("\n=== Dataset statistics (Table 3 / Appendix A) ===")
    import pandas as pd

    xlsx = EVAL / "new_updated.xlsx"
    df = pd.read_excel(xlsx, sheet_name="Cleaned_up_208_planning_dataset")
    norm = lambda col: df[col].astype(str).str.strip().str.lower()

    print(f"  cases: {len(df)} (paper: 208)")
    for col, paper in [
        ("Shape Complexity", "easy 108, medium 77, hard 23 (52/37/11%)"),
        ("Document Quality", "good 176, bad 32 (85/15%)"),
    ]:
        vc = norm(col).value_counts()
        cells = ", ".join(f"{k} {v} ({100 * v / len(df):.0f}%)" for k, v in vc.items())
        print(f"  {col.lower()}: {cells}   paper: {paper}")
    colour = (
        norm("Document Colour").map(lambda x: "yellow" if x == "yellow" else "white").value_counts()
    )
    cells = ", ".join(f"{k} {v}" for k, v in colour.items())
    print(
        f"  colour (2 odd colours folded into white, as in the figure): "
        f"{cells}   paper: white 170, yellow 38"
    )

    dates = pd.to_datetime(df["Document Date"], errors="coerce", dayfirst=True)
    years = dates.dt.year
    extra = df.loc[years.isna(), "Document Date"].astype(str).str.extract(r"(\d{4})")[0]
    years = years.fillna(pd.to_numeric(extra, errors="coerce"))
    print(
        f"  document years: {int(years.min())}-{int(years.max())}, "
        f"median {years.median():.0f} (paper: 1958-2025, median 1997)"
    )

    n_pages = len(json.loads((REPO / "training/eval/predictions/sam_kfold.json").read_text()))
    print(
        f"  annotated map pages in the segmentation pool: {n_pages} "
        f"(paper: 211; 208 cases after multi-page collapse)"
    )


# -------------------------------------------------------------------- main

SECTIONS = {
    "table1": lambda a: table1(a.run_dir),
    "table2": lambda a: table2(a.run_dir),
    "table4": lambda a: table4(),
    "table9": lambda a: table9(),
    "table11": lambda a: table11(),
    "table12": lambda a: table12(),
    "fig3": lambda a: fig3(),
    "fig4": lambda a: fig4(a.run_dir),
    "costs": lambda a: costs(a.run_dir),
    "dataset": lambda a: dataset(),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "sections",
        nargs="+",
        choices=[*SECTIONS, "all"],
        metavar="section",
        help=f"one or more of: {' '.join(SECTIONS)} all",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN,
        help="benchmark run to aggregate (default: %(default)s)",
    )
    args = ap.parse_args()

    wanted = list(SECTIONS) if "all" in args.sections else args.sections
    for name in wanted:
        SECTIONS[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
