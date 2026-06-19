"""Compute every paper table/figure number from the cached run artifacts on disk.

Run with no arguments to compute every section; pass section names to compute a
subset. Sections whose input data isn't on disk are skipped.

    uv run scripts/compute_tables.py                       # everything
    uv run scripts/compute_tables.py table1 costs          # just these
    uv run scripts/compute_tables.py --run-dir results/...  # a different run
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from _pricing import PRICES, token_cost  # noqa: E402 (scripts/ on sys.path when run as a file)
from geoplanagent.metrics import (  # noqa: E402
    feret_diameter_m,
    aggregate_spatial_metrics,
    geojson_to_shape,
    load_case_ground_truth,
    load_run_metrics,
    load_sam_iou_by_case,
    pre_critic_iou_by_case,
    seg_iou_by_case,
    worker_first,
)
from geoplanagent.paths import (  # noqa: E402
    DATA_DIR,
    FOLD_ASSIGNMENT,
    MAIN_RUN_DIR,
    TRAINING_DATASET_DIR,
    EVAL_PREDICTIONS_DIR,
    SAM_KFOLD_PREDICTIONS,
    ABL_NO_READER,
    ABL_VLM_E2E,
    ABL_LOCATE_ONLY,
    ABL_SAM_BASE,
    ABL_VLM_SEG,
    VLM_E2E_SUBSET,
)
from geoplanagent.utils import (  # noqa: E402
    route_key,
    page_to_case,
    aggregate_pages_to_cases,
    load_case_labels,
    CASE_LABEL_BUCKETS,
)


# ---------------------------------------------------------------- geometry


_feret_cache: dict[str, float] = {}


def gt_feret(case: str) -> float:
    if case not in _feret_cache:
        gt = load_case_ground_truth(DATA_DIR / case)
        _feret_cache[case] = feret_diameter_m(geojson_to_shape(gt)) if gt else 0.0
    return _feret_cache[case]


# ------------------------------------------------------------- run loading


def print_row(label, stats, cost, secs):
    print(
        f"  {label:<28} n={stats['n_cases']:<4} %IoU>0 {stats['pct_grt_0']:5.1f}  "
        f"mean {stats['mean_IoU']:.3f}  med {stats['median_IoU']:.3f}  "
        f"%>=0.8 {stats['pct_grt_08']:5.1f}  medErr {stats['median_centroid_distance_m']:7.1f} m  "
        f"Acc@0.1D {stats['acc_01d']:5.1f}  ${cost:.3f}/doc  {secs:.0f} s"
    )


def _stage_cost(case_metrics, stage) -> float:
    """$/doc for one pipeline stage (reader/worker/locate) of one case (gemini-flash prices)."""
    agent_stats = case_metrics.get("agent_stats", {}) or {}
    pin, pout = PRICES["gemini-flash"]
    return token_cost(
        int(agent_stats.get(f"{stage}_request_tokens", 0) or 0),
        int(agent_stats.get(f"{stage}_response_tokens", 0) or 0),
        pin,
        pout,
    )


def _critic_cost(case_metrics) -> float:
    """$/doc of the critic's own LLM call(s), from agent_stats.critic.tokens."""
    tokens = ((case_metrics.get("agent_stats") or {}).get("critic") or {}).get("tokens") or {}
    pin, pout = PRICES["gemini-flash"]
    return token_cost(int(tokens.get("request", 0) or 0), int(tokens.get("response", 0) or 0), pin, pout)


def _critic_wall(case_metrics) -> float:
    """Wall-clock seconds the critic spent on one case (0 if it didn't run)."""
    critic = (case_metrics.get("agent_stats") or {}).get("critic") or {}
    return sum(iteration.get("wall_s") or 0.0 for iteration in (critic.get("iterations") or []))


def _doc_cost(metrics, with_critic=False) -> float:
    """Mean $/doc over a run: reader+worker+locate, plus the critic when with_critic."""
    return float(
        np.mean(
            [
                sum(_stage_cost(case_metrics, stage) for stage in ("reader", "worker", "locate"))
                + (_critic_cost(case_metrics) if with_critic else 0.0)
                for case_metrics in metrics.values()
            ]
        )
    )


def _doc_time(metrics, with_critic=False) -> float:
    """Mean wall-clock seconds per case; drop the critic's wall_s when with_critic is False."""
    return float(
        np.mean(
            [
                case_metrics["processing_time"] - (0.0 if with_critic else _critic_wall(case_metrics))
                for case_metrics in metrics.values()
            ]
        )
    )


# ------------------------------------------------------------------ tables


def table1(run_dir: Path):
    print("\n=== Table 1: main results ===")
    metrics = load_run_metrics(run_dir)
    if len(metrics) != 208:
        print(f"  warning: {len(metrics)} cases under {run_dir}, expected 208")

    ferets = [gt_feret(case) for case in metrics]
    worker_firsts = [worker_first(case_metrics) for case_metrics in metrics.values()]
    # GeoPlanAgent is the pre-critic result: cost/time exclude the critic.
    # "+ Critic" adds the critic's own tokens (agent_stats.critic.tokens) and wall_s.
    print_row(
        "GeoPlanAgent",
        aggregate_spatial_metrics(
            [iou for iou, _err in worker_firsts],
            [err for _iou, err in worker_firsts],
            ferets,
        ),
        cost=_doc_cost(metrics),
        secs=_doc_time(metrics),
    )
    print_row(
        "+ Critic",
        aggregate_spatial_metrics(
            [case_metrics["iou"] for case_metrics in metrics.values()],
            [case_metrics.get("centroid_distance_m") for case_metrics in metrics.values()],
            ferets,
        ),
        cost=_doc_cost(metrics, with_critic=True),
        secs=_doc_time(metrics, with_critic=True),
    )

    # Cases where the critic changed the final IoU (for the text's delta claim).
    changed = [
        (case, wf_iou, case_metrics["iou"])
        for (case, case_metrics), (wf_iou, _wf_err) in zip(metrics.items(), worker_firsts)
        if abs(case_metrics["iou"] - wf_iou) > 1e-9
    ]
    delta = np.mean([case_metrics["iou"] for case_metrics in metrics.values()]) - np.mean(
        [iou for iou, _err in worker_firsts]
    )
    print(
        f"  critic interventions: {len(changed)} cases "
        f"{[(case, round(before, 3), round(after, 3)) for case, before, after in changed]}, "
        f"mean IoU delta {delta:+.4f}"
    )

    # Collapsed Reader ablation (optional — its own run dir may not be on disk)
    try:
        nr_metrics = load_run_metrics(ABL_NO_READER / "gemini-flash")
        print_row(
            "Collapsed Reader",
            aggregate_spatial_metrics(
                [case_metrics["iou"] for case_metrics in nr_metrics.values()],
                [case_metrics.get("centroid_distance_m") for case_metrics in nr_metrics.values()],
                [gt_feret(case) for case in nr_metrics],
            ),
            cost=_doc_cost(nr_metrics),
            secs=_doc_time(nr_metrics),
        )
        nr_tokens = np.mean([case_metrics["agent_stats"]["total_tokens"] for case_metrics in nr_metrics.values()])
        main_tokens = np.mean([case_metrics["agent_stats"]["total_tokens"] for case_metrics in metrics.values()])
        print(
            f"  collapsed-reader tokens/case: {nr_tokens:.0f} vs {main_tokens:.0f} "
            f"= {100 * (nr_tokens / main_tokens - 1):+.0f}%"
        )
    except FileNotFoundError:
        print("  Collapsed Reader: skipped — ablation run not on disk")

    # VLM end-to-end baselines + GeoPlanAgent on the 40-case subset
    subset_cases = {case["folder"] for case in json.loads(VLM_E2E_SUBSET.read_text())["cases"]}
    print("\n  VLM end-to-end baselines (PDF -> GeoJSON, single call):")
    for model in ["gemini-flash", "gemini-pro", "claude-opus", "gpt-5.5-pro"]:
        rows = list(csv.DictReader(open(ABL_VLM_E2E / model / "results.csv")))
        for n_cases, selected in [
            (40, [row for row in rows if row["case"] in subset_cases]),
            (208, rows if len(rows) == 208 else None),
        ]:
            if not selected:
                continue
            pin, pout = PRICES[model]
            ious = [float(row["iou"]) for row in selected]
            errs = [
                float(row["centroid_distance_m"]) if row["centroid_distance_m"] else None
                for row in selected
            ]
            cost = np.mean(
                [
                    token_cost(int(row["vlm_request_tokens"]), int(row["vlm_response_tokens"]), pin, pout)
                    for row in selected
                ]
            )
            secs = np.mean([float(row["call_seconds"]) for row in selected])
            print_row(
                f"{model} ({n_cases})",
                aggregate_spatial_metrics(ious, errs, [gt_feret(row["case"]) for row in selected]),
                cost=cost,
                secs=secs,
            )

    subset_metrics = {case: metrics[case] for case in subset_cases}
    subset_worker_firsts = [worker_first(case_metrics) for case_metrics in subset_metrics.values()]
    print_row(
        "GeoPlanAgent (40 subset)",
        aggregate_spatial_metrics(
            [iou for iou, _err in subset_worker_firsts],
            [err for _iou, err in subset_worker_firsts],
            [gt_feret(case) for case in subset_metrics],
        ),
        cost=_doc_cost(subset_metrics),
        secs=_doc_time(subset_metrics),
    )


def table2(run_dir: Path):
    print("\n=== Table 2: locate-stage centroid error ===")

    def stats(errs_m, label):
        errs_arr = np.asarray(errs_m, float)
        print(
            f"  {label:<32} n={len(errs_arr):<4} median {np.median(errs_arr):7.1f} m  "
            f"<500m {100 * np.mean(errs_arr < 500):5.1f}%  <1km {100 * np.mean(errs_arr < 1000):5.1f}%"
        )

    for config, label in [
        ("min_1_tool", "Place only (production)"),
        ("full", "All 6 geocoder tools"),
        ("vlm_direct_gemini-flash", "VLM-direct (Flash)"),
        ("vlm_direct_gemini-pro", "VLM-direct (Pro)"),
    ]:
        path = ABL_LOCATE_ONLY / config / "locate_picks.csv"
        errs = [float(row["err_km"]) * 1000 for row in csv.DictReader(open(path)) if row.get("err_km")]
        stats(errs, label)

    # Full-pipeline row: per-case centroid_distance_m under the same
    # pre-critic convention as Table 1.
    metrics = load_run_metrics(run_dir)
    errs = [worker_first(case_metrics)[1] for case_metrics in metrics.values()]
    errs = [err if err is not None else float("inf") for err in errs]
    stats(errs, "Full pipeline (+ match_at)")


def table4():
    print("\n=== Table 4: stratified 40-case subset ===")
    subset = json.loads(VLM_E2E_SUBSET.read_text())
    counts: dict[str, int] = {}
    for case in subset["cases"]:
        counts[case["stratum"]] = counts.get(case["stratum"], 0) + 1
    for stratum in sorted(counts):
        print(f"  {stratum:<18} {counts[stratum]}")
    print(f"  total {sum(counts.values())}")


def _fold_table(per_page: dict[str, dict], value_keys: list[str], label: str):
    """Collapse page-level k-fold predictions to cases, aggregate per fold."""
    case_vals = aggregate_pages_to_cases(
        {page: [rec[key] for key in value_keys] for page, rec in per_page.items()}
    )
    case_fold = {page_to_case(page): rec["fold"] for page, rec in per_page.items()}
    folds: dict[int, list] = {}
    for case, vals in case_vals.items():
        folds.setdefault(case_fold[case], []).append(vals)

    print(f"  {label}:")
    means = []
    for fold_id in sorted(folds):
        fold_values = np.asarray(folds[fold_id])
        means.append(fold_values.mean(axis=0))
        cells = "  ".join(f"{val:.4f}" for val in fold_values.mean(axis=0))
        print(f"    fold {fold_id}: |V|={len(fold_values):<3} {cells}")
    means = np.asarray(means)
    agg = "  ".join(
        f"{metric_mean:.4f} +/- {metric_std:.4f}"
        for metric_mean, metric_std in zip(means.mean(axis=0), means.std(axis=0))
    )
    print(f"    mean over folds: {agg}")


def table9():
    print("\n=== Table 9: rotation classifier (5-fold, case-level) ===")
    labels = json.loads((TRAINING_DATASET_DIR / "rotation_annotations.json").read_text())
    fold_assignment = json.loads(FOLD_ASSIGNMENT.read_text())
    for filename, label in [
        ("rotation_kfold.json", "single view"),
        ("rotation_kfold_tta.json", "4-way TTA (deployed)"),
    ]:
        preds = json.loads((EVAL_PREDICTIONS_DIR / filename).read_text())
        fold_by_route = {route_key(key): fold for key, fold in fold_assignment.items()}
        per_page = {
            page: {"fold": fold_by_route[route_key(page)], "acc": float(preds[page] == labels[page])}
            for page in preds
        }
        _fold_table(per_page, ["acc"], label)


def table11():
    print("\n=== Table 11: SAM3-LoRA out-of-fold segmentation ===")
    per_page = json.loads(SAM_KFOLD_PREDICTIONS.read_text())
    _fold_table(per_page, ["sem_iou", "sem_f1"], "pixel IoU / F1")


def table12():
    print("\n=== Table 12: vanilla SAM3 prompt sweep (208 cases) ===")
    for prompt_dir in sorted(p for p in ABL_SAM_BASE.iterdir() if p.is_dir()):
        ious = seg_iou_by_case(prompt_dir / "results.csv")
        print(
            f"  {prompt_dir.name:<28} n={len(ious)}  mean {ious.mean():.3f}  "
            f"median {np.median(ious):.3f}  >=0.5 {100 * np.mean(ious >= 0.5):.1f}%  "
            f">=0.8 {100 * np.mean(ious >= 0.8):.1f}%"
        )


def fig3():
    print("\n=== Figure 3: segmentation method comparison (case-level) ===")
    lora = np.asarray(list(load_sam_iou_by_case().values()))
    bars = [
        ("VLM-direct (Flash)", seg_iou_by_case(ABL_VLM_SEG / "gemini-flash" / "results.csv")),
        ("VLM-direct (Pro)", seg_iou_by_case(ABL_VLM_SEG / "gemini-pro" / "results.csv")),
        ("Vanilla SAM3 (best prompt)", seg_iou_by_case(ABL_SAM_BASE / "highlighted_marked_area" / "results.csv")),
        ("SAM3-LoRA (ours)", lora),
    ]
    for label, vals in bars:
        print(
            f"  {label:<28} n={len(vals)}  mean IoU {vals.mean():.4f}  "
            f">=0.8 {100 * np.mean(vals >= 0.8):.1f}%"
        )


def fig4(run_dir: Path):
    print("\n=== Figure 4: IoU by document attribute (pre-critic) ===")
    df = load_case_labels()
    df["iou"] = df["folder"].map(pre_critic_iou_by_case(run_dir))
    missing = df[df["iou"].isna()]["folder"].tolist()
    if missing:
        print(f"  warning: no metrics for {missing}")

    total = df["iou"].dropna()
    print(f"  total           n={len(total):<4} mean {total.mean():.3f}  >=0.8 {100 * (total >= 0.8).mean():.1f}%")
    for col, order in CASE_LABEL_BUCKETS.items():
        n_bucketed = 0
        for bucket in order:
            bucket_ious = df.loc[df[col] == bucket, "iou"].dropna()
            n_bucketed += len(bucket_ious)
            print(
                f"  {col}={bucket:<8} n={len(bucket_ious):<4} mean {bucket_ious.mean():.3f}  "
                f">=0.8 {100 * (bucket_ious >= 0.8).mean():.1f}%"
            )
        # Every scored case must land in exactly one bucket — guards against the
        # "?"-label drop that silently shrank these breakdowns.
        assert n_bucketed == len(total), (
            f"{col}: buckets {order} cover {n_bucketed} scored cases but Total is "
            f"{len(total)} — a case fell outside the buckets (unexpected label?)"
        )


# -------------------------------------------------------------------- main

SECTIONS = {
    "table1": lambda args: table1(args.run_dir),
    "table2": lambda args: table2(args.run_dir),
    "table4": lambda args: table4(),
    "table9": lambda args: table9(),
    "table11": lambda args: table11(),
    "table12": lambda args: table12(),
    "fig3": lambda args: fig3(),
    "fig4": lambda args: fig4(args.run_dir),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sections",
        nargs="*",
        metavar="section",
        help=f"sections to compute (default: all). one or more of: {' '.join(SECTIONS)} all",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=MAIN_RUN_DIR,
        help="benchmark run to aggregate (default: %(default)s)",
    )
    args = parser.parse_args()

    sections = args.sections or ["all"]
    unknown = [section for section in sections if section != "all" and section not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choose from: {' '.join([*SECTIONS, 'all'])}")
    wanted = list(SECTIONS) if "all" in sections else sections
    for name in wanted:
        try:
            SECTIONS[name](args)
        except FileNotFoundError as error:
            print(f"\n=== {name}: skipped — input data not on disk ({error}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
