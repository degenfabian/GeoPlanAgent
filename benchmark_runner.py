"""
Benchmark Runner - Evaluate planning document GeoJSON extraction

Runs the unified tool-calling agent on the evaluation dataset.
The agent reads each PDF, geocodes locations, positions the map via MINIMA,
extracts boundaries with SAM3, and verifies — all through tool calls.

Usage:
    uv run benchmark_runner.py --model gemini-flash --enable-critic   # paper configuration
    uv run benchmark_runner.py --max-cases 5                          # quick smoke test
    uv run benchmark_runner.py --cases 12:00116:ART4                  # specific case
    uv run benchmark_runner.py --max-iterations 3                     # limit agent turns
"""

import time
import json
import traceback
import pandas as pd
from pathlib import Path, PurePosixPath
from datetime import datetime

from geoplanagent.tools.pdf import resolve_case_pdf
from geoplanagent.metrics import load_geojson, calculate_spatial_metrics, aggregate_stats
from geoplanagent.utils import PRODUCTION_LOCATE_DISABLED_TOOLS

# Duplicates removed from disk; filtered out of the dataset at load time.
DUPLICATE_SL_NOS = {9, 68, 83, 232, 253}


# Model Loading


def load_models():
    """Load SAM3 fine-tuned model and MINIMA matcher."""
    from geoplanagent.tools.segment import load_sam3_ft
    from geoplanagent.tools.matching import load_minima

    state = {}
    state["sam3_ft"] = load_sam3_ft()
    state["minima"] = load_minima()
    return state


# Main Runner


def _load_dataset(dataset_path, eval_dir, only_cases, start_from, max_cases):
    """Build the case list: Excel rows that exist under eval_dir, plus
    *_merged folders on disk that aren't in the spreadsheet, minus the
    physically-removed duplicates; then apply the case/range filters."""
    dataset = pd.read_excel(dataset_path, sheet_name="0_planning_dataset_list")
    n_total = len(dataset)

    # Filter to cases that exist in eval_dir
    eval_path = Path(eval_dir)
    dataset = dataset[
        dataset["Unique ID (Folder_Name)"].apply(lambda f: (eval_path / str(f)).exists())
    ]
    n_exists = len(dataset)

    # Inject *_merged folders that exist on disk but aren't in the Excel.
    # Each merged folder is its own complete case (own PDF + GT geojson).
    excel_folders = set(dataset["Unique ID (Folder_Name)"].astype(str))
    merged_extras = []
    for sub in sorted(eval_path.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_merged"):
            continue
        if sub.name in excel_folders:
            continue
        gj = sub / f"{sub.name}.geojson"
        if not gj.exists():
            continue
        merged_extras.append(
            {
                "Sl no": 9000 + len(merged_extras) + 1,
                "Unique ID (Folder_Name)": sub.name,
                "geojson ID (for sanity check)": gj.name,
            }
        )
    if merged_extras:
        dataset = pd.concat([dataset, pd.DataFrame(merged_extras)], ignore_index=True)
        print(
            f"Injected {len(merged_extras)} *_merged cases not in Excel: "
            f"{[e['Unique ID (Folder_Name)'] for e in merged_extras]}"
        )

    # Drop physically-removed duplicates. There is no training-case
    # exclusion: k-fold SAM3 routes each case to its held-out fold's
    # adapter at inference, so cases that appear in the training pool
    # are still scored leak-free.
    dataset = dataset[~dataset["Sl no"].isin(DUPLICATE_SL_NOS)]
    print(
        f"Dataset: {len(dataset)} cases "
        f"({n_total} in Excel, {n_total - n_exists} missing from disk, "
        f"{len(DUPLICATE_SL_NOS)} duplicates dropped)"
    )

    # Filter to specific cases if requested
    if only_cases:
        dataset = dataset[dataset["Unique ID (Folder_Name)"].apply(lambda f: str(f) in only_cases)]
        print(f"Filtered to {len(dataset)} specific cases: {only_cases}")
    else:
        dataset = dataset.iloc[start_from:]
        if max_cases:
            dataset = dataset.head(max_cases)

    return dataset


def _run_case(
    row,
    case_idx,
    n_cases,
    eval_path,
    output_path,
    models_state,
    all_results,
    *,
    model_name,
    dpi,
    max_iterations,
    force,
    enable_critic,
    critic_max_iters,
    locate_model_name,
    locate_disabled_tools,
    folded,
):
    """Run one case (or load it from cache), appending its summary row to
    ``all_results``. Returns True only on a fatal error that should stop
    the whole benchmark (invalid model ID)."""
    from geoplanagent.run import run_agent

    folder_name = str(row["Unique ID (Folder_Name)"])
    sl_no = int(row["Sl no"])
    # The xlsx cells were authored on POSIX; PurePosixPath always uses
    # '/' regardless of host OS, so the basename extraction works the
    # same on Windows or Linux.
    geojson_file = PurePosixPath(str(row["geojson ID (for sanity check)"])).name

    print(f"\n{'─' * 70}")
    print(f"[{case_idx + 1}/{n_cases}] Sl {sl_no}: {folder_name}")

    folder_path = eval_path / folder_name
    pdf_path = resolve_case_pdf(folder_path)
    if pdf_path is None:
        print("  SKIP: no PDF")
        all_results.append({"folder": folder_name, "sl_no": sl_no, "error": "no PDF"})
        return False
    gt_files = list(folder_path.glob(geojson_file))
    if not gt_files:
        gt_files = list(folder_path.glob("*.geojson"))
    gt_geojson = load_geojson(str(gt_files[0])) if gt_files else None

    # Check for cached result
    case_dir = output_path / folder_name
    cached_metrics = case_dir / "metrics.json"
    if cached_metrics.exists() and not force:
        prev = json.loads(cached_metrics.read_text())
        # Cache-mode mismatch detection: a cached entry that contains
        # worker_first_iou came from a --enable-critic run. If the
        # current invocation is in a different mode, the cached IoU
        # is not comparable — force re-run to avoid silently mixing
        # critic and no-critic results. district_lookup cases are
        # mode-agnostic: the critic never runs on them, so their cached
        # entry has no worker_first_iou in either mode and stays valid.
        cached_had_critic = "worker_first_iou" in prev
        is_district = (prev.get("agent_stats") or {}).get("outcome_status") == "district_lookup"
        if cached_had_critic != enable_critic and not is_district:
            print(
                f"  [cache mode mismatch — re-running] "
                f"cached_had_critic={cached_had_critic} "
                f"current={enable_critic}"
            )
        else:
            print(f"  [cached] IoU={prev.get('iou', 0):.3f}")
            all_results.append(
                {
                    "folder": folder_name,
                    "sl_no": sl_no,
                    **{k: v for k, v in prev.items() if k not in ("sl_no",)},
                }
            )
            return False

    # ── Run the agent ──
    try:
        t0 = time.time()
        result = run_agent(
            pdf_path=str(pdf_path),
            models_state=models_state,
            model_name=model_name,
            max_iterations=max_iterations,
            dpi=dpi,
            verbose=True,
            case_name=folder_name,
            case_dir=case_dir,
            enable_critic=enable_critic,
            critic_max_iters=critic_max_iters,
            locate_model_name=locate_model_name,
            locate_disabled_tools=locate_disabled_tools,
            folded=folded,
        )
        dt = time.time() - t0

        if not result.get("success"):
            err = result.get("error", "agent failed")
            print(f"  Error: {err}")
            # Fail fast only on invalid model ID (don't waste 200 cases)
            if "not a valid model ID" in str(err):
                print("\n  FATAL: Invalid model ID, stopping benchmark.")
                return True
            # Still save what we can from failed cases
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "sl_no": sl_no,
                        "error": err,
                        "processing_time": dt,
                        "agent_stats": result.get("agent_stats", {}),
                    },
                    indent=2,
                    default=str,
                )
            )
            msg_log = result.get("message_log", [])
            if msg_log:
                (case_dir / "message_log.json").write_text(
                    json.dumps(msg_log, indent=2, default=str)
                )
            # Save partial geojson if any
            partial_gj = result.get("geojson")
            if partial_gj:
                (case_dir / "predicted.geojson").write_text(json.dumps(partial_gj, indent=2))
            all_results.append(
                {
                    "folder": folder_name,
                    "sl_no": sl_no,
                    "error": err,
                    "processing_time": dt,
                }
            )
            return False

        geojson = result.get("geojson")
        mi = result.get("match_info", {})

        # Compute metrics on the (final) geojson — this is the
        # critic_iou when the critic is enabled, or the worker_iou
        # when it isn't.
        metrics = {}
        if gt_geojson and geojson:
            metrics = calculate_spatial_metrics(gt_geojson, geojson)

        iou = metrics.get("iou", 0)

        # When critic was enabled, also compute the worker's
        # first-commit IoU (no-critic baseline) and stash it.
        worker_first_iou = None
        worker_first_metrics = None
        worker_first_gj = result.get("worker_first_geojson")
        if worker_first_gj is not None and gt_geojson:
            worker_first_metrics = calculate_spatial_metrics(gt_geojson, worker_first_gj)
            worker_first_iou = worker_first_metrics.get("iou")

        if worker_first_iou is not None:
            delta = (iou or 0) - (worker_first_iou or 0)
            print(
                f"  IoU={iou:.3f} (critic) vs {worker_first_iou:.3f} "
                f"(worker_first) Δ={delta:+.3f}  "
                f"inliers={mi.get('n_inliers', 0)}  t={dt:.1f}s  "
                f"reason={result.get('agent_reason', '')[:60]}"
            )
        else:
            print(
                f"  IoU={iou:.3f}  inliers={mi.get('n_inliers', 0)}  "
                f"t={dt:.1f}s  reason={result.get('agent_reason', '')[:60]}"
            )

        # Persist the only two release artifacts: the predicted boundary
        # and the scores/telemetry. Nothing else is written to disk —
        # per-case visualisation is on-demand via scripts/visualize_case.py.
        case_dir.mkdir(parents=True, exist_ok=True)
        if geojson:
            (case_dir / "predicted.geojson").write_text(json.dumps(geojson, indent=2))

        # Core metrics (used for cache-hit detection on re-runs)
        metrics_payload = {
            "sl_no": sl_no,
            "match_info": mi,
            "processing_time": dt,
            "agent_stats": result.get("agent_stats", {}),
            **metrics,
        }
        # Paired no-critic / with-critic snapshot from a single run
        # (only present when --enable-critic was set).
        if worker_first_metrics is not None:
            metrics_payload["worker_first_iou"] = worker_first_iou
            metrics_payload["worker_first_metrics"] = worker_first_metrics
        (case_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=str))

        # Record the result before any optional side-effect writes
        # that could raise and wrongly tag this case as a crash;
        # metrics.json is already on disk at this point.
        # Mirror metrics.json's full payload (minus sl_no) so the
        # fresh-run per_case entry has the SAME schema as the
        # cache-hit path's entry — otherwise summary.json["per_case"]
        # is sparse for fresh runs and complete for cached runs,
        # which breaks downstream analyses that read e.g.
        # ``worker_first_iou`` from per_case.
        all_results.append(
            {
                "folder": folder_name,
                "sl_no": sl_no,
                **{k: v for k, v in metrics_payload.items() if k != "sl_no"},
            }
        )

    except Exception as e:
        traceback.print_exc()
        all_results.append({"folder": folder_name, "sl_no": sl_no, "error": str(e)})
    return False


def run_benchmark(
    model_name,
    output_dir,
    max_cases=None,
    start_from=0,
    dpi=200,
    max_iterations=12,
    dataset_path="evaluation_data/0_planning_dataset_list.xlsx",
    eval_dir="evaluation_data",
    only_cases=None,
    force=False,
    enable_critic=False,
    critic_max_iters=2,
    locate_model_name="google/gemini-3-flash-preview",
    locate_disabled_tools=PRODUCTION_LOCATE_DISABLED_TOOLS,
    folded=False,
):
    """Run benchmark using the unified tool-calling agent.

    Args:
        model_name: OpenRouter model identifier (reader + worker).
        output_dir: Base output directory for results.
        max_cases: Limit number of cases to run.
        start_from: Skip first N cases.
        dpi: PDF rendering DPI.
        max_iterations: Max agent turns per case.
        only_cases: If set, only run these specific folder names.
        locate_model_name: Model for the locate sub-agent (independent of
            model_name). Default google/gemini-3-flash-preview.
    """
    dataset = _load_dataset(dataset_path, eval_dir, only_cases, start_from, max_cases)
    print(f"Running: {len(dataset)} cases\n")

    models_state = load_models()

    eval_path = Path(eval_dir)
    output_path = Path(output_dir) / model_name.replace("/", "_")
    all_results = []

    for case_idx, (_, row) in enumerate(dataset.iterrows()):
        fatal = _run_case(
            row,
            case_idx,
            len(dataset),
            eval_path,
            output_path,
            models_state,
            all_results,
            model_name=model_name,
            dpi=dpi,
            max_iterations=max_iterations,
            force=force,
            enable_critic=enable_critic,
            critic_max_iters=critic_max_iters,
            locate_model_name=locate_model_name,
            locate_disabled_tools=locate_disabled_tools,
            folded=folded,
        )
        if fatal:
            break

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {model_name}")
    print(f"{'=' * 70}")

    summary = _compute_summary(all_results)
    summary_path = output_path / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    s = summary
    print(
        f"\n  {s['polygons_produced']} polygons / {s['no_polygon']} no-polygon / "
        f"{s['crashed']} crashed   (total {s['total']})"
    )
    if s.get("metrics") and s["metrics"].get("iou"):
        m = s["metrics"]["iou"]
        print(f"  IoU (failures=0):    mean={m['mean']:.3f}  median={m['median']:.3f}")
    if s.get("metrics_successful_only") and s["metrics_successful_only"].get("iou"):
        m2 = s["metrics_successful_only"]["iou"]
        print(f"  IoU (polygon-only):  mean={m2['mean']:.3f}  median={m2['median']:.3f}")


def _compute_summary(results):
    """Compute aggregate stats.

    Headline metrics count every failure mode as IoU=0: cases that ran but
    produced no polygon, and cases that crashed outright. Either way the
    operator ends up with no result, so both score 0 in the denominator.

    The 'successful_only' block restricts to cases that produced a
    polygon, which separates "did the produced polygons score well" from
    "how often did we produce one at all".
    """
    crashes = [r for r in results if "error" in r]
    non_crashed = [r for r in results if "error" not in r]
    honest = [(r["iou"] if r.get("iou") is not None else 0.0) for r in non_crashed] + [0.0] * len(
        crashes
    )
    no_polygon = [r for r in non_crashed if r.get("iou") is None]
    polygons = [r for r in non_crashed if r.get("iou") is not None]

    summary = {
        "total": len(results),
        "polygons_produced": len(polygons),
        "no_polygon": len(no_polygon),
        "crashed": len(crashes),
        "timestamp": datetime.now().isoformat(),
    }

    if honest:
        # Production-honest IoU (rejections counted as 0).
        summary["metrics"] = {"iou": aggregate_stats(honest)}

        if polygons:
            summary["metrics_successful_only"] = {
                "iou": aggregate_stats([r["iou"] for r in polygons]),
                "precision": aggregate_stats([r["precision"] for r in polygons]),
                "recall": aggregate_stats([r["recall"] for r in polygons]),
            }
            pos_errs = [
                r["centroid_distance_m"]
                for r in polygons
                if r.get("centroid_distance_m") is not None
            ]
            if pos_errs:
                summary["metrics_successful_only"]["centroid_distance_m"] = aggregate_stats(pos_errs)

    summary["per_case"] = results
    return summary


# CLI

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark planning document GeoJSON extraction")
    parser.add_argument(
        "--model",
        default="gemini-flash",
        help="OpenRouter model identifier (reader + worker). "
        "Default matches the paper configuration.",
    )
    parser.add_argument(
        "--locate-model",
        default="google/gemini-3-flash-preview",
        help="Model alias or OpenRouter identifier for the locate "
        "sub-agent (independent of --model). Default: "
        "google/gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--locate-disabled-tools",
        default=None,
        help="Comma-separated locate-agent tools to disable for the "
        "locate sub-agent (e.g. 'la_check' or "
        "'postcode,grid_ref,road,intersect,la_check' for min_1_tool, "
        "or '' for the full 6-tool kit). Default (flag not passed) = "
        "production place-only kit. Vocabulary: postcode, grid_ref, "
        "place, road, intersect, la_check.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=12, help="Max agent turns per case")
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument(
        "--cases", nargs="+", default=None, help="Only run these specific case folder names"
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if cached results exist")
    parser.add_argument(
        "--enable-critic",
        action="store_true",
        help="Run an independent LLM critic after the worker "
        "submits. The critic compares all stored "
        "match candidates (pairwise) and may direct the "
        "worker to switch candidates or re-locate. The "
        "worker's first-commit polygon is also captured "
        "(snapshot) so metrics.json carries paired "
        "no-critic and with-critic IoUs from one run.",
    )
    parser.add_argument(
        "--critic-max-iters",
        type=int,
        default=2,
        help="Max critic-rejection iterations per case "
        "before forcing accept. Ignored without "
        "--enable-critic.",
    )
    parser.add_argument(
        "--no-reader",
        action="store_true",
        help="Folded ablation: skip the dedicated reader "
        "phase. The worker receives the PDF binary "
        "and must call submit_pdf_info as its first "
        "tool call to populate PDFInfo before "
        "positioning. Suggested --output-dir: "
        "ablations/no_reader/.",
    )
    args = parser.parse_args()

    # Flag not passed → fall through to run_benchmark's production default
    # (place-only). Flag passed (even as empty string) → use the requested
    # kit explicitly, including the empty-string case for the full 6-tool kit.
    run_kwargs = dict(
        model_name=args.model,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        start_from=args.start_from,
        dpi=args.dpi,
        max_iterations=args.max_iterations,
        only_cases=args.cases,
        force=args.force,
        enable_critic=args.enable_critic,
        critic_max_iters=args.critic_max_iters,
        locate_model_name=args.locate_model,
        folded=args.no_reader,
    )
    if args.locate_disabled_tools is not None:
        _KNOWN_LOCATE_TOOLS = frozenset(
            {"postcode", "grid_ref", "place", "road", "intersect", "la_check"}
        )
        disabled_set = frozenset(
            t.strip() for t in args.locate_disabled_tools.split(",") if t.strip()
        )
        unknown = disabled_set - _KNOWN_LOCATE_TOOLS
        if unknown:
            parser.error(
                f"--locate-disabled-tools: unknown tool(s) {sorted(unknown)}. "
                f"Known: {sorted(_KNOWN_LOCATE_TOOLS)}"
            )
        run_kwargs["locate_disabled_tools"] = disabled_set

    run_benchmark(**run_kwargs)
