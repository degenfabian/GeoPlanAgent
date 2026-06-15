"""
Benchmark Runner - Evaluate planning document GeoJSON extraction

Runs the unified tool-calling agent on the evaluation dataset.
The agent reads each PDF, geocodes locations, positions the map via MINIMA,
extracts boundaries with SAM3, and verifies — all through tool calls.

Usage:
    uv run benchmark_runner.py --model gemini-flash --enable-critic   # paper configuration
    uv run benchmark_runner.py --max-cases 5                          # quick smoke test
    uv run benchmark_runner.py --cases 12:00116:ART4                  # specific case
"""

import time
import json
import traceback
import pandas as pd
from pathlib import Path, PurePosixPath
from datetime import datetime
from dotenv import load_dotenv

from geoplanagent.tools.pdf import resolve_case_pdf
from geoplanagent.metrics import load_geojson, calculate_spatial_metrics, aggregate_stats
from geoplanagent.paths import DATA_DIR, DATASET_XLSX, DATASET_SHEET

# Load .env once at the entry point so HF_TOKEN, OPENROUTER_API_KEY, etc. are in
# os.environ before any model load or agent call.
load_dotenv()


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
    """Build the case-list DataFrame from the cleaned 208-case sheet, keeping
    only rows whose folder exists under eval_dir, then apply the case/range
    filters. A full run (no filters) yields all 208 cases.

    k-fold SAM3 routes each case to its held-out fold's adapter at inference,
    so cases that appear in the training pool are still scored without leakage.

    Args:
        dataset_path: path to the dataset .xlsx (the DATASET_SHEET tab is read).
        eval_dir: directory with one folder per case; rows whose folder is
            absent here are dropped.
        only_cases: the --cases flag. If given, keep only these exact folder
            names (e.g. to re-run a single case); overrides start_from/max_cases.
        start_from: the --start-from flag. Skip the first N rows (e.g. to resume
            a partial run).
        max_cases: the --max-cases flag. Keep at most this many rows after
            start_from (e.g. a quick smoke test). None means no limit.

    Returns the filtered dataset DataFrame.
    """
    dataset = pd.read_excel(dataset_path, sheet_name=DATASET_SHEET)
    eval_path = Path(eval_dir)
    dataset = dataset[dataset["Folder Name"].apply(lambda f: (eval_path / str(f)).exists())]
    print(f"Dataset: {len(dataset)} cases under {eval_path}")

    if only_cases:
        dataset = dataset[dataset["Folder Name"].apply(lambda f: str(f) in only_cases)]
        print(f"Filtered to {len(dataset)} specific cases: {only_cases}")
    else:
        dataset = dataset.iloc[start_from:]
        if max_cases:
            dataset = dataset.head(max_cases)

    return dataset


def _cached_entry(metrics_path, force, retry_failed, enable_critic):
    """Return the cached per-case row to reuse, or None to (re-)run the case.

    A cached metrics.json is reused unless one of these forces a re-run:
      - force: --force was passed;
      - retry_failed: --retry-failed was passed and the cached run crashed
        (ok / no_prediction cases stay cached);
      - critic-mode mismatch: a cached worker_first_iou means the cache came
        from a --enable-critic run, whose IoU isn't comparable to a no-critic
        run. district_lookup cases are exempt (the critic never runs on them,
        so their cached entry is valid in either mode).

    Args:
        metrics_path: path to the case's cached metrics.json.
        force / retry_failed / enable_critic: the matching CLI flags.

    Returns the cached row (minus sl_no, ready to merge) on a cache hit, else None.
    """
    if force or not metrics_path.exists():
        return None
    prev = json.loads(metrics_path.read_text())
    if retry_failed and prev.get("status") == "crashed":
        print("  [retry-failed — re-running crashed case]")
        return None
    cached_had_critic = "worker_first_iou" in prev
    is_district = (prev.get("agent_stats") or {}).get("outcome_status") == "district_lookup"
    if cached_had_critic != enable_critic and not is_district:
        print(
            f"  [cache mode mismatch — re-running] "
            f"cached_had_critic={cached_had_critic} current={enable_critic}"
        )
        return None
    print(f"  [cached] IoU={prev.get('iou', 0):.3f}")
    return {k: v for k, v in prev.items() if k != "sl_no"}


def _score_prediction(gt_geojson, result):
    """Score a run_agent result against ground truth.

    Computes IoU/precision/recall/centroid on the final polygon, and — when the
    critic ran — the worker's first-commit polygon too (the no-critic baseline).
    Downgrades status 'ok' to 'no_prediction' when a polygon was produced but is
    unscorable (invalid geometry that buffer(0) can't repair), or when none was
    produced; both then count as IoU 0 in the headline aggregate.

    Args:
        gt_geojson: ground-truth GeoJSON (None if the case has no GT).
        result: the dict returned by run_agent.

    Returns {status, metrics, worker_first_iou, worker_first_metrics}; the
    worker_first_* values are None unless the critic produced a paired result.
    """
    geojson = result.get("geojson")
    status = result.get("status", "ok")
    metrics = {}
    if gt_geojson and geojson:
        try:
            metrics = calculate_spatial_metrics(gt_geojson, geojson)
        except ValueError as e:
            status = "no_prediction"
            print(f"  WARNING: produced geometry is unscorable ({e}); recording no_prediction (IoU 0).")
    elif not geojson:
        print("  WARNING: no polygon produced; recording no_prediction (IoU 0).")

    worker_first_iou = worker_first_metrics = None
    worker_first_gj = result.get("worker_first_geojson")
    if worker_first_gj is not None and gt_geojson:
        worker_first_metrics = calculate_spatial_metrics(gt_geojson, worker_first_gj)
        worker_first_iou = worker_first_metrics.get("iou")

    return {
        "status": status,
        "metrics": metrics,
        "worker_first_iou": worker_first_iou,
        "worker_first_metrics": worker_first_metrics,
    }


def _run_case(
    row,
    case_idx,
    n_cases,
    eval_path,
    output_path,
    models_state,
    all_results,
    model_name,
    dpi,
    max_requests,
    force,
    retry_failed,
    enable_critic,
    critic_max_iters,
    locate_model_name,
    folded,
):
    """Run (or load from cache) one benchmark case and append its summary row.

    Resolves the case PDF + ground truth, reuses a cached result when valid
    (see ``_cached_entry``), otherwise runs the agent, scores it (see
    ``_score_prediction``), and writes ``predicted.geojson`` + ``metrics.json``.
    Every case leaves exactly one metrics.json — including crashes, which are
    recorded with a traceback.

    Returns True only on a fatal error that should stop the whole benchmark
    (an invalid model ID); False otherwise.
    """
    from geoplanagent.run import run_agent

    folder_name = str(row["Folder Name"])
    sl_no = int(row["Sl no (Unique ID)"])
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

    # Reuse a valid cached result; otherwise (re-)run below.
    case_dir = output_path / folder_name
    cached = _cached_entry(case_dir / "metrics.json", force, retry_failed, enable_critic)
    if cached is not None:
        all_results.append({"folder": folder_name, "sl_no": sl_no, **cached})
        return False

    # ── Run the agent ──
    try:
        t0 = time.time()
        result = run_agent(
            pdf_path=str(pdf_path),
            models_state=models_state,
            model_name=model_name,
            max_requests=max_requests,
            dpi=dpi,
            verbose=True,
            case_name=folder_name,
            enable_critic=enable_critic,
            critic_max_iters=critic_max_iters,
            locate_model_name=locate_model_name,
            folded=folded,
        )
        dt = time.time() - t0

        if result.get("status") == "crashed":
            err = result.get("error", "agent failed")
            print(f"  Error: {err}")
            # Fail fast only on invalid model ID (don't waste 200 cases)
            if "not a valid model ID" in str(err):
                print("\n  FATAL: Invalid model ID, stopping benchmark.")
                return True
            # Record the crash: status + stack trace + whatever partial state exists.
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "sl_no": sl_no,
                        "status": "crashed",
                        "error": err,
                        "traceback": result.get("traceback"),
                        "processing_time": dt,
                        "agent_stats": result.get("agent_stats", {}),
                    },
                    indent=2,
                    default=str,
                )
            )
            # Save partial geojson if any
            partial_gj = result.get("geojson")
            if partial_gj:
                (case_dir / "predicted.geojson").write_text(json.dumps(partial_gj, indent=2))
            all_results.append(
                {
                    "folder": folder_name,
                    "sl_no": sl_no,
                    "status": "crashed",
                    "error": err,
                    "processing_time": dt,
                }
            )
            return False

        geojson = result.get("geojson")
        mi = result.get("match_info", {})

        scored = _score_prediction(gt_geojson, result)
        status = scored["status"]
        metrics = scored["metrics"]
        iou = metrics.get("iou", 0)
        worker_first_iou = scored["worker_first_iou"]
        worker_first_metrics = scored["worker_first_metrics"]

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
            "status": status,
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
        # Last-resort guard: a failure in the per-case harness itself (scoring,
        # file IO) rather than inside run_agent, which handles its own crashes.
        # Record it as a crash with a trace so every case leaves exactly one
        # metrics.json instead of silently vanishing.
        tb = traceback.format_exc()
        traceback.print_exc()
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "metrics.json").write_text(
            json.dumps(
                {"sl_no": sl_no, "status": "crashed", "error": str(e), "traceback": tb},
                indent=2,
                default=str,
            )
        )
        all_results.append(
            {"folder": folder_name, "sl_no": sl_no, "status": "crashed", "error": str(e)}
        )
    return False


def run_benchmark(
    model_name,
    output_dir,
    max_cases=None,
    start_from=0,
    dpi=200,
    max_requests=30,
    dataset_path=DATASET_XLSX,
    eval_dir=DATA_DIR,
    only_cases=None,
    force=False,
    retry_failed=False,
    enable_critic=False,
    critic_max_iters=2,
    locate_model_name="google/gemini-3-flash-preview",
    folded=False,
):
    """Run benchmark using the unified tool-calling agent.

    Args:
        model_name: OpenRouter model identifier (reader + worker).
        output_dir: Base output directory for results.
        max_cases: Limit number of cases to run.
        start_from: Skip first N cases.
        dpi: PDF rendering DPI.
        max_requests: Max worker LLM requests (model calls) per case.
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
            max_requests=max_requests,
            force=force,
            retry_failed=retry_failed,
            enable_critic=enable_critic,
            critic_max_iters=critic_max_iters,
            locate_model_name=locate_model_name,
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
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=30,
        help="Max worker LLM requests (model calls) per case",
    )
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument(
        "--cases", nargs="+", default=None, help="Only run these specific case folder names"
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if cached results exist")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only cases whose cached metrics.json has status='crashed' "
        "(e.g. reader failures); ok / no_prediction cases stay cached.",
    )
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

    run_kwargs = dict(
        model_name=args.model,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        start_from=args.start_from,
        dpi=args.dpi,
        max_requests=args.max_requests,
        only_cases=args.cases,
        force=args.force,
        retry_failed=args.retry_failed,
        enable_critic=args.enable_critic,
        critic_max_iters=args.critic_max_iters,
        locate_model_name=args.locate_model,
        folded=args.no_reader,
    )
    run_benchmark(**run_kwargs)
