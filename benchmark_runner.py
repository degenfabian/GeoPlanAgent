"""
Benchmark Runner - Evaluate planning document GeoJSON extraction

Runs the unified tool-calling agent on the evaluation dataset.
The agent reads each PDF, geocodes locations, positions the map via MINIMA,
extracts boundaries with SAM3, and verifies — all through tool calls.

Usage:
    uv run benchmark_runner.py --model claude-sonnet                  # default
    uv run benchmark_runner.py --max-cases 5                          # quick test
    uv run benchmark_runner.py --cases 12:00116:ART4                  # specific case
    uv run benchmark_runner.py --max-iterations 3                     # limit agent turns
"""

import time
import json
import signal
import traceback
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime

from tools.metrics.geojson import load_geojson, calculate_spatial_metrics

# ── Legacy training-data list (historical, no longer enforced) ───────────────
# Pre-v8 (single-adapter SAM3) these 27 cases were the LoRA's training set
# and had to be excluded from evaluation to avoid leakage. v8 is k-fold, so
# each case is held out from the fold that does its inference — no leakage
# even when this set is included. As of 2026-05-14 the --include-training-cases
# flag was retired and these cases are ALWAYS included. The constant is
# retained for paper-reproducibility (documenting which cases used to be
# the training-only subset for single-adapter ablations).
EXCLUDE_SL_NOS = {1, 3, 5, 6, 11, 13, 15, 21, 22, 23, 33, 34, 49, 54, 59,
                  79, 84, 86, 88, 89, 125, 139, 230, 236, 246, 255, 256}

# ── Removed cases (duplicates of other cases in the dataset) ─────────────────
# Always filtered. Cases were physically removed from disk on 2026-05-13;
# kept here as a manifest for paper-reproducibility.
DUPLICATE_SL_NOS = {9, 68, 83, 232, 253}


# ── Model Loading ────────────────────────────────────────────────────────────

def load_models():
    """Load SAM3 fine-tuned model and MINIMA matcher."""
    from tools.extraction.sam3 import load_sam3_ft
    from tools.matching import load_minima

    state = {}
    state["sam3_ft"] = load_sam3_ft()
    state["minima"] = load_minima()
    return state


# ── Visualization ────────────────────────────────────────────────────────────

def save_visualizations(result_dir, map_img, boundary_mask, predicted_geojson,
                         gt_geojson):
    """Save per-case visualizations."""
    result_dir = Path(result_dir)

    if map_img is not None:
        cv2.imwrite(str(result_dir / "viz_map.png"), map_img)

    if boundary_mask is not None and map_img is not None:
        overlay = map_img.copy()
        overlay[boundary_mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(map_img, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(result_dir / "viz_boundary.png"), blended)

    if predicted_geojson is not None:
        viz_path = result_dir / "viz_comparison.png"
        try:
            from tools.metrics.visualization import visualize_comparison
            visualize_comparison(
                predicted_geojson=predicted_geojson,
                ground_truth_geojson=gt_geojson,
                output_path=str(viz_path),
            )
        except Exception as e:
            # Write a stub image so silent absence becomes a visible failure
            # at review time. v10 case DE5A30DA had viz silently missing.
            print(f"  Viz failed: {e}")
            try:
                stub = np.full((400, 800, 3), 240, dtype=np.uint8)
                msg = f"viz_comparison failed: {type(e).__name__}: {str(e)[:120]}"
                cv2.putText(stub, msg, (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1,
                            cv2.LINE_AA)
                cv2.imwrite(str(viz_path), stub)
            except Exception:
                pass


# ── Critic-trace writer (rigorous post-mortem analysis) ─────────────────────

def _iou_vs_gt(gt_geojson, pred_geojson):
    """Compute IoU against ground truth. Returns None if either side missing."""
    if gt_geojson is None or pred_geojson is None:
        return None
    try:
        from tools.metrics.geojson import calculate_spatial_metrics
        m = calculate_spatial_metrics(gt_geojson, pred_geojson)
        return float(m.get("iou", 0) or 0)
    except Exception:
        return None


# ── Main Runner ──────────────────────────────────────────────────────────────

def run_benchmark(model_name, output_dir, max_cases=None, start_from=0,
                  dpi=200, max_iterations=8,
                  dataset_path="evaluation_data/0_planning_dataset_list.xlsx",
                  eval_dir="evaluation_data",
                  only_cases=None, force=False,
                  hard_first=False, prev_results_dir=None):
    """Run benchmark using the unified tool-calling agent.

    Args:
        model_name: OpenRouter model identifier.
        output_dir: Base output directory for results.
        max_cases: Limit number of cases to run.
        start_from: Skip first N cases.
        dpi: PDF rendering DPI.
        max_iterations: Max agent turns per case.
        only_cases: If set, only run these specific folder names.
    """
    from tools.agent import run_agent

    dataset = pd.read_excel(dataset_path, sheet_name="0_planning_dataset_list")
    n_total = len(dataset)

    # Filter to cases that exist in eval_dir
    eval_path = Path(eval_dir)
    dataset = dataset[dataset["Unique ID (Folder_Name)"].apply(
        lambda f: (eval_path / str(f)).exists())]
    n_exists = len(dataset)

    # Inject *_merged folders that exist on disk but aren't in the Excel.
    # Each merged folder is its own complete case (own PDF + GT geojson).
    excel_folders = set(dataset["Unique ID (Folder_Name)"].astype(str))
    merged_extras = []
    for sub in sorted(eval_path.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_merged"): continue
        if sub.name in excel_folders: continue
        gj = sub / f"{sub.name}.geojson"
        if not gj.exists(): continue
        merged_extras.append({
            "Sl no": 9000 + len(merged_extras) + 1,
            "Unique ID (Folder_Name)": sub.name,
            "geojson ID (for sanity check)": gj.name,
        })
    if merged_extras:
        dataset = pd.concat([dataset, pd.DataFrame(merged_extras)],
                              ignore_index=True)
        print(f"Injected {len(merged_extras)} *_merged cases not in Excel: "
              f"{[e['Unique ID (Folder_Name)'] for e in merged_extras]}")

    # Drop physically-removed duplicates. The legacy training-case
    # exclusion (EXCLUDE_SL_NOS) was retired 2026-05-14 — k-fold SAM3 routes
    # each case to its held-out fold's adapter at inference, so no leakage
    # even when the former training-only set is included. The constant
    # itself is kept for paper-reproducibility.
    dataset = dataset[~dataset["Sl no"].isin(DUPLICATE_SL_NOS)]
    print(f"Dataset: {len(dataset)} cases "
          f"({n_total} in Excel, {n_total - n_exists} missing from disk, "
          f"{len(DUPLICATE_SL_NOS)} duplicates dropped)")

    # Filter to specific cases if requested
    if only_cases:
        dataset = dataset[dataset["Unique ID (Folder_Name)"].apply(
            lambda f: str(f) in only_cases)]
        print(f"Filtered to {len(dataset)} specific cases: {only_cases}")
    else:
        dataset = dataset.iloc[start_from:]
        if max_cases:
            dataset = dataset.head(max_cases)
    # ── Hard-first ordering: prioritize previously failing cases ──────────
    if hard_first:
        # Find previous results to sort by IoU (worst first)
        if prev_results_dir is None:
            prev_results_dir = Path(output_dir) / model_name.replace("/", "_")
        else:
            prev_results_dir = Path(prev_results_dir)

        prev_ious = {}
        if prev_results_dir.exists():
            for case_dir in prev_results_dir.iterdir():
                mf = case_dir / "metrics.json"
                if mf.exists():
                    try:
                        m = json.loads(mf.read_text())
                        prev_ious[case_dir.name] = m.get("iou", 1.0)
                    except Exception:
                        prev_ious[case_dir.name] = 1.0

        def _sort_key(row):
            folder = str(row["Unique ID (Folder_Name)"])
            iou = prev_ious.get(folder)
            if iou is None:
                return (1, 0.5)  # unseen cases go after failures but before successes
            return (0 if iou < 0.5 else 2, iou)

        dataset = dataset.assign(
            _sort_key=dataset.apply(_sort_key, axis=1)
        ).sort_values("_sort_key").drop(columns=["_sort_key"])

        n_hard = sum(1 for f in dataset["Unique ID (Folder_Name)"]
                     if prev_ious.get(str(f), 1.0) < 0.5)
        n_unseen = sum(1 for f in dataset["Unique ID (Folder_Name)"]
                       if str(f) not in prev_ious)
        print(f"Hard-first ordering: {n_hard} hard cases (IoU<0.5), "
              f"{n_unseen} unseen, {len(dataset) - n_hard - n_unseen} good")

    print(f"Running: {len(dataset)} cases\n")

    models_state = load_models()

    output_path = Path(output_dir) / model_name.replace("/", "_")
    all_results = []

    for case_idx, (_, row) in enumerate(dataset.iterrows()):
        folder_name = str(row["Unique ID (Folder_Name)"])
        sl_no = int(row["Sl no"])
        geojson_file = str(row["geojson ID (for sanity check)"]).split("/")[-1]

        print(f"\n{'─' * 70}")
        print(f"[{case_idx+1}/{len(dataset)}] Sl {sl_no}: {folder_name}")

        folder_path = eval_path / folder_name
        pdf_files = list(folder_path.glob("*.pdf")) if folder_path.exists() else []
        if not pdf_files:
            print("  SKIP: no PDF")
            all_results.append({
                "folder": folder_name, "sl_no": sl_no, "error": "no PDF"
            })
            continue

        # Prefer PDFs whose filename hints at a map/plan (dedicated diagram).
        # "plan" catches cases like A4Da2 where the map PDF is named
        # "..._Direction_Plan.pdf" and the other PDF is a notice.
        _MAP_TOKENS = ("map", "plan")
        map_pdfs = [p for p in pdf_files
                    if any(t in p.name.lower() for t in _MAP_TOKENS)]
        pdf_path = map_pdfs[0] if map_pdfs else pdf_files[0]
        gt_files = list(folder_path.glob(geojson_file))
        if not gt_files:
            gt_files = list(folder_path.glob("*.geojson"))
        gt_geojson = load_geojson(str(gt_files[0])) if gt_files else None

        # Check for cached result
        case_dir = output_path / folder_name
        cached_metrics = case_dir / "metrics.json"
        if cached_metrics.exists() and not force:
            prev = json.loads(cached_metrics.read_text())
            print(f"  [cached] IoU={prev.get('iou', 0):.3f}")
            all_results.append({
                "folder": folder_name, "sl_no": sl_no,
                **{k: v for k, v in prev.items()
                   if k not in ("sl_no",)}
            })
            continue

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
            )
            dt = time.time() - t0

            if not result.get("success"):
                err = result.get("error", "agent failed")
                print(f"  Error: {err}")
                # Fail fast only on invalid model ID (don't waste 200 cases)
                if "not a valid model ID" in str(err):
                    print("\n  FATAL: Invalid model ID, stopping benchmark.")
                    break
                # Still save what we can from failed cases
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "metrics.json").write_text(json.dumps({
                    "sl_no": sl_no, "error": err,
                    "processing_time": dt,
                    "agent_stats": result.get("agent_stats", {}),
                }, indent=2, default=str))
                msg_log = result.get("message_log", [])
                if msg_log:
                    (case_dir / "message_log.json").write_text(
                        json.dumps(msg_log, indent=2, default=str))
                # Save partial geojson if any
                partial_gj = result.get("geojson")
                if partial_gj:
                    (case_dir / "predicted.geojson").write_text(
                        json.dumps(partial_gj, indent=2))
                all_results.append({
                    "folder": folder_name, "sl_no": sl_no,
                    "error": err,
                    "processing_time": dt,
                })
                continue

            geojson = result.get("geojson")
            mi = result.get("match_info", {})

            # Compute metrics
            metrics = {}
            if gt_geojson and geojson:
                metrics = calculate_spatial_metrics(gt_geojson, geojson)

            iou = metrics.get("iou", 0)
            print(f"  IoU={iou:.3f}  inliers={mi.get('n_inliers', 0)}  "
                  f"t={dt:.1f}s  reason={result.get('agent_reason', '')[:60]}")

            # Save results — cache everything for offline analysis
            case_dir.mkdir(parents=True, exist_ok=True)
            if geojson:
                (case_dir / "predicted.geojson").write_text(
                    json.dumps(geojson, indent=2)
                )

            # Core metrics (used for cache-hit detection on re-runs)
            (case_dir / "metrics.json").write_text(json.dumps({
                "sl_no": sl_no,
                "match_info": mi,
                "processing_time": dt,
                "agent_accepted": result.get("agent_accepted"),
                "agent_reason": result.get("agent_reason"),
                "agent_stats": result.get("agent_stats", {}),
                **metrics,
            }, indent=2, default=str))

            # Full message log (every tool call, return, and reasoning text)
            msg_log = result.get("message_log", [])
            if msg_log:
                (case_dir / "message_log.json").write_text(
                    json.dumps(msg_log, indent=2, default=str)
                )

            # Reader phase extraction (what the LLM read from the PDF)
            pdf_info = result.get("agent_stats", {}).get("pdf_info")
            if pdf_info:
                (case_dir / "pdf_info.json").write_text(
                    json.dumps(pdf_info, indent=2, default=str)
                )

            # Boundary mask (binary, for re-projection experiments)
            mask = result.get("mask")
            if mask is not None:
                cv2.imwrite(str(case_dir / "boundary_mask.png"), mask)

            # Affine transform (for re-projection without re-running MINIMA)
            affine_H = result.get("affine_H")
            if affine_H is not None:
                np.save(str(case_dir / "affine_H.npy"), affine_H)

            # Tile info metadata (zoom, tx/ty_min for coordinate conversion)
            tile_meta = result.get("tile_info_meta", {})
            if tile_meta:
                (case_dir / "tile_info.json").write_text(
                    json.dumps(tile_meta, indent=2, default=str)
                )

            # Final selected boundary overlay
            selected_overlay = result.get("selected_overlay")
            if selected_overlay is not None:
                cv2.imwrite(str(case_dir / "selected_boundary.png"),
                            selected_overlay)

            # Visualization (with timeout). The agent cleans up map_img
            # before returning, so only comparison viz runs here.
            old_handler = signal.signal(signal.SIGALRM,
                lambda s, f: (_ for _ in ()).throw(TimeoutError))
            signal.alarm(60)
            try:
                save_visualizations(
                    case_dir, None, None, geojson, gt_geojson
                )
            except TimeoutError:
                print("  Viz timed out")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            all_results.append({
                "folder": folder_name, "sl_no": sl_no,
                "processing_time": dt, **metrics,
            })

        except Exception as e:
            traceback.print_exc()
            all_results.append({
                "folder": folder_name, "sl_no": sl_no, "error": str(e)
            })

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {model_name}")
    print(f"{'=' * 70}")

    summary = _compute_summary(all_results)
    summary_path = output_path / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    s = summary
    print(f"\n  {s['polygons_produced']} polygons / {s['no_polygon']} no-polygon / "
          f"{s['crashed']} crashed   (total {s['total']})")
    if s.get("metrics") and s["metrics"].get("iou"):
        m = s["metrics"]["iou"]
        print(f"  IoU (failures=0):    mean={m['mean']:.3f}  "
              f"median={m['median']:.3f}")
    if s.get("metrics_successful_only") and s["metrics_successful_only"].get("iou"):
        m2 = s["metrics_successful_only"]["iou"]
        print(f"  IoU (polygon-only):  mean={m2['mean']:.3f}  "
              f"median={m2['median']:.3f}")


def _compute_summary(results):
    """Compute aggregate stats.

    Headline metrics count "no polygon produced" cases as IoU=0 — this is
    the production-honest number since the operator ends up with no result
    for that case. Pre-2026-05-14 this counter was 'rejected_by_agent';
    rejection has since been removed from the schema, so a no-polygon case
    now reflects an upstream failure (empty SAM mask, projection failure,
    validator exhaustion), not the agent giving up.

    Pipeline crashes (entries with 'error' set) are excluded entirely.

    The 'successful_only' block is the old behaviour (mean over cases that
    produced a polygon). Useful for separating "did the produced polygons
    score well" from "how often did we produce one at all".
    """
    crashes = [r for r in results if "error" in r]
    non_crashed = [r for r in results if "error" not in r]
    # Production-honest: produced polygon → real IoU; no-polygon → 0.
    honest = [(r["iou"] if r.get("iou") is not None else 0.0)
              for r in non_crashed]
    no_polygon = [r for r in non_crashed if r.get("iou") is None]
    polygons = [r for r in non_crashed if r.get("iou") is not None]

    summary = {
        "total": len(results),
        "polygons_produced": len(polygons),
        "no_polygon": len(no_polygon),
        "crashed": len(crashes),
        "timestamp": datetime.now().isoformat(),
    }

    if non_crashed:
        def _stats(values):
            arr = np.array(values)
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        # Production-honest IoU (rejections counted as 0).
        summary["metrics"] = {"iou": _stats(honest)}

        if polygons:
            summary["metrics_successful_only"] = {
                "iou": _stats([r["iou"] for r in polygons]),
                "f1_score": _stats([r["f1_score"] for r in polygons]),
                "precision": _stats([r["precision"] for r in polygons]),
                "recall": _stats([r["recall"] for r in polygons]),
            }
            pos_errs = [r["positioning_error_m"] for r in polygons
                         if r.get("positioning_error_m") is not None]
            if pos_errs:
                summary["metrics_successful_only"]["positioning_error_m"] = _stats(pos_errs)

    summary["per_case"] = results
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark planning document GeoJSON extraction"
    )
    parser.add_argument("--model", default="gemini-pro",
                        help="OpenRouter model identifier")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=8,
                        help="Max agent turns per case")
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Only run these specific case folder names")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached results exist")
    parser.add_argument("--hard-first", action="store_true",
                        help="Run previously failing cases (IoU<0.5) first, "
                             "then unseen, then successful cases last")
    parser.add_argument("--prev-results", default=None,
                        help="Path to previous results dir for --hard-first "
                             "ordering (default: same as output-dir/model)")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        start_from=args.start_from,
        dpi=args.dpi,
        max_iterations=args.max_iterations,
        only_cases=args.cases,
        force=args.force,
        hard_first=args.hard_first,
        prev_results_dir=args.prev_results,
    )
