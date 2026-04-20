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

from tools.geojson_metrics import load_geojson, calculate_spatial_metrics

# ── Training data exclusion (SAM3 + MapSAM contamination) ────────────────────
EXCLUDE_SL_NOS = {1, 3, 5, 6, 11, 13, 15, 21, 22, 23, 33, 34, 49, 54, 59,
                  79, 84, 86, 88, 89, 125, 139, 230, 236, 246, 255, 256}


# ── Model Loading ────────────────────────────────────────────────────────────

def load_models():
    """Load SAM3 fine-tuned model and MINIMA matcher."""
    from tools.sam3_boundary import load_sam3_ft
    from tools.positioning import load_minima

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
        try:
            from tools.visualization_tools import visualize_comparison
            visualize_comparison(
                predicted_geojson=predicted_geojson,
                ground_truth_geojson=gt_geojson,
                output_path=str(result_dir / "viz_comparison.png"),
            )
        except Exception as e:
            print(f"  Viz failed: {e}")


# ── Main Runner ──────────────────────────────────────────────────────────────

def run_benchmark(model_name, output_dir, max_cases=None, start_from=0,
                  dpi=200, max_iterations=8,
                  dataset_path="evaluation_data/0_planning_dataset_list.xlsx",
                  eval_dir="evaluation_data",
                  only_cases=None, force=False,
                  hard_first=False, prev_results_dir=None,
                  enable_critic=True):
    """Run benchmark using the unified tool-calling agent.

    Args:
        model_name: OpenRouter model identifier.
        output_dir: Base output directory for results.
        max_cases: Limit number of cases to run.
        start_from: Skip first N cases.
        dpi: PDF rendering DPI.
        max_iterations: Max agent turns per case.
        only_cases: If set, only run these specific folder names.
        enable_critic: Run Phase 3 Commenter VLM critic after worker finishes.
    """
    from tools.agent import run_agent

    dataset = pd.read_excel(dataset_path, sheet_name="0_planning_dataset_list")
    n_total = len(dataset)

    # Filter to cases that exist in eval_dir
    eval_path = Path(eval_dir)
    dataset = dataset[dataset["Unique ID (Folder_Name)"].apply(
        lambda f: (eval_path / str(f)).exists())]
    n_exists = len(dataset)

    # Exclude training data (SAM3/MapSAM contamination)
    dataset = dataset[~dataset["Sl no"].isin(EXCLUDE_SL_NOS)]
    print(f"Dataset: {len(dataset)} cases "
          f"({n_total} in Excel, {n_total - n_exists} missing from disk, "
          f"{len(EXCLUDE_SL_NOS)} training excluded)")

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

        # Prefer PDFs with "map" in filename (dedicated map files)
        map_pdfs = [p for p in pdf_files if "map" in p.name.lower()]
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
                enable_critic=enable_critic,
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
            crit = result.get("critic_final_decision") or "-"
            rot_applied = result.get("critic_applied_rotation_deg")
            crit_extra = ""
            if rot_applied:
                crit_extra += f" rot={rot_applied}"
            if result.get("critic_worker_reentered"):
                crit_extra += " worker_re"
            print(f"  IoU={iou:.3f}  inliers={mi.get('n_inliers', 0)}  "
                  f"critic={crit}{crit_extra}  "
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

            # Instance candidate overlays (each candidate on the map)
            candidate_overlays = result.get("candidate_overlays", [])
            if candidate_overlays:
                for i, overlay in enumerate(candidate_overlays):
                    cv2.imwrite(str(case_dir / f"candidate_{i}.png"), overlay)

            # Final selected boundary overlay
            selected_overlay = result.get("selected_overlay")
            if selected_overlay is not None:
                cv2.imwrite(str(case_dir / "selected_boundary.png"),
                            selected_overlay)

            # Which candidate indices were selected
            selected_indices = result.get("selected_indices")
            if selected_indices is not None:
                (case_dir / "selected_indices.json").write_text(
                    json.dumps({"selected_indices": selected_indices})
                )

            # Phase 3 critic artifacts
            critic_iters = result.get("critic_iterations") or []
            if critic_iters:
                (case_dir / "critic_log.json").write_text(json.dumps({
                    "iterations": critic_iters,
                    "final_decision": result.get("critic_final_decision"),
                    "changed_mask": result.get("critic_changed_mask"),
                    "applied_rotation_deg": result.get("critic_applied_rotation_deg"),
                    "suspected_wrong_location": result.get(
                        "critic_suspected_wrong_location"),
                    "worker_reentered": result.get("critic_worker_reentered"),
                    "tokens": result.get("critic_tokens"),
                }, indent=2, default=str))
            critic_panel = result.get("critic_panel_img")
            if critic_panel is not None:
                cv2.imwrite(str(case_dir / "critic_panel.png"), critic_panel)

            # Geocoding transparency log
            centers_tried = result.get("centers_tried") or []
            if centers_tried:
                (case_dir / "centers_tried.json").write_text(
                    json.dumps(centers_tried, indent=2, default=str))

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
    print(f"\n  {s['successful']}/{s['total']} success")
    if s["successful"] > 0:
        m = s["metrics"]
        print(f"  IoU:  mean={m['iou']['mean']:.3f}  "
              f"median={m['iou']['median']:.3f}")


def _compute_summary(results):
    """Compute aggregate stats."""
    successful = [r for r in results
                  if "error" not in r and r.get("iou") is not None]
    failed = [r for r in results if "error" in r or r.get("iou") is None]

    summary = {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "timestamp": datetime.now().isoformat(),
    }

    if successful:
        def _stats(values):
            arr = np.array(values)
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        summary["metrics"] = {
            "iou": _stats([r["iou"] for r in successful]),
            "f1_score": _stats([r["f1_score"] for r in successful]),
            "precision": _stats([r["precision"] for r in successful]),
            "recall": _stats([r["recall"] for r in successful]),
        }

        pos_errs = [r["positioning_error_m"] for r in successful
                     if r.get("positioning_error_m") is not None]
        if pos_errs:
            summary["metrics"]["positioning_error_m"] = _stats(pos_errs)

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
    parser.add_argument("--max-iterations", type=int, default=6,
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
    parser.add_argument("--no-critic", action="store_true",
                        help="Disable Phase 3 Commenter critic loop (A/B testing)")
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
        enable_critic=not args.no_critic,
    )
