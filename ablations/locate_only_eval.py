"""Locate-only ablation harness.

Skips the worker, MINIMA, SAM3, commit/critic — just calls the locate
sub-agent once per case and scores its picked (lat, lon) against the
nearest GT polygon-part centroid (haversine km).

Two configs (the only locate rows the paper reports, Table 2):
  - default          — the single ``place`` geocoder (-> min_1_tool/).
  - ``--all-tools``  — all six geocoders (-> full/).
The VLM-direct baseline is a separate harness (locate_vlm_direct.py).

Inputs:
  ablations/cached_pdf_info_for_locate_ablations.json (frozen reader
  output, identical across configs — isolates locate-side variation
  from reader-side noise).
  data/<case>/<gt>.geojson for scoring.

Outputs (per config):
  results/ablations/locate_only_eval/{min_1_tool,full}/locate_picks.csv
    one row per case: err_km, picked coord, source, confidence,
    sigma, evidence.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from ablations.utils import (  # noqa: E402
    gt_part_centroids,
    nearest_part_err_km,
    print_err_km_summary,
    LOCATE_PICKS_FIELDNAMES,
)
from geoplanagent.agents.locate import run_locate  # noqa: E402
from geoplanagent.tools.pdf import resolve_case_pdf  # noqa: E402
from geoplanagent.tools.pdf import render_map_page  # noqa: E402
from geoplanagent.metrics import load_case_ground_truth  # noqa: E402
from geoplanagent.paths import ABL_LOCATE_ONLY, ABL_PDF_INFO_CACHE, DATA_DIR  # noqa: E402

load_dotenv()

DEFAULT_LOCATE_MODEL = "gemini-flash"

# Per-case CSV schema (shared by both locate harnesses).
CSV_FIELDNAMES = LOCATE_PICKS_FIELDNAMES

# Main eval


def evaluate(args: argparse.Namespace) -> int:
    """Run the locate sub-agent over every cached case and score its pick.

    For each case in the pdf_info cache: render the map page, call the locate
    agent (just the `place` geocoder, or all six under --all-tools), and measure
    the haversine error (km) from its picked (lat, lon) to the nearest GT
    polygon-part centroid. Writes one row per case to
    <out>/{min_1_tool,full}/locate_picks.csv (append-aware under --resume).

    Returns 0 on success, 1 if the pdf_info cache is missing.
    """
    all_tools = args.all_tools
    label = "full" if all_tools else "min_1_tool"
    out_dir = Path(args.out) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "locate_picks.csv"

    print(f"Config:        {label}/ (all_tools={all_tools})")
    print(f"Locate model:  {args.locate_model}")
    print(f"Output CSV:    {out_csv}")

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}", file=sys.stderr)
        return 1
    cache = json.loads(cache_path.read_text())
    print(f"Cache:         {len(cache)} entries from {cache_path}")

    cases = sorted(cache.keys())
    if args.cases:
        wanted = set(args.cases)
        cases = [case for case in cases if case in wanted]
        missing_subset = wanted - set(cases)
        if missing_subset:
            print(f"WARNING: --cases not in cache: {sorted(missing_subset)}")
    if args.max_cases:
        cases = cases[: args.max_cases]

    # Resume: skip cases already in the CSV.
    already_done: set[str] = set()
    if args.resume and out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                already_done.add(row["case"])
        if already_done:
            print(f"--resume:      {len(already_done)} cases already in CSV")

    eval_root = Path(args.eval_dir)
    fieldnames = CSV_FIELDNAMES

    # Open CSV in append mode when resuming, write+header when starting fresh.
    csv_mode = "a" if (args.resume and already_done) else "w"
    t0 = time.time()
    n_ok = n_err = 0

    with open(out_csv, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        for i, case in enumerate(cases, start=1):
            if case in already_done:
                continue

            print(f"\n[{i}/{len(cases)}] {case}")

            pdf_info_full = cache[case]
            # Strip _* telemetry keys before passing to locate, matching
            # production's _public() helper (geoplanagent/run.py:36).
            pdf_info = {
                key: value
                for key, value in pdf_info_full.items()
                if not key.startswith("_")
            }

            case_dir = eval_root / case
            pdf_path = resolve_case_pdf(case_dir)
            gt_geojson = load_case_ground_truth(case_dir)
            centroids = gt_part_centroids(gt_geojson) if gt_geojson else []

            row = {field: "" for field in fieldnames}
            row["case"] = case
            row["n_gt_parts"] = len(centroids)

            if pdf_path is None:
                row["error"] = "no PDF"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print("  -> SKIP (no PDF)")
                continue

            map_pages = pdf_info.get("map_pages") or []
            if not map_pages:
                row["error"] = "no map_pages in pdf_info"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print("  -> SKIP (no map_pages)")
                continue

            try:
                rendered = render_map_page(
                    str(pdf_path),
                    int(map_pages[0]),
                    verbose=False,
                    case_name=case,
                )
            except Exception as error:
                row["error"] = f"render failed: {error!s:.140}"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print(f"  -> SKIP (render failed: {error!s:.80})")
                continue

            if rendered is None:
                row["error"] = "render returned None"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print("  -> SKIP (render returned None)")
                continue

            page_img, _rot = rendered
            _, png_buf = cv2.imencode(".png", page_img)
            png_bytes = png_buf.tobytes()

            try:
                pick, _ = run_locate(
                    pdf_info=pdf_info,
                    map_img_bytes=png_bytes,
                    model_name=args.locate_model,
                    all_tools=all_tools,
                    request_limit=25,  # headroom for the all-tools agent; production keeps the default 15
                )
            except Exception as error:
                traceback.print_exc()
                row["error"] = f"run_locate raised: {error!s:.140}"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print(f"  -> ERROR (run_locate raised: {error!s:.80})")
                continue

            err = nearest_part_err_km(pick.top_lat, pick.top_lon, centroids)
            row.update(
                {
                    "err_km": (f"{err:.3f}" if err is not None else ""),
                    "picked_lat": f"{pick.top_lat:.6f}",
                    "picked_lon": f"{pick.top_lon:.6f}",
                    "picked_source": pick.picked_source[:120],
                    "confidence": pick.confidence,
                    "sigma_m": pick.sigma_m,
                    "evidence": pick.evidence[:240],
                }
            )
            writer.writerow(row)
            f.flush()
            n_ok += 1

            if err is not None:
                print(
                    f"  -> ok | err={err:.2f} km | conf={pick.confidence} "
                    f"| src={pick.picked_source[:50]}"
                )
            else:
                print(f"  -> ok (no GT centroids) | conf={pick.confidence}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min. n_ok={n_ok}, n_err={n_err}.")
    print(f"Wrote {out_csv}")

    print_err_km_summary(out_csv)
    return 0


# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all-tools",
        action="store_true",
        help="Use all six geocoders (the paper's 'full' config). "
        "Default: just the `place` geocoder (the paper's "
        "'min_1_tool' / production config).",
    )
    parser.add_argument(
        "--cache",
        default=str(ABL_PDF_INFO_CACHE),
        help=f"Cached pdf_info JSON. Default: {ABL_PDF_INFO_CACHE.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DATA_DIR),
        help=f"Eval data root. Default: {DATA_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--locate-model",
        default=DEFAULT_LOCATE_MODEL,
        help=f"Model alias or OpenRouter identifier for the locate "
        f"sub-agent. Default: {DEFAULT_LOCATE_MODEL}.",
    )
    parser.add_argument(
        "--out",
        default=str(ABL_LOCATE_ONLY),
        help=f"Parent dir for outputs; this run writes to a min_1_tool/ "
        f"or full/ subdir under it (per --all-tools). Default: "
        f"{ABL_LOCATE_ONLY.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Space-separated case names; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke limit — evaluate only the first N cases.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already in the output CSV.",
    )
    args = parser.parse_args()

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
