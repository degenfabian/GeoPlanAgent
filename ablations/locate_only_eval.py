"""Locate-only ablation harness.

Skips the worker, MINIMA, SAM3, commit/critic — just calls the locate
sub-agent once per case and scores its picked (lat, lon) against the
nearest GT polygon-part centroid (haversine km).

Two configs (the only locate rows the paper reports, Table 2):
  - ``production`` — the single ``place`` geocoder (-> min_1_tool/).
  - ``all_tools``  — all six geocoders (-> full/).
The VLM-direct baseline is a separate harness (locate_vlm_direct.py).

Inputs:
  ablations/cached_pdf_info_for_locate_ablations.json (frozen reader
  output, identical across configs — isolates locate-side variation
  from reader-side noise).
  evaluation_data/<case>/<gt>.geojson for scoring.

Outputs (per config):
  ablations/locate_only_eval/{min_1_tool,full}/locate_picks.csv
    one row per case: err_km, picked coord, source, confidence,
    sigma, evidence.

Usage (from repo root):

  # Production config (place only)
  uv run python ablations/locate_only_eval.py --config production

  # All-six-tools config
  uv run python ablations/locate_only_eval.py --config all_tools

  # Smoke (first 3 cases)
  uv run python ablations/locate_only_eval.py --max-cases 3

  # Specific case(s)
  uv run python ablations/locate_only_eval.py --only-cases A4D4A1
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from ablations._shared import (  # noqa: E402
    CSV_FIELDNAMES,
    add_subset_args,
    gt_part_centroids,
    nearest_part_err_km,
    print_err_km_summary,
)
from geoplanagent.agents.locate import run_locate  # noqa: E402
from geoplanagent.tools.pdf import resolve_case_pdf  # noqa: E402
from geoplanagent.tools.pdf import render_map_page  # noqa: E402
from geoplanagent.metrics import load_geojson  # noqa: E402
from geoplanagent.paths import DATA_DIR  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "ablations" / "cached_pdf_info_for_locate_ablations.json"
DEFAULT_EVAL_DIR = DATA_DIR
DEFAULT_LOCATE_MODEL = "gemini-flash"
DEFAULT_OUT_ROOT = REPO_ROOT / "ablations" / "locate_only_eval"

# config name -> (output dir label, all_tools flag). These are the only
# locate configs the paper reports: production = the single `place`
# geocoder; all_tools = all six geocoders.
_CONFIGS: dict[str, tuple[str, bool]] = {
    "production": ("min_1_tool", False),
    "all_tools": ("full", True),
}


# GT-centroid extraction + nearest-part scoring live in ablations._shared
# so the locate / VLM-direct / aggregation harnesses all agree on the
# metric byte-for-byte. Imported above as gt_part_centroids and
# nearest_part_err_km.


# Main eval


def evaluate(args: argparse.Namespace) -> int:
    label, all_tools = _CONFIGS[args.config]
    out_dir = Path(args.out_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "locate_picks.csv"

    print(f"Config:        {args.config} -> {label}/ (all_tools={all_tools})")
    print(f"Locate model:  {args.locate_model}")
    print(f"Output CSV:    {out_csv.relative_to(REPO_ROOT)}")

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}", file=sys.stderr)
        return 1
    cache = json.loads(cache_path.read_text())
    print(f"Cache:         {len(cache)} entries from {cache_path.relative_to(REPO_ROOT)}")

    cases = sorted(cache.keys())
    if args.only_cases:
        wanted = {c.strip() for c in args.only_cases.split(",") if c.strip()}
        cases = [c for c in cases if c in wanted]
        missing_subset = wanted - set(cases)
        if missing_subset:
            print(f"WARNING: --only-cases not in cache: {sorted(missing_subset)}")
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

            pi_full = cache[case]
            # Strip _* telemetry keys before passing to locate, matching
            # production state-population convention (runtime.py:109).
            pi = {k: v for k, v in pi_full.items() if not k.startswith("_")}

            case_dir = eval_root / case
            pdf_path = resolve_case_pdf(case_dir)
            gt_files = list(case_dir.glob("*.geojson"))
            gt_geojson = load_geojson(str(gt_files[0])) if gt_files else None
            centroids = gt_part_centroids(gt_geojson) if gt_geojson else []

            row = {fn: "" for fn in fieldnames}
            row["case"] = case
            row["n_gt_parts"] = len(centroids)

            if pdf_path is None:
                row["error"] = "no PDF"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print("  -> SKIP (no PDF)")
                continue

            map_pages = pi.get("map_pages") or []
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
                    dpi=args.dpi,
                    verbose=False,
                    case_name=case,
                )
            except Exception as e:
                row["error"] = f"render failed: {e!s:.140}"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print(f"  -> SKIP (render failed: {e!s:.80})")
                continue

            if rendered is None:
                row["error"] = "render returned None"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print("  -> SKIP (render returned None)")
                continue

            page_img, _rot = rendered
            _, buf = cv2.imencode(".png", page_img)
            png_bytes = buf.tobytes()

            try:
                pick, _ = run_locate(
                    pdf_info=pi,
                    map_img_bytes=png_bytes,
                    model_name=args.locate_model,
                    all_tools=all_tools,
                )
            except Exception as e:
                traceback.print_exc()
                row["error"] = f"run_locate raised: {e!s:.140}"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print(f"  -> ERROR (run_locate raised: {e!s:.80})")
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
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")

    print_err_km_summary(out_csv)
    return 0


# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        choices=sorted(_CONFIGS),
        default="production",
        help="Which locate config to run. 'production' = the single "
        "`place` geocoder (-> min_1_tool/); 'all_tools' = all six "
        "geocoders (-> full/). Default: production.",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help=f"Cached pdf_info JSON. Default: {DEFAULT_CACHE.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DEFAULT_EVAL_DIR),
        help=f"Eval data root. Default: {DEFAULT_EVAL_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--locate-model",
        default=DEFAULT_LOCATE_MODEL,
        help=f"Model alias or OpenRouter identifier for the locate "
        f"sub-agent. Default: {DEFAULT_LOCATE_MODEL}.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Output root (a per-config subdir is created). "
        f"Default: {DEFAULT_OUT_ROOT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="PDF rendering DPI. Default: 200 (matches production)."
    )
    add_subset_args(parser)
    args = parser.parse_args()

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
