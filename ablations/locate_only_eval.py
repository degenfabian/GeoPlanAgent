"""Locate-only ablation harness.

Skips the worker, MINIMA, SAM3, commit/critic — just calls the locate
sub-agent once per case and scores its picked (lat, lon) against the
nearest GT polygon-part centroid (haversine km).

Used for:
  1. Locate LOO ablation: run with ``--disabled-tools postcode`` (etc.)
     for each of the 6 tools and compare per-tool error deltas against
     the no-disabled baseline.
  2. Locate vs VLM-direct geocode (sibling script — coming next).

Inputs:
  ablations/cached_pdf_info_for_locate_ablations.json (frozen reader
  output, identical across LOO variants — isolates locate-side
  variation from reader-side noise).
  evaluation_data/<case>/<gt>.geojson for scoring.

Outputs (per --disabled-tools config):
  ablations/locate_only_eval/<config>/locate_picks.csv
    one row per case: err_km, picked coord, source, confidence,
    sigma, verified_inside_admin_region, evidence.

The harness has a ``--dump-prompts`` mode that writes all 7 prompt
variants to disk and exits without LLM calls. Use this for pre-run
verification — read the prompts, confirm the LOO variants look clean,
then approve the actual runs.

Usage (from repo root):

  # Dump prompts for review, no LLM calls
  uv run python ablations/locate_only_eval.py --dump-prompts

  # Full baseline (no tool disabled)
  uv run python ablations/locate_only_eval.py

  # LOO variant
  uv run python ablations/locate_only_eval.py --disabled-tools postcode

  # Smoke (first 3 cases)
  uv run python ablations/locate_only_eval.py --max-cases 3

  # Specific case(s)
  uv run python ablations/locate_only_eval.py --only-cases A4D4A1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from ablations._shared import (  # noqa: E402
    CSV_FIELDNAMES, gt_part_centroids, nearest_part_err_km,
)
from tools.agent.locate_agent import (  # noqa: E402
    _LOCATE_TOOL_NAMES, _build_locate_prompt, run_locate,
)
from tools.agent.runtime import extract_message_log_from_msgs  # noqa: E402
from tools.io.eval_case import resolve_case_pdf  # noqa: E402
from tools.io.map_page import render_map_page  # noqa: E402
from tools.metrics.geojson import load_geojson  # noqa: E402


DEFAULT_CACHE = (
    REPO_ROOT / "ablations" / "cached_pdf_info_for_locate_ablations.json"
)
DEFAULT_EVAL_DIR = REPO_ROOT / "evaluation_data"
DEFAULT_LOCATE_MODEL = "gemini-flash"
DEFAULT_PROMPTS_DIR = REPO_ROOT / "ablations" / "prompts"
DEFAULT_OUT_ROOT = REPO_ROOT / "ablations" / "locate_only_eval"


# ── Helpers ────────────────────────────────────────────────────────────────


def _config_label(disabled: frozenset) -> str:
    """Filesystem-safe label for a config: 'full' or 'no_<tool>'."""
    if not disabled:
        return "full"
    return "no_" + "_".join(sorted(disabled))


def _parse_disabled(s: Optional[str]) -> frozenset[str]:
    """Parse a comma-separated tool list. Empty / None → no tools disabled."""
    if not s:
        return frozenset()
    tools = {t.strip() for t in s.split(",") if t.strip()}
    unknown = tools - _LOCATE_TOOL_NAMES
    if unknown:
        raise ValueError(
            f"Unknown locate tool(s) in --disabled-tools: {sorted(unknown)}. "
            f"Valid names: {sorted(_LOCATE_TOOL_NAMES)}"
        )
    return frozenset(tools)


# GT-centroid extraction + nearest-part scoring live in ablations._shared
# so the locate / VLM-direct / aggregation harnesses all agree on the
# metric byte-for-byte. Imported above as gt_part_centroids and
# nearest_part_err_km.


# ── Prompt dump (no LLM calls) ─────────────────────────────────────────────


def dump_prompts(out_dir: Path) -> None:
    """Write all 7 prompt variants + a diff overview to ``out_dir``.

    For each variant: ``locate_prompt_<config>.md`` — the literal prompt
    that pydantic-ai would send to the locate sub-agent for that config.
    Plus ``locate_prompt_diffs.md`` listing the lines each LOO variant
    removes vs the full prompt, so a reviewer can confirm "disabling X
    actually scrubbed all mentions of X".
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = (
        [frozenset()]
        + [frozenset({t}) for t in sorted(_LOCATE_TOOL_NAMES)]
    )

    written: list[tuple[str, Path, int, int]] = []
    full_prompt: Optional[str] = None
    prompts_by_label: dict[str, str] = {}
    for cfg in configs:
        label = _config_label(cfg)
        prompt = _build_locate_prompt(cfg)
        prompts_by_label[label] = prompt
        if not cfg:
            full_prompt = prompt
        path = out_dir / f"locate_prompt_{label}.md"
        path.write_text(prompt)
        written.append(
            (label, path, len(prompt), prompt.count("\n") + 1)
        )

    # Diff view: for each LOO variant, lines removed vs full.
    full_lines = set((full_prompt or "").splitlines())
    diff_lines: list[str] = [
        "# Locate prompt variants — diff vs full",
        "",
        "Each section lists lines present in the FULL prompt but NOT in "
        "the LOO variant. Use this to sanity-check that disabling a tool "
        "actually removes all references to it (tool description, signal-"
        "priority bullets, protocol-step references).",
        "",
        f"Full prompt: {len(full_prompt or '')} chars, "
        f"{(full_prompt or '').count(chr(10)) + 1} lines",
        "",
    ]

    for cfg in configs:
        if not cfg:
            continue
        label = _config_label(cfg)
        variant_lines = set(prompts_by_label[label].splitlines())
        removed = sorted(full_lines - variant_lines)
        added = sorted(variant_lines - full_lines)
        diff_lines.append(f"## {label}")
        diff_lines.append("")
        diff_lines.append(f"**Removed from full ({len(removed)} lines):**")
        diff_lines.append("```")
        diff_lines.extend(removed)
        diff_lines.append("```")
        if added:
            diff_lines.append("")
            diff_lines.append(f"**Added (not in full, {len(added)} lines):**")
            diff_lines.append("```")
            diff_lines.extend(added)
            diff_lines.append("```")
        diff_lines.append("")

    diff_path = out_dir / "locate_prompt_diffs.md"
    diff_path.write_text("\n".join(diff_lines))

    print(f"Wrote {len(written)} prompt variants to "
          f"{out_dir.relative_to(REPO_ROOT)}/")
    for label, path, n_chars, n_lines in written:
        print(f"  {label:<14} {path.name:<32} ({n_chars} chars, {n_lines} lines)")
    print(f"  + diff view:  {diff_path.relative_to(REPO_ROOT)}")


# ── Main eval ──────────────────────────────────────────────────────────────


def evaluate(args: argparse.Namespace) -> int:
    disabled = _parse_disabled(args.disabled_tools)
    # Allow caller to override the auto-derived dir name. Useful for
    # named subset ablations (e.g. "min_3_tool" instead of the verbose
    # "no_grid_ref_intersect_road").
    label = args.config_label or _config_label(disabled)
    out_root = Path(args.out_root)
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "locate_picks.csv"
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:        {label!r} (disabled={sorted(disabled) or 'none'})")
    print(f"Locate model:  {args.locate_model}")
    print(f"Output CSV:    {out_csv.relative_to(REPO_ROOT)}")
    print(f"Trajectories:  {traj_dir.relative_to(REPO_ROOT)}/<case>.json")

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}", file=sys.stderr)
        return 1
    cache = json.loads(cache_path.read_text())
    print(f"Cache:         {len(cache)} entries from "
          f"{cache_path.relative_to(REPO_ROOT)}")

    cases = sorted(cache.keys())
    if args.only_cases:
        wanted = {c.strip() for c in args.only_cases.split(",") if c.strip()}
        cases = [c for c in cases if c in wanted]
        missing_subset = wanted - set(cases)
        if missing_subset:
            print(f"WARNING: --only-cases not in cache: "
                  f"{sorted(missing_subset)}")
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
                writer.writerow(row); f.flush()
                n_err += 1
                print("  -> SKIP (no PDF)")
                continue

            map_pages = pi.get("map_pages") or []
            if not map_pages:
                row["error"] = "no map_pages in pdf_info"
                writer.writerow(row); f.flush()
                n_err += 1
                print("  -> SKIP (no map_pages)")
                continue

            try:
                rendered = render_map_page(
                    str(pdf_path), int(map_pages[0]),
                    dpi=args.dpi, verbose=False, case_name=case,
                )
            except Exception as e:
                row["error"] = f"render failed: {e!s:.140}"
                writer.writerow(row); f.flush()
                n_err += 1
                print(f"  -> SKIP (render failed: {e!s:.80})")
                continue

            if rendered is None:
                row["error"] = "render returned None"
                writer.writerow(row); f.flush()
                n_err += 1
                print("  -> SKIP (render returned None)")
                continue

            page_img, _rot = rendered
            _, buf = cv2.imencode(".png", page_img)
            png_bytes = buf.tobytes()

            try:
                pick, msgs = run_locate(
                    pdf_info=pi,
                    map_img_bytes=png_bytes,
                    model_name=args.locate_model,
                    disabled_tools=disabled,
                )
            except Exception as e:
                traceback.print_exc()
                row["error"] = f"run_locate raised: {e!s:.140}"
                writer.writerow(row); f.flush()
                n_err += 1
                print(f"  -> ERROR (run_locate raised: {e!s:.80})")
                continue

            err = nearest_part_err_km(pick.top_lat, pick.top_lon, centroids)
            row.update({
                "err_km": (f"{err:.3f}" if err is not None else ""),
                "picked_lat": f"{pick.top_lat:.6f}",
                "picked_lon": f"{pick.top_lon:.6f}",
                "picked_source": pick.picked_source[:120],
                "confidence": pick.confidence,
                "sigma_m": pick.sigma_m,
                "verified_inside_admin_region": pick.verified_inside_admin_region,
                "evidence": pick.evidence[:240],
            })
            writer.writerow(row); f.flush()
            n_ok += 1

            # Trajectory dump — per-case JSON capturing the full pick
            # plus the locate sub-agent's tool-call trajectory. Binary
            # content (e.g. the map PNG sent on the first user message)
            # is summarised by extract_message_log_from_msgs, so the
            # JSON stays small (~10-30 KB per case).
            try:
                trajectory, traj_stats = extract_message_log_from_msgs(msgs)
                traj_payload = {
                    "case": case,
                    "config": {
                        "disabled_tools": sorted(disabled),
                        "locate_model": args.locate_model,
                    },
                    "pick": pick.model_dump(),
                    "err_km": err,
                    "gt_centroids": [
                        {"lat": lat, "lon": lon} for lat, lon in centroids
                    ],
                    "trajectory_stats": traj_stats,
                    "trajectory": trajectory,
                }
                # Filesystem safety: a few case names contain ':' which
                # is illegal on some filesystems. Replace with '_'.
                fs_case = case.replace("/", "_").replace(":", "_")
                (traj_dir / f"{fs_case}.json").write_text(
                    json.dumps(traj_payload, indent=2, default=str)
                )
            except Exception as _e:
                # Don't fail the whole case if trajectory serialisation
                # hiccups — the CSV row is already written. Note it so
                # we can debug post-hoc.
                print(f"  WARN: trajectory dump failed: {_e!s:.80}")

            if err is not None:
                print(f"  -> ok | err={err:.2f} km | conf={pick.confidence} "
                      f"| src={pick.picked_source[:50]}")
            else:
                print(f"  -> ok (no GT centroids) | conf={pick.confidence}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. n_ok={n_ok}, n_err={n_err}.")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")

    # Quick aggregate stats (mean + median err_km).
    if out_csv.exists():
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        errs = [float(r["err_km"]) for r in rows
                if r.get("err_km") and r["err_km"]]
        if errs:
            errs.sort()
            mean = sum(errs) / len(errs)
            median = errs[len(errs) // 2]
            print(f"err_km: n={len(errs)}  mean={mean:.2f} km  "
                  f"median={median:.2f} km  min={errs[0]:.2f}  "
                  f"max={errs[-1]:.2f}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache", default=str(DEFAULT_CACHE),
        help=f"Cached pdf_info JSON. Default: "
             f"{DEFAULT_CACHE.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--eval-dir", default=str(DEFAULT_EVAL_DIR),
        help=f"Eval data root. Default: "
             f"{DEFAULT_EVAL_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--locate-model", default=DEFAULT_LOCATE_MODEL,
        help=f"Model alias or OpenRouter identifier for the locate "
             f"sub-agent. Default: {DEFAULT_LOCATE_MODEL}.",
    )
    parser.add_argument(
        "--disabled-tools", default=None,
        help="Comma-separated locate tool names to disable. Valid: "
             "postcode, grid_ref, place, road, intersect, la_check. "
             "Empty / omitted = full baseline.",
    )
    parser.add_argument(
        "--config-label", default=None,
        help="Override the auto-derived output dir name. Default: "
             "'full' / 'no_<tool>' / 'no_<tool1>_<tool2>'. Set this "
             "(e.g. 'min_3_tool') when running multi-tool subsets so "
             "the output path is human-readable.",
    )
    parser.add_argument(
        "--out-root", default=str(DEFAULT_OUT_ROOT),
        help=f"Output root (a per-config subdir is created). "
             f"Default: {DEFAULT_OUT_ROOT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument("--dpi", type=int, default=200,
                        help="PDF rendering DPI. Default: 200 (matches production).")
    parser.add_argument(
        "--only-cases", default=None,
        help="Comma-separated case names; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Smoke limit — evaluate only the first N cases.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cases already in the output CSV.",
    )
    parser.add_argument(
        "--dump-prompts", action="store_true",
        help=f"Write all 7 prompt variants to "
             f"{DEFAULT_PROMPTS_DIR.relative_to(REPO_ROOT)}/ and exit. "
             f"No LLM calls, no eval.",
    )
    args = parser.parse_args()

    if args.dump_prompts:
        dump_prompts(DEFAULT_PROMPTS_DIR)
        return 0

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
