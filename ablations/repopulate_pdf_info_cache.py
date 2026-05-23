"""Repopulate the pdf_info cache by running the current production reader
on every case folder in evaluation_data/.

The locate-stage ablations need pdf_info that matches what the current
production pipeline produces. Reusing a pdf_info cache from an older
benchmark run risks schema drift — e.g. the MAXIMALLYFINALVERSION run
emitted fields like ``directional_modifier`` and ``n_pages`` that no
longer exist in the current PDFInfo schema. Running the reader fresh
guarantees the cache matches what ``run_locate`` and the worker would
see today.

This script invokes ONLY the reader phase
(:func:`tools.agent.runtime.read_pdf_phase`) per case — the exact same
code path production uses for Phase 1. No MINIMA, no SAM3, no worker
loop.

Outputs (relative to repo root):
    ablations/cached_pdf_info_for_locate_ablations.json
        {case_name: pdf_info_dict, ...}. ``_reader_tokens`` is preserved
        for retrospective cost telemetry — the ablation harness strips
        ``_*``-prefixed keys before passing to ``run_locate`` to mirror
        production state population.
    ablations/locate_ablation_missing_cases.txt
        Human-readable report of cases the reader could not produce a
        usable pdf_info for, one per line with a reason.

The cache is rewritten after every successful case, so a crash, ctrl-C,
or credit exhaustion leaves a partial cache that ``--resume`` picks up.

Usage (from repo root):
    uv run python ablations/repopulate_pdf_info_cache.py            # full fresh run
    uv run python ablations/repopulate_pdf_info_cache.py --resume   # skip cached cases
    uv run python ablations/repopulate_pdf_info_cache.py --only-cases A4D4A1
    uv run python ablations/repopulate_pdf_info_cache.py --reader-model gemini-pro
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# Add repo root to sys.path so ``tools.*`` resolves when this script is
# invoked directly (mirrors the convention in the other ablation scripts
# in this directory).
sys.path.insert(0, str(REPO_ROOT))

# Importing tools.agent triggers tool-decorator registration on the
# worker agent. We don't actually use the worker here (we only call
# read_pdf_phase), but the import path is the same as production.
from tools.agent import runtime as _rt  # noqa: E402
from tools.io.eval_case import resolve_case_pdf  # noqa: E402
DEFAULT_EVAL_DIR = REPO_ROOT / "evaluation_data"
# Match the production default — the rest of the pipeline (locate, worker)
# runs against pdf_info produced by gemini-flash today.
DEFAULT_MODEL = "gemini-flash"

OUT_CACHE = REPO_ROOT / "ablations" / "cached_pdf_info_for_locate_ablations.json"
OUT_MISSING = REPO_ROOT / "ablations" / "locate_ablation_missing_cases.txt"


def _load_existing(path: Path) -> dict[str, dict]:
    """Load an existing cache JSON, returning {} if absent/malformed."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, default=str))


def _write_missing_report(
    path: Path,
    model_name: str,
    expected: list[str],
    cache: dict,
    missing: list[tuple[str, str]],
) -> None:
    lines: list[str] = [
        "# Locate-ablation pdf_info cache — missing / unusable cases",
        f"# Source: live reader via tools.agent.runtime.read_pdf_phase",
        f"# Reader model: {model_name}",
        f"# Total expected: {len(expected)}",
        f"# Usable in cache: {len(cache)}",
        f"# Missing/unusable: {len(missing)}",
        "",
    ]
    for case, reason in missing:
        lines.append(f"{case}\t{reason}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir", default=str(DEFAULT_EVAL_DIR),
        help=f"Eval data root (case folders). Default: "
             f"{DEFAULT_EVAL_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--reader-model", default=DEFAULT_MODEL,
        help=f"Model alias or full OpenRouter identifier. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing cache and skip cases already in it. Use after a "
             "crash / credit-out / partial run.",
    )
    parser.add_argument(
        "--only-cases", default=None,
        help="Comma-separated case names; run only these. Wipes the cache "
             "by default (clean subset output); combine with --resume to "
             "preserve other entries.",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_dir)
    if not eval_root.is_dir():
        print(f"ERROR: eval dir not found: {eval_root}", file=sys.stderr)
        return 1

    all_cases = sorted(p.name for p in eval_root.iterdir() if p.is_dir())

    if args.only_cases:
        wanted = {c.strip() for c in args.only_cases.split(",") if c.strip()}
        cases = [c for c in all_cases if c in wanted]
        not_found = wanted - set(cases)
        if not_found:
            print(f"WARNING: --only-cases not found in eval dir: "
                  f"{sorted(not_found)}")
    else:
        cases = all_cases

    # Cache state — wipe semantics are the default; --resume is the
    # explicit opt-in to incremental behavior:
    #   default                       -> wipe, run all 208 fresh
    #   --only-cases X                -> wipe, run only X (clean smoke output)
    #   --resume                      -> load, skip cases already present
    #   --resume --only-cases X       -> load, run X only if not cached
    # The "force-update a few cases inside an otherwise-good cache" use
    # case is intentionally not supported via a flag; manually delete
    # those entries from the JSON and re-run with --resume.
    if args.resume:
        cache = _load_existing(OUT_CACHE)
        if cache:
            print(f"--resume: loaded existing cache ({len(cache)} entries)")
    else:
        cache = {}

    missing: list[tuple[str, str]] = []
    t0 = time.time()

    for i, case in enumerate(cases, start=1):
        if args.resume and case in cache:
            print(f"[{i}/{len(cases)}] {case}: cached, skip")
            continue

        case_dir = eval_root / case
        pdf_path = resolve_case_pdf(case_dir)
        if pdf_path is None:
            missing.append((case, "no PDF in case folder"))
            print(f"[{i}/{len(cases)}] {case}: SKIP (no PDF)")
            continue

        print(f"[{i}/{len(cases)}] {case}: reader on {pdf_path.name}")
        try:
            pi = _rt.read_pdf_phase(
                str(pdf_path), args.reader_model, verbose=False)
        except Exception as e:
            traceback.print_exc()
            missing.append((case, f"reader exception: {e!s:.160}"))
            continue

        if pi.get("error"):
            missing.append((case, f"reader error: {str(pi['error'])[:160]}"))
            print(f"    -> reader error: {str(pi['error'])[:80]}")
            continue
        if not pi.get("map_pages"):
            missing.append(
                (case, "no map_pages (reader returned empty selection)"))
            print("    -> empty map_pages")
            continue

        cache[case] = pi
        # Persist incrementally so a mid-run crash leaves a usable cache.
        _write_cache(OUT_CACHE, cache)

        # Quick per-case telemetry line.
        rt = pi.get("_reader_tokens") or {}
        print(f"    -> ok | map_pages={pi.get('map_pages')} | "
              f"postcodes={len(pi.get('postcodes') or [])} | "
              f"places={len(pi.get('place_names') or [])} | "
              f"roads={len(pi.get('road_names') or [])} | "
              f"tokens: req={rt.get('request')} resp={rt.get('response')}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. "
          f"Cache: {len(cache)} entries. Missing: {len(missing)}.")

    _write_missing_report(
        OUT_MISSING, args.reader_model, expected=cases, cache=cache,
        missing=missing)
    print(f"Wrote {OUT_MISSING.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_CACHE.relative_to(REPO_ROOT)}")

    # Cost retrospective (rough): sum _reader_tokens across cache.
    total_req = sum(((v.get("_reader_tokens") or {}).get("request") or 0)
                    for v in cache.values())
    total_resp = sum(((v.get("_reader_tokens") or {}).get("response") or 0)
                     for v in cache.values())
    print(f"Token totals: req={total_req:,} resp={total_resp:,} "
          f"(use OpenRouter dashboard for ground-truth $ figure)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
