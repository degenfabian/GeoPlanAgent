"""Extract per-case pdf_info into one consolidated JSON for locate ablations.

The locate-stage ablations (LOO over geocoder tools, locate-vs-VLM-direct)
need to hold the reader output IDENTICAL across all variants. Re-running
the reader per ablation config would add reader cost + reader noise on
top of the locate-side variation we actually care about. This script
freezes the reader output from a known-good production run.

Source convention:
    <src>/<model_subdir>/<case>/pdf_info.json

Outputs (relative to repo root):
    ablations/cached_pdf_info_for_locate_ablations.json
        Single JSON object {case_name: pdf_info_dict, ...}
    ablations/locate_ablation_missing_cases.txt
        Human-readable report of cases that were expected in the
        evaluation_data folder list but had no usable pdf_info (missing
        / malformed / reader error / empty map_pages), one per line
        with a reason.

Usage (from repo root):
    uv run python ablations/extract_pdf_info_cache.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SRC = REPO_ROOT / "results" / "benchmark_std_post_fix"
DEFAULT_MODEL_SUBDIR = "gemini-flash"
# Canonical case list comes from the on-disk evaluation_data folders.
# The xlsx has rows that never had case folders created and would
# inflate the "missing" count with non-issues.
DEFAULT_EVAL_DIR = REPO_ROOT / "evaluation_data"

OUT_CACHE = REPO_ROOT / "ablations" / "cached_pdf_info_for_locate_ablations.json"
OUT_MISSING = REPO_ROOT / "ablations" / "locate_ablation_missing_cases.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", default=str(DEFAULT_SRC),
        help=f"Benchmark run dir (default: {DEFAULT_SRC.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--model-subdir", default=DEFAULT_MODEL_SUBDIR,
        help=f"Subdirectory under --src (default: {DEFAULT_MODEL_SUBDIR})",
    )
    parser.add_argument(
        "--eval-dir", default=str(DEFAULT_EVAL_DIR),
        help=f"Eval data root (canonical case list = its subfolders). "
             f"Default: {DEFAULT_EVAL_DIR.relative_to(REPO_ROOT)}",
    )
    args = parser.parse_args()

    src_root = Path(args.src) / args.model_subdir
    if not src_root.exists():
        print(f"ERROR: source dir not found: {src_root}", file=sys.stderr)
        return 1

    eval_root = Path(args.eval_dir)
    if not eval_root.is_dir():
        print(f"ERROR: eval dir not found: {eval_root}", file=sys.stderr)
        return 1

    expected_cases = sorted(
        p.name for p in eval_root.iterdir() if p.is_dir()
    )
    print(
        f"Expected: {len(expected_cases)} cases "
        f"(case folders under {eval_root.relative_to(REPO_ROOT)})"
    )

    cache: dict[str, dict] = {}
    missing: list[tuple[str, str]] = []  # (case, reason)

    for case in expected_cases:
        pi_path = src_root / case / "pdf_info.json"
        if not pi_path.exists():
            missing.append((case, "no pdf_info.json"))
            continue
        try:
            pi = json.loads(pi_path.read_text())
        except Exception as e:
            missing.append((case, f"malformed JSON: {e!s:.100}"))
            continue
        # Reader emits "error" on Phase 1 failure; the body is essentially
        # empty in that case so it isn't useful for the locate stage.
        if pi.get("error"):
            missing.append(
                (case, f"reader error: {str(pi.get('error', ''))[:120]}")
            )
            continue
        # Sanity: must have non-empty map_pages, otherwise the locate
        # harness has nothing to render as the primary page.
        if not pi.get("map_pages"):
            missing.append((case, "no map_pages (reader returned empty selection)"))
            continue
        cache[case] = pi

    # Inventory: src dirs that have pdf_info.json but are NOT in the
    # evaluation_data folder list (e.g. excluded cases that were still attempted,
    # or stale dirs from a prior naming scheme). Informational only.
    skipped_other: list[str] = []
    if src_root.is_dir():
        expected_set = set(expected_cases)
        for sub in src_root.iterdir():
            if (sub.is_dir() and sub.name not in expected_set
                    and (sub / "pdf_info.json").exists()):
                skipped_other.append(sub.name)
    skipped_other.sort()

    OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OUT_CACHE.write_text(json.dumps(cache, indent=2, default=str))
    print(f"Wrote cache:   {OUT_CACHE.relative_to(REPO_ROOT)} "
          f"({len(cache)} cases)")

    lines: list[str] = [
        "# Locate-ablation pdf_info cache — missing / unusable cases",
        f"# Source:           {src_root.relative_to(REPO_ROOT)}",
        f"# Total expected:   {len(expected_cases)}",
        f"# Usable in cache:  {len(cache)}",
        f"# Missing/unusable: {len(missing)}",
        f"# In src but not in eval_data (informational): {len(skipped_other)}",
        "",
    ]
    for case, reason in missing:
        lines.append(f"{case}\t{reason}")
    if skipped_other:
        lines.append("")
        lines.append("# Cases present in src but not in eval_data:")
        for case in skipped_other:
            lines.append(f"{case}")
    OUT_MISSING.write_text("\n".join(lines) + "\n")
    print(f"Wrote missing: {OUT_MISSING.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
