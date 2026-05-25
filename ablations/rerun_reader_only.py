"""Re-run the reader phase only on every eval case, writing fresh PDFInfo
JSON files to an isolated output directory. No worker, no positioning —
just the LLM extraction of PDFInfo against the current
``READER_SYSTEM_PROMPT``.

Used after a reader-prompt edit (e.g. tightening the
``is_district_wide`` field guidance) to measure which cases produce a
different PDFInfo under the new prompt, WITHOUT re-running the full
agent on all 208 cases. After this script writes the new PDFInfos, run
``ablations/diff_reader_output.py`` to identify the subset that actually
needs a full agent rerun (cases where ``is_district_wide`` flipped, or
``map_pages`` changed, etc.).

Each case writes::

    <out_dir>/<case>/pdf_info.json

Plus a per-run summary file ``_summary.json`` with token counts and
reader errors. ``--max-cases`` and ``--cases`` are honoured for
incremental / cherry-picked runs; ``--force`` overwrites cached entries.

Usage (from repo root)::

    uv run python ablations/rerun_reader_only.py \\
        --out-dir ablations/reader_rerun_post_fix \\
        --model gemini-flash
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.agent.runtime import read_pdf_phase  # noqa: E402
from tools.io.eval_case import resolve_case_pdf  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="ablations/reader_rerun_post_fix",
        help="Output directory for per-case pdf_info.json files "
             "(relative to repo root).",
    )
    parser.add_argument(
        "--model", default="gemini-flash",
        help="Reader model alias / OpenRouter id. Default: gemini-flash.",
    )
    parser.add_argument(
        "--eval-dir", default="evaluation_data",
        help="Eval data root (canonical case list = its subfolders).",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Cap the number of cases (for smoke runs).",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Only re-run these specific case folder names.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing pdf_info.json files in --out-dir "
             "(default: skip cases that already have one).",
    )
    args = parser.parse_args()

    eval_root = REPO_ROOT / args.eval_dir
    if not eval_root.is_dir():
        print(f"ERROR: --eval-dir not found: {eval_root}", file=sys.stderr)
        return 1
    out_root = REPO_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    all_cases = sorted(p.name for p in eval_root.iterdir() if p.is_dir())
    if args.cases:
        wanted = set(args.cases)
        all_cases = [c for c in all_cases if c in wanted]
        unknown = wanted - set(all_cases)
        if unknown:
            print(f"WARNING: requested cases not in eval dir: {sorted(unknown)}",
                  file=sys.stderr)
    if args.max_cases:
        all_cases = all_cases[: args.max_cases]

    print(f"Reader rerun: {len(all_cases)} cases  model={args.model}  "
          f"out={out_root.relative_to(REPO_ROOT)}")

    from tools.agent._model import resolve_model_name
    model_name = resolve_model_name(args.model)

    summary: dict = {
        "model": model_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": {},
    }
    n_done = n_skipped = n_error = 0
    n_dw_true = n_dw_false = 0
    t0 = time.time()

    for i, case in enumerate(all_cases, 1):
        case_dir = out_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / "pdf_info.json"

        if out_path.exists() and not args.force:
            try:
                cached = json.loads(out_path.read_text())
                summary["cases"][case] = {
                    "status": "cached",
                    "is_district_wide": cached.get("is_district_wide"),
                }
                n_skipped += 1
                if cached.get("is_district_wide"):
                    n_dw_true += 1
                else:
                    n_dw_false += 1
                continue
            except Exception:
                pass  # malformed cache → re-run

        case_folder = eval_root / case
        pdf_path = resolve_case_pdf(case_folder)
        if pdf_path is None:
            summary["cases"][case] = {"status": "no_pdf"}
            n_error += 1
            print(f"  [{i}/{len(all_cases)}] {case}: SKIP no PDF")
            continue

        t1 = time.time()
        try:
            pi = read_pdf_phase(str(pdf_path), model_name, verbose=False)
        except Exception as e:
            summary["cases"][case] = {"status": "error", "error": str(e)[:200]}
            n_error += 1
            print(f"  [{i}/{len(all_cases)}] {case}: ERROR {type(e).__name__}: "
                  f"{str(e)[:100]}")
            continue
        dt = time.time() - t1

        # Strip private (_-prefixed) keys before writing — keeps the on-disk
        # file shape identical to what benchmark_runner produces.
        public = {k: v for k, v in pi.items() if not k.startswith("_")}
        out_path.write_text(json.dumps(public, indent=2, default=str))

        dw = public.get("is_district_wide")
        err = public.get("error")
        tok = pi.get("_reader_tokens", {})
        summary["cases"][case] = {
            "status": "ok" if not err else "reader_error",
            "is_district_wide": dw,
            "district_name": public.get("district_name"),
            "map_pages": public.get("map_pages"),
            "n_map_page_details": len(public.get("map_page_details") or []),
            "n_road_names": len(public.get("road_names") or []),
            "n_place_names": len(public.get("place_names") or []),
            "n_visible_labels": len(public.get("visible_map_labels") or []),
            "tokens_request": tok.get("request"),
            "tokens_response": tok.get("response"),
            "seconds": round(dt, 1),
            "error": err if err else None,
        }
        n_done += 1
        if dw:
            n_dw_true += 1
        else:
            n_dw_false += 1
        flag = "[DW]" if dw else "    "
        print(f"  [{i}/{len(all_cases)}] {case}: {flag} "
              f"map_pages={public.get('map_pages')} "
              f"req={tok.get('request', '?')}tk t={dt:.1f}s"
              + (f"  err={err[:60]}" if err else ""))

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["n_done"] = n_done
    summary["n_skipped"] = n_skipped
    summary["n_error"] = n_error
    summary["n_is_district_wide_true"] = n_dw_true
    summary["n_is_district_wide_false"] = n_dw_false
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"Done: {n_done} new, {n_skipped} cached, {n_error} errors  "
          f"({elapsed/60:.1f} min)")
    print(f"is_district_wide=True: {n_dw_true}/{len(all_cases)} "
          f"({n_dw_true/max(len(all_cases),1)*100:.0f}%)")
    print(f"Summary: {(out_root / '_summary.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
