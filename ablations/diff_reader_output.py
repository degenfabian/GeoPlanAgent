"""Diff freshly-re-run reader output against an existing benchmark's
pdf_info.json files, and bucket each case by the kind of change so we
can decide which need a full agent rerun.

The decision tree applied per case:

  Bucket A — IDENTICAL on all positioning-relevant fields:
      no rerun needed. The worker would make the same decisions.

  Bucket B — is_district_wide FLIPPED (True → False or False → True):
      MANDATORY rerun. The worker takes a completely different path
      (lookup_district vs propose_centers / match_at).

  Bucket C — map_pages changed (added / removed / re-ranked):
      MANDATORY rerun. Worker positions a different image, possibly
      via a different match_at page argument.

  Bucket D — Other field changes (road_names, place_names, visible_map_labels,
      site_address, likely_town_or_city, etc.):
      OPTIONAL rerun. The worker's positioning attempts depend on these
      via the locate sub-agent, but small wording changes often won't
      materially shift the locate pick. Flagged for review.

  Bucket E — Old or new is MISSING / has reader error:
      Inspect manually.

Usage (from repo root)::

    uv run python ablations/diff_reader_output.py \\
        --new-dir ablations/reader_rerun_post_fix \\
        --old-dir results/benchmark_std_post_fix/gemini-flash
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fields whose change does NOT require a worker rerun on its own — the
# locate stage and downstream tools tolerate small wording shifts.
_INFORMATIONAL_FIELDS = ("scale", "site_address")

# Fields whose change SHOULD trigger a worker rerun.
_POSITIONING_FIELDS = (
    "map_pages",
    "is_district_wide",
    "district_name",
)

# Fields whose change is locate-relevant but might or might not matter.
_LOCATE_FIELDS = (
    "postcodes",
    "grid_refs",
    "road_names",
    "place_names",
    "visible_map_labels",
    "house_number_road_pairs",
    "parish_names",
    "admin_region",
    "likely_town_or_city",
    "adjacency_hints",
)


def _norm(v):
    """Normalise for comparison: lists → sorted-tuples, None==''."""
    if v is None:
        return ""
    if isinstance(v, list):
        return tuple(sorted(str(x).strip() for x in v))
    if isinstance(v, str):
        return v.strip()
    return v


def _changed_fields(old: dict, new: dict, fields) -> list[str]:
    return [f for f in fields if _norm(old.get(f)) != _norm(new.get(f))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-dir", default="ablations/reader_rerun_post_fix",
        help="Directory containing fresh per-case pdf_info.json files "
             "from rerun_reader_only.py.",
    )
    parser.add_argument(
        "--old-dir", default="results/benchmark_std_post_fix/gemini-flash",
        help="Directory containing the existing per-case pdf_info.json "
             "files (the std baseline you want to patch).",
    )
    parser.add_argument(
        "--out-report", default="ablations/reader_diff_report.json",
        help="Where to write the structured diff report.",
    )
    parser.add_argument(
        "--out-rerun-list", default="ablations/cases_needing_worker_rerun.txt",
        help="Where to write the case-name list (one per line) for the "
             "full agent rerun.",
    )
    args = parser.parse_args()

    new_root = REPO_ROOT / args.new_dir
    old_root = REPO_ROOT / args.old_dir
    if not new_root.is_dir():
        print(f"ERROR: --new-dir not found: {new_root}", file=sys.stderr)
        return 1
    if not old_root.is_dir():
        print(f"ERROR: --old-dir not found: {old_root}", file=sys.stderr)
        return 1

    cases = sorted(p.name for p in new_root.iterdir()
                   if p.is_dir() and (p / "pdf_info.json").exists())

    bucket_a: list[str] = []   # identical, no rerun
    bucket_b: list[dict] = []  # is_district_wide flipped
    bucket_c: list[dict] = []  # map_pages changed
    bucket_d: list[dict] = []  # locate fields changed only
    bucket_e: list[dict] = []  # missing / error

    for case in cases:
        new_p = new_root / case / "pdf_info.json"
        old_p = old_root / case / "pdf_info.json"
        try:
            new = json.loads(new_p.read_text())
        except Exception as e:
            bucket_e.append({"case": case, "reason": f"new unreadable: {e!s:.80}"})
            continue
        if not old_p.exists():
            bucket_e.append({"case": case, "reason": "old missing"})
            continue
        try:
            old = json.loads(old_p.read_text())
        except Exception as e:
            bucket_e.append({"case": case, "reason": f"old unreadable: {e!s:.80}"})
            continue
        if new.get("error"):
            bucket_e.append({"case": case, "reason": f"new reader error: {new['error'][:80]}"})
            continue

        pos_changed = _changed_fields(old, new, _POSITIONING_FIELDS)

        dw_flip = (
            _norm(old.get("is_district_wide")) !=
            _norm(new.get("is_district_wide"))
        )
        map_pages_changed = (
            _norm(old.get("map_pages")) != _norm(new.get("map_pages"))
        )

        if dw_flip:
            bucket_b.append({
                "case": case,
                "old_is_district_wide": old.get("is_district_wide"),
                "new_is_district_wide": new.get("is_district_wide"),
                "old_district_name": old.get("district_name"),
                "new_district_name": new.get("district_name"),
                "old_map_pages": old.get("map_pages"),
                "new_map_pages": new.get("map_pages"),
                "old_iou": None,  # filled below if available
            })
            continue

        if map_pages_changed:
            bucket_c.append({
                "case": case,
                "old_map_pages": old.get("map_pages"),
                "new_map_pages": new.get("map_pages"),
            })
            continue

        locate_changed = _changed_fields(old, new, _LOCATE_FIELDS)
        if locate_changed or pos_changed:
            bucket_d.append({
                "case": case,
                "changed_fields": sorted(set(locate_changed + pos_changed)),
            })
            continue

        bucket_a.append(case)

    # Attach old IoU to bucket_b entries (helpful for prioritising rerun).
    for entry in bucket_b:
        m_path = old_root / entry["case"] / "metrics.json"
        if m_path.exists():
            try:
                m = json.loads(m_path.read_text())
                entry["old_iou"] = m.get("iou")
            except Exception:
                pass

    report = {
        "new_dir": str(new_root.relative_to(REPO_ROOT)),
        "old_dir": str(old_root.relative_to(REPO_ROOT)),
        "n_cases_compared": len(cases),
        "bucket_a_identical": {"n": len(bucket_a), "cases": bucket_a},
        "bucket_b_is_district_wide_flipped": {
            "n": len(bucket_b),
            "cases": bucket_b,
        },
        "bucket_c_map_pages_changed": {
            "n": len(bucket_c),
            "cases": bucket_c,
        },
        "bucket_d_locate_fields_only": {
            "n": len(bucket_d),
            "cases": bucket_d,
        },
        "bucket_e_inspect": {"n": len(bucket_e), "cases": bucket_e},
    }

    out_report = REPO_ROOT / args.out_report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, default=str))

    rerun_cases = sorted({e["case"] for e in bucket_b} |
                          {e["case"] for e in bucket_c})
    out_rerun = REPO_ROOT / args.out_rerun_list
    out_rerun.write_text("\n".join(rerun_cases) + ("\n" if rerun_cases else ""))

    print(f"Compared:               {len(cases)} cases")
    print(f"  A. identical:         {len(bucket_a):>3d}  (no rerun)")
    print(f"  B. is_district_wide:  {len(bucket_b):>3d}  (MANDATORY rerun)")
    print(f"  C. map_pages changed: {len(bucket_c):>3d}  (MANDATORY rerun)")
    print(f"  D. locate fields:     {len(bucket_d):>3d}  (optional — review)")
    print(f"  E. inspect:           {len(bucket_e):>3d}")
    print()
    print(f"Mandatory rerun set: {len(rerun_cases)} cases")
    print(f"Report: {out_report.relative_to(REPO_ROOT)}")
    print(f"Rerun list: {out_rerun.relative_to(REPO_ROOT)}")
    print()
    print("To rerun the worker on just the mandatory set:")
    print(f"  uv run python benchmark_runner.py --model gemini-flash \\")
    print(f"    --max-iterations 12 --output-dir {args.old_dir.replace('/gemini-flash', '')} \\")
    print(f"    --force --cases $(cat {args.out_rerun_list} | tr '\\n' ' ')")
    print()
    print('NOTE: the worker rerun will read pdf_info from Phase 1 of the')
    print("standard pipeline, NOT from --new-dir. The fix only takes effect")
    print("because the prompt has changed — Phase 1 will reproduce the new")
    print("pdf_info on its own. If you want to bypass Phase 1 and inject the")
    print("pre-computed PDFInfo, see the note in the script source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
