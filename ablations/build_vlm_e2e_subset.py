"""Build the stratified 40-case subset for the VLM-direct PDF-to-GeoJSON ablation.

Reads ``evaluation_data/new_updated.xlsx`` (sheet "Cleaned_up_208_planning_dataset"),
fixes the 5 merged-folder naming mismatches so labels join cleanly against the
``results/benchmark_v_post_refactor/gemini-flash/`` per-case dict, filters to
``Shape Matches correctly`` in {``yes``, ``yes - across …``}, strata-samples 40
cases over Document Quality × Shape Complexity with a floor of 2 per stratum,
and writes:

    ablations/vlm_e2e_pdf_to_geojson/
        subset_40.json                     # 40 case folders + stratum + GT path
        subset_40_pipeline_baseline.json   # same 40 with existing benchmark IoU/F1
        subset_40_summary.md               # stratum counts + pipeline mean per stratum

Deterministic: seed=42, sorted folder list before sampling. Running twice
overwrites with identical content.

No API calls — pure file I/O. Safe to run.

Usage (from repo root):

    uv run python ablations/build_vlm_e2e_subset.py
    uv run python ablations/build_vlm_e2e_subset.py --n 40 --seed 42
    uv run python ablations/build_vlm_e2e_subset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = REPO_ROOT / "evaluation_data" / "new_updated.xlsx"
DEFAULT_BENCHMARK_SUMMARY = (
    REPO_ROOT / "results" / "benchmark_v_post_refactor" / "gemini-flash" / "summary.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "ablations" / "vlm_e2e_pdf_to_geojson"
DEFAULT_EVAL_DIR = REPO_ROOT / "evaluation_data"


# Excel-label folder name → benchmark folder name. The Excel labels the merged
# cases by their child IDs; the benchmark stores them under a shorter merged
# name. Without this rename the join drops 5 cases.
MERGED_RENAME = {
    "12_A_B_C_merged":   "12_merged",
    "A4D5A1_A2_merged":  "A4D5A_merged",
    "A4D6A2_A3_merged":  "A4D6A_merged",
    "A4D8A1_A2_merged":  "A4D8A_merged",
    "Ar4.1_4.7a_merged": "Ar4.1_7a_merged",
}


def load_labels(xlsx_path: Path, include_ambiguous: bool = False) -> pd.DataFrame:
    """Load the cleaned 208-case sheet, normalise label columns, apply merged
    rename. By default returns the 200-case clean subset (plain 'yes' +
    'yes - across' merged). When ``include_ambiguous=True``, returns all
    208 rows including the 8 with ambiguous/broken GT
    ('yes - almost', 'almost', 'yes - shape not outlined in pdf')."""
    df = pd.read_excel(xlsx_path, sheet_name="Cleaned_up_208_planning_dataset")
    df = df.rename(columns={"Unique ID (Folder_Name)": "folder"})
    for col in ("Document Quality", "Shape Complexity", "Shape Matches correctly"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["Boundary Shape"] = df["Boundary Shape"].astype(str).str.strip()
    df["folder"] = df["folder"].astype(str).str.strip().map(
        lambda f: MERGED_RENAME.get(f, f)
    )
    if include_ambiguous:
        df["stratum"] = (
            df["Document Quality"] + "_x_" + df["Shape Complexity"]
        )
        return df
    clean = df[
        (df["Shape Matches correctly"] == "yes")
        | (df["Shape Matches correctly"].str.startswith("yes - across"))
    ].copy()
    clean["stratum"] = (
        clean["Document Quality"] + "_x_" + clean["Shape Complexity"]
    )
    return clean


def load_benchmark_per_case(summary_path: Path) -> dict:
    """folder → {iou, precision, recall, f1_score, positioning_error_m}."""
    summary = json.loads(summary_path.read_text())
    out = {}
    for r in summary["per_case"]:
        out[r["folder"]] = {
            "iou": r.get("iou"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "f1_score": r.get("f1_score"),
            "positioning_error_m": r.get("positioning_error_m"),
        }
    return out


def allocate_strata(strata_counts: dict[str, int], n_total: int,
                    floor: int = 2) -> dict[str, int]:
    """Proportional allocation across strata with a per-stratum floor.

    Strategy: first guarantee ``floor`` per stratum (capped at its population
    so we never ask for more than exists). Distribute the remaining budget
    proportionally to remaining-population, breaking ties / handling rounding
    by topping up the largest strata until the totals match.
    """
    pop = {s: int(c) for s, c in strata_counts.items()}
    pop_total = sum(pop.values())
    if n_total > pop_total:
        raise ValueError(
            f"asked for {n_total} cases but only {pop_total} clean candidates")

    # Step 1: floor allocation.
    alloc = {s: min(floor, p) for s, p in pop.items()}
    used = sum(alloc.values())
    remaining = n_total - used
    if remaining < 0:
        raise ValueError(
            f"floor={floor} across {len(pop)} strata exceeds n_total={n_total}")

    # Step 2: proportional fill on remaining capacity.
    remaining_cap = {s: pop[s] - alloc[s] for s in pop}
    cap_total = sum(remaining_cap.values())
    if cap_total == 0:
        return alloc

    raw = {s: remaining * (remaining_cap[s] / cap_total) for s in pop}
    floored = {s: int(math.floor(raw[s])) for s in pop}
    # Cap by remaining_cap.
    floored = {s: min(floored[s], remaining_cap[s]) for s in pop}
    for s in pop:
        alloc[s] += floored[s]

    # Step 3: distribute the residual to strata with the largest fractional
    # remainder until totals match. Deterministic tie-break: stratum name.
    remainder_after_floor = n_total - sum(alloc.values())
    if remainder_after_floor > 0:
        fractional = sorted(
            ((s, raw[s] - math.floor(raw[s])) for s in pop),
            key=lambda kv: (-kv[1], kv[0]),
        )
        i = 0
        while remainder_after_floor > 0 and i < len(fractional) * 4:
            s = fractional[i % len(fractional)][0]
            if alloc[s] < pop[s]:
                alloc[s] += 1
                remainder_after_floor -= 1
            i += 1

    if sum(alloc.values()) != n_total:
        raise RuntimeError(
            f"allocation failed: got {sum(alloc.values())}, want {n_total} "
            f"(alloc={alloc}, pop={pop})")
    return alloc


def sample_subset(clean: pd.DataFrame, alloc: dict[str, int],
                  seed: int) -> pd.DataFrame:
    """Pull ``alloc[s]`` rows from each stratum. Deterministic: sort by
    folder name first, then sample with ``random.Random(seed)``."""
    rng = random.Random(seed)
    picked = []
    for stratum, k in sorted(alloc.items()):
        pool = sorted(clean[clean["stratum"] == stratum]["folder"].tolist())
        if k > len(pool):
            raise ValueError(
                f"stratum {stratum}: asked for {k} but only {len(pool)} in pool")
        picked.extend(rng.sample(pool, k))
    return clean[clean["folder"].isin(picked)].copy()


def resolve_gt_geojson(eval_dir: Path, folder: str) -> Path | None:
    """First *.geojson under evaluation_data/<folder>/, or None."""
    case_dir = eval_dir / folder
    if not case_dir.is_dir():
        return None
    gjs = sorted(case_dir.glob("*.geojson"))
    return gjs[0] if gjs else None


def write_outputs(subset: pd.DataFrame, bench_per_case: dict, out_dir: Path,
                  eval_dir: Path, seed: int, n_total: int,
                  alloc: dict[str, int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_records = []
    pipeline_baseline = {}
    for _, row in subset.sort_values("folder").iterrows():
        folder = row["folder"]
        gt_path = resolve_gt_geojson(eval_dir, folder)
        rec = {
            "folder": folder,
            "stratum": row["stratum"],
            "document_quality": row["Document Quality"],
            "shape_complexity": row["Shape Complexity"],
            "boundary_shape": row["Boundary Shape"],
            "gt_geojson_relpath": (
                str(gt_path.relative_to(REPO_ROOT)) if gt_path else None
            ),
        }
        subset_records.append(rec)
        bench = bench_per_case.get(folder, {})
        pipeline_baseline[folder] = {
            "iou": bench.get("iou"),
            "precision": bench.get("precision"),
            "recall": bench.get("recall"),
            "f1_score": bench.get("f1_score"),
            "positioning_error_m": bench.get("positioning_error_m"),
        }

    subset_json = {
        "config": {
            "source_xlsx": "evaluation_data/new_updated.xlsx",
            "source_sheet": "Cleaned_up_208_planning_dataset",
            "merged_rename": MERGED_RENAME,
            "filter": "Shape Matches correctly in {'yes', 'yes - across …'}",
            "n_total": n_total,
            "stratification": "Document Quality x Shape Complexity",
            "floor_per_stratum": 2,
            "seed": seed,
            "allocation": alloc,
        },
        "cases": subset_records,
    }
    (out_dir / f"subset_{n_total}.json").write_text(
        json.dumps(subset_json, indent=2, default=str))

    (out_dir / f"subset_{n_total}_pipeline_baseline.json").write_text(
        json.dumps({
            "config": {
                "source": "results/benchmark_v_post_refactor/gemini-flash/summary.json",
                "n_total": n_total,
            },
            "per_case": pipeline_baseline,
        }, indent=2, default=str))

    # Markdown summary so we can eyeball without parsing JSON.
    lines = [
        f"# VLM-E2E subset (N={n_total})",
        "",
        f"- source: `evaluation_data/new_updated.xlsx` → `Cleaned_up_208_planning_dataset`",
        f"- filter: `Shape Matches correctly` ∈ {{`yes`, `yes - across …`}}",
        f"- stratification: Document Quality × Shape Complexity, floor=2",
        f"- seed: {seed}",
        "",
        "## Stratum allocation",
        "",
        "| Stratum | Pop | Allocated | Pipeline mean IoU |",
        "|---|---|---|---|",
    ]
    pipeline_iou_by_stratum: dict[str, list[float]] = {}
    for _, r in subset.iterrows():
        v = bench_per_case.get(r["folder"], {}).get("iou")
        if v is not None:
            pipeline_iou_by_stratum.setdefault(r["stratum"], []).append(v)

    for s in sorted(alloc.keys()):
        pop = int((subset["stratum"] == s).sum())  # in subset
        # Population in the clean candidate pool, not the picked subset.
        n_alloc = alloc[s]
        ious = pipeline_iou_by_stratum.get(s, [])
        pipe_mean = sum(ious) / len(ious) if ious else float("nan")
        lines.append(f"| {s} | n/a | {n_alloc} | {pipe_mean:.3f} |")

    pipeline_ious = [v["iou"] for v in pipeline_baseline.values()
                     if v["iou"] is not None]
    overall = sum(pipeline_ious) / len(pipeline_ious) if pipeline_ious else float("nan")
    lines += [
        "",
        f"**Overall pipeline mean IoU on the {len(pipeline_ious)} subset cases: "
        f"{overall:.3f}**",
        "",
        "## Cases (alphabetical)",
        "",
    ]
    for rec in subset_records:
        lines.append(f"- `{rec['folder']}` — {rec['stratum']}")

    (out_dir / f"subset_{n_total}_summary.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--xlsx", default=str(DEFAULT_XLSX),
                    help=f"Excel path. Default: {DEFAULT_XLSX.relative_to(REPO_ROOT)}")
    ap.add_argument("--benchmark-summary", default=str(DEFAULT_BENCHMARK_SUMMARY),
                    help="Path to the benchmark summary.json for the pipeline "
                         "baseline column.")
    ap.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR),
                    help="evaluation_data root; used to resolve GT geojson paths.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"Output dir. Default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)}")
    ap.add_argument("--n", type=int, default=40, help="Subset size. Default: 40.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed. Default: 42.")
    ap.add_argument("--floor", type=int, default=2,
                    help="Minimum cases per stratum. Default: 2.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the allocation + sampled case list; write no files.")
    ap.add_argument("--all", action="store_true",
                    help="Output every clean case (no stratified sampling). "
                         "By default this is the 200 clean cases (filtered to "
                         "'yes' / 'yes - across'). Combine with "
                         "--include-ambiguous to include all 208.")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="When set with --all, include the 8 cases with "
                         "ambiguous/broken GT ('yes - almost', 'almost', "
                         "'yes - shape not outlined in pdf') for a full-208 "
                         "subset. The 8 cases will score against problematic "
                         "GT — caveat to disclose downstream.")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    bench_path = Path(args.benchmark_summary)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)

    print(f"Loading labels:        {xlsx.relative_to(REPO_ROOT)}")
    clean = load_labels(xlsx, include_ambiguous=args.include_ambiguous)
    label = "candidates (incl. ambiguous)" if args.include_ambiguous else "clean candidates"
    print(f"  {label}: {len(clean)}")

    print(f"Loading benchmark:     {bench_path.relative_to(REPO_ROOT)}")
    bench_per_case = load_benchmark_per_case(bench_path)
    print(f"  benchmark cases:     {len(bench_per_case)}")

    missing = sorted(set(clean["folder"]) - set(bench_per_case))
    if missing:
        print(f"WARNING: {len(missing)} label folders not in benchmark — "
              f"cannot fetch pipeline baseline for these:")
        for f in missing:
            print(f"  {f!r}")

    strata_counts = clean.groupby("stratum").size().to_dict()
    print()
    print(f"Stratum population (clean candidates, n={sum(strata_counts.values())}):")
    for s in sorted(strata_counts):
        print(f"  {s:24s} {strata_counts[s]:4d}")

    # --all path: emit every clean case, no stratified sampling. Reuses
    # write_outputs so the JSON shape matches subset_40.json byte-for-byte
    # (just with more rows). Output filename: subset_full.json.
    if args.all:
        full_alloc = dict(strata_counts)  # for the summary table header
        print()
        print(f"--all: writing every clean case (n={len(clean)}); "
              f"skipping stratified sampling.")
        if args.dry_run:
            for _, r in clean.sort_values("folder").iterrows():
                bench = bench_per_case.get(r["folder"], {})
                iou = bench.get("iou")
                iou_s = f"{iou:.3f}" if iou is not None else "  n/a"
                print(f"  [{r['stratum']:18s}] {r['folder']:40s}  pipeline IoU={iou_s}")
            print("\n(dry-run; no files written)")
            return 0
        write_outputs(clean, bench_per_case, out_dir, eval_dir,
                      seed=args.seed, n_total=len(clean), alloc=full_alloc)
        print()
        print(f"Wrote:")
        print(f"  {out_dir.relative_to(REPO_ROOT)}/subset_{len(clean)}.json")
        print(f"  {out_dir.relative_to(REPO_ROOT)}/subset_{len(clean)}_pipeline_baseline.json")
        print(f"  {out_dir.relative_to(REPO_ROOT)}/subset_{len(clean)}_summary.md")
        return 0

    alloc = allocate_strata(strata_counts, args.n, floor=args.floor)
    print()
    print(f"Allocation (n={args.n}, floor={args.floor}):")
    for s in sorted(alloc):
        print(f"  {s:24s} {alloc[s]:4d}")
    print(f"  {'TOTAL':24s} {sum(alloc.values()):4d}")

    subset = sample_subset(clean, alloc, args.seed)
    assert len(subset) == args.n, f"sampled {len(subset)}, expected {args.n}"

    print()
    print(f"Sampled {len(subset)} cases (seed={args.seed}):")
    for _, r in subset.sort_values(["stratum", "folder"]).iterrows():
        bench = bench_per_case.get(r["folder"], {})
        iou = bench.get("iou")
        iou_s = f"{iou:.3f}" if iou is not None else "  n/a"
        print(f"  [{r['stratum']:18s}] {r['folder']:40s}  pipeline IoU={iou_s}")

    if args.dry_run:
        print("\n(dry-run; no files written)")
        return 0

    write_outputs(subset, bench_per_case, out_dir, eval_dir,
                  seed=args.seed, n_total=args.n, alloc=alloc)
    print()
    print(f"Wrote:")
    print(f"  {out_dir.relative_to(REPO_ROOT)}/subset_{args.n}.json")
    print(f"  {out_dir.relative_to(REPO_ROOT)}/subset_{args.n}_pipeline_baseline.json")
    print(f"  {out_dir.relative_to(REPO_ROOT)}/subset_{args.n}_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
