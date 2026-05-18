"""Build the stratified 40-case subset for the VLM-direct PDF-to-GeoJSON ablation.

Reads the cleaned 208-case sheet (``data/0_planning_dataset_list.xlsx``, sheet
DATASET_SHEET) and keeps the 200 cases with a well traced boundary
(``Shape Matches correctly`` = ``yes`` or ``yes - across … ``), dropping the
8 with imperfect/absent GT (``yes - almost``, ``shape not outlined``). It then
strata-samples 40 over Document Quality × Shape Complexity (floor 2 per stratum)
and writes ``ablations/subset_40.json`` (40 case folders + stratum).

Deterministic: seed=42, sorted folder list before sampling.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from geoplanagent.paths import (  # noqa: E402
    DATASET_XLSX,
    DATASET_SHEET,
    VLM_E2E_SUBSET,
)
from geoplanagent.utils import normalise_label, load_dataset_labels  # noqa: E402


def load_labels(xlsx_path: Path) -> pd.DataFrame:
    """Load the cleaned 208-case sheet, normalise label columns, and return the
    200 cases with a well traced boundary (Shape Matches correctly = 'yes' or
    'yes - across …'); the 8 'yes - almost' / 'not outlined' are dropped."""
    
    df = load_dataset_labels(xlsx_path)
    df = df.rename(columns={"Folder Name": "folder"})
    # Document Quality / Shape Complexity get the full label normalisation, which
    # folds the annotator's stray "bad?"/"medium?" marks onto the canonical
    # {good, bad} x {easy, medium, hard} the stratum is built from (keeping the
    # seed-42 selection reproducible without editing the sheet). "Shape Matches
    # correctly" only needs lower+strip — it carries no "?" marks.
    df["Shape Matches correctly"] = df["Shape Matches correctly"].astype(str).str.strip().str.lower()
    for col in ("Document Quality", "Shape Complexity"):
        df[col] = df[col].map(normalise_label)
    df["Boundary Shape"] = df["Boundary Shape"].astype(str).str.strip()
    df["folder"] = df["folder"].astype(str).str.strip()
    clean = df[
        (df["Shape Matches correctly"] == "yes")
        | (df["Shape Matches correctly"].str.startswith("yes - across"))
    ].copy()
    clean["stratum"] = clean["Document Quality"] + "_x_" + clean["Shape Complexity"]
    return clean


def allocate_strata(strata_counts: dict[str, int], n_total: int, floor: int = 2) -> dict[str, int]:
    """Proportional allocation across strata with a per-stratum floor.

    Strategy: first guarantee ``floor`` per stratum (capped at its population
    so we never ask for more than exists). Distribute the remaining budget
    proportionally to remaining-population, breaking ties / handling rounding
    by topping up the largest strata until the totals match.

    Args:
        strata_counts: ``{stratum -> population}`` — how many clean candidates
            fall in each Document Quality × Shape Complexity stratum.
        n_total: total cases to hand out across all strata (the subset size).
        floor: minimum cases per stratum, each capped at that stratum's
            population so a sparse stratum is never over-asked.

    Returns:
        ``{stratum -> n_allocated}`` summing exactly to ``n_total``. Raises
        ``ValueError`` if ``n_total`` exceeds the total population, or
        ``RuntimeError`` if the residual top-up cannot reach ``n_total``.
    """
    pop = {stratum: int(count) for stratum, count in strata_counts.items()}
    pop_total = sum(pop.values())
    if n_total > pop_total:
        raise ValueError(f"asked for {n_total} cases but only {pop_total} clean candidates")

    # Floor allocation.
    alloc = {stratum: min(floor, population) for stratum, population in pop.items()}
    used = sum(alloc.values())
    remaining = n_total - used
    if remaining < 0:
        raise ValueError(f"floor={floor} across {len(pop)} strata exceeds n_total={n_total}")

    # Proportional fill on remaining capacity.
    remaining_cap = {stratum: pop[stratum] - alloc[stratum] for stratum in pop}
    cap_total = sum(remaining_cap.values())
    if cap_total == 0:
        return alloc

    raw = {stratum: remaining * (remaining_cap[stratum] / cap_total) for stratum in pop}
    floored = {stratum: int(math.floor(raw[stratum])) for stratum in pop}
    # Cap by remaining_cap.
    floored = {stratum: min(floored[stratum], remaining_cap[stratum]) for stratum in pop}
    for stratum in pop:
        alloc[stratum] += floored[stratum]

    # Distribute the residual to strata with the largest fractional
    # remainder until totals match. Deterministic tie-break: stratum name.
    remainder_after_floor = n_total - sum(alloc.values())
    if remainder_after_floor > 0:
        fractional = sorted(
            ((stratum, raw[stratum] - math.floor(raw[stratum])) for stratum in pop),
            key=lambda item: (-item[1], item[0]),
        )
        i = 0
        while remainder_after_floor > 0 and i < len(fractional):
            stratum = fractional[i % len(fractional)][0]
            if alloc[stratum] < pop[stratum]:
                alloc[stratum] += 1
                remainder_after_floor -= 1
            i += 1

    if sum(alloc.values()) != n_total:
        raise RuntimeError(
            f"allocation failed: got {sum(alloc.values())}, want {n_total} "
            f"(alloc={alloc}, pop={pop})"
        )
    return alloc


def sample_subset(clean: pd.DataFrame, alloc: dict[str, int], seed: int) -> pd.DataFrame:
    """Draw the per-stratum sample from the clean candidate pool.

    Deterministic: each stratum's folder list is sorted before sampling, and
    the strata are walked in sorted order through a single ``random.Random(seed)``
    — so identical inputs always select the same folders.

    Args:
        clean: cleaned candidate rows from ``load_labels``; must carry the
            ``stratum`` and ``folder`` columns.
        alloc: ``{stratum -> n_to_pick}`` from ``allocate_strata``.
        seed: seed for ``random.Random`` (42 reproduces the shipped subset).

    Returns:
        The subset of ``clean`` rows whose ``folder`` was picked
        (``sum(alloc.values())`` rows).
    """
    rng = random.Random(seed)
    picked = []
    for stratum, n_pick in sorted(alloc.items()):
        pool = sorted(clean[clean["stratum"] == stratum]["folder"].tolist())
        if n_pick > len(pool):
            raise ValueError(f"stratum {stratum}: asked for {n_pick} but only {len(pool)} in pool")
        picked.extend(rng.sample(pool, n_pick))
    return clean[clean["folder"].isin(picked)].copy()


def write_outputs(
    subset: pd.DataFrame,
    out_dir: Path,
    seed: int,
    n_total: int,
    alloc: dict[str, int],
    xlsx_path: Path,
) -> None:
    """Write ``subset_<n_total>.json`` — the tracked subset definition.

    Each case row records its stratum + label fields; the ``config`` block
    records provenance (source sheet, filter, seed, allocation) so the subset
    is reproducible from the file alone. The GT geojson is not pinned here —
    consumers resolve it from the case folder via ``load_case_ground_truth``,
    which also unions the multi-file merged cases.

    Args:
        subset: the sampled rows from ``sample_subset``.
        out_dir: directory the ``subset_<n_total>.json`` file is written into.
        seed: RNG seed, recorded in the ``config`` block for provenance.
        n_total: subset size — drives both the filename and the ``config`` block.
        alloc: ``{stratum -> n_allocated}``, recorded in the ``config`` block.
        xlsx_path: source spreadsheet, recorded (repo-relative) in ``config``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_records = []
    for _, row in subset.sort_values("folder").iterrows():
        subset_records.append(
            {
                "folder": row["folder"],
                "stratum": row["stratum"],
                "document_quality": row["Document Quality"],
                "shape_complexity": row["Shape Complexity"],
                "boundary_shape": row["Boundary Shape"],
            }
        )

    subset_json = {
        "config": {
            "source_xlsx": str(xlsx_path.relative_to(REPO_ROOT)),
            "source_sheet": DATASET_SHEET,
            "filter": "Shape Matches correctly in {'yes', 'yes - across …'}",
            "n_total": n_total,
            "stratification": "Document Quality x Shape Complexity",
            "floor_per_stratum": 2,
            "seed": seed,
            "allocation": alloc,
        },
        "cases": subset_records,
    }
    (out_dir / f"subset_{n_total}.json").write_text(json.dumps(subset_json, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xlsx",
        default=str(DATASET_XLSX),
        help=f"Excel path. Default: {DATASET_XLSX.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--out-dir",
        default=str(VLM_E2E_SUBSET.parent),
        help=f"Output dir. Default: {VLM_E2E_SUBSET.parent.relative_to(REPO_ROOT)}",
    )
    parser.add_argument("--n", type=int, default=40, help="Subset size. Default: 40.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed. Default: 42.")
    parser.add_argument("--floor", type=int, default=2, help="Minimum cases per stratum. Default: 2.")
    args = parser.parse_args()

    xlsx = Path(args.xlsx)
    out_dir = Path(args.out_dir)

    print(f"Loading labels: {xlsx}")
    clean = load_labels(xlsx)
    print(f"  clean candidates: {len(clean)}")

    strata_counts = clean.groupby("stratum").size().to_dict()
    print()
    print(f"Stratum population (clean candidates, n={sum(strata_counts.values())}):")
    for stratum in sorted(strata_counts):
        print(f"  {stratum:24s} {strata_counts[stratum]:4d}")

    alloc = allocate_strata(strata_counts, args.n, floor=args.floor)
    print()
    print(f"Allocation (n={args.n}, floor={args.floor}):")
    for stratum in sorted(alloc):
        print(f"  {stratum:24s} {alloc[stratum]:4d}")
    print(f"  {'TOTAL':24s} {sum(alloc.values()):4d}")

    subset = sample_subset(clean, alloc, args.seed)
    assert len(subset) == args.n, f"sampled {len(subset)}, expected {args.n}"

    print()
    print(f"Sampled {len(subset)} cases (seed={args.seed}):")
    for _, row in subset.sort_values(["stratum", "folder"]).iterrows():
        print(f"  [{row['stratum']:18s}] {row['folder']}")

    write_outputs(subset, out_dir, seed=args.seed, n_total=args.n, alloc=alloc, xlsx_path=xlsx)
    print()
    print(f"Wrote {out_dir / f'subset_{args.n}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
