"""Assemble training/dataset/ from the user-annotated boundary_annotations/.

Inputs (one per case; boundary_annotations/ is not distributed with the release):
  boundary_annotations/<case>/map.png          — rendered planning map
  boundary_annotations/<case>/edited_mask.png  — user-annotated binary mask

Outputs:
  training/dataset/maps/<case>.png            — copied map
  training/dataset/boundary_masks/<case>.png  — copied mask
  models/fold_assignment.json                 — {case_name: fold}, the one
      canonical fold map (one eval-form key per case; readers resolve name
      variants via geoplanagent.utils.route_key)

Fold assignment uses LPT (longest-processing-time-first) bin-packing for
balanced fold sizes while keeping these "stay-together" groups intact:
- Multi-page renders from one source (A108P_p4/p5/p6, A4D6A2_A3_merged_p9/p10
  etc.) — auto-detected via the _p<N> suffix.
- Hand-identified twin sets that share a planning site (the 6-case G3 set
  and the 12:00141 / 12:00117 pair).

Re-running the script is idempotent (same input → bit-identical output).

Run:   uv run python training/build_sam3_training_set.py
"""

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from geoplanagent.paths import FOLD_ASSIGNMENT, TRAINING_DATASET_DIR

REPO = Path(__file__).resolve().parent.parent
ANNOT_ROOT = REPO / "boundary_annotations"
OUT_ROOT = TRAINING_DATASET_DIR
N_FOLDS = 5

# Hand-identified twin cases that share a planning site or map; splitting
# them across folds would leak. Each member maps to a canonical group key.
_EXPLICIT_GROUPS: list[list[str]] = [
    ["12:00141:ART4", "12:00117:ART4"],
    [
        "095AB379-F04E-473A-BC0D-8948B58E4090",
        "3DA282A7-E829-47CF-B842-E03E0C704072",
        "4AB36890-E52B-4CCC-9CDE-FB1476FCEB82",
        "B9CDCF90-EC6A-4B66-A967-DEBF3B72D58D",
        "DE5A30DA-29A4-45BE-B60A-C201A5F11C6F",
        "FDBC0FDC-D090-4778-A123-232EB71DF3C6",
    ],
]
EXPLICIT_MAP: dict[str, str] = {
    member: group[0] for group in _EXPLICIT_GROUPS for member in group
}

# Case names with ':' or other punctuation become filenames; sanitise.
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(name: str) -> str:
    return _FILENAME_SAFE_RE.sub("_", name)


# Strip trailing _p<N> page-split suffix so A108P_p4/p5/p6 → "A108P".
_PAGE_SUFFIX_RE = re.compile(r"_p\d+$")


def _group_key(case_name: str) -> str:
    if case_name in EXPLICIT_MAP:
        return EXPLICIT_MAP[case_name]
    return _PAGE_SUFFIX_RE.sub("", case_name)


def _assign_folds_balanced(group_to_members: dict, n_folds: int) -> dict:
    """LPT (longest-processing-time-first) bin-packing for balanced folds.

    Sort groups by (size desc, key asc); for each group, assign it to the
    fold with the currently smallest total. Deterministic — same input
    yields bit-identical assignment across runs. Optimal-or-near-optimal
    for the multi-way partition problem under realistic group sizes
    (largest single group is 6 cases here, so worst-case spread ≤ 6).
    """
    fold_sizes = [0] * n_folds
    assignment = {}
    for group_key, members in sorted(
        group_to_members.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        best = min(range(n_folds), key=lambda f: (fold_sizes[f], f))
        assignment[group_key] = best
        fold_sizes[best] += len(members)
    return assignment


def main() -> int:
    if not ANNOT_ROOT.exists():
        print(f"ERROR: {ANNOT_ROOT} does not exist", file=sys.stderr)
        return 1

    maps_out = OUT_ROOT / "maps"
    masks_out = OUT_ROOT / "boundary_masks"
    maps_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: discover cases and bucket by group ──
    case_records = []  # (case_name, safe_name, group_key, map_path, mask_path)
    group_to_members = defaultdict(list)
    skipped = []
    for case_dir in sorted(ANNOT_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
        map_path = case_dir / "map.png"
        mask_path = case_dir / "edited_mask.png"
        if not (map_path.exists() and mask_path.exists()):
            skipped.append(
                (
                    case_dir.name,
                    f"missing {'map.png' if not map_path.exists() else 'edited_mask.png'}",
                )
            )
            continue
        case_name = case_dir.name
        group_key = _group_key(case_name)
        case_records.append(
            (case_name, _safe_filename(case_name), group_key, map_path, mask_path)
        )
        group_to_members[group_key].append(case_name)

    # ── Pass 2: LPT bin-pack groups onto folds for balanced sizes ──
    group_to_fold = _assign_folds_balanced(group_to_members, N_FOLDS)

    # ── Pass 3: copy files + assemble fold_assignment ──
    # Map/mask file stems are filesystem-safe (parens/colons replaced with
    # `_`), but fold_assignment.json records one key per case in the original
    # eval-form case name. Readers reduce any name variant (filesystem-safe
    # stem, colon form) to that key via geoplanagent.utils.route_key, so a
    # single written key resolves every lookup path.
    fold_map = {}
    cases_summary = []  # for the post-run report only
    for case_name, safe_name, group_key, map_path, mask_path in case_records:
        fold = group_to_fold[group_key]
        filename = f"{safe_name}.png"
        shutil.copy(map_path, maps_out / filename)
        shutil.copy(mask_path, masks_out / filename)

        # One eval-form key per case; readers reduce any name variant to it
        # via geoplanagent.utils.route_key, so no safe/per-page forms needed.
        fold_map[case_name] = fold
        cases_summary.append({"case": case_name, "fold": fold, "group_key": group_key})

    # Write the one canonical fold map (models/fold_assignment.json) — the only
    # fold_assignment.json in the repo, read by inference and training alike.
    FOLD_ASSIGNMENT.write_text(json.dumps(fold_map, indent=2, sort_keys=True))

    # ── Report ──
    by_fold = Counter(case["fold"] for case in cases_summary)
    by_group = defaultdict(list)
    for case in cases_summary:
        by_group[case["group_key"]].append(case["case"])

    print(f"\nDataset built at {OUT_ROOT}")
    print(f"  cases:  {len(cases_summary)}")
    print(f"  groups: {len(by_group)}  (each group → exactly one fold)")
    print(f"  skipped: {len(skipped)}" + (": " + str(skipped[:5]) if skipped else ""))
    print("\nFold distribution:")
    for fold in range(N_FOLDS):
        print(f"  fold {fold}: {by_fold[fold]} cases")

    multi_groups = sorted(
        (
            (group_key, members)
            for group_key, members in by_group.items()
            if len(members) > 1
        ),
        key=lambda item: -len(item[1]),
    )
    if multi_groups:
        print(f"\nMulti-member groups ({len(multi_groups)} total):")
        for group_key, members in multi_groups:
            print(f"  fold {group_to_fold[group_key]}  ({group_key}): {len(members)} cases")
            for member in members:
                print(f"      {member}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
