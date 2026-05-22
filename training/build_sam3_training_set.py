"""Assemble training/dataset/ from the user-annotated boundary_annotations/.

Inputs (one per case):
  boundary_annotations/<case>/map.png          — rendered planning map
  boundary_annotations/<case>/edited_mask.png  — user-annotated binary mask

Outputs:
  training/dataset/maps/<case>.png            — copied map
  training/dataset/boundary_masks/<case>.png  — copied mask
  training/dataset/fold_assignment.json       — {case_name: fold} for production lookup

Fold assignment uses LPT (longest-processing-time-first) bin-packing for
balanced fold sizes while keeping these "stay-together" groups intact:
- Multi-page renders from one source (A108P_p4/p5/p6, A4D6A_merged_p9/p10
  etc.) — auto-detected via the _p<N> suffix.
- User-identified twin sets that share a planning site (the 6-case G3 set
  and the 12:00141 / 12:00117 pair, post the 2026-05-13 duplicate removal).

Re-running the script is idempotent (same input → bit-identical output).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANNOT_ROOT = REPO / "boundary_annotations"
OUT_ROOT = REPO / "training" / "dataset"
N_FOLDS = 5

# Explicit stay-together groups (the user identified these as twin cases that
# share a planning site or map; splitting across folds would leak). Each member
# is mapped to the canonical group key.
_EXPLICIT_GROUPS: list[list[str]] = [
    # Remaining twin-set groups after the 5 duplicate twins were removed
    # 2026-05-13. The 5 deleted twins were: 05D21091, 74D9394B, 5797F9C9,
    # B76BCA2D, F3632728 — so their pairs collapse to single-member groups
    # and no longer need an explicit entry.
    ["12:00141:ART4", "12:00117:ART4"],
    ["095AB379-F04E-473A-BC0D-8948B58E4090",
     "3DA282A7-E829-47CF-B842-E03E0C704072",
     "4AB36890-E52B-4CCC-9CDE-FB1476FCEB82",
     "B9CDCF90-EC6A-4B66-A967-DEBF3B72D58D",
     "DE5A30DA-29A4-45BE-B60A-C201A5F11C6F",
     "FDBC0FDC-D090-4778-A123-232EB71DF3C6"],
]
EXPLICIT_MAP: dict[str, str] = {m: g[0] for g in _EXPLICIT_GROUPS for m in g}

# Case names with ':' or other punctuation become filenames; sanitise.
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")
def _safe_filename(s: str) -> str:
    return _FILENAME_SAFE_RE.sub("_", s)

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
    for gk, members in sorted(group_to_members.items(),
                                key=lambda kv: (-len(kv[1]), kv[0])):
        best = min(range(n_folds), key=lambda f: (fold_sizes[f], f))
        assignment[gk] = best
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
    case_records = []   # (case_name, safe_name, group_key, map_p, mask_p)
    group_to_members = defaultdict(list)
    skipped = []
    for d in sorted(ANNOT_ROOT.iterdir()):
        if not d.is_dir():
            continue
        map_p = d / "map.png"
        mask_p = d / "edited_mask.png"
        if not (map_p.exists() and mask_p.exists()):
            skipped.append((d.name, f"missing {'map.png' if not map_p.exists() else 'edited_mask.png'}"))
            continue
        case_name = d.name
        group_key = _group_key(case_name)
        case_records.append((case_name, _safe_filename(case_name),
                              group_key, map_p, mask_p))
        group_to_members[group_key].append(case_name)

    # ── Pass 2: LPT bin-pack groups onto folds for balanced sizes ──
    group_to_fold = _assign_folds_balanced(group_to_members, N_FOLDS)

    # ── Pass 3: copy files + assemble fold_assignment ──
    # File stems are filesystem-safe (parens/colons replaced with `_`).
    # The mapping back to boundary_annotations/<case>/ is implicit in the
    # original case name, recorded as a key in fold_assignment.json
    # alongside the canonical-underscore form AND the filesystem-safe form
    # — so either lookup path resolves: production looks up by case name
    # (with colon→underscore canonicalisation handled in tools.core.
    # fold_routing); training/eval lookups use the map filename's stem.
    fold_map = {}
    cases_summary = []  # for the post-run report only
    for case_name, safe_name, group_key, map_p, mask_p in case_records:
        fold = group_to_fold[group_key]
        filename = f"{safe_name}.png"
        shutil.copy(map_p, maps_out / filename)
        shutil.copy(mask_p, masks_out / filename)

        # All forms of the case name resolve to the same fold:
        canonical = case_name.replace(":", "_").replace("/", "_")
        for key in {case_name, canonical, safe_name}:
            fold_map[key] = fold
        cases_summary.append({"case": case_name, "fold": fold,
                               "group_key": group_key})

    (OUT_ROOT / "fold_assignment.json").write_text(
        json.dumps(fold_map, indent=2, sort_keys=True))

    # ── Report ──
    by_fold = Counter(c["fold"] for c in cases_summary)
    by_group = defaultdict(list)
    for c in cases_summary:
        by_group[c["group_key"]].append(c["case"])

    print(f"\nDataset built at {OUT_ROOT}")
    print(f"  cases:  {len(cases_summary)}")
    print(f"  groups: {len(by_group)}  (each group → exactly one fold)")
    print(f"  skipped: {len(skipped)}" + (": " + str(skipped[:5]) if skipped else ""))
    print(f"\nFold distribution:")
    for f in range(N_FOLDS):
        print(f"  fold {f}: {by_fold[f]} cases")

    multi_groups = sorted(((g, m) for g, m in by_group.items() if len(m) > 1),
                           key=lambda x: -len(x[1]))
    if multi_groups:
        print(f"\nMulti-member groups ({len(multi_groups)} total):")
        for gk, members in multi_groups:
            print(f"  fold {group_to_fold[gk]}  ({gk}): {len(members)} cases")
            for m in members: print(f"      {m}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
