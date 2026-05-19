"""Merge a partial VLM ablation rerun into the original full-run results.

Use case: the original run (`results/ablation_vlm_seg/gemini-flash/`) hit
HTTP 413 on three oversized maps and pydantic validation retry-exhaustion
on three more. We re-ran the 18 affected cases (15 with longest-side
> 4096 px get resized via `--max-image-dim 4096`; 3 validation failures
get retried at native res) to a separate dir. This script replaces those
18 rows in the original CSV and regenerates `summary.json` so the paper
can quote one consistent table.

Cases NOT in the patch keep their original row verbatim — methodologically
clean since their input was below the resize cap and the script wouldn't
have touched them anyway.

Usage:
    uv run python scripts/merge_vlm_results.py \\
        --base   results/ablation_vlm_seg/gemini-flash \\
        --patch  results/ablation_vlm_seg_patch18/gemini-flash \\
        --out    results/ablation_vlm_seg_merged/gemini-flash
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Match ablations/vlm_segmentation.py:summarise verbatim.
def summarise(name: str, xs: List[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"name": name, "n": 0}
    s = sorted(xs)
    return {
        "name": name,
        "n": n,
        "mean": sum(xs) / n,
        "median": s[n // 2],
        "ge_0.50": sum(1 for x in xs if x >= 0.50) / n,
        "ge_0.70": sum(1 for x in xs if x >= 0.70) / n,
        "ge_0.80": sum(1 for x in xs if x >= 0.80) / n,
        "ge_0.90": sum(1 for x in xs if x >= 0.90) / n,
    }


def _read_csv(path: Path) -> List[Dict]:
    with path.open() as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def _parse_iou(v: Optional[str]) -> Optional[float]:
    import math
    if v is None or v == "" or v.lower() in ("none", "nan"):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="Original (full 211-case) run dir")
    ap.add_argument("--patch", required=True,
                    help="Partial-rerun dir (the cases to replace)")
    ap.add_argument("--out", required=True,
                    help="Where to write the merged results")
    ap.add_argument("--model", default="gemini-flash",
                    help="Stored in summary.json for paper traceability")
    args = ap.parse_args()

    base_dir = Path(args.base)
    patch_dir = Path(args.patch)
    out_dir = Path(args.out)

    base_csv = base_dir / "results.csv"
    patch_csv = patch_dir / "results.csv"
    if not base_csv.exists():
        sys.exit(f"missing: {base_csv}")
    if not patch_csv.exists():
        sys.exit(f"missing: {patch_csv}")

    base_rows = _read_csv(base_csv)
    patch_rows = _read_csv(patch_csv)
    patch_by_case = {r["case"]: r for r in patch_rows}

    print(f"Base run:  {len(base_rows)} rows from {base_csv}")
    print(f"Patch run: {len(patch_rows)} rows from {patch_csv}")

    # Sanity: every patch case must exist in base
    missing_in_base = sorted(set(patch_by_case) - {r["case"] for r in base_rows})
    if missing_in_base:
        sys.exit(f"patch contains cases not in base: {missing_in_base}")

    # Merge: replace base rows whose case appears in the patch
    merged_rows: List[Dict] = []
    replaced: List[str] = []
    for r in base_rows:
        if r["case"] in patch_by_case:
            merged_rows.append(patch_by_case[r["case"]])
            replaced.append(r["case"])
        else:
            merged_rows.append(r)

    if len(replaced) != len(patch_rows):
        sys.exit(
            f"replaced {len(replaced)} but patch had {len(patch_rows)} — "
            f"investigate before trusting merge"
        )

    # Recompute summary from merged rows
    valid_ious = [
        _parse_iou(r.get("iou"))
        for r in merged_rows
        if _parse_iou(r.get("iou")) is not None
    ]
    fails = sum(1 for r in merged_rows
                if _parse_iou(r.get("iou")) is None)
    summary_all = summarise("VLM-direct pixel IoU (all)", valid_ious)

    # Per-fold summary (mirror the script's logic)
    folds = sorted({r.get("fold") for r in merged_rows
                    if r.get("fold") not in (None, "", "None")})
    fold_summaries = []
    for f in folds:
        xs = [_parse_iou(r["iou"]) for r in merged_rows
              if r.get("fold") == f and _parse_iou(r.get("iou")) is not None]
        if xs:
            fold_summaries.append(summarise(f"fold {f}", xs))

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    fieldnames = list(merged_rows[0].keys())
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in merged_rows:
            w.writerow(r)

    out_summary = out_dir / "summary.json"
    out_summary.write_text(json.dumps({
        "model": args.model,
        "merge_provenance": {
            "base": str(base_dir),
            "patch": str(patch_dir),
            "replaced_cases": sorted(replaced),
        },
        "n_cases": len(merged_rows),
        "n_failures": fails,
        "summary": summary_all,
        "per_fold": fold_summaries,
    }, indent=2))

    # Copy pred_masks. Patch's masks override base's for any matching case.
    pred_out = out_dir / "pred_masks"
    pred_out.mkdir(exist_ok=True)
    base_masks = base_dir / "pred_masks"
    patch_masks = patch_dir / "pred_masks"
    n_from_base = 0
    n_from_patch = 0
    if base_masks.exists():
        for p in base_masks.iterdir():
            if p.is_file():
                shutil.copy2(p, pred_out / p.name)
                n_from_base += 1
    if patch_masks.exists():
        for p in patch_masks.iterdir():
            if p.is_file():
                shutil.copy2(p, pred_out / p.name)  # overwrites if dup
                n_from_patch += 1
    n_total = len(list(pred_out.iterdir()))

    # Report
    print(f"\nReplaced {len(replaced)} rows: {sorted(replaced)}")
    print(f"\nMerged: {len(merged_rows)} rows, {fails} failures")
    print(f"  pred_masks: {n_from_base} from base, {n_from_patch} from patch, "
          f"{n_total} total in {pred_out}")
    print(f"\n  mean   = {summary_all.get('mean', float('nan')):.4f}")
    print(f"  median = {summary_all.get('median', float('nan')):.4f}")
    print(f"  >=0.50 = {summary_all.get('ge_0.50', 0)*100:.1f}%")
    print(f"  >=0.70 = {summary_all.get('ge_0.70', 0)*100:.1f}%")
    print(f"  >=0.80 = {summary_all.get('ge_0.80', 0)*100:.1f}%")
    print(f"  >=0.90 = {summary_all.get('ge_0.90', 0)*100:.1f}%")
    print(f"\nWrote:\n  {out_csv}\n  {out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
