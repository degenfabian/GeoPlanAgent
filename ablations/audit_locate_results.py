"""Identify cases that should be re-run after the three locate fixes:
HTTP retry, image downscale, L2 cross-check validator.

For each config directory under ``ablations/locate_only_eval/``, find:

  - Bucket A — emergency fallbacks: rows where ``picked_source`` contains
    ``emergency_la_centroid`` (the locate agent hit an HTTP error and fell
    back to the LA centroid). These will retry with the new HTTP retry +
    image downscale and hopefully succeed.

  - Bucket B — L2-catchable: rows where the most recent ``la_check`` call
    in the trajectory has a coord >1 km from the final pick. These are
    sign-flip / lat-lon-swap LLM output bugs the new L2 validator would
    catch.

Writes per-config rerun lists to
``ablations/locate_only_eval/<cfg>/rerun_cases.txt``, plus an aggregate
``ablations/locate_only_eval/AUDIT.md`` summarising counts and totals.

Usage (from repo root):
    uv run python ablations/audit_locate_results.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.geo.coords import haversine_km   # noqa: E402

EVAL_ROOT = REPO_ROOT / "ablations" / "locate_only_eval"
L2_THRESHOLD_KM = 1.0


def _last_la_check_coord(trajectory: list) -> tuple[float, float] | None:
    """Return (lat, lon) from the most recent la_check tool call, or None."""
    for entry in reversed(trajectory):
        if "ToolCall" not in entry.get("kind", ""):
            continue
        if entry.get("tool") != "la_check":
            continue
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            continue
        try:
            return float(args["lat"]), float(args["lon"])
        except (KeyError, ValueError, TypeError):
            continue
    return None


def audit_config(cfg_dir: Path) -> dict:
    csv_path = cfg_dir / "locate_picks.csv"
    traj_dir = cfg_dir / "trajectories"
    if not csv_path.exists():
        return {"config": cfg_dir.name, "skipped": "no CSV"}

    bucket_a: list[dict] = []   # emergency fallbacks
    bucket_b: list[dict] = []   # L2-catchable

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        case = r["case"]
        src = r.get("picked_source") or ""
        err = r.get("err_km")
        if "emergency_la_centroid" in src:
            bucket_a.append({"case": case, "err_km": err, "source": src[:60]})
            continue

        # Bucket B requires trajectory inspection
        fs_case = case.replace("/", "_").replace(":", "_")
        traj_path = traj_dir / f"{fs_case}.json"
        if not traj_path.exists():
            continue
        try:
            j = json.loads(traj_path.read_text())
        except Exception:
            continue
        last = _last_la_check_coord(j.get("trajectory") or [])
        if last is None:
            continue
        try:
            pick_lat = float(r["picked_lat"])
            pick_lon = float(r["picked_lon"])
        except (KeyError, ValueError, TypeError):
            continue
        drift = haversine_km(pick_lat, pick_lon, last[0], last[1])
        if drift > L2_THRESHOLD_KM:
            bucket_b.append({
                "case": case, "err_km": err, "drift_km": f"{drift:.2f}",
                "pick": f"({pick_lat:.4f}, {pick_lon:.4f})",
                "la_check": f"({last[0]:.4f}, {last[1]:.4f})",
            })

    return {
        "config": cfg_dir.name,
        "n_rows": len(rows),
        "bucket_a": bucket_a,
        "bucket_b": bucket_b,
    }


def main() -> int:
    if not EVAL_ROOT.is_dir():
        print(f"ERROR: {EVAL_ROOT} not found", file=sys.stderr)
        return 1

    cfg_dirs = sorted(d for d in EVAL_ROOT.iterdir() if d.is_dir())
    # We audit locate configs only; VLM-direct doesn't use la_check.
    cfg_dirs = [d for d in cfg_dirs if not d.name.startswith("vlm_direct")]

    print(f"Auditing {len(cfg_dirs)} configs...\n")

    results = []
    total_a = 0
    total_b = 0
    rerun_total = 0

    md = ["# Locate LOO post-hoc audit — cases to rerun after fixes\n"]
    md.append("Two buckets per config:\n")
    md.append("- **A**: ``picked_source`` contains ``emergency_la_centroid`` "
              "(HTTP error fell back to LA centroid). Fix: HTTP retry + "
              "image downscale.\n")
    md.append("- **B**: most recent la_check coord differs from final pick "
              f"by >{L2_THRESHOLD_KM} km. Fix: L2 cross-check validator.\n\n")
    md.append("| Config | A (HTTP) | B (sign-flip) | Total to rerun |\n")
    md.append("|---|---:|---:|---:|\n")

    for cfg in cfg_dirs:
        r = audit_config(cfg)
        results.append(r)
        if "skipped" in r:
            continue
        a, b = len(r["bucket_a"]), len(r["bucket_b"])
        # A case in both buckets only needs to be rerun once.
        union = {x["case"] for x in r["bucket_a"]} | {x["case"] for x in r["bucket_b"]}
        rerun_list = sorted(union)

        total_a += a
        total_b += b
        rerun_total += len(rerun_list)

        rerun_file = cfg / "rerun_cases.txt"
        rerun_file.write_text("\n".join(rerun_list) + ("\n" if rerun_list else ""))

        md.append(f"| {cfg.name} | {a} | {b} | {len(rerun_list)} |\n")

        print(f"=== {cfg.name} ===")
        print(f"  Bucket A (HTTP fallback):    {a:3d} cases")
        for x in r["bucket_a"][:5]:
            print(f"    {x['case']:<42} err={x['err_km']} src={x['source']}")
        if a > 5:
            print(f"    ... + {a - 5} more")
        print(f"  Bucket B (L2-catchable):     {b:3d} cases")
        for x in r["bucket_b"][:5]:
            print(f"    {x['case']:<42} err={x['err_km']} drift={x['drift_km']}km "
                  f"pick={x['pick']} la_check={x['la_check']}")
        if b > 5:
            print(f"    ... + {b - 5} more")
        print(f"  Union to rerun:              {len(rerun_list):3d} cases "
              f"→ {rerun_file.relative_to(REPO_ROOT)}")
        print()

    md.append(f"| **TOTAL** | **{total_a}** | **{total_b}** | **{rerun_total}** |\n")
    out_md = EVAL_ROOT / "AUDIT.md"
    out_md.write_text("".join(md))
    print(f"Wrote {out_md.relative_to(REPO_ROOT)}")
    print(f"\nTotal to rerun across {len(cfg_dirs)} configs: {rerun_total} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
