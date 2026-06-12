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

from geoplanagent.utils import haversine_km   # noqa: E402

EVAL_ROOT = REPO_ROOT / "ablations" / "locate_only_eval"

# Empirically calibrated (see investigate_bucket_b notebook): sign-flips
# on UK lon produce drift > 20 km; "agent picked a different candidate
# after la_check" cases stay < 5 km from at least one tool return. A 5 km
# threshold cleanly separates them.
L2_THRESHOLD_KM = 5.0


def _collect_tool_return_coords(trajectory: list) -> list[tuple[float, float]]:
    """Collect every (lat, lon) any locate tool returned in this trajectory.

    Returns the union of:
      - single-coord returns (postcode, grid_ref, la_check)
      - multi-hit returns (place.hits, road.hits)
      - intersection returns (intersect.intersections)
    """
    coords: list[tuple[float, float]] = []
    for entry in trajectory:
        if "ToolReturn" not in entry.get("kind", ""):
            continue
        ret = entry.get("return") or {}
        if not isinstance(ret, dict):
            continue
        if "lat" in ret and "lon" in ret:
            try:
                coords.append((float(ret["lat"]), float(ret["lon"])))
            except (ValueError, TypeError):
                pass
        for h in ret.get("hits") or []:
            if isinstance(h, dict):
                try:
                    coords.append((float(h["lat"]), float(h["lon"])))
                except (KeyError, ValueError, TypeError):
                    pass
        for h in ret.get("intersections") or []:
            if isinstance(h, dict):
                try:
                    coords.append((float(h["lat"]), float(h["lon"])))
                except (KeyError, ValueError, TypeError):
                    pass
    return coords


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

        # Bucket B: pick is far from EVERY coord any tool returned —
        # almost certainly a sign-flip / lat-lon-swap / number-corruption
        # bug on output. Cases where the agent legitimately picked a
        # different candidate after la_check have pick close to at
        # least one tool return, so they don't trigger.
        fs_case = case.replace("/", "_").replace(":", "_")
        traj_path = traj_dir / f"{fs_case}.json"
        if not traj_path.exists():
            continue
        try:
            j = json.loads(traj_path.read_text())
        except Exception:
            continue
        coords = _collect_tool_return_coords(j.get("trajectory") or [])
        if not coords:
            continue
        try:
            pick_lat = float(r["picked_lat"])
            pick_lon = float(r["picked_lon"])
        except (KeyError, ValueError, TypeError):
            continue
        min_drift = min(
            haversine_km(pick_lat, pick_lon, c_lat, c_lon)
            for c_lat, c_lon in coords
        )
        if min_drift > L2_THRESHOLD_KM:
            bucket_b.append({
                "case": case, "err_km": err,
                "min_drift_km": f"{min_drift:.2f}",
                "n_tool_coords": len(coords),
                "pick": f"({pick_lat:.4f}, {pick_lon:.4f})",
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
    md.append("- **B**: pick is > "
              f"{L2_THRESHOLD_KM} km from EVERY coord any tool returned. Fix: L2 cross-check validator.\n\n")
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

        # Only materialise a rerun list when there is something to rerun;
        # a clean config leaves no file behind.
        rerun_file = cfg / "rerun_cases.txt"
        if rerun_list:
            rerun_file.write_text("\n".join(rerun_list) + "\n")
        elif rerun_file.exists():
            rerun_file.unlink()

        md.append(f"| {cfg.name} | {a} | {b} | {len(rerun_list)} |\n")

        print(f"=== {cfg.name} ===")
        print(f"  Bucket A (HTTP fallback):    {a:3d} cases")
        for x in r["bucket_a"][:5]:
            print(f"    {x['case']:<42} err={x['err_km']} src={x['source']}")
        if a > 5:
            print(f"    ... + {a - 5} more")
        print(f"  Bucket B (L2-catchable):     {b:3d} cases")
        for x in r["bucket_b"][:5]:
            print(f"    {x['case']:<42} err={x['err_km']} "
                  f"min_drift_to_any_tool={x['min_drift_km']}km "
                  f"pick={x['pick']} n_tool_coords={x['n_tool_coords']}")
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
