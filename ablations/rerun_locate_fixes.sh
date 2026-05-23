#!/usr/bin/env bash
#
# Surgically rerun the cases identified by audit_locate_results.py.
# For each LOO config:
#   1. Load rerun_cases.txt (one case per line)
#   2. Remove those rows from locate_picks.csv (Python one-liner)
#   3. Delete those cases' trajectory JSONs
#   4. Invoke locate_only_eval.py --resume --only-cases <comma-sep-list>
#
# The fixes shipped earlier (HTTP retry, image downscale, L2 cross-check
# validator) now apply to each rerun. Cases that previously hit emergency
# fallback should succeed; cases with sign-flip output bugs should be
# caught and re-emitted.
#
# Usage (from repo root):
#   bash ablations/rerun_locate_fixes.sh
#
# Wall-clock estimate: ~80 min sequential (118 cases × ~40s with retries).
# Cost: ~$2-3 on gemini-flash.

set -u
cd "$(dirname "$0")/.."   # repo root

CONFIGS=(full no_postcode no_grid_ref no_place no_road no_intersect no_la_check)

echo "================================================================"
echo "Locate LOO post-hoc rerun (after HTTP retry + downscale + L2 fixes)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

n_failed=0
n_total_reran=0

for cfg in "${CONFIGS[@]}"; do
    cfg_dir="ablations/locate_only_eval/$cfg"
    rerun_file="$cfg_dir/rerun_cases.txt"

    if [[ ! -f "$rerun_file" ]]; then
        echo "  [$cfg] no rerun_cases.txt, skipping"
        continue
    fi

    # Read non-empty case lines (bash 3.2 compatible — macOS default bash
    # doesn't have ``mapfile``).
    cases=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "${line//[[:space:]]/}" ]] && continue
        cases+=("$line")
    done < "$rerun_file"
    n=${#cases[@]}
    if [[ "$n" == "0" ]]; then
        echo "  [$cfg] rerun list is empty, skipping"
        continue
    fi

    echo
    echo "================================================================"
    echo "[$cfg]  rerunning $n cases"
    echo "================================================================"

    # Surgical removal of bad rows + trajectories
    uv run python - "$cfg_dir" "${cases[@]}" <<'PYEOF'
import csv, sys
from pathlib import Path
cfg_dir = Path(sys.argv[1])
to_drop = set(sys.argv[2:])
csv_path = cfg_dir / "locate_picks.csv"
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else []
kept = [r for r in rows if r["case"] not in to_drop]
print(f"  CSV: {len(rows)} → {len(kept)} rows (dropped {len(rows)-len(kept)})")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fieldnames))
    w.writeheader()
    w.writerows(kept)
traj_dir = cfg_dir / "trajectories"
dropped_traj = 0
for case in to_drop:
    fs_case = case.replace("/", "_").replace(":", "_")
    p = traj_dir / f"{fs_case}.json"
    if p.exists():
        p.unlink()
        dropped_traj += 1
print(f"  trajectories: dropped {dropped_traj} JSONs")
PYEOF

    # Build the comma-separated --only-cases argument
    only=$(IFS=,; echo "${cases[*]}")

    # Determine --disabled-tools flag for this config
    if [[ "$cfg" == "full" ]]; then
        disabled_flag=""
    else
        # cfg is like "no_postcode" → strip "no_" prefix
        disabled_flag="--disabled-tools ${cfg#no_}"
    fi

    log_file="$cfg_dir/rerun.log"
    echo "  invoking locate_only_eval.py --resume $disabled_flag --only-cases <$n cases>"
    echo "  log: $log_file"

    uv run python -u ablations/locate_only_eval.py \
        --resume \
        $disabled_flag \
        --only-cases "$only" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}

    if [[ "$rc" != "0" ]]; then
        echo "  !!! $cfg rerun exited with code $rc — continuing"
        n_failed=$((n_failed + 1))
    fi

    # Verify the row count after the rerun matches the original (208).
    final_rows=$(($(wc -l < "$cfg_dir/locate_picks.csv") - 1))
    echo "  [$cfg] post-rerun row count: $final_rows / 208"
    n_total_reran=$((n_total_reran + n))
done

echo
echo "================================================================"
echo "Rerun complete."
echo "End:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Configs failed: $n_failed"
echo "Cases reran:    $n_total_reran"
echo "================================================================"
