#!/usr/bin/env bash
#
# Run two multi-tool subset ablations to test if the locate sub-agent's
# 6-tool kit can be trimmed without meaningful performance loss.
#
#   min_3_tool = keep {place, postcode, la_check}
#              = disable {grid_ref, road, intersect}
#
#   min_2_tool = keep {place, la_check}
#              = disable {grid_ref, postcode, road, intersect}
#
# Each writes into ablations/locate_only_eval/<label>/ with the same
# CSV + trajectories schema as the LOO configs, so aggregation later
# unions them in cleanly.
#
# Usage (from repo root):
#   bash ablations/run_subset_ablations.sh
#   nohup bash ablations/run_subset_ablations.sh > ablations/subset_master.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

CONFIGS=(
    "min_3_tool|grid_ref,road,intersect"
    "min_2_tool|grid_ref,postcode,road,intersect"
)

START_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "================================================================"
echo "Locate subset ablations (min_3_tool + min_2_tool)"
echo "Start: $START_TS"
echo "================================================================"

n_failed=0
for entry in "${CONFIGS[@]}"; do
    label=${entry%%|*}
    disabled=${entry#*|}

    out_dir="ablations/locate_only_eval/$label"
    mkdir -p "$out_dir"
    log_file="$out_dir/run.log"

    echo
    echo "================================================================"
    echo "Config: $label  (disabled: $disabled)"
    echo "Start: $(date '+%H:%M:%S')"
    echo "Log:    $log_file"
    echo "================================================================"

    uv run python -u ablations/locate_only_eval.py \
        --disabled-tools "$disabled" \
        --config-label "$label" \
        2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}

    if [[ "$rc" != "0" ]]; then
        echo "  !!! $label exited with code $rc — continuing"
        n_failed=$((n_failed + 1))
    fi

    n_rows=$(($(wc -l < "$out_dir/locate_picks.csv") - 1))
    echo "  [$label] post-run row count: $n_rows / 208"
done

END_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo
echo "================================================================"
echo "Subset ablations complete."
echo "Start: $START_TS"
echo "End:   $END_TS"
echo "Failures: $n_failed"
echo "================================================================"
