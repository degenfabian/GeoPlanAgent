#!/usr/bin/env bash
#
# Run all 7 locate-LOO ablation configs sequentially.
#
# Order: full baseline first, then each disabled-tool variant. Each
# config writes its own CSV + per-case trajectories under
#   ablations/locate_only_eval/<label>/
# plus a tee'd stdout log at
#   ablations/locate_only_eval/<label>/run.log
#
# Resume-safe — if you ctrl-C mid-run or a config errors out, re-run
# with RESUME=1 to pick up where the partial CSVs left off:
#   RESUME=1 bash ablations/run_locate_loo.sh
#
# Usage:
#   bash ablations/run_locate_loo.sh                # fresh run
#   RESUME=1 bash ablations/run_locate_loo.sh      # incremental
#   nohup bash ablations/run_locate_loo.sh > all.log 2>&1 &   # background
#
# Per-config wall-clock estimate (gemini-flash, 208 cases): ~30-60 min.
# All 7 configs sequential: ~3.5-7 hours.

set -u   # fail on undefined vars; NOT `set -e` — we want to finish
         # remaining configs even if one errors out.

cd "$(dirname "$0")/.."   # cd to repo root regardless of where invoked

RESUME_FLAG=""
if [[ "${RESUME:-}" == "1" ]]; then
    RESUME_FLAG="--resume"
    echo "RESUME=1 set — passing --resume to each config"
fi

# Baseline (empty) + 6 LOOs. Order matters for two reasons:
#   - Baseline first so a partial run still has the "control".
#   - la_check last because it's the most distinctive variant
#     (verifier vs signal generators) and most likely to surface
#     unexpected behaviour worth inspecting before going further.
CONFIGS=("" "postcode" "grid_ref" "place" "road" "intersect" "la_check")

START_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "================================================================"
echo "Locate LOO ablation — all 7 configs"
echo "Start: $START_TS"
echo "================================================================"

n_failed=0
for cfg in "${CONFIGS[@]}"; do
    if [[ -z "$cfg" ]]; then
        flag=""
        label="full"
    else
        flag="--disabled-tools $cfg"
        label="no_$cfg"
    fi

    out_dir="ablations/locate_only_eval/$label"
    mkdir -p "$out_dir"
    log_file="$out_dir/run.log"

    echo
    echo "================================================================"
    echo "Config: $label  (start: $(date '+%H:%M:%S'))"
    echo "Log:    $log_file"
    echo "================================================================"

    # Tee combined stdout + stderr to the per-config log.
    # PIPESTATUS[0] preserves the python exit code through tee.
    uv run python ablations/locate_only_eval.py $flag $RESUME_FLAG 2>&1 \
        | tee "$log_file"
    rc=${PIPESTATUS[0]}

    if [[ "$rc" != "0" ]]; then
        echo
        echo "!!! Config '$label' exited with code $rc — see $log_file"
        echo "    Continuing with remaining configs; rerun with RESUME=1 later."
        n_failed=$((n_failed + 1))
    fi
done

END_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo
echo "================================================================"
echo "All 7 configs done."
echo "Start: $START_TS"
echo "End:   $END_TS"
if [[ "$n_failed" != "0" ]]; then
    echo "Failures: $n_failed config(s) exited non-zero."
fi
echo "================================================================"
echo
echo "Per-config row counts in locate_picks.csv:"
for cfg in "${CONFIGS[@]}"; do
    label=$([[ -z "$cfg" ]] && echo "full" || echo "no_$cfg")
    csv="ablations/locate_only_eval/$label/locate_picks.csv"
    if [[ -f "$csv" ]]; then
        # subtract 1 for header
        n=$(($(wc -l < "$csv") - 1))
        printf "  %-14s %4d rows\n" "$label" "$n"
    else
        printf "  %-14s (no CSV)\n" "$label"
    fi
done
