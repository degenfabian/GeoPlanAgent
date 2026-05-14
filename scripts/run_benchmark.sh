#!/usr/bin/env bash
# Run the full evaluation benchmark.
#
# All previous env-gate feature flags have been baked into the code as
# defaults. No flags needed.
#
# Usage:
#   scripts/run_benchmark.sh                       # writes to results/benchmark
#   scripts/run_benchmark.sh results/my_run        # custom output dir
#
# Estimated cost: ~$3-5 (Gemini Flash, 215 cases × ~12 iterations avg)
# Estimated runtime: ~3.5h on Apple Silicon

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:-results/benchmark}"

uv run benchmark_runner.py \
  --model gemini-flash \
  --max-iterations 12 \
  --output-dir "$OUT_DIR" \
  --force
