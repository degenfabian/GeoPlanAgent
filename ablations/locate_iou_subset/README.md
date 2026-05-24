# Locate-stage IoU regression test: min_1_tool vs full kit (subset)

Goal: check whether dropping the locate sub-agent from 6 tools to 1 tool
(`place` only) causes the full pipeline to lose IoU on the cases most
likely to regress.

## The 11-case set

From `ablations/locate_only_eval/`: cases where `min_1_tool`'s locate-stage
error crossed the 1km boundary upward vs `full` (full ≤ 1km, min_1_tool > 1km).
These are the cases where MINIMA recovery is most at risk if locate
degrades:

```
23:00006:REG_4    full=0.001km → min1=2.748km    full_IoU=0.998
23:53155:ART4     full=0.806km → min1=3.237km    full_IoU=0.924
CPA4(2a)          full=0.087km → min1=2.401km    full_IoU=0.934
A4D14A1           full=0.002km → min1=2.198km    full_IoU=0.999
12:00117:ART4     full=0.121km → min1=1.513km    full_IoU=0.897
12:00141:ART4     full=0.129km → min1=1.247km    full_IoU=0.783
11                full=0.726km → min1=1.563km    full_IoU=0.000
ARTICLE:210       full=0.692km → min1=1.387km    full_IoU=0.000
A4D6              full=0.354km → min1=1.027km    full_IoU=0.937
SSA409            full=0.864km → min1=1.361km    full_IoU=0.976
SSA416            full=0.881km → min1=1.191km    full_IoU=0.000
```

8 of these currently pass with IoU ≥ 0.5 in the 6-tool benchmark
(`results/benchmark_v_post_refactor/gemini-flash`); the other 3 are
already zero. The test is whether any of the 8 flip to zero (or drop
meaningfully) when locate is restricted to `place` only.

## How to run

The benchmark runner now accepts `--locate-disabled-tools` (comma-separated).
For `min_1_tool` (place only), disable the other five:

```bash
uv run benchmark_runner.py \
    --model gemini-flash \
    --max-iterations 12 \
    --output-dir results/benchmark_min1_subset \
    --cases $(tr '\n' ' ' < ablations/locate_iou_subset/cases.txt) \
    --locate-disabled-tools postcode,grid_ref,road,intersect,la_check \
    --force
```

That runs all 11 cases end-to-end with the min_1_tool locate kit. Pre-cache
all the heavy upstream artefacts (reader, SAM3) are loaded once per
process so wall clock is ~10-15 min × the longest case.

## Compare the IoUs

After the run finishes:

```bash
uv run python ablations/locate_iou_subset/compare_iou.py
```

Defaults read from `results/benchmark_v_post_refactor/gemini-flash`
(the 6-tool baseline) and `results/benchmark_min1_subset/gemini-flash`
(the new run), and the case list from `cases.txt`. Override via
`--full-dir`, `--min1-dir`, `--cases-file`.

Output shows side-by-side IoUs, Δ per case, and counts of "lost"
(≥0.5 → <0.5) vs "held" (|Δ|≤0.05) cases. Δmean over the paired set
is the headline number to use when deciding whether to switch the
production kit.
