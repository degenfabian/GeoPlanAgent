# Paper audit — fresh-eyes cross-check

You are a fresh Claude Code instance with no prior context on this repository
or paper. Your job is to **audit `paper.tex` against the codebase that
produced it** and report every place where the paper's description, every
constant it cites, or every metric it reports fails to match the actual
implementation or the actual numerical data on disk. You are not writing
code, not running experiments, and not touching the paper. You produce one
markdown report and exit.

The previous Claude Code instance just rewrote large parts of the paper
methodology and added Results / Ablations sections with placeholders. The
person who asked for this audit specifically wants an independent
verification pass — assume the paper is wrong until you have read the code
that proves otherwise.

## Repository orientation (read these first, in order)

1. `README.md` — top-level orientation. What the pipeline does, current
   tool list, output layout, model alias table. Use as a navigation aid,
   **not** as ground truth — verify everything against code.
2. `tools/README.md` — package map. Same caveat.
3. `tools/agent/README.md` — the agent pipeline. Same caveat.
4. `paper.tex` — the document you are auditing.

The three README files were written by the same person whose paper you are
auditing, so they may share staleness. **Source of truth is always code +
on-disk data, never prose.**

## What to check (full coverage)

### 1. Tool names and counts

For every statement of the form "the worker has N tools" / "the locate
sub-agent has M geocoder tools" / "tools 1, 2, 3 are X, Y, Z":

- Worker tools are decorated `@_agent.tool` under `tools/agent/tools/*.py`.
  Enumerate them.
- Locate sub-agent tools are decorated `@_locate_agent.tool_plain` in
  `tools/agent/locate_agent.py`. Enumerate them.
- Cross-reference against Table 5 (`tab:worker-tools`) in `paper.tex` and
  the prose in §4.3, §4.6, §4.7 Stage 2, §6.

### 2. Numerical constants in prose

For every numeric claim in the paper (a non-exhaustive list to seed you):

- Window stride / target window count (`WINDOW_STRIDE_TARGET` in
  `tools/matching/_core.py`)
- RANSAC reprojection threshold ("10-pixel" — verify in
  `tools/matching/_core.py:estimate_affine`)
- 6-DOF affine gates (`GATE_RATIO_6DOF`, `SCALE_6DOF_MIN/MAX` —
  `tools/matching/_core.py`)
- Delaunay-consistency filter band `[0.5, 2.0]` — `tools/delaunay_filter.py`
- Weak-retry trigger ("fewer than 25 inliers or overall score below 0.4"
  + "re-runs at 2σ") — `tools/agent/tools/match.py:_match_single_page`
- `OUTSIDE_LA_PENALTY` (paper says 0.3) — `tools/scoring.py`
- Composite window score (eq:composite-window) `score = V · Q/4 · 1/(1+d_km)`
  — `tools/scoring.py:composite_window_score`
- Commit gate (eq:commit-score) — `tools/scoring.py:commit_attempt_score`
- Strict gate ("fewer than 18 inliers or mask covers less than 0.2% of
  the image") — verify in `tools/agent/tools/match.py:commit_match`. **This
  is a known historical claim that may be stale.**
- `match_at_budget` (paper says 5) — `tools/agent/state.py`
- `REFINE_BUDGET_PER_CASE` (paper says 3) — `tools/agent/tools/refine.py`
- Locate sub-agent budget (paper says 8 geocode calls) —
  `tools/agent/locate_agent.py` `LOCATE_SYSTEM_PROMPT` + agent config
- Locate sub-agent σ values (200 / 300-500 / 800-1500 / LA-radius) —
  `LOCATE_SYSTEM_PROMPT` in `tools/agent/locate_agent.py`
- Letterhead postcode filter ("5 km from named LA") — verify the rule
  text in `LOCATE_SYSTEM_PROMPT` matches the paper
- Auto-rotation confidence-abstain threshold (paper says 0.80) —
  `tools/io/rotation_classifier.py`
- Auto-rotation TTA mechanic (4-rotation, cyclically shifted, mean
  softmax) — `tools/io/rotation_classifier.py`
- Rotation classifier training data ("808 samples from 202 cases") —
  verify in `training/train_rotation_kfold.py` + the rotation dataset
  files
- LoRA rank, learning rate, weight decay, grad clip, batch size, grad
  accum, patience, epochs — `training/train_sam3_kfold.py` (search for
  `--lr`, `--rank`, default values, AdamW)
- Loss weights (5/5 for sem focal+dice, ramped 0.5 surface, 5/5/2/1 for
  instance focal+dice+cls+pres) — `LOSS_WEIGHT_*` constants at the top
  of `training/train_sam3_kfold.py`
- Focal α / γ for each head (sem: α=0.6 γ=1.6, inst: α=0.25 γ=2) —
  `training/train_sam3_kfold.py` (search `sigmoid_focal_loss`,
  `semantic_loss`, `instance_loss`)
- Hungarian matching cost `cost = -IoU - 0.05·σ(cls_i)` — verify the
  0.05 weight against `train_sam3_kfold.py`
- Soft positive cls target `σ(cls_best)^0.25 · IoU_best^0.75` — verify
  exponents and operands
- Tile rendering: "5% bbox inflation", road widths at z=17, colour codes
  (`#F4CCCC` pink buildings, etc.) — `tools/io/os_tiles.py`
- Mask cleanup primitives: "5% largest-component floor", "100-pixel
  absolute floor", "10% bbox foreground triggers thin-outline expand",
  "1.5% of bbox smaller dim" kernel, "1% of smaller image dim" closing
  kernel, "Douglas-Peucker ε=3px" — `tools/extraction/mask_ops.py` and
  `tools/matching/_core.py:mask_to_geojson_affine`

### 3. SAM3 cross-validation numbers (TABLE tab:sam3-cv in §8)

The paper reports per-fold values:

| Fold | $|V|$ | sem_iou | sem_f1 | inst_iou |
|---|---|---|---|---|
| 0 | 43 | 0.877 | 0.908 | 0.867 |
| 1 | 42 | 0.922 | 0.946 | 0.922 |
| 2 | 42 | 0.827 | 0.860 | 0.827 |
| 3 | 42 | 0.879 | 0.914 | 0.876 |
| 4 | 42 | 0.953 | 0.974 | 0.954 |
| Mean | 211 | 0.892 | 0.920 | 0.889 |
| Std  |     | 0.043 | 0.038 | 0.044 |

**Verify every cell** against `models/sam3_lora/cv_summary.json` to 3
decimal places. If a cell mismatches, report paper value vs JSON value
and flag a fix.

### 4. Rotation classifier numbers (§A.2 + §8 abl-rotation-cv)

The paper claims per-fold accuracy: 0.924 / 0.976 / 0.988 / 0.946 /
0.964, mean **0.960 ± 0.024**. Verify against
`models/rotation_classifier_kfold/kfold_summary.json` (look for
`best_val_acc` per fold).

### 5. Dataset size claims

- Paper says **270 cases** total. Verify with
  `evaluation_data/0_planning_dataset_list.xlsx` row count (header
  excluded). The benchmark runner at `benchmark_runner.py` also drops
  some duplicates (`DUPLICATE_SL_NOS`); count what actually gets
  benchmarked. The paper says **208 cases** are benchmarked — verify.
- Paper says **211 cases** in the SAM3 fine-tune training pool. Verify
  with `wc -l training/dataset/manifest.json` (or count entries in the
  JSON).
- Geographic distribution (London 37%, Kent 15%, Norfolk 15%, etc.) —
  these are in §3 Dataset which the auditor is **not** asked to
  verify (it was written by a different author). Skip.

### 6. Pipeline architecture (§4.7)

Verify each stage's description against the actual run flow:

- Stage 1 (Reader) — `tools/agent/__init__.py:run_agent` ->
  `tools/agent/runtime.py:read_pdf_phase`. Does it really use both PDF
  binary and OCR text? Does OCR cascade really go fitz → macOS Vision
  → PaddleOCR?
- Stage 2 (Locate) — `tools/agent/tools/locate.py:propose_centers`
  → `tools/agent/locate_agent.py:run_locate`. Does the sub-agent really
  emit a `LocatePick` schema with the claimed fields (top_lat, top_lon,
  sigma_m, confidence, picked_source, evidence, la_check_passed)?
- Stage 3 (Match) — `tools/agent/tools/match.py:match_at` →
  `tools/matching/_core.py:sliding_window_position`. Multi-zoom, scale
  perturbation, RANSAC, Delaunay filter all in there?
- Stage 4 (Segment) — paper says segmentation happens inside `match_at`,
  not as a post-commit stage. Verify in
  `tools/agent/tools/match.py:_match_single_page` and
  `_get_or_compute_mask`. Verify the text prompt is locked to
  `"planning boundary"`.

### 7. Schemas + output validator

- `PDFInfo` fields: paper says "approximately twenty fields". Count
  them in `tools/agent/schemas.py`.
- `BoundaryOutcome` fields and the validator preconditions: paper says
  "every borderline match in the 25-100 inlier band must be visually
  verified" — verify in `tools/agent/worker_agent.py:validate_boundary_outcome`.
  Verify the validator also enforces `lookup_district` success when
  `status="district_lookup"`.
- `LocatePick` fields — `tools/agent/locate_agent.py`.

### 8. Equations (cite paper line, find code)

- `eq:sam3-loss` (§A.6) — sem 5·focal + 5·dice + ramped 0.5·surf;
  inst 5·focal + 5·dice + 2·cls + 1·pres. Check **every coefficient**
  and α/γ value. Code is in `training/train_sam3_kfold.py`.
- `eq:composite-window` (§4.8) — `V · Q/4 · 1/(1+d_km)`. Code in
  `tools/scoring.py:composite_window_score`.
- `eq:commit-score` (§4.9) — `n_inliers · {1.0 if in LA, 0.3 if not}`.
  Code in `tools/scoring.py:commit_attempt_score`.
- IoU / Precision / Recall / F1 equations (§4.5) — verify against
  `tools/metrics/geojson.py:calculate_spatial_metrics`.

### 9. Citations referenced but possibly missing

The paper now references the citation key `mhclg_extract` (§7
Results). Verify the entry exists in `custom.bib`. If not, flag for
the author to add. Also verify `os_code_point_open`,
`os_open_names`, `os_boundary_line`, `os_open_zoomstack`, and any
other Ordnance Survey citation keys exist.

### 10. Placeholder discipline

The Results (§7) and Ablations (§8) sections deliberately use
`\placeholder` for results not yet computed. Verify:

- Every `\placeholder` corresponds to a result the user has explicitly
  said "we'll fill this when the benchmark/ablation finishes".
- No claim made WITHOUT a placeholder is unverifiable.

Specifically, the **SAM3 LoRA cross-val table is filled with real
numbers** — these must match `cv_summary.json`. The **rotation
classifier numbers are filled** — must match `kfold_summary.json`.
Everything else in §7-§8 should be `\placeholder` or framing prose only.

## Stale-reference watch list

The previous instance fixed these. Verify they remain fixed (zero
occurrences in `paper.tex`):

- `submit_pick` (locate sub-agent terminates via structured output)
- `pdf_info_text` (was renamed to `pdf_info`)
- `critic` / `Phase 3 critic` / `VLM critic` / `critic_log` /
  `critic_panel`
- `extract_boundary` / `project_boundary` (no longer separate worker
  tools)
- `render_page` / `geocode` / `visualize` (no longer worker tools)
- `analytical short-circuit` / `analytical_affine`
- `OSM` / `Nominatim` / `Overpass` (in district-lookup context)
- `auto-labeller` / `Path A` / `Path B` / `Path C` (was the auto-
  labelling pipeline; replaced by hand-annotation)
- `MapSAM` (model variant that didn't pan out, should not appear)
- `--include-training-cases` flag (retired)

Plus check for items that **may still be stale** (run-it-yourself):

- "fewer than 18 inliers" / "MIN_INLIERS_COMMIT" / "0.2% mask
  coverage" / "MIN_MASK_FRAC_COMMIT" — these described an older strict
  commit gate. The newer code refuses commits only when **no group
  produced a valid affine** (no inlier threshold). Verify the paper's
  current §4.9 commit-gate description against the actual code in
  `tools/agent/tools/match.py:commit_match`.
- "eleven tools" / "11 tools" — should be 6.

## Output format

Write a single markdown report to `paper_audit_report.md` in the repo
root, with these sections:

```
# Paper-vs-code audit report

## A. Stale references (highest priority)
Each item: paper section + line + the stale claim + what the code
actually does + suggested fix.

## B. Numerical drift
Each item: paper section + line + value in paper + value in code +
file:line of code source + suggested fix.

## C. Tool / API drift
Each item: paper claim about a tool, schema field, or interface +
what the code actually exposes + file:line + suggested fix.

## D. Confirmed-correct items
Bullet list, ≤ 80 chars each, of substantive claims you verified are
accurate. Include file:line citations.

## E. Number cross-check table

| # | Claim | Paper § / line | Code source (file:line) | Paper value | Code value | Match? |
|---|---|---|---|---|---|---|

(Include EVERY numerical claim you encountered. Aim for completeness —
40+ rows is fine.)

## F. Stale-watch results

| Term | Occurrences in paper.tex | Notes |
|---|---|---|

(Use the watch list from this prompt. One row per term.)

## G. Unverifiable / cannot tell
Each item: paper claim + what you'd need to verify it (e.g. "would
need to run the benchmark", "would need to compile the prose against
the figure").
```

## Constraints

- **READ-ONLY**. Do not edit any file. Do not run `git`, `uv`, or any
  command that mutates state. `grep`, `cat`, `find`, `wc`, `head`,
  `tail`, `Read`, `Glob` only.
- **Do not run** the benchmark, train models, call the OpenRouter API,
  or anything that costs money or time.
- **Trust order**: data files > code > READMEs > paper. If the code
  and a README disagree, code wins.
- **Be skeptical of yourself**. If you find yourself thinking "this
  looks right, skip it", that's exactly the kind of claim that
  drifts. Verify it.
- **Cite file:line for every code source you use**. The author will
  use these citations to fix things — bare claims like "the code says
  otherwise" without a citation are useless.

## What the paper is about (orientation, ≤ 100 words)

GeoPlanAgent: a three-LLM-agent pipeline (Reader → Worker → Locate
sub-agent) for extracting UK planning-permission application
boundaries from PDFs into WGS84 GeoJSON. The Reader parses the PDF
binary + per-page OCR text. The Worker calls 6 tools that drive
MINIMA-LoFTR feature matching against rendered OS OpenData tiles,
RANSAC affine recovery, a LoRA-fine-tuned SAM 3 for boundary
segmentation, and an OS BoundaryLine fallback for district-wide
documents. The Locate sub-agent uses 6 offline geocoder tools
(Code-Point Open, OS Open Names, OS OpenMap Local roads, OS
BoundaryLine LA polygons). The dataset is 270 hand-annotated UK
Article 4 Direction documents spanning 1973-2024; SAM 3 is fine-tuned
with 5-fold CV on 211 of them. Target venue: EMNLP 2026 main long
paper (8 pages + unlimited appendix).

## When you're done

Print the absolute path of `paper_audit_report.md` and stop. The
person reviewing your report will read it manually; they will not
expect you to fix anything. If your report is empty (everything
verified), say so explicitly in section D rather than leaving an
empty file.
