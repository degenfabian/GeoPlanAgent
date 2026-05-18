# Paper-vs-code audit report

Audit subject: `paper.tex` at commit `bf61a86` (main).
Source-of-truth order applied: data files > code > READMEs > paper.
All citations include `file:line` references for the author to act on.

---

## A. Stale references (highest priority)

### A1. §4.9 strict commit gate description is stale
**Paper §4.9 (paper.tex:405):**
> "The strict gate refuses any commit whose match has fewer than 18 inliers or whose accompanying SAM mask covers less than 0.2% of the rendered image — this band corresponds to commits whose mean IoU on a held-out audit was 0.44 versus 0.75 for accepted matches above the threshold ..."

**Code says otherwise:** `tools/agent/tools/match.py:653-664` only rejects when **no group produced a valid affine** (`n_groups_committed == 0`). There is no inlier-count threshold and no mask-fraction check. Constants `MIN_INLIERS_COMMIT` / `MIN_MASK_FRAC_COMMIT` do not exist anywhere in `tools/`.

**Cross-check:** `tools/agent/README.md:54` and `tools/README.md:84-86` correctly describe the current gate ("rejects commits where no group produced a valid affine").

**Fix:** Replace the entire "fewer than 18 inliers / 0.2 % mask coverage / 0.44 vs 0.75 audit IoU" passage with the current behaviour, e.g. "The strict gate refuses any commit where MINIMA failed to recover a usable affine for every area_group on the candidate; the worker must retry with a different page or a fresh `propose_centers` pick."

---

### A2. §A.1 PDFInfo description lists two fields that do not exist
**Paper §A.1 (paper.tex:381):**
> "(the printed scale, the colour of the boundary line, the page on which the canonical site map appears, the rotation needed to make the map upright, and an `is_district_wide` flag)"

**Code says otherwise:** `tools/agent/schemas.py:90-284` defines exactly 18 fields on `PDFInfo`. There is no `boundary_colour` / `boundary_line_colour` field, and there is no `rotation` field. Rotation is handled by a separate ResNet50 classifier in `tools/io/rotation_classifier.py` and applied at page-render time (`tools/agent/tools/match.py:74` checks `rot_info["applied"]`, not a reader-extracted field). The closest thing to "boundary line colour" is the per-page `MapPageMeta.boundary_clarity` enum `{clear, ambiguous, none}` (`tools/agent/schemas.py:54-60`), which is about visual clarity, not colour.

**Fix:** Drop "the colour of the boundary line" and "the rotation needed to make the map upright" from the §A.1 enumeration. Replace with the actual document-level fields present in the schema: `n_pages`, `district_name`, `admin_region`, `likely_town_or_city`, `directional_modifier`.

---

### A3. §A.5 fold-assignment description is stale
**Paper §A.5 (paper.tex:644):**
> "We use 5-fold cross-validation, splitting cases by a deterministic hash of the case identifier so that each case is held out from training in exactly one fold."

**Code says otherwise:** Training-time fold assignment is done by **longest-processing-time-first (LPT) bin-packing** that respects "stay-together" groups, not by hashing. See `scripts/build_sam3_training_set.py:71-87` (`_assign_folds_balanced`). The md5 hash in `tools/extraction/sam3.py:90-104` (`_fold_for_case`) is only the **inference-time fallback** for cases not present in `fold_assignment.json`.

Confusingly, §A.7 (paper.tex:654) describes the LPT bin-packing correctly. So §A.5 and §A.7 contradict each other.

**Fix:** Replace §A.5's "deterministic hash" sentence with a reference forward to §A.7, e.g. "We use 5-fold cross-validation; the per-case fold assignment is produced by an LPT bin-packing scheme described in §A.7."

---

### A4. `mhclg_extract` is cited but missing from `custom.bib`
**Paper §7 Headline Results (paper.tex:454):**
> `MHCLG Extract (public)~\cite{mhclg_extract} & 270 & --- & --- & --- & 0.82\textsuperscript{$\dagger$} & ...`

**`custom.bib` says otherwise:** no `@misc{mhclg_extract, ...}` entry exists in `custom.bib` (verified by enumerating all `@type{key,` blocks; only 29 entries, none with key `mhclg_extract`). Compilation will emit `?` for the citation key.

**Fix:** Add an `@misc{mhclg_extract, ...}` (or `@techreport`) entry in `custom.bib` pointing to the MHCLG / DLUHC public extract that the 0.82 figure was taken from.

---

## B. Numerical drift

### B1. Rotation-classifier abstain threshold: paper 0.80, code 0.50
**Paper §6 (paper.tex:433):** "...a ResNet50 page-rotation classifier...used with 4-rotation test-time augmentation + a 0.80 confidence-abstain rule"
**Paper §A.2 (paper.tex:604):** "...abstain (return zero rotation) whenever the top-class probability falls below 0.80."
**Paper §8.3 (paper.tex:549):** "...paired at inference with 4-rotation TTA and an 0.80-softmax abstain rule"

**Code says otherwise:** `tools/io/rotation_classifier.py:54` — `_DEFAULT_CONFIDENCE_THRESHOLD = 0.50`. The 0.50 default is what `predict_rotation_with_confidence` uses when called without an explicit `threshold=` kwarg, which is how every caller in the production pipeline calls it (no override found in `tools/`, `benchmark_runner.py`, or `tools/agent/`). Both the module docstring (tools/io/rotation_classifier.py:18, :27) and the README (`tools/io/README.md:14, :94`) say 0.80 — the discrepancy is internal to the repo, but the **constant the production code actually evaluates is 0.50**.

**Fix:** either flip the constant to 0.80 in code (and update docstring/README), or change every "0.80" in §6 / §A.2 / §8.3 to "0.50". The constant is the load-bearing thing — pick one and align.

---

### B2. Rotation classifier training-set size: paper 202 cases / 808 samples, actual 211 / 844
**Paper §A.2 (paper.tex:604):** "applying each of the four rotations to **202 maps** that have been visually confirmed upright, yielding **808 samples**"

**Data says otherwise:**
- `rotation_annotations.json` contains **211** labelled cases (verified with `python3 -c "import json; print(len([k for k,v in json.load(open('rotation_annotations.json')).items() if not k.startswith('__') and isinstance(v,int) and v in (0,90,180,270)]))"` → 211).
- `models/rotation_classifier_kfold/kfold_summary.json` per-fold `n_train + n_val` sums to 211 (168+43, 169+42×4) for every fold.
- Training script: `training/train_rotation_kfold.py:14` documents "Each labelled case generates 4 training samples by applying all 4 CW rotations" → 4×211 = **844 samples**.

The label distribution `{0: 189, 90: 15, 180: 1, 270: 6}` (sums to 211) matches the comment in `train_rotation_kfold.py:18` ("189/15/1/6 in our case").

**Fix:** "**211 maps** that have been visually confirmed upright, yielding **844 samples**". Also note that the training and SAM 3 fine-tune pool are the same 211 cases (a useful detail).

---

### B3. Rotation classifier std: paper 0.024, computed 0.022 (pop) / 0.025 (sample)
**Paper §8.3 (paper.tex:549):** "Mean validation top-1 accuracy is **0.960 ± 0.024**"

**Data says otherwise:** Computing from `models/rotation_classifier_kfold/kfold_summary.json` `best_val_acc`:
- per-fold: 0.9244, 0.9762, 0.9881, 0.9464, 0.9643
- mean = 0.9599 → rounds to 0.960 ✓
- population std (ddof=0) = 0.0224 → rounds to 0.022
- sample std (ddof=1) = 0.0251 → rounds to 0.025

Neither matches 0.024 exactly. Minor (within rounding tolerance of either convention) but worth picking a convention and stating it.

**Fix:** State explicitly which std (pop vs sample) and use the matching value. Recommended: "0.960 ± 0.022 (population std over 5 folds)".

---

### B4. Tile-renderer building colour: paper #F4CCCC, code BGR(179, 179, 255) = RGB #FFB3B3
**Paper §A.4 (paper.tex:630):** "pink (**#F4CCCC**) building footprints with grey outlines"

**Code says otherwise:** `tools/io/os_tiles.py:143` — `"building": (179, 179, 255),  # salmon/pink fill` in BGR. Converting BGR(179, 179, 255) to RGB gives (255, 179, 179) = **#FFB3B3**. The paper's claimed `#F4CCCC` = RGB(244, 204, 204), a noticeably paler pink.

**Fix:** "pink (**#FFB3B3**) building footprints with grey outlines" (or update the code constant if #F4CCCC is what the renderer is supposed to use).

---

### B5. Worker turn cap: paper "soft cap of 12", code default 8
**Paper §7 (paper.tex:438):** "Each case is allowed a soft cap of **12 worker turns**"

**Code says otherwise:** Default `max_iterations=8` in both `benchmark_runner.py:100, :488` and `tools/agent/__init__.py:58`. The README example uses `--max-iterations 12` (`README.md:94`), and the user memory note "Default to `--max-iterations 12`" matches the production invocation rather than the code default. This claim is **only correct if every benchmark run reported in §7 was actually launched with `--max-iterations 12`**; otherwise the paper overstates the budget.

**Fix:** Either confirm that the `results/benchmark_v_R21/` run was launched with `--max-iterations 12` and add a footnote stating the CLI flag, or change the paper text to "Each case is allowed a soft cap of 8 worker turns by default (12 in the runs reported below)".

---

## C. Tool / API drift

### C1. `PDFInfo` no longer carries rotation
Paper §A.1 lists "the rotation needed to make the map upright" as a `PDFInfo` field. As detailed in A2, no such field exists. Rotation classification is a separate, downstream concern (`tools/io/rotation_classifier.py`).

### C2. `PDFInfo` no longer carries boundary-line colour
Paper §A.1 lists "the colour of the boundary line" as a `PDFInfo` field. No such field exists; closest is `MapPageMeta.boundary_clarity` (`tools/agent/schemas.py:54-60`), which is an enum about *clarity* not colour.

### C3. "rural override" rule for the 0.40 commit threshold doesn't exist as such
**Paper §4.3 (paper.tex:319):** "...reject on overall scores below 0.40 unless an **explicit rural override** applies"

**Code says otherwise:** `tools/agent/prompts.py:221` has the rule "< 0.40 on the first try → reject; try another center." There is no explicit "rural override". The prompt does flag that `road_name_agreement = 0.5` with verdict "no OS roads within radius" is a *rural-sparsity* signal that shouldn't be treated as a wrong-area indicator (lines 232-236) — but that affects the multi-axis verdict, not the 0.40 reject threshold.

**Fix:** Either point to a real override rule in the prompt and quote it, or drop the "unless an explicit rural override applies" clause.

---

## D. Confirmed-correct items

- SAM 3 LoRA 5-fold CV per-fold and summary stats (Table `sam3-cv`) match `models/sam3_lora/cv_summary.json` to 3 dp.
- Rotation classifier per-fold accuracies (0.924 / 0.976 / 0.988 / 0.946 / 0.964) and mean 0.960 match `models/rotation_classifier_kfold/kfold_summary.json`.
- Worker tool count = 6 (`propose_centers`, `match_at`, `commit_match`, `verify_position`, `lookup_district`, `reader_refine`) — `tools/agent/tools/{locate,match,verify,refine}.py`.
- Locate sub-agent tool count = 6 (`postcode`, `grid_ref`, `place`, `road`, `intersect`, `la_check`) — `tools/agent/locate_agent.py:153,177,201,241,292,391`.
- Locate sub-agent budget of 8 geocode calls — `tools/agent/locate_agent.py:129`.
- σ buckets (200 / 300-500 / 800-1500 / LA-radius) and 5-km letterhead drop — `tools/agent/locate_agent.py:109-122`.
- `match_at_budget = 5` — `tools/agent/state.py:91`.
- `REFINE_BUDGET_PER_CASE = 3` — `tools/agent/tools/refine.py:26`.
- `WINDOW_STRIDE_TARGET = 100` — `tools/matching/_core.py:61`.
- RANSAC reproj threshold = 10 px — `tools/matching/_core.py:124,150`.
- 6-DOF gates (GATE_RATIO_6DOF=1.3, aspect≥0.85, scale∈[0.3,3.0], det>0, shear<0.15) — `tools/matching/_core.py:51-53, 162-174`.
- Delaunay-consistency band [0.5, 2.0] and survival floor max(8, n//3) — `tools/delaunay_filter.py:17,37`, `tools/matching/_core.py:185-194`.
- Weak-retry trigger (< 25 inliers OR overall_score < 0.4, retry at 2σ) — `tools/agent/tools/match.py:483-489`.
- `OUTSIDE_LA_PENALTY = 0.3` and `commit_attempt_score` form — `tools/scoring.py:103-118`.
- `composite_window_score = V · Q/4 · 1/(1+d_km)` — `tools/scoring.py:22-47`.
- "+5 cases at IoU≥0.8 on 211-case overnight sweep" — supported by docstring at `tools/scoring.py:37`.
- Output validator: 25-100 band requires `verify_position` + ≥20 char notes — `tools/agent/worker_agent.py:125-143`.
- Output validator: `district_lookup` requires GeoJSON — `tools/agent/worker_agent.py:104-110`.
- `LocatePick` schema fields (`top_lat`, `top_lon`, `sigma_m`, `confidence`, `picked_source`, `evidence`, `la_check_passed`) — `tools/agent/locate_agent.py:36-64`.
- Dataset = 270 cases in Excel; 208 actually benchmarked after dropping 5 duplicates and adding 5 `*_merged` extras — verified by running the `benchmark_runner.py` selection logic against `evaluation_data/`.
- Training pool = 211 cases — `training/dataset/manifest.json` has 211 entries.
- SAM 3 text prompt fixed to "planning boundary" both in training and inference — `training/train_sam3_kfold.py:111` and `tools/agent/tools/match.py:45`.
- SAM 3 base model `facebook/sam3` — `tools/extraction/sam3.py:158-159`.
- SAM 3 mask cached per page — `tools/agent/tools/match.py:84-99`.
- Fold routing via `set_fold_for_case` — `tools/extraction/sam3.py:278-305`.
- LoRA target modules `q_proj, k_proj, v_proj, o_proj, fc1, fc2` (paper "{q,k,v,o} + fc_1, fc_2") — `training/train_sam3_kfold.py:95`.
- LoRA rank=16, AdamW lr=2e-4, weight_decay=0.01, batch=1, grad_accum=4, grad_clip=0.1, patience=6, ≤20 epochs, cosine→5%, bf16 — `training/train_sam3_kfold.py:489,492,796-824`.
- Loss weights 5/5 sem (focal+dice), 5/5/2/1 inst (focal+dice+cls+pres), 0.5 surf max — `training/train_sam3_kfold.py:102-108`.
- Surface ramp `r(e) = min(1, e/15)` (`SURFACE_LOSS_RAMP = 15`) — `training/train_sam3_kfold.py:109, 203`.
- Focal α/γ: sem α=0.6 γ=1.6, inst α=0.25 γ=2 — `training/train_sam3_kfold.py:198, 306, 336`.
- Hungarian matching cost `-IoU - 0.05·σ(cls)` — `training/train_sam3_kfold.py:285-294`.
- Soft positive cls target `σ(cls_best)^0.25 · IoU_best^0.75` — `training/train_sam3_kfold.py:330-334`.
- ±15 % scale perturbations on the chosen mpp — `tools/matching/_core.py:489-490, 508-509`.
- 4-rotation TTA with cyclic-shift ensemble — `tools/io/rotation_classifier.py:298-322`.
- Mask cleanup: 5 % largest-component floor + 100 px absolute floor — `tools/extraction/mask_ops.py:119-162`.
- Thin-outline expand triggered at < 10 % of bbox foreground, safety-capped at 50 % — `tools/extraction/mask_ops.py:90,111`.
- Closing kernel ≈ 1 % of smaller image dim — `tools/extraction/mask_ops.py:38`.
- Douglas-Peucker ε = 3 px and 100-px external-contour filter — `tools/matching/_core.py:303, 333-335`.
- Tile bbox inflation 5 % — `tools/io/os_tiles.py:108`.
- Road widths at z=17: Local Street 1.5 px fill (paper's claim of "1.5 px wide") — `tools/io/os_tiles.py:161` (fill, casing) = (1.5, 2.5).
- IoU / precision / recall / F1 equations — `tools/metrics/geojson.py:197-215`.
- Stale-watch list — every term in the prompt has zero occurrences in `paper.tex` (see §F).

---

## E. Number cross-check table

| # | Claim | Paper § / line | Code source (file:line) | Paper value | Code value | Match? |
|---|---|---|---|---|---|---|
| 1 | SAM 3 fold 0 sem_iou | §8.2 / 534 | models/sam3_lora/cv_summary.json | 0.877 | 0.8773 → 0.877 | ✓ |
| 2 | SAM 3 fold 0 sem_f1 | §8.2 / 534 | cv_summary.json | 0.908 | 0.9081 → 0.908 | ✓ |
| 3 | SAM 3 fold 0 inst_iou | §8.2 / 534 | cv_summary.json | 0.867 | 0.867 | ✓ |
| 4 | SAM 3 fold 1 sem_iou | §8.2 / 535 | cv_summary.json | 0.922 | 0.9221 → 0.922 | ✓ |
| 5 | SAM 3 fold 1 sem_f1 | §8.2 / 535 | cv_summary.json | 0.946 | 0.9462 → 0.946 | ✓ |
| 6 | SAM 3 fold 1 inst_iou | §8.2 / 535 | cv_summary.json | 0.922 | 0.9221 → 0.922 | ✓ |
| 7 | SAM 3 fold 2 sem_iou | §8.2 / 536 | cv_summary.json | 0.827 | 0.8274 → 0.827 | ✓ |
| 8 | SAM 3 fold 2 sem_f1 | §8.2 / 536 | cv_summary.json | 0.860 | 0.8596 → 0.860 | ✓ |
| 9 | SAM 3 fold 2 inst_iou | §8.2 / 536 | cv_summary.json | 0.827 | 0.8266 → 0.827 | ✓ |
| 10 | SAM 3 fold 3 sem_iou | §8.2 / 537 | cv_summary.json | 0.879 | 0.8788 → 0.879 | ✓ |
| 11 | SAM 3 fold 3 sem_f1 | §8.2 / 537 | cv_summary.json | 0.914 | 0.9139 → 0.914 | ✓ |
| 12 | SAM 3 fold 3 inst_iou | §8.2 / 537 | cv_summary.json | 0.876 | 0.8763 → 0.876 | ✓ |
| 13 | SAM 3 fold 4 sem_iou | §8.2 / 538 | cv_summary.json | 0.953 | 0.9531 → 0.953 | ✓ |
| 14 | SAM 3 fold 4 sem_f1 | §8.2 / 538 | cv_summary.json | 0.974 | 0.9737 → 0.974 | ✓ |
| 15 | SAM 3 fold 4 inst_iou | §8.2 / 538 | cv_summary.json | 0.954 | 0.9535 → 0.954 | ✓ |
| 16 | SAM 3 mean sem_iou | §8.2 / 539 | cv_summary.json | 0.892 | 0.8917 → 0.892 | ✓ |
| 17 | SAM 3 mean sem_f1 | §8.2 / 539 | cv_summary.json | 0.920 | 0.9203 → 0.920 | ✓ |
| 18 | SAM 3 mean inst_iou | §8.2 / 539 | cv_summary.json | 0.889 | 0.8891 → 0.889 | ✓ |
| 19 | SAM 3 std sem_iou | §8.2 / 540 | cv_summary.json | 0.043 | 0.0429 → 0.043 | ✓ |
| 20 | SAM 3 std sem_f1 | §8.2 / 540 | cv_summary.json | 0.038 | 0.0385 → 0.038 | ✓ |
| 21 | SAM 3 std inst_iou | §8.2 / 540 | cv_summary.json | 0.044 | 0.0443 → 0.044 | ✓ |
| 22 | SAM 3 mean |V| | §8.2 / 539 | cv_summary.json `n_total_val` | 211 | 211 | ✓ |
| 23 | Rotation fold 0 acc | §8.3 / 549 | kfold_summary.json fold 0 | 0.924 | 0.9244 → 0.924 | ✓ |
| 24 | Rotation fold 1 acc | §8.3 / 549 | kfold_summary.json fold 1 | 0.976 | 0.9762 → 0.976 | ✓ |
| 25 | Rotation fold 2 acc | §8.3 / 549 | kfold_summary.json fold 2 | 0.988 | 0.9881 → 0.988 | ✓ |
| 26 | Rotation fold 3 acc | §8.3 / 549 | kfold_summary.json fold 3 | 0.946 | 0.9464 → 0.946 | ✓ |
| 27 | Rotation fold 4 acc | §8.3 / 549 | kfold_summary.json fold 4 | 0.964 | 0.9643 → 0.964 | ✓ |
| 28 | Rotation mean acc | §8.3 / 549 | kfold_summary.json | 0.960 | 0.9599 → 0.960 | ✓ |
| 29 | Rotation std | §8.3 / 549 | computed from kfold_summary.json | 0.024 | 0.022 pop / 0.025 sample | ✗ |
| 30 | Rotation training set size | §A.2 / 604 | rotation_annotations.json | 202 maps | 211 maps | ✗ |
| 31 | Rotation training samples | §A.2 / 604 | rotation_annotations.json × 4 | 808 | 844 | ✗ |
| 32 | Rotation abstain threshold | §6 / §A.2 / §8.3 | tools/io/rotation_classifier.py:54 | 0.80 | 0.50 | ✗ |
| 33 | Dataset total | §7 / 438 + §3 / 153 | evaluation_data/0_planning_dataset_list.xlsx | 270 | 270 | ✓ |
| 34 | Dataset benchmarked | §7 / 451 | benchmark_runner.py selection logic | 208 | 208 | ✓ |
| 35 | SAM 3 training pool | §A.7 / 654 + §A.5 | training/dataset/manifest.json | 211 | 211 | ✓ |
| 36 | LoRA rank | §A.5 / 635 | training/train_sam3_kfold.py:799 | 16 | 16 | ✓ |
| 37 | AdamW lr | §A.5 / 642 | train_sam3_kfold.py:800 | 2e-4 | 2e-4 | ✓ |
| 38 | weight decay | §A.5 / 642 | train_sam3_kfold.py:489 | 0.01 | 0.01 | ✓ |
| 39 | grad clip | §A.5 / 642 | train_sam3_kfold.py:803 | 0.1 | 0.1 | ✓ |
| 40 | batch size | §A.5 / 642 | train_sam3_kfold.py:801 | 1 | 1 | ✓ |
| 41 | grad accum | §A.5 / 642 | train_sam3_kfold.py:802 | 4 | 4 | ✓ |
| 42 | patience | §A.5 / 642 | train_sam3_kfold.py:821 | 6 | 6 | ✓ |
| 43 | max epochs | §A.5 / 642 | train_sam3_kfold.py:796 | 20 | 20 | ✓ |
| 44 | cosine min lr | §A.5 / 642 | train_sam3_kfold.py:492 | 5 % of base | `eta_min=lr*0.05` | ✓ |
| 45 | sem focal α | §A.5 / 637 | train_sam3_kfold.py:198 | 0.6 | 0.6 | ✓ |
| 46 | sem focal γ | §A.5 / 637 | train_sam3_kfold.py:198 | 1.6 | 1.6 | ✓ |
| 47 | inst focal α | §A.5 / 637 | train_sam3_kfold.py:306,336 | 0.25 | 0.25 | ✓ |
| 48 | inst focal γ | §A.5 / 637 | train_sam3_kfold.py:306,336 | 2 | 2.0 | ✓ |
| 49 | sem focal weight | §A.5 / 637 | train_sam3_kfold.py:102 | 5 | 5.0 | ✓ |
| 50 | sem dice weight | §A.5 / 637 | train_sam3_kfold.py:103 | 5 | 5.0 | ✓ |
| 51 | surface weight max | §A.5 / 637 | train_sam3_kfold.py:104 | 0.5 | 0.5 | ✓ |
| 52 | inst focal weight | §A.5 / 638 | train_sam3_kfold.py:105 | 5 | 5.0 | ✓ |
| 53 | inst dice weight | §A.5 / 638 | train_sam3_kfold.py:106 | 5 | 5.0 | ✓ |
| 54 | inst cls weight | §A.5 / 638 | train_sam3_kfold.py:107 | 2 | 2.0 | ✓ |
| 55 | inst pres weight | §A.5 / 638 | train_sam3_kfold.py:108 | 1 | 1.0 | ✓ |
| 56 | surface ramp epochs | §A.5 / 642 | train_sam3_kfold.py:109 | 15 | 15 | ✓ |
| 57 | Hungarian cls coefficient | §A.5 / 642 | train_sam3_kfold.py:294 | 0.05 | 0.05 | ✓ |
| 58 | soft cls target exponent on prob | §A.5 / 642 | train_sam3_kfold.py:333 | 0.25 | 0.25 | ✓ |
| 59 | soft cls target exponent on iou | §A.5 / 642 | train_sam3_kfold.py:333 | 0.75 | 0.75 | ✓ |
| 60 | match_at budget | §4.3 / 319, §7 / 438 | tools/agent/state.py:91 | 5 | 5 | ✓ |
| 61 | reader_refine budget | §6 (table) / 427 | tools/agent/tools/refine.py:26 | 3 | 3 | ✓ |
| 62 | Locate geocode budget | §4.7 / 384, §A.3 / 620 | tools/agent/locate_agent.py:129 | 8 | 8 | ✓ |
| 63 | Letterhead distance | §4.7 / 384, §A.3 / 620 | tools/agent/locate_agent.py:109 | 5 km | 5 km | ✓ |
| 64 | WINDOW_STRIDE_TARGET | §A.4 / 625 | tools/matching/_core.py:61 | 100 px | 100 | ✓ |
| 65 | RANSAC reproj threshold | §A.4 / 625 | tools/matching/_core.py:124,150 | 10 px | 10.0 | ✓ |
| 66 | 6-DOF ratio gate | §4.7 / 387 | tools/matching/_core.py:51 (GATE_RATIO_6DOF) | "30 % more" | 1.3× | ✓ |
| 67 | 6-DOF aspect gate | §4.7 / 387 | tools/matching/_core.py:173 | 0.85 | 0.85 | ✓ |
| 68 | 6-DOF scale min | §4.7 / 387 | tools/matching/_core.py:52 | 0.3 | 0.3 | ✓ |
| 69 | 6-DOF scale max | §4.7 / 387 | tools/matching/_core.py:53 | 3.0 | 3.0 | ✓ |
| 70 | 6-DOF shear gate | §4.7 / 387 | tools/matching/_core.py:174 | < 0.15 | < 0.15 | ✓ |
| 71 | Delaunay band lo | §4.7 / 387, §A.4 / 625 | tools/delaunay_filter.py:37 | 0.5 | 0.5 | ✓ |
| 72 | Delaunay band hi | §4.7 / 387, §A.4 / 625 | tools/delaunay_filter.py:37 | 2.0 | 2.0 | ✓ |
| 73 | Delaunay survival floor | §A.4 / 625 | tools/matching/_core.py:189 | max(8, n/3) | `max(8, n_inliers//3)` | ✓ |
| 74 | Weak-retry inlier trigger | §4.7 / 387 | tools/agent/tools/match.py:483 | < 25 | < 25 | ✓ |
| 75 | Weak-retry score trigger | §4.7 / 387 | tools/agent/tools/match.py:484 | < 0.4 | < 0.4 | ✓ |
| 76 | Weak-retry σ multiplier | §4.7 / 387 | tools/agent/tools/match.py:487-489 | 2σ | `sigma_m * 2.0` | ✓ |
| 77 | Outside-LA penalty | §4.9 (eq:commit-score) | tools/scoring.py:103 | 0.3 | 0.3 | ✓ |
| 78 | Scale perturbation | §A.4 / 625 | tools/matching/_core.py:489-490 | ±15 % | ×0.85 / ×1.15 | ✓ |
| 79 | Strict-gate inlier threshold | §4.9 / 405 | tools/agent/tools/match.py:653-664 | 18 | (none — only n_groups_committed==0) | ✗ |
| 80 | Strict-gate mask-frac threshold | §4.9 / 405 | tools/agent/tools/match.py:653-664 | 0.2 % | (none) | ✗ |
| 81 | Composite-window score form | eq:composite-window | tools/scoring.py:46-47 | V·Q/4·1/(1+d_km) | identical | ✓ |
| 82 | Commit-score form | eq:commit-score | tools/scoring.py:106-118 | n·{1 if LA else 0.3} | identical | ✓ |
| 83 | "211-case +5 cases" | §4.8 / 400 | tools/scoring.py:37 | +5 at IoU≥0.8 | "+5 cases at IoU ≥ 0.8 vs the v13 raw-metric ranking (125 → 130)" | ✓ |
| 84 | Building colour hex | §A.4 / 630 | tools/io/os_tiles.py:143 | #F4CCCC | BGR(179,179,255) = #FFB3B3 | ✗ |
| 85 | Local Street width at z=17 | §A.4 / 630 | tools/io/os_tiles.py:161 | 1.5 px | (fill 1.5, casing 2.5) | ✓ |
| 86 | Tile bbox inflation | §A.4 / 630 | tools/io/os_tiles.py:108 | 5 % | 0.05 | ✓ |
| 87 | mask-cleanup 5 % floor | §A.6 / 649 | tools/extraction/mask_ops.py:120 | 5 % | 0.05 | ✓ |
| 88 | mask-cleanup 100-px abs floor | §A.6 / 649 | tools/extraction/mask_ops.py:149 | 100 px | 100 | ✓ |
| 89 | thin-outline expand trigger | §A.6 / 649 | tools/extraction/mask_ops.py:90 | < 10 % bbox | `fill_vs_bbox > 0.10: return` | ✓ |
| 90 | thin-outline expand kernel | §A.6 / 649 | tools/extraction/mask_ops.py:95 | ≈ 1.5 % bbox | `min(bw,bh) // 60` ≈ 1.67 % | ≈ |
| 91 | closing kernel | §A.6 / 649 | tools/extraction/mask_ops.py:38 | 1 % image dim | `min(h,w) // 100` = 1 % | ✓ |
| 92 | Douglas-Peucker ε | §A.6 / 649 | tools/matching/_core.py:303 | 3 px | 3.0 | ✓ |
| 93 | external-contour filter | §A.6 / 649 | tools/matching/_core.py:333 | > 100 px | < 100 px area dropped | ✓ |
| 94 | worker-turn soft cap (paper claim) | §7 / 438 | tools/agent/__init__.py:58 | 12 | 8 (default; 12 via CLI flag) | ≈ |
| 95 | SAM 3 LoRA |V|=43 for fold 0 | §8.2 / 534 | cv_summary.json fold 0 `n_val` | 43 | 43 | ✓ |
| 96 | SAM 3 LoRA |V|=42 for folds 1-4 | §8.2 / 535-538 | cv_summary.json folds 1-4 `n_val` | 42 each | 42, 42, 42, 42 | ✓ |
| 97 | OS Open Zoomstack size | §A.4 / 630 | external; OS website | ≈ 750 MB | (cannot verify offline) | n/a |

---

## F. Stale-watch results

| Term | Occurrences in paper.tex | Notes |
|---|---|---|
| `submit_pick` | 0 | Cleaned. |
| `pdf_info_text` | 0 | Cleaned. |
| `critic` | 1 | One occurrence at line 482 — the word "interpretable", not the deleted critic role. Verified via context. |
| `Phase 3 critic` | 0 | Cleaned. |
| `VLM critic` | 0 | Cleaned. |
| `critic_log` | 0 | Cleaned. |
| `critic_panel` | 0 | Cleaned. |
| `extract_boundary` | 0 | Cleaned. |
| `project_boundary` | 0 | Cleaned. |
| `render_page` | 0 | Cleaned. |
| `geocode` | 0 | Cleaned (as a tool name). The word "geocoder" appears, but that's the locate sub-agent tool category. |
| `visualize` | 0 | Cleaned. |
| `analytical short-circuit` | 0 | Cleaned. |
| `analytical_affine` | 0 | Cleaned. |
| `OSM` | 0 | Cleaned. |
| `Nominatim` | 0 | Cleaned. |
| `Overpass` | 0 | Cleaned. |
| `auto-labeller` | 0 | Cleaned. |
| `Path A` / `Path B` / `Path C` | 0 | Cleaned. |
| `MapSAM` | 0 | Cleaned. |
| `--include-training-cases` | 0 | Cleaned. |
| `MIN_INLIERS_COMMIT` | 0 | The *constant name* is gone — but the *concept* (18 inliers gate) still appears in §4.9; see A1. |
| `MIN_MASK_FRAC_COMMIT` | 0 | Same — concept (0.2 % mask floor) still in §4.9; see A1. |
| `0.2 \%` (mask coverage claim) | 1 | line 405 — stale, see A1. |
| `18 inliers` | 1 | line 405 — stale, see A1. |
| `eleven tools` / `11 tools` | 0 | Cleaned. |

The grep search confirms that the previous Claude instance removed the stale tool / phase names, but **left in the §4.9 strict-commit-gate description**, which is the load-bearing inaccuracy that A1 calls out.

---

## G. Unverifiable / cannot tell

- **§7 headline benchmark numbers** — every cell in `tab:main-result` and `tab:stratified` is `\placeholder`. Would need to run the full benchmark or inspect `results/benchmark_v_R21/` (gitignored; not on disk in the audit environment).
- **§7 MHCLG Extract "0.82" comparison** — depends on an unpublished comparison protocol. Would need an actual reference to the MHCLG report (currently missing the `mhclg_extract` bib entry — see A4).
- **§7.3 failure-mode counts** — all `\placeholder`. Would need to run the benchmark and categorise each failing case manually.
- **§8.1 VLM-direct and Vanilla SAM 3 rows** — `\placeholder`. The `ablations/` directory exists in the repo but the eval JSONs are not committed (gitignored).
- **§8.4 pipeline ablations (A1, A2, A3)** — all `\placeholder`. Would need three 50-case stratified runs and one 208-case prompt-stripped run.
- **§8.5 frontier-model contrast** — `\placeholder`. Would need 10 cases × 4 models.
- **§8.6 three-measurement decomposition** — `\placeholder`. Would need to re-score the existing per-case `results/.../<case>/metrics.json` files with the contour-IoU + position-error decomposition.
- **§7 "12 worker turns" claim** — would need to inspect the actual launch command for `results/benchmark_v_R21/` (or `summary.json` if it logs the CLI flags) to confirm `--max-iterations 12` was used; otherwise the default 8 makes this claim incorrect. See B5.
- **OS Open Zoomstack file size ("~750 MB")** — paper §A.4. Could not verify offline because `os_opendata/` is gitignored.
- **"5–20 ms per tile cold cache" wall-clock claim** in §A.4 — requires benchmark instrumentation; not verifiable from code review.
