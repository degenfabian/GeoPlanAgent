# v14 Plan — Focus: 36 sub-0.3 IoU cases

Updated 2026-05-07 after building verification_checks.py + 6 research agents (3 yesterday, 3 today). Offline validation results below.

## Sub-0.3 case anatomy (36 cases)

| Pattern | Count | Best lever |
|---|---|---|
| Wrong town/county (0% inside named LA) | 8 | LA-boundary check (built, validated) |
| Single-property polygon "too big" | 4 | Area+description heuristics (brittle alone) |
| Wrong scale_factor (extreme) | 5 | Scale check (miscalibrated, see notes) |
| In right LA but wrong location | 14 | Need stronger anchor — postcode, parish, road-side |
| Conservation-area shape mismatch | 5 | Mask refinement (Florence-2, SoM) |

## What I built today (offline-deliverable)

### `tools/verification_checks.py` (5 pure functions + aggregator)
- `check_area_consistency` — vs description band + explicit area text
- `check_postcode_in_polygon` — gated to single-property descriptors only
- `check_la_boundary` — uses OS BoundaryLine (602 LA polygons, 705MB downloaded)
- `check_inlier_scatter` — needs v14 logging of inlier coords (placeholder ready)
- `check_scale_factor` — currently miscalibrated (passing cases have scale 0.42-0.5)
- `verification_score(...)` aggregator returns score + per-check breakdown + diagnosis

### Offline validation results

**Hard-gate veto (any check confidence <0.10)**: 15/36 sub-0.3 caught, but 43/133 passing FPs — too noisy for sole veto.

**LA-only hard gate**: 8/36 sub-0.3 caught (22% recall), 5/133 FP (4%) — usable but marginal.

**Weighted-mean score**: poor discrimination (sub-0.3 median 0.92, passing median 0.83) because most sub-0.3 cases pass most checks even when wrong.

**Conclusion**: verification_checks must be used as an **advisory signal to the LLM critic**, not as a deterministic veto. Architecture: append verification breakdown to the critic context block; LLM decides retry. This matches the CRITIC pattern (Gou ICLR 2024) — tool-grounded critique outperforms self-critique alone.

## NEW v14 levers from today's 3 agents (synthesized)

### Visual-perception agent
1. **Cartographic-convention-conditioned mask scoring** — score SAM3 candidates by red-edge density × interior-color density. Catches: B9CDCF90 (IoU 0.01, "pink-shaded"), 095AB379 (IoU 0.06, "pink"), A4Ha1 (IoU 0.38). Different from Phase ZE1: conditions reranking, doesn't replace SAM3. **Cost: 4-6h, est +3-5 cases**.
2. **DocLayout-YOLO page pre-segmentation** — kill title-block + legend before SAM3 sees the image. Catches A018S, A4EC3a1 (high-recall low-precision = SAM3 grabbed legend). Apache-2.0 model, MPS-supported. **Cost: 5-7h, est +2-4**.
3. **Set-of-Mark VLM second opinion on SAM3 candidates** — overlay numbered marks on top-k SAM3 masks, ask Gemini Flash "which is the boundary". Catches A002S, A097S, A100S (the THE-SITE-callout failures). **Cost: 3-5h, est +3-5**.
4. **Annotation/callout edge subtraction via skeletonization** — remove leader-line stubs from SAM3 mask. **Cost: 2-3h, est +2-3**.
5. **Florence-2 cross-validation** — independent visual second opinion. **Cost: 6-8h, est +3-5**.

### Disambiguation agent (NEW reward axes)
1. **Application-form-area gate as posterior multiplier** — `axis_area_form_agreement` reads pdf_info.site_area_text, compares to candidate's projected polygon area. **Cost: 3-4h, est +6-10 (HIGHEST)**.
2. **Postcode-in-polygon CONSISTENCY axis** (not anchor) — flags candidates where nearest postcode is >20km away as wrong-county. **Cost: 2h, est +3-5**.
3. **Top-k disambiguation tool** — when top1.score - top2.score < 0.10, present k=2-3 candidates with structured evidence to LLM (this is the human-in-the-loop disambiguation move). **Cost: 3h, est +3-5**.
4. **INSPIRE freehold-edge alignment axis** — score candidates by % of polygon boundary lying on freehold edges. **Cost: 4h, est +5-8**.
5. **Town-specificity prior** — multiply candidate score by Gaussian(d_km from likely_town centroid, σ=25km). **Cost: 2h, est +2-4**.

### Multi-document agent
1. **Parent-geometry phrase extractor** — new PDFInfo field for "co-extensive with X Conservation Area" → use as positional anchor (NOT direct geometry, that's GT-leak). Catches Ar4.4, A4D8A_merged, A4D6A_merged, A4Da2, CB:75:00001. **Cost: 3-4h, est +5-8**.
2. **Multi-page map composition** — extend cached preflight test from `reference_multipage_handler_2026_05_07.md`, then process ALL map_pages not just [0]. **Cost: 2h test + 2h wire, est +5**.
3. **Adjacent-case prior** — cluster cases by (admin_region, document_filename_prefix), use highest-confidence sibling's centroid as 20km Gaussian prior. **Cost: 4h, est +8 (cleverest free idea)**.
4. **Citation-aware reading (GraphRAG-lite)** — second LLM pass to extract (page_n → reference_target) tuples; rerank map_pages by inbound citations. **Cost: 3h, est +1-2 page-routing**.
5. **INSPIRE-freehold prior at matching time** — soft prior, not snap. **Cost: 6h, est +5-10**.

## Updated v14 execution roadmap

### Phase 0 — Wire what's built today (1-2h)
- [ ] Add `verification_score(...)` call to `tools/critic.py` context block
- [ ] Append diagnosis + per-check breakdown to critic prompt
- [ ] Env-gate by `GEOMAP_USE_VERIFICATION_CHECKS=1`
- [ ] Add inlier-coord logging to `tools/positioning.py:estimate_affine` (5-line change) so check_inlier_scatter activates next run

### Phase 1 — Quick wins (5-7h)
- Town-specificity prior (Disamb #5, 2h, +2-4)
- Application-form-area gate (Disamb #1, 3-4h, +6-10)
- Postcode-consistency axis (Disamb #2, 2h, +3-5)

### Phase 2 — Multi-document (5-8h)
- Multi-page composition (Multi-doc #2, 4h, +5)
- Parent-geometry phrase (Multi-doc #1, 3-4h, +5-8)

### Phase 3 — Visual perception (8-12h)
- Cartographic-convention reranker (Vision #1, 4-6h, +3-5)
- Set-of-Mark VLM picker (Vision #3, 3-5h, +3-5)

### Phase 4 — Big bets (10-20h)
- INSPIRE freehold-edge alignment axis (Disamb #4, 4h, +5-8)
- Adjacent-case prior (Multi-doc #3, 4h, +8)
- Florence-2 cross-validation (Vision #5, 6-8h, +3-5)
- Diagnostic retry router with Reflexion memory (~6h, +6-10)

### Combined estimate
Phase 0+1: ~+12-19 cases. Phase 0-3: ~+25-37. All phases: ~+40-60 (some overlap discounted, more than enough to hit 90% target IF lifts compound).

## What's been confirmed/refuted today

**Confirmed:**
- LA-boundary check works on the WRONG_TOWN failure mode (8 sub-0.3 catches)
- OS BoundaryLine integration is solid (602 LAs loaded)
- Verification-as-advisory is the right architecture (not as veto)

**Refuted/limited:**
- Pure scale-factor check is miscalibrated for this dataset (passing cases have scale 0.42-0.5)
- Pure area-band check has 30%+ FP rate (description heuristics too crude)
- Hard-veto pattern is too noisy alone
