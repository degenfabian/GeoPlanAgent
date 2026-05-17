# Refactor + Codebase Audit — 2026-05-18

Read-only audit of the schema/tool refactor and surrounding correctness paths in
`/Users/fabiandegen/Documents/VSCODE/GeoMapAgent_autonomous`. Five parallel subagents
were dispatched (refactor surface, data leakage, metric correctness, match.py edge
cases, verify_position + critic). No code was modified, no benchmark or LLM call was
launched.

---

## 1. Refactor correctness

| Check | Result | Notes |
|---|---|---|
| (a) Legacy `state.*` field refs (map_img/active_page/current_mask/…) | **PASS** | No hits under `tools/` |
| (b) `tools.agent.tools.render` imports | **PASS** | Only `tools.io.map_page.render_map_page` (utility) survives, used via `match.py:56` `_get_or_render_page` |
| (c) `match_at(ctx, page, name, lat, lon, sigma_m?, scale_ratio?)` — `page` mandatory | **PASS** | [tools/agent/tools/match.py:196](tools/agent/tools/match.py:196) |
| (d) Schema enum ↔ prompt text alignment | **PARTIAL FAIL** | See §6 |
| (e) Worker prompt scrubbed of `render_page` / `role='detail'` / `role='context'` | **PASS** | |
| (f) Reader prompt scrubbed of `role` field | **PASS** | |
| (g) `tools/agent/tools/__init__.py` | **PASS** | Empty file (intentional — tools auto-register via `@_agent.tool`) |
| (h) Toolset has exactly the 6 declared tools | **PASS** | propose_centers, match_at, commit_match, verify_position, lookup_district, reader_refine |
| (i) `scripts/` / `benchmark_runner.py` refs to old `role` | **FAIL** | [scripts/viz_reader_page_ranking.py:166](scripts/viz_reader_page_ranking.py:166) and [scripts/viz_reader_page_ranking_v2.py:218](scripts/viz_reader_page_ranking_v2.py:218) still call `meta.get("role", "?")` |

**Other surprises**:
- Stale `tools/agent/tools/__pycache__/render.cpython-310.pyc` should be deleted.
- Reader prompt at [tools/agent/prompts.py:40-45](tools/agent/prompts.py:40) only documents 5 of the 8 `discard_reason` values defined in schemas.py (missing: `decorative`, `no_boundary`, `other`). The schema still accepts them, but reader precision degrades.

---

## 2. Data-leakage audit

**No leaks detected.** This is a clean bill of health on a path the user has been
bitten on before (0.7716 → 0.7569). Highlights:

- **SAM3 k-fold routing** ([tools/extraction/sam3.py:90-105, 278-318](tools/extraction/sam3.py:90)):
  `md5(canonical_case_name) % N_FOLDS`. Canonicalisation handles colon→underscore
  consistently. Fallback to lowest-available fold is invoked only when a fold file is
  *missing*, not when a case is missing from `fold_assignment.json` — safe.
- **Training set construction** ([training/train_sam3_kfold.py:354-367](training/train_sam3_kfold.py:354),
  [scripts/build_sam3_training_set.py](scripts/build_sam3_training_set.py)):
  Trains on `boundary_annotations/` only, never on `evaluation_data/`. Train/val split
  excludes the case's own fold correctly.
- **Rotation classifier** ([tools/io/rotation_classifier.py:205-235, 353-387](tools/io/rotation_classifier.py:205))
  uses identical fold logic to SAM3 and shares the same `fold_assignment.json`. A case
  routes to the same held-out fold for both models.
- **Agent inference** ([tools/agent/runtime.py](tools/agent/runtime.py),
  [tools/agent/tools/match.py:80](tools/agent/tools/match.py:80)): Reader sees PDF +
  OCR text only. `set_fold_for_case(case_name)` runs before any SAM3 inference. No
  GT geojson is read.
- **Benchmark runner** ([benchmark_runner.py:350, 416](benchmark_runner.py:350)): GT
  geojson is loaded *after* `run_agent` completes, used only for metric scoring.
- **Critic** ([tools/agent/critic_agent.py](tools/agent/critic_agent.py)): operates on
  predicted geojson only.
- **Caches** (`cache/text_extraction/`, OS tiles, OSM overpass): no GT-derived content.

---

## 3. Metric-correctness audit

- **BLOCKER** — [tools/metrics/geojson.py:134-136](tools/metrics/geojson.py:134):
  `calculate_positioning_error_m` reads `.centroid.x / .centroid.y` **outside** the
  try/except at lines 126–128. On an empty `GeometryCollection` (or
  `POINT EMPTY`) shapely raises `GEOSException` — uncaught. A multi-group commit
  where every per-group polygon is degenerate but the union itself is "valid" (type
  is valid, but `is_empty` is true) hits this. Worst case: benchmark_runner crashes
  for that case and silently drops it from the mean.
- **SUSPICIOUS** — [tools/metrics/geojson.py:492-531](tools/metrics/geojson.py:492)
  / [tools/agent/tools/match.py:347](tools/agent/tools/match.py:347): An empty
  MultiPolygon from `_union_geojsons` passes the `if geojson:` guard
  ([benchmark_runner.py:416](benchmark_runner.py:416)) and enters
  `calculate_spatial_metrics`. Either reject empty geometries in `_union_geojsons`
  or add `is_empty` guards in the metrics functions.
- **SUSPICIOUS** — [tools/metrics/geojson.py:113-114](tools/metrics/geojson.py:113):
  `calculate_iou` returns `0.0` when `union.area == 0`. Probably correct, but worth
  confirming CRS sanity (pred + GT both in EPSG:4326; areas in deg² are very small
  but non-zero for real polygons, so this guard fires only for degenerate cases).
- **Cleared**: `calculate_iou` arithmetic is correct (same-CRS assumed, `buffer(0)` fix
  for invalid geometry, MultiPolygon handled transparently by shapely). Benchmark
  aggregation distinguishes `iou is None` (no polygon produced) from `iou == 0`
  (produced but no overlap) at [benchmark_runner.py:566-603](benchmark_runner.py:566)
  using `r.get("iou") is not None`. Strict/non-strict inequalities are consistent.

---

## 4. match.py edge-case audit

- **BUG (medium)** — [tools/agent/tools/match.py:105-108](tools/agent/tools/match.py:105):
  `_groups_to_match` silently falls back to `[(0, requested_page)]` when the requested
  page isn't in `map_pages`. Masks worker errors. Should `raise ValueError(...)` or at
  least log loudly.
- **BUG (low)** — [tools/agent/tools/match.py:525-530](tools/agent/tools/match.py:525):
  `_union_geojsons` only normalises `Polygon → MultiPolygon`. If
  `unary_union` returns `Point`/`LineString`/empty (all reachable for pathological
  inputs), the function silently passes it downstream. Add a type check + return
  None on non-polygonal results.
- **QUESTION** — [tools/agent/tools/match.py:570-576](tools/agent/tools/match.py:570):
  `commit_match._attempt_score` inside-LA check picks the first `center_latlon` in
  `per_group`, not the *requested-group's* centre. For multi-group commits this may
  not be what was intended; confirm.
- **QUESTION** — [tools/agent/tools/match.py:341-343](tools/agent/tools/match.py:341):
  Analytical methods bypass `n_inliers ≥ 18` / `mask_frac ≥ 0.002` strict gate. Likely
  intentional (analytical is closure-based, not feature-matched), but worth a code
  comment.
- **Cleared**: empty `map_page_details` path; SAM3 cache never cleared mid-case;
  re-entrancy on same page reuses cache correctly; no leftover legacy-pointer writes;
  budget decremented once per `match_at()` call, not per group.

---

## 5. verify_position + critic audit

- **HIGH** — Critic visual contract is broken for multi-group commits.
  [tools/agent/critic_agent.py:78-82](tools/agent/critic_agent.py:78) tells the critic
  it will see "A 2-panel image. LEFT: planning map with SAM mask. RIGHT: OS map." But
  `build_critic_panel` ([critic_agent.py:195-203](tools/agent/critic_agent.py:195))
  calls `_committed_primary_view` and only shows the *primary* group. Meanwhile
  `verify_position` ([tools/agent/tools/verify.py:38-44](tools/agent/tools/verify.py:38))
  produces an N-stacked layout. **Result: for multi-group commits, the critic is blind
  to SAM3 errors in secondary groups.** Fix is either (a) have `build_critic_panel`
  mirror the verify_position N-panel layout, or (b) update `CRITIC_INSTRUCTIONS` to
  document the current single-primary contract.
- **HIGH** — Same prompt gap for `lookup_district` (zero per_group): the critic sees a
  1-panel image but the prompt promises 2. `build_critic_panel` falls back gracefully
  ([critic_agent.py:257](tools/agent/critic_agent.py:257)) but the prompt should
  describe this explicitly.
- **Cleared**: Single-group `verify_position` layout is correct; mask resolution
  resampling uses `INTER_NEAREST` and matches dims via `cv2.resize`
  ([critic_agent.py:210-212](tools/agent/critic_agent.py:210)); GeoJSON CRS is WGS84
  throughout (`mask_to_geojson_affine`, `_draw_geojson_on_tiles`); panel height fixed
  to 360px with per-panel widths via aspect ratio; multi-group wide-image scaling
  capped at 1800px ([verify.py:119-121](tools/agent/tools/verify.py:119)); temporary
  files cleaned up in `cleanup_temp_pages` ([runtime.py:320](tools/agent/runtime.py:320));
  `_committed_primary_view` returns `(None, None, None)` cleanly and all callers
  guard for it.

---

## 6. Action items

- **BLOCKER** — [tools/metrics/geojson.py:134-136](tools/metrics/geojson.py:134) —
  Wrap `.centroid.x/.y` access in the existing try/except, or short-circuit with
  `if geom.is_empty: return None` before centroid extraction.
- **HIGH** — [tools/agent/critic_agent.py:78-82, 195-203](tools/agent/critic_agent.py:78) —
  Either rebuild critic panel to show N planning maps (matching `verify_position`)
  or rewrite `CRITIC_INSTRUCTIONS` to document the primary-only contract for multi-group
  and district_lookup cases.
- **HIGH** — [scripts/viz_reader_page_ranking.py:166](scripts/viz_reader_page_ranking.py:166)
  and [scripts/viz_reader_page_ranking_v2.py:218](scripts/viz_reader_page_ranking_v2.py:218) —
  Replace `meta.get("role", "?")` with the new field set (`category`, `area_group`,
  `boundary_clarity`, `detail_level`).
- **MEDIUM** — [tools/agent/tools/match.py:105-108](tools/agent/tools/match.py:105) —
  Raise on unknown page in `_groups_to_match` instead of silent fallback.
- **MEDIUM** — [tools/agent/tools/match.py:570-576](tools/agent/tools/match.py:570) —
  Decide and document whether LA filter should use the requested group's centre vs
  the first available.
- **LOW** — [tools/agent/tools/match.py:525-530](tools/agent/tools/match.py:525) —
  Guard `_union_geojsons` against `Point`/`LineString`/empty union results.
- **LOW** — [tools/agent/prompts.py:40-45](tools/agent/prompts.py:40) — Document the
  3 missing `discard_reason` values (`decorative`, `no_boundary`, `other`).
- **LOW** — Empty MultiPolygon flowing into metrics
  ([benchmark_runner.py:416](benchmark_runner.py:416)) — reject upstream in
  `_union_geojsons` or add `is_empty` guard in metric fns.
- **LOW** — Delete stale `tools/agent/tools/__pycache__/render.cpython-310.pyc`.

---

## 7. Verdict

**Not paper-ready yet, but the path to ready is short.** The data-leakage audit is
clean — k-fold routing for SAM3 and the rotation classifier is consistent, training
uses only `boundary_annotations/`, and inference never touches the GT geojson. That's
the most important result, given prior history.

The headline risk is the **BLOCKER** in
[tools/metrics/geojson.py:134-136](tools/metrics/geojson.py:134): an unhandled
`GEOSException` on `.centroid` access for empty geometries. The impact depends on
whether your test set ever produces empty unions in practice — if it does, those
cases will silently drop from `mean(IoU)` or crash the run. Audit one recent
`results/*/results.json` for missing or `None` IoU entries before publishing
numbers, and patch the guard. Per memory, the positioning-error metric is the
"broken" one and you're reporting IoU only, but the same function path is invoked
by `calculate_spatial_metrics`, so the crash is still live.

The **HIGH**-severity critic blind-spot for multi-group commits is the second
priority. It doesn't change the metric, but it weakens Phase 3 quality control on
multi-area documents. If the paper's contribution explicitly markets the
multi-group capability, this should be either fixed or candidly explained.

Refactor surface looks mostly clean. Two visualization scripts still reference the
old `role` field — annoying but non-load-bearing. Schema-vs-prompt drift on
`discard_reason` is a small precision degradation on the reader, not a correctness
bug.

Recommendation: fix the BLOCKER + the two HIGH items, decide on the two MEDIUM
questions, then run a paper-grade benchmark.
