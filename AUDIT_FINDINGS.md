# Fresh-eyes audit — findings

## Verdict: **READY FOR BENCHMARK**

All R12 deletions are clean: zero references to `check_building_overlap`,
`filter_centers`, `_deduplicate_centers`, `cross_validate_centers`,
`run_deterministic_critic`, or any of the deleted env vars (`GEOMAP_USE_*`,
`GEOMAP_CRITIC`, `GEOMAP_LOCATE_MODEL`, `GEOMAP_REFINE_MODEL`,
`GEOMAP_MAX_CENTERS`, `GEOMAP_6DOF_GATE_RATIO`, etc.). Named constants
(`GATE_RATIO_6DOF`, `MIN_INLIERS_COMMIT`, `MIN_MASK_FRAC_COMMIT`,
`WINDOW_STRIDE_TARGET`, `REFINE_BUDGET_PER_CASE`, `OUTSIDE_LA_PENALTY`)
are single-source. `tools/data/` is deleted + gitignored. Production
imports resolve cleanly. Nothing below blocks the benchmark — all findings
are cleanup-ish (stale comments / dead state fields / minor magic numbers
that the v3 results already reflect). One contract bug (#11) is
pre-existing and known.

---

## 1. Phantom-consumer dead code

- **`tools/matching/__init__.py:38`** — comment says `_expand_thin_mask` is
  "used by critic.retry_projection"; `retry_projection` no longer exists
  in `critic_agent.py` (only `retry_extract_bbox` / `retry_match_at`). The
  symbol is now only called from `mask_to_geojson_affine` itself.
  Severity: `fix` (delete the comment).

## 2. Stub functions returning hardcoded values

- None found. `check_building_overlap` is deleted; remaining
  `verification_checks.py` checks all have a real implementation path.

## 3. Disabled features with leftover plumbing

- **`tools/agent/state.py:67` (`candidate_overlays`)** — initialized to
  `[]`; the only write site lived in the deleted instance-mask branch.
  Read & passed through to `runtime.py:529` and benchmark_runner:477 but
  always empty.
  Severity: `delete` (drop field + downstream wiring).
- **`tools/agent/state.py:69` (`selected_indices`)** — only ever set to
  `None` (in `extract.py:65`); was driven by the deleted instance-mask
  branch. Always None at output.
  Severity: `delete`.
- **`tools/agent/state.py:89` (`critic_applied_rotation_deg`)** —
  initialized to None; only assigned None inside `critic_agent.py`
  (`{"applied_rotation_deg": None}`). The rotation-retry critic action was
  deleted with the critic redesign.
  Severity: `delete`.
- **`tools/agent/state.py:90` (`critic_suspected_wrong_location`)** —
  `runtime.py:285` reads `critic_result.get("suspected_wrong_location",
  False)`, but `run_critic_agent` never emits that key. Always False.
  Severity: `delete`.
- **`tools/agent/state.py:94` (`centers_tried`)** — initialized to `[]`
  and never written by any tool. Read by `critic_agent.py:282` for the
  CENTRES `[tried|untried]` display (so `tried_names` is always an empty
  set) and propagated to outputs in `runtime.py:227/550` and
  benchmark_runner:516.
  Severity: `consider` (either wire it from `match_at`'s `name` argument
  or delete + tighten the critic prompt).

## 4. Version-suffixed identifiers in production

- None in production tools. v3-suffixed paths only appear in (a) the
  empirical-derivation pointer comments (`results/benchmark_v3/...`) and
  (b) the eval/training scripts under `scripts/`, which is acceptable.
  `scripts/eval_sam_kfold_v2.py` retains the `_v2` name as the canonical
  k-fold evaluator — fine.

## 5. Magic numbers

Module-level + well-commented in `tools/matching/_core.py` and
`tools/agent/tools/match.py` (R12 promoted them). Remaining ones to
consider:

- **`tools/matching/_core.py:531-532`** — `MAX_CANDIDATES = 5`,
  `PER_BUCKET = 1` live inside `sliding_window_position`. Suggest
  promoting to module-level (`SLIDING_WINDOW_TOP_K`,
  `SLIDING_WINDOW_PER_BUCKET`) — canonical home `tools/matching/_core.py`
  alongside `GATE_RATIO_6DOF` etc.
  Severity: `consider`.
- **`tools/matching/_core.py:584`** — `EARLY_STOP_METRIC = 75.0` inside
  the function. Suggest module-level `SLIDING_WINDOW_EARLY_STOP_METRIC`.
  Severity: `consider`.
- **`tools/matching/_core.py:318`** — `if scale_factor < 0.3 or > 3.0`
  inside `resize_map_to_match_zoom` shadows `SCALE_6DOF_MIN/MAX`. Same
  numeric value, but separately named at separate sites. Either reuse the
  constants or rename (`MAP_RESIZE_SCALE_MIN/MAX`).
  Severity: `consider`.
- **`tools/agent/worker_agent.py:34`** — `KEEP_RECENT = 4` lives inside
  `_strip_old_images`. Fine at function scope, but a one-line module
  constant would let test code reference it. Suggest
  `IMAGE_HISTORY_KEEP_RECENT`.
  Severity: `note`.
- **Borderline-inliers band (25, 100)** repeated in
  `worker_agent.py:136`, `runtime.py:486` (`< 25`), `match.py:274`
  (`< 25`). All three should reference a single constant
  `BORDERLINE_INLIERS_MIN = 25` (canonical home: `tools/agent/tools/match.py`
  or a new `tools/agent/constants.py`).
  Severity: `fix`.
- **`tools/agent/state.py:109`** — `self.match_at_budget: int = 5`. Hard
  cap referenced by message at `match.py:135` ("5 attempts"). Suggest
  module-level `MATCH_AT_BUDGET_DEFAULT` on state.py or pull into
  `match.py` next to `MIN_INLIERS_COMMIT`.
  Severity: `note`.
- **`tools/agent/tools/match.py:274-275`** — `n_inliers < 25 OR
  overall_score < 0.4` triggers 2× sigma retry. Numbers are
  uncommented relative to a calibration source.
  Severity: `note` (add brief comment).
- **`tools/agent/locate_agent.py:445`** — `sigma = max(2000, min(radius_m,
  50_000))` for the emergency LA-centroid pick. The 2000/50000 lower
  bounds aren't documented; LocatePick's schema has `ge=100, le=50000`,
  so 2000 is a deliberate "wider than tight" floor for the fallback.
  Severity: `note` (add a one-line comment).
- **`tools/agent/critic_agent.py:436` and `:149`** — Default model
  hardcoded to `google/gemini-2.5-flash-preview-09-2025` while R12
  hardcoded `google/gemini-3-flash-preview` for locate + refine.
  Inconsistent intent.
  Severity: `fix` (either move all three to a `_FLASH_MODEL` constant in
  `_model.py`, or update the critic default to gemini-3).

## 6. Stale comments referencing deleted code

- **`tools/__init__.py:13`** — "title-block crop" mentioned in the io/
  description; R7 deleted the title-block crop. Severity: `fix`.
- **`tools/agent/tools/render.py:5`** — docstring says pages come with
  "auto-rotate + title-block crop applied"; only auto-rotate runs now.
  Severity: `fix`.
- **`tools/matching/__init__.py:38`** — "used by critic.retry_projection"
  (see §1). Severity: `fix`.
- **`tools/README.md`** — significantly stale:
  - line 32 says `extract_boundary` produces "(+ INSPIRE snap)" — INSPIRE
    snap is gone.
  - lines 42-49 list critic actions `retry_sam`, `retry_projection`,
    `retry_rotation`, `retry_in_worker`, `flag_low_confidence` — all
    deleted. Real actions are `approve | retry_extract_bbox |
    retry_match_at`.
  - lines 54-58 describe the hard/soft-failure escalation pattern that
    the deterministic critic used — also gone.
  - line 73 references `_BG_RGB`, `_BG_TOL`, `_BG_FRAC_THR` background-skip
    constants that were removed 2026-05-12.
  Severity: `fix` (rewrite the file).
- **`tools/verification_checks.py:573`** — `__main__` smoke test still
  globs `results/benchmark_v13/...` (probably wants `v3`).
  Severity: `note`.
- **`tools/matching/source_priorities.py:10/78/158`** — `"inspire":`
  source-priority entries: the INSPIRE snap code is gone, and no live
  code emits an `inspire:` prefixed source today. Dead keys (harmless,
  but mention them when next touching this file).
  Severity: `note`.

## 7. Duplicated logic

- **Haversine 3×**: `tools/geo/coords.py:30` (`haversine_m`),
  `tools/scoring.py:77` (`haversine_km`), `tools/verification_checks.py:146`
  (`_haversine_m`). Different signatures (lat/lon order, units), same
  math. Severity: `consider` (collapse into the `tools/geo/coords.py`
  pair with a `_km` wrapper).
- **`_resolve_la` consumers**: `tools/verification_checks.py` (canonical),
  `tools/matching/source_priorities.py:137`, `tools/agent/locate_agent.py`
  (4 call sites at 261/309/402/437). Each does its own lazy-import.
  Defensible but slightly ceremonial. Severity: `note`.
- **`buffer(0)` validity-fix**: `tools/metrics/geojson.py:81/106/108`,
  `tools/agent/critic_agent.py:313`, `tools/verification_checks.py:589`.
  Severity: `note`.
- **`tempfile.NamedTemporaryFile(suffix=".png", delete=False) +
  cv2.imwrite`**: `runtime.py:161-163` and `render.py:64-66` are
  near-identical (the render tool's flow when the page isn't
  pre-cached). One helper `_dump_temp_png(img: np.ndarray) -> str`
  removes the duplication. Severity: `consider`.

## 8. Vague names

- **`tools/agent/tools/locate.py:geocode`** — the `propose_centers`
  docstring contrasts it with this tool, but the LLM sees a tool called
  `geocode` that only handles postcodes/grid_refs. Consider renaming to
  `geocode_postcode_or_grid_ref` or restricting via two narrower tools.
  Severity: `consider`.
- **`tools/agent/state.py`** — name is generic but the module ALSO
  defines several re-export side-effect imports at the bottom. The
  re-exports are documented in the docstring. Fine.
  Severity: `note`.
- **`tools/agent/tools/extract.py:visualize`** — `visualize` does
  in-conversation image display, not file output; mostly intuitive.
  `lookup_district` and `verify_position` are clear. Severity: `note`.

## 9. Backward-compat shims that have lost their purpose

- **`tools/agent/__init__.py:55`** — `from tools.agent.tools.locate import
  geocode  # noqa: F401`. No external caller imports `tools.agent.geocode`
  anywhere in the repo (verified via grep). Stale shim.
  Severity: `delete`.
- **`tools/agent/__init__.py:35-38`** — `PDFInfo`, `BoundaryOutcome`,
  `AgentState` re-exports. No script imports them from `tools.agent`
  directly (only from `tools.agent.schemas` and `tools.agent.state`).
  Severity: `consider` (drop, or keep for the API surface).
- **`tools/agent/state.py:116-125`** — the `_agent`, `_img_to_binary`,
  `_dedup_check`, `_create_boundary_overlay`, `_draw_geojson_on_tiles`,
  `_run_sync_with_retry` re-exports are actively used by every
  `tools/agent/tools/*.py` (verified). Leave intact.
- **`tools/matching/__init__.py` + `_core.py`** — re-exports from
  `source_priorities`, `road_verify`, `mask_ops`, `coords` are all
  actively consumed. Leave intact.
- **`run_critic_loop` signature in `critic_agent.py:535-551`** —
  parameters `worker_agent`, `sam3`, `minima_matcher`, `max_inner` are
  accepted but never forwarded to `run_critic_agent`. Caller
  (`runtime.py:270-275`) passes them all anyway. Dead parameters.
  Severity: `consider` (drop them or rename to `*_unused`).
- **`_ensure_refine_agent(model_name)`** in `refine.py:47-57` — `model_name`
  is accepted but unused; the agent is initialized with the `"test"`
  placeholder and `model=` is overridden at `run_sync` time.
  Severity: `consider` (drop the parameter).

## 10. Empty/near-empty files

- `tools/metrics/__init__.py` (0 bytes)
- `tools/io/__init__.py` (0 bytes)
- `tools/extraction/__init__.py` (0 bytes)
- `tools/agent/tools/__init__.py` (0 bytes)

Harmless; they exist to make the directories importable. If you want
parity with `tools/__init__.py` (one-line package docstring), 4 small
one-liners would cover it.
Severity: `note`.

## 11. Pre-existing bugs caught during audit

- **Locate σ overwrite contract bug** (the one called out in the brief).
  `tools/matching/_core.py:517` calls `effective_sigma(name, scale_ratio)`
  with `name = "live_locate:postcode:..."` (set by
  `tools/agent/tools/locate.py:157`). `sigma_from_source` strips the
  prefix to `"live_locate"`, which is NOT in `_SOURCE_SIGMA_M`, so it
  falls through to the 5000m default. Then `effective_sigma = max(5000,
  sigma_from_scale)`. The locate sub-agent's calibrated σ (e.g. 200 for
  tight consensus, 300-500 for a clean postcode) is silently destroyed.
  Severity: `fix` (either register `"live_locate" → <something tighter>`
  in `_SOURCE_SIGMA_M`, or — better — derive σ from `LocatePick.sigma_m`
  itself and skip `effective_sigma` for live-locate centers).
- **Same contract: `match_at` ignores candidate's `sigma_m`**.
  `tools/agent/tools/match.py:194-198` defaults `sigma_m` to
  `sigma_from_scale(scale_ratio)` when the LLM doesn't pass it. The
  candidate dict that `propose_centers` writes carries `sigma_m:
  float(pick.sigma_m)`, but `match_at` is a regular tool — the LLM has
  to copy it across. With nothing in the prompt instructing it to do so,
  the calibrated σ is dropped. Closely related to the bug above.
  Severity: `fix` (read `state.proposed_centers` for the matching center
  and pull σ from there when the LLM doesn't supply one).
- **`commit_match` rejection-message arithmetic**
  `tools/agent/tools/match.py:425-426` — prints
  `"(score={best_score - 0.01:.1f}). Candidate_id={best_id} has a
  better commit-score..."` as the candidate's own score. The "best_score
  - 0.01" is a stand-in for the user-submitted candidate's score; in
  practice it understates by 0.01 OR is wrong when the rejected score
  was higher than 0.01 below `best_score`. Cosmetic but misleading.
  Severity: `note`.

## 12. Checked-in caches / large assets

- **`tools/data/`** — already deleted + added to `.gitignore` in R12.
  Clean.
- **`tools/oml_road_index.json`** (443 MB) and
  **`tools/oml_road_geom_subset.json`** (43 MB) — present in the working
  tree but `.gitignore`-d (verified via `git check-ignore`). Both are
  consumed by `tools/agent/locate_agent.py` (`road`, `intersect` tools).
  Regenerable via `tools/build_oml_road_index.py`. Severity: `note`
  (consider a fallback that warns instead of returning `"OML road geom
  missing"` errors when the file isn't present).
- **`cache/`**, **`results/`**, **`evaluation_data/*`**, **`os_opendata/`**
  — all gitignored. Fine.

## 13. Overlapping responsibilities between tools

- **`geocode` (worker) vs `postcode`/`grid_ref` (locate sub-agent)**.
  The worker's `geocode` exists for postcodes/grid_refs the worker spots
  on the map after `propose_centers`. Defensible — the locate agent has
  already returned and isn't in the loop. But: the worker's `geocode`
  uses **postcodes.io HTTP** (`match.py` wait no — `locate.py:53`)
  whereas the sub-agent's `postcode` tool uses **Code-Point Open offline**.
  Different sources for the same lookup. Suggest making the worker's
  `geocode` also use Code-Point Open so the two stay consistent.
  Severity: `consider`.
- **Critic vs worker output-validator coverage** — the validator only
  enforces things on `accepted`/`district_lookup` outcomes. Critic adds a
  Phase 3 visual check. The two never collide.
  Severity: `note`.

---

## Recommended pre-benchmark touch-ups (5 min, zero risk)

1. Fix the stale comments at `tools/__init__.py:13`, `render.py:5`, and
   `tools/matching/__init__.py:38`.
2. Update `tools/agent/critic_agent.py:149/436` to use the
   `google/gemini-3-flash-preview` model (consistency with R12 intent).
3. Decide on the locate-σ contract bug (§11) — either accept it for
   this run and patch after, or apply a 5-line fix:
   - Add `"live_locate": 500` (or similar) to `_SOURCE_SIGMA_M`, OR
   - Replace line 517 of `_core.py` with a pass-through that keeps the
     candidate's input σ when it's already finite and ≥ `sigma_from_scale`.

The dead state fields (§3) and unused parameters (§9) are zero-risk
cleanups that don't gate the benchmark.
