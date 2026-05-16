# Fresh-eyes audit — read this first

Paste this entire file as your first prompt in a new Claude Code session
in this repo. You are the auditor; the previous session was the
refactor-er. **Do not make changes.** Produce a markdown report
(`AUDIT_FINDINGS.md`) at the end. The user reads it and decides what
to act on.

## Background — what this codebase does

`GeoMapAgent_autonomous` is a research pipeline that extracts UK
planning-permission boundaries from PDF planning documents into GeoJSON.
It runs three LLM agents (reader → worker → optional critic) plus a
locate sub-agent and a reader-refine sub-agent. The worker drives a
tool loop: `propose_centers` (delegates to the locate sub-agent) →
`match_at` (MINIMA LoFTR matcher) → `commit_match` → `extract_boundary`
(SAM3 LoRA) → `project_boundary` → `verify_position`. Production
benchmark = `scripts/run_benchmark.sh` (writes `results/benchmark_v3/...`).

## What's recently been refactored (last ~20 commits, log shows them)

Over a long cleanup session we removed ~9,600 LOC by deleting dead code,
merged the `tools/geocoding/` and `tools/geo/` directories, renamed
`agents.py` → `reader_agent.py` + `worker_agent.py`, renamed `critic.py`
→ `critic_agent.py`, dropped the title-block crop step, deleted
`wikidata`/`gpkg`/`nominatim`-as-dispatcher dead helpers, deleted the
HSV colour-line fallback, persisted the locate-agent's message history
across worker re-calls, and added live console output for the locate
sub-agent. Read `git log --oneline -25` for the full sequence (commits
R0–R11 plus earlier batches).

## Your job

Do a **fresh-eyes audit**. **DO NOT MAKE CHANGES.** Read code, then flag
findings in a concise, actionable list. The user will decide what to act
on. Focus on these specific patterns we missed (or might still be
missing):

### 1. Phantom-consumer dead code
Functions/modules with docstrings or comments claiming "used by
overnight/X.py" or "called by Y" where the claimed consumer doesn't
exist or has been deleted. We found `_geocode_os_open_names` claiming a
non-existent `overnight/phaseZQ_full_v14_replay.py`. Grep for "used by",
"consumed by", "see overnight/", verify the referenced file/symbol
exists.

### 2. Stub functions returning hardcoded values
A function that always returns the same neutral value because its real
dependency was deleted but the stub stayed. We found
`check_building_overlap` returning `(0.5, "")` permanently. Look for
short functions that return constant tuples or constant scores; check
whether the surrounding aggregator filters them out (might be silently
zero-weighting them).

### 3. Disabled features with leftover plumbing
A state field, dict, or list that's declared in a class but never
written to by anything live. We found `state.instance_masks` populated
only by the disabled `extract_boundary(mode='instance')` branch. Grep
every `self.X = ...` in `tools/agent/state.py`'s `AgentState.__init__`
and verify it's written to at least once elsewhere in production code.

### 4. Version-suffixed identifiers still in production
Things named `_v2`, `_v3`, `propose_centers_v2`, `locate_v2`, `MapSAM2`
outside of `training/` (kept for reproducibility). The user wants the
production pipeline to look canonical, not historical.

### 5. Magic numbers
Constants embedded in production code paths without an explanatory
comment naming what they're tuned against or why. Look especially in:
- `tools/matching/_core.py` — thresholds like `1.3`, `0.85`, `0.15`,
  `0.3`, `3.0`, `500m`, `5km`, `200m`, `1000m`
- `tools/matching/source_priorities.py` — per-source σ defaults (100,
  300, 800, 2500, 3000, 4000, 5000, 8000, 15000m)
- `tools/agent/worker_agent.py` — `25 <= final_inl <= 100`,
  `KEEP_RECENT = 4`
- `tools/agent/tools/match.py` — strict commit gates: `n_inliers < 18`,
  `mask_frac < 0.002`, smart-commit weights
- `tools/agent/tools/extract.py` — mask-area thresholds
- `tools/agent/critic_agent.py` — `0.5`, `0.6`, `0.3`, `0.45`, `0.25`,
  `0.7` decision thresholds
- `tools/verification_checks.py` — `_expected_area_band_m2` numeric
  bands; `DEFAULT_WEIGHTS`
- `tools/agent/locate_agent.py` — sigma_m bounds [100, 50000],
  LA-distance thresholds
- `tools/agent/tools/refine.py` — `REFINE_BUDGET_PER_CASE = 3`

Flag the ones that lack explanation or look arbitrary, especially if
they appear in multiple places (suggests they should be a named
constant in one canonical home).

### 6. Stale comments referencing deleted code
Comments mentioning `_position_boundary_disabled`, `FALLBACK_ANCHOR`,
`state.centers`, `_fallback_geojson_at_anchor`, `GEOMAP_USE_*` env vars
that were never wired, `INSPIRE snap`, `tools.snap.*`, or "deleted in
v18". `git log -p` may show the underlying code is gone but the comments
survived.

### 7. Duplicated logic
Sequences that should be one helper. We caught render → auto_rotate →
map_crop duplicated 4 times. Common candidates: image encoding
(`cv2.imencode('.png', ...)` followed by tempfile creation), GeoJSON
loading + `buffer(0)` validity-fixing, LA polygon lookup via
`_resolve_la`. Three identical-ish blocks > one helper.

### 8. Vague names
`build_dataset.py` (build *what* dataset?) was an example; user renamed
it. Look for:
- Module/file names: `dispatchers.py` (deleted but the pattern might
  recur), `_helpers.py`, `_core.py`, `runtime.py`, `state.py` — do they
  describe what they do?
- Function names: `lookup_district`, `visualize`, `verify_position`,
  `project_boundary` — clear-ish, flag any that aren't
- Tool names exposed to the LLM: the worker has ~11 tools; flag any
  that are ambiguous to an LLM reading the docstring

### 9. Backward-compat shims that have lost their purpose
`tools/agent/state.py` re-exports a dozen names from `worker_agent.py`
/ `reader_agent.py` / `_helpers.py` / `_retry.py` / `_model.py` for
backward compatibility. Audit whether every re-export is still needed
by external callers (production code, scripts, training,
benchmark_runner), or whether the consumers could just import from the
canonical home.

### 10. Empty/near-empty files
`__init__.py` files that are empty (`tools/extraction/__init__.py`,
`tools/io/__init__.py`, `tools/metrics/__init__.py`,
`tools/agent/tools/__init__.py`). Harmless but worth flagging if any
could meaningfully re-export the package's public API instead.

### 11. Pre-existing bugs caught during audit (don't fix — list)
We noted that `pick.sigma_m` from the locate sub-agent is overwritten
in `tools/matching/_core.py` around line 649 by `effective_sigma(name,
scale_ratio)` because the source label `live_locate:...` isn't in
`_SOURCE_SIGMA_M` → falls through to default 5000m. The worker also
doesn't pass `cand["sigma_m"]` to `match_at`. Look for similar contract
bugs where one component writes a value and another silently overwrites
it.

### 12. Checked-in caches / large assets
`tools/data/{adjacency_websearch_cache,refinement_agent_cache,websearch_landmark_cache}.json`
are committed but look like regenerable caches. Verify whether they're
needed (any module reads them?) or should be `.gitignore`d and
regenerated on demand.

### 13. Overlapping responsibilities between tools
The worker has a `geocode()` tool (postcode/grid_ref) AND the locate
sub-agent has its own `postcode`/`grid_ref` tools. Defensible (worker
uses geocode for things spotted on the map that PDFInfo missed) but
worth flagging if the boundary is unclear.

## How to report

Produce ONE markdown file `AUDIT_FINDINGS.md` grouped by the 13
sections above. For each finding:
- File path with line number (`tools/foo.py:123`)
- One-line description of the issue
- Severity: `delete`, `fix`, `consider`, or `note`
- (For magic numbers) suggested name + canonical home

Keep the report under 1,500 words. Don't include code, don't make
edits, don't run benchmarks. The user reads, decides, then comes back
to act.

**Verify your claims**: before flagging "X has zero callers", actually
`grep -rn` for it. We were burned earlier by deleting things that
turned out to be load-bearing in main but not in a worktree.

## Constraints
- No deletions, no renames, no edits. Read-only audit.
- No API calls (no LLM calls, no benchmark runs).
- Don't suggest the obvious patterns we already addressed (the 20
  commits R0–R11 + earlier batches cover them).
- Don't propose new features. Pipeline behaviour must stay the same.
- If you find something that's clearly worth fixing immediately and is
  100% safe, still just flag it — let the user batch decisions.
