# `tools/agent.py`

**~3550 lines.** The orchestrator. Defines the LLM agent (PydanticAI),
the agent's `AgentState`, the 9 tools the LLM can call, the input/output
schemas, the system prompt, and the top-level `run_agent` entry point
that `benchmark_runner.py` invokes per case.

This is the file you read top-to-bottom to understand the pipeline. The
other modules in `tools/` are libraries; this is the conductor.

## Table of contents (line ranges)

| Lines | Section |
|---|---|
| 1-46 | Imports + module-level config |
| 53-340 | Pydantic schemas (PDFInfo, BoundaryOutcome, AgentState, …) |
| 341-410 | Internal helpers (_get_instance_masks, _get_instance_masks_rich) |
| 412-484 | AgentState class |
| 484-602 | Reader-agent (Phase 1) definition + helpers |
| 605-836 | Worker-agent (Phase 2) definition + system prompt |
| 869-2828 | The 9 tool definitions (one @_agent.tool per tool) |
| 2828-3057 | Helpers for run_agent (model loading, retry wrapper) |
| 3057-3550 | run_agent: end-to-end per-case driver |

## Module setup

### Imports (lines 1-46)

Standard fare: numpy, cv2, torch, fitz (now removed after refactor),
typing, pydantic_ai, plus internal modules. The pydantic_ai imports
(`Agent`, `RunContext`, `ToolReturn`, `BinaryContent`, `ModelRetry`,
`UsageLimits`) are the framework's primitives.

`_FIXED_QUERY = "planning boundary"` is the default SAM3 prompt for the
single-prompt path; multi-prompt overrides it.

## Schemas

### `PDFInfo` (line ~53)

The structured output of the reader agent. Every benchmark run starts by
running the reader to produce a `PDFInfo` for the case PDF. Fields:

- `map_pages: List[int]` — 1-indexed page numbers where the actual
  boundary map is drawn.
- `postcodes: List[str]`
- `road_names: List[str]`
- `place_names: List[str]`
- `grid_refs: List[str]` — OSGB grid references found in the body text.
- `scale: Optional[str]` — "1:2500" or similar.
- `site_address: Optional[str]`
- `district_name: Optional[str]`
- `is_district_wide: bool`
- … plus a few more flags and a `confidence_*` field per group.

Heavily commented so the reader-LLM knows what to put in each field.

### `BoundaryOutcome` (line ~140)

The structured output the worker agent must produce:

```python
{
  "status": "accepted" | "rejected_no_match" | "rejected_low_quality",
  "rationale": str,              # one paragraph
  "n_inliers_final": int,
  "boundary_geojson_attached": bool,
}
```

The actual GeoJSON is stored on `state.current_result["geojson"]`; the
schema just says whether the agent intends it as the final answer.

### `Center` (line ~254, NamedTuple)

`(name: str, lat: float, lon: float, sigma_m: int)`. Used everywhere
candidate centres are passed around. NamedTuple instead of class for
back-compat (a lot of older code unpacks `(n, lat, lon, sig) = center`).

### `CenterInput` (line ~259)

The Pydantic version exposed to the LLM as a tool argument shape — same
fields, but Pydantic-validated so the LLM gets a clear schema.

### `AgentState` (line 412)

The mutable shared context across tool calls within one case:

```python
class AgentState:
    pdf_path: str
    case_name: str
    dpi: int = 200
    pdf_info: dict
    map_img: Optional[np.ndarray]      # rendered planning page
    map_crop_path: Optional[str]       # PNG path for SAM3
    current_mask: Optional[np.ndarray] # latest extraction
    instance_masks: list               # SAM3 candidate pool
    selected_indices: Optional[list]   # which candidates the LLM picked
    current_result: dict               # affine_H, tile_info, geojson, match_info
    centers, centers_tried: lists       # geocoder transparency
    proposed_centers: list             # what propose_centers returned
    match_attempts: dict               # match_at history (cid → ...)
    sam3_state, sam3_processor, sam3_model, device — model handles
    minima_matcher                     — MINIMA-LoFTR handle
    rotation_checked: bool             # whether auto_rotate ran
    accepted: Optional[bool]           # final decision
    accept_reason: Optional[str]
    critic_iterations: list
    position_calls, match_at_budget, ...  # budget tracking
```

The agent's tools all take `ctx: RunContext[AgentState]` and read/mutate
`ctx.deps`. This is how state propagates between tool calls in PydanticAI.

## Internal helpers

### `_get_instance_masks(map_crop_path, processor, model, device, query, top_k, bbox)` (line 374)

Wraps `extract_candidates` in `sam3_boundary.py` to return just the
masks (no metadata). Single-prompt path.

### `_get_instance_masks_rich(map_crop_path, processor, model, device, top_k, bbox, plan_img_bgr)` (line 384)

The multi-prompt + colour-fallback wrapper added during recovery
integration. Returns `(masks, labels)`:

1. Calls `extract_candidates_multi_prompt` for SAM3 candidates with
   tagged sources.
2. Appends `extract_color_boundary(plan_img_bgr)` if not None.
3. Returns parallel lists of masks + their source labels.

The agent's `extract_boundary` instance-mode tool calls this, surfaces
the source labels in candidate overlays so the LLM sees `Cand 3
[site outline]` etc.

### `_try_analytical_affine(state)` (line 1601)

Construct an affine analytically from `pdf_info.grid_refs` (parsed via
`parse_easting_northing`) + `state.scale_ratio` + the SAM-mask centroid.
Returns a `sliding_window_position`-shaped result dict, or None if any
precondition is missing.

Called inside the legacy `position_boundary` tool to short-circuit MINIMA.

### `_try_analytical_match_at(state, name, lat, lon, scale_ratio, tolerance_m=50)` (line ~1599)

The v2-flow analogue of `_try_analytical_affine`. Same idea but invoked
inside `match_at` when the probe `(lat, lon)` is within `tolerance_m`
of an OS easting/northing parsed from `pdf_info.grid_refs`. Returns a
match-attempt dict to write into `state.match_attempts`.

## Reader agent (lines 484-602)

### `_reader_agent` (line 486)

```python
_reader_agent = Agent(
    "test",  # placeholder, model overridden at runtime
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    model_settings={"temperature": 0},
    instructions="""You are a UK planning document reader. ...""",
)
```

The system prompt is a long block (≈200 lines) explaining what each
PDFInfo field should contain. Examples like "grid_refs: OS grid references
on map edges (e.g. TG 210 080, TR 34 SE)" are critical — without them
the LLM puts wrong things in the wrong fields.

### `run_reader(pdf_path, model, dpi, max_pages, verbose)` (≈line 540)

Renders the first 8 pages of the PDF + extracts text via
`tools.text_extraction`, hands both to `_reader_agent.run_sync`, returns
the validated `PDFInfo`.

## Worker agent (lines 605-836)

### `_agent` (line 607)

```python
_agent = Agent(
    "test",
    deps_type=AgentState,
    output_type=BoundaryOutcome,
    retries=5,
    output_retries=3,
    history_processors=[_strip_old_images],
    model_settings={"temperature": 0},
)
```

`history_processors=[_strip_old_images]` is critical: image bytes in the
agent's conversation history balloon the context window. The
`_strip_old_images` processor removes images from old turns once the
agent has moved on, keeping context manageable.

### `_strip_old_images` (line 583)

Walk the agent's message history; for any image bytes more than 3 turns
old, replace them with a placeholder string. The most-recent image is
always kept.

### `validate_boundary_outcome` (line 622)

Output validator: enforces that `accepted` cases have a non-empty
geojson and `n_inliers_final >= some_threshold`. Raises `ModelRetry` if
the LLM tries to accept without proof.

### `build_system_prompt(ctx)` (line 719)

The big system prompt for the worker agent. ≈400 lines. Documents:

- The pipeline order (render → geocode → propose_centers → match_at × N
  → commit_match → extract_boundary → project_boundary → verify_position
  → submit).
- When to use each tool.
- Common failure modes and how to handle them ("if MINIMA gives <50
  inliers, propose new centers; do not retry with the same centers").
- The legacy `position_boundary` fallback path for when v2 returns no
  candidates.

This prompt evolved heavily during benchmarking — every wrong agent
behaviour observed got a paragraph here telling the agent not to do that
again.

## The 9 tools

Each is a `@_agent.tool` decorator on a function that takes
`ctx: RunContext[AgentState]` + tool-specific args, mutates state, and
returns a JSON-able dict (or a `ToolReturn` with image content).

### 1. `render_page(page)` (line 870)

1. 1-indexed → 0-indexed page conversion.
2. Render via `tools.pdf_tools.render_pdf_page` (200 DPI default).
3. Auto-rotate via `tools.rotation_classifier.auto_rotate`.
4. Title-block crop via `tools.map_crop.detect_title_block_crop`.
5. Save to `state.map_crop_path` (PNG, used by SAM3 which expects file
   paths) and `state.map_img` (in-memory ndarray for tools that work
   on bytes).

Returns `{"success": True, "shape": [h, w], "rotation_applied": int}`.

### 2. `geocode(query, kind="auto")` (line 967, `@_agent.tool_plain`)

Tool-plain (no state access — just a function) that geocodes a single
query string. Tries gpkg → Photon → Nominatim. Returns a result dict
with lat/lon + the source.

Used by the agent for one-off lookups; bulk candidate generation goes
through `propose_centers`.

### 3. `propose_centers(extra_terms, skip_sources)` (line 1061)

Returns a deduplicated, ranked list of candidate centres for `match_at`
to probe. Sources combined:

1. Postcodes from `pdf_info.postcodes` → Photon/Nominatim.
2. Grid refs from `pdf_info.grid_refs` → `parse_easting_northing`,
   `os_grid_ref_to_latlon`.
3. Road names + place names → gpkg, Photon, Wikidata cascades.
4. `tools.locate.locate_map` if scale and DPI are known.
5. `extra_terms` parameter — agent can add more queries if it noticed
   something the reader missed.
6. Deduplicate within 500m, rank by source-specificity (postcode > road >
   place > district).
7. Cross-validate with `cross_validate_centers` (drop outlier centres).

Stores everything in `state.proposed_centers` and `state.centers_tried`
for transparency. Returns the top-N with metadata.

### 4. `match_at(name, lat, lon, sigma_m, scale_ratio, rotation)` (line 1346)

The v2 positioning tool — runs MINIMA at one center.

1. **Validate** the (lat, lon) is within 100m of a `propose_centers`
   entry (rejects the LLM hallucinating coordinates).
2. **Decrement budget** (`match_at_budget`, default 5).
3. **Try analytical short-circuit** via `_try_analytical_match_at` —
   if (lat, lon) is near a parsed E/N anchor and scale is known, build
   the affine analytically + return immediately.
4. **Otherwise run MINIMA** via `sliding_window_position` constrained to
   this single center.
5. **2× sigma retry** if the result is weak (n_inliers < 25 OR
   overall_score < 0.4).
6. **Compute reward** via `tools.reward.compute_match_reward`.
7. **Store** as `state.match_attempts[cid]`.
8. **Render visual panel** (`_build_match_at_panel`) — left = planning
   map, right = OS tiles with matched window highlighted. The agent
   accumulates these across calls so it can compare visually.
9. Return `{"candidate_id": cid, "overall_score": ...,
    "match_summary": {...}}` plus the panel image.

### 5. `commit_match(candidate_id)` (line 1593)

Finalise one of the stored match attempts as the active result. Sets
`state.current_result` to the chosen one's affine + tile_info + geojson.
The LLM is allowed to call `commit_match` multiple times — it overwrites
the active result each time.

### 6. `position_boundary(...)` (line 1771, legacy fallback)

The pre-v2 single-call positioning tool. Takes optional scale_ratio,
road_names, extra_centers + skip_sources, runs `propose_centers`-equivalent
internally, then one big `sliding_window_position` call across all
centers.

Now also tries `_try_analytical_affine` first (lines 2118-2143) to skip
MINIMA when the analytical preconditions are met.

The agent's system prompt tells it to prefer `match_at` + `commit_match`
(v2) and only fall back here if `propose_centers` returns nothing.

### 7. `extract_boundary(mode, select_indices, bbox)` (line 2324)

The SAM3 boundary extraction tool.

- `mode="instance"` (default): two-step.
  - **First call** (no `select_indices`, no `bbox`): runs
    `_get_instance_masks_rich` → multi-prompt SAM3 + colour fallback,
    stores 5-8 candidates in `state.instance_masks`, returns each as a
    captioned overlay image so the LLM can see them.
  - **Second call** with `select_indices=[i, j]`: combines the chosen
    candidates via `np.maximum`, sets `state.current_mask`.
  - **Optional third call** with `bbox=[x1,y1,x2,y2]`: re-extract within
    that bbox if none of the original candidates looked right.
- `mode="semantic"`: single-mask. Runs `extract_boundary_sam3_semantic`
  with `_FIXED_QUERY`. If the resulting mask covers > 60% of the image
  (suggesting it grabbed the whole sheet), raises `ModelRetry` telling
  the agent to switch to instance mode.

In both modes, `set_fold_for_case(state.sam3_state, state.case_name)` is
called first to switch the LoRA to the case's k-fold-correct adapter.

### 8. `project_boundary()` (line 2548)

Project `state.current_mask` through `state.current_result["affine_H"]`
via `mask_to_geojson_affine`. Stores the result in
`state.current_result["geojson"]`. No arguments — uses the most recent
mask + affine.

The mask cleanup (largest CC, hull, hole-fill) happens inside
`mask_to_geojson_affine`, so the projected polygon is always cleaned.

### 9. `verify_position(zoom_change)` (line 2705)

Render an OS tile at the matched location at a chosen zoom level and
overlay the projected boundary. Used for visual sanity check by the LLM.
Returns the rendered image to the agent's conversation.

### 10. `lookup_district(district_name)` (line 2768)

Pull the admin polygon for a named district via Nominatim. Used as a
last-resort fallback when MINIMA can't match anything but the case is
plausibly district-wide. Stores the district polygon as the prediction.

### 11. `visualize()` (line 2828)

Render the final overlay panel: predicted boundary on OS basemap, with
optional GT comparison if `evaluation_data/<case>/<gt>.geojson` exists.
Returns the panel image. Called by the agent to confirm output before
submitting.

## `run_agent(...)` (line 3057)

The end-to-end driver:

1. **Load models** if not already cached (SAM3 + LoRA + MINIMA + verifier).
2. **Build `AgentState`** with the case path, models, dpi, etc.
3. **Run the reader** to populate `state.pdf_info` (Phase 1).
4. **Render the first map page** so the worker has an image to start
   with.
5. **Run the worker agent** with the system prompt + initial state. The
   LLM autonomously decides which tools to call.
6. **If the worker accepts**: optionally run the critic loop
   (`tools.critic.run_critic_loop`) to second-opinion the verdict.
7. **Compute spatial metrics** via `tools.geojson_metrics.calculate_spatial_metrics`
   if a GT polygon exists.
8. **Write metrics.json** + various debug artefacts to the output dir.
9. **Return** the result dict.

The retry wrapper `_run_sync_with_retry` (line 2934 ish) handles
transient HTTP errors from the model provider (Gemini Flash had a ~22%
rate of "Provider returned error" in v11 — wrapping in retries got us
to ~98% case-completion).

## Why this design

**Why is everything in one file?** The PydanticAI Agent definition
needs to see all the tool decorators and the AgentState class. Splitting
across files would mean importing everything in one location anyway.
That said, this file is right at the edge of "too big" — at ~3550 lines
it's slow to load mentally. Future refactor: split tools into separate
modules but keep the Agent definition central.

**Why temperature=0?** Reproducibility. The recovery experiments showed
~half the achievable wins came from cross-run LLM variance (`best of N`).
With temp=0, those go away — but every run is reproducible from the
same inputs, which makes A/B-ing changes (this file mods, prompt
mods, etc.) much cleaner.

**Why both v2 (`match_at` + `commit_match`) and legacy `position_boundary`?**
v2 lets the agent probe candidates one by one with reward signals before
committing — useful when there are multiple plausible anchors. Legacy is
a one-shot "throw everything at MINIMA" that's faster when the case is
obvious. The system prompt routes the agent through v2 by default.

**Why `_strip_old_images`?** Each map render is ~500KB; SAM3 candidate
overlays are ~1MB each. Keeping them all in the agent's context across
20 turns blows past Gemini's 1M-token limit. Stripping old images keeps
context bounded while preserving the *text* of past turns (so the LLM
remembers what it tried).

**Why a separate output validator (`validate_boundary_outcome`)?** The
LLM occasionally tries to set `status="accepted"` with no actual GeoJSON
— validator catches it and `ModelRetry` forces a correction.
