# tools/agent/

The pipeline's orchestrator. Two top-level LLM agents (reader → worker)
plus a sub-agent called inline from the worker's `propose_centers` tool.

## Public API

```python
from tools.agent import run_agent, PDFInfo, BoundaryOutcome

result = run_agent(
    pdf_path="evaluation_data/12:00116:ART4/document.pdf",
    models_state={"sam3_ft": ..., "minima": ...},
    model_name="gemini-flash",
    max_iterations=12,
    dpi=200,
    verbose=True,
    case_name="12:00116:ART4",   # used for SAM3 k-fold adapter routing
    case_dir=Path(...),          # where to flush pdf_info.json / partial_state.json
)
```

`result` is a dict with `geojson`, `mask`, `match_info`, `affine_H`,
`tile_info_meta`, `selected_overlay`, `agent_accepted`, `agent_reason`,
`agent_stats`, `message_log`.

## Files

| File | What lives here |
|---|---|
| `__init__.py` | Thin `run_agent` orchestrator. Imports the worker-tool modules at module-load time so the `@_agent.tool` decorators fire and every tool is registered before the agent runs. |
| `reader_agent.py` | Phase 1 pydantic-ai Agent (`output_type=PDFInfo`). Reads the raw PDF + per-page OCR text. |
| `worker_agent.py` | Phase 2 Agent (`output_type=BoundaryOutcome`). Defines `_agent`, the `_strip_old_images` history processor (keeps token cost flat), and `validate_boundary_outcome` — the output validator that enforces `lookup_district` success for the district-fallback path and that `status="accepted"` has a committed geojson. Post-commit visual review is delegated to the optional critic (`enable_critic=True`). |
| `critic_agent.py` | Optional Phase 3 LLM critic (`enable_critic=True` in `run_agent`). Pairwise comparison across all stored match candidates with action ∈ {approve, switch, retry_locate}; opaque to the worker during initial exploration. See module docstring for details. |
| `locate_agent.py` | Sub-agent invoked from `propose_centers`. Pydantic-ai Agent (`output_type=LocatePick`) with six offline-geocoder `@_locate_agent.tool_plain` tools. `run_locate(pdf_info, map_img_bytes, model_name, match_context?, prior_messages?)` is the callable. |
| `runtime.py` | Phase-specific helpers used by `run_agent` (`read_pdf_phase`, `prepare_worker_state`, `invoke_worker`, `dump_partial_state`, `extract_message_log`, `collect_agent_stats`, `apply_quality_gate`, `build_run_agent_return`). |
| `state.py` | `AgentState` — mutable per-case state passed to every worker tool as `RunContext.deps`. Re-exports `_agent`, `_img_to_binary`, `_dedup_check`, `_create_boundary_overlay`, `_draw_geojson_on_tiles`, `_run_sync_with_retry` so the worker-tool modules don't need to reach across the package. Also exposes `primary_match_page(state)` and `committed_primary_page(state)`. |
| `schemas.py` | Pydantic models: `MapPageMeta`, `PDFInfo`, `BoundaryOutcome`. Field descriptions are authoritative (the system prompts reference them). |
| `prompts.py` | `READER_SYSTEM_PROMPT` and `WORKER_SYSTEM_PROMPT`. The locate sub-agent's prompt lives inside `locate_agent.py` next to the schema. |
| `_model.py` | `resolve_model` / `resolve_model_name` + the alias table (`gemini-flash` → `google/gemini-3-flash-preview`, etc.). |
| `_helpers.py` | Image-to-binary, dedup-detection, overlay-render helpers. |
| `_retry.py` | Transient-HTTP-error retry wrapper around `agent.run_sync`. |
| `tools/` | Worker tool modules (one tool per file). |

## Worker tools (`tools/agent/tools/`)

Each module registers its tool against `_agent` via `@_agent.tool` at
import time. `tools.agent.__init__` imports all of them so they're
registered before `run_agent` is called.

| Tool | Module | Public surface |
|---|---|---|
| `propose_centers` | `locate.py` | `(extra_terms?, match_context?) → {candidate_id, lat, lon, sigma_m, source, evidence}`. Always returns ONE candidate per call. Internally calls `tools.agent.locate_agent.run_locate`. |
| `match_at` | `match.py` | `(page, name, lat, lon, sigma_m?, scale_ratio?) → dict` with per-group reward (numbers only — `total_inliers`, `per_group[]` incl. `n_inliers`, `road_name_agreement`, `road_name_verdict`, `scale_consistency`). For multi-area-group documents, internally runs MINIMA at the same centre on every group's primary page and UNIONs the resulting polygons. |
| `commit_match` | `match.py` | `(candidate_id) → {committed: …}`. Smart-commit gate (`commit_attempt_score` = `total_inliers` × inside-LA-weighted) redirects to a better candidate when the worker has ≥2 stored attempts. Strict gate rejects commits where no group produced a valid affine. |
| `lookup_district` | `verify.py` | `(district_name) → {success, matched_variant?, instruction?}`. OS BoundaryLine offline lookup; supports `'|'`-separated name alternates. On success, the district polygon is committed to internal state and the worker submits `status="district_lookup"`. |

## Locate sub-agent

Defined in `locate_agent.py`. A separate pydantic-ai Agent with
`output_type=LocatePick` and six offline-geocoder tools:

| Tool | Source | Note |
|---|---|---|
| `postcode(pc)` | Code-Point Open (offline) | Sub-100 m precision for full UK postcodes |
| `grid_ref(gr)` | OS BNG parser | Accepts many formats: `TL 150 067`, `TR3559`, `485700 148600`, etc. |
| `place(q, la?)` | OS Open Names (offline) | Villages, churches, schools, named buildings |
| `road(q, la?)` | OS OpenMap Local index | Road-instance centroids, LA-bbox-filtered |
| `intersect(road_a, road_b, la?, road_c?)` | OS OpenMap Local geometry | Geometric road-road junction, sub-100 m |
| `la_check(lat, lon, la)` | OS BoundaryLine | LA-polygon containment + distance to boundary |

The agent sees the rendered map image + a JSON dump of the reader's
PDFInfo. Protocol: view map → scan pdf_info → letterhead-check
postcodes via `la_check` → build a 2-4 candidate pool → cluster + pick →
final `la_check` → emit `LocatePick` directly (pydantic-ai consumes the
structured output; there is no explicit "submit" tool).

Budget: 8 geocode tool calls per case. On agent-loop failure
`run_locate` emits an emergency LA-centroid `LocatePick` so the
worker is guaranteed at least one candidate.

When the worker re-invokes `propose_centers` after a weak `match_at`,
`run_locate` is called with `prior_messages=state.locate_message_history`
so the sub-agent sees its own previous reasoning + tool calls + pick.
The new `match_context` (worker's feedback in plain English) tells the
sub-agent to pick from a DIFFERENT signal type this time.

## Pydantic schemas (`schemas.py`)

### `PDFInfo` — reader output

Site address, postcodes, grid_refs, scale, ranked `map_pages` with
per-page `map_page_details` entries carrying `category` /
`area_group` / `boundary_clarity` / `detail_level` /
`area_signature` / `caption`. Plus locate-stage signals: `road_names`,
`place_names`, `house_number_road_pairs`, `parish_names`,
`admin_region`, `likely_town_or_city`, `directional_modifier`,
`visible_map_labels`, `adjacency_hints`, `is_district_wide`,
`district_name`. Strict ASCII validator on string list fields rejects
CJK/Cyrillic/etc.; a `_critical_fields_not_all_empty` model validator
catches partial-generation failures.

### `BoundaryOutcome` — worker output

`status` (`"accepted"` or `"district_lookup"`) + `final_n_inliers` +
`rotation_checked` + `reasoning`. The output validator at
`worker_agent.py:validate_boundary_outcome`:

- For `status="accepted"`: requires that a commit_match call has
  produced a geojson on `state.current_result`.
- For `status="district_lookup"`: requires that `lookup_district`
  produced a GeoJSON.
- Overwrites `rotation_checked` / `final_n_inliers` from real state if
  the model misreports them (the validator doesn't trust the model's
  flags).

Post-commit visual review is delegated to the optional independent
critic (`enable_critic=True`), not the worker. Rejection was removed
from the schema 2026-05-14. The pipeline always
emits a polygon.

## Multi-page + multi-area-group handling

Some planning docs are split across multiple map pages (`map_pages =
[3, 5, 7]`). Each page has an integer `area_group`. Pages sharing
`area_group` are duplicate views of the same site; pages in different
`area_group`s show different sites that get UNIONed in the final
polygon.

The worker doesn't iterate groups — a single `match_at` call internally
runs MINIMA at the supplied centre on each group's primary page, caches
SAM3 masks per page, and unions the resulting polygons. To retry just
one group whose mask looks wrong, call `match_at` again with
`page=<next alternate page in that group>` — the other groups are
re-matched at the same centre but reuse their cached SAM3 masks.

## Conversation cost control

The `_strip_old_images` history processor on the worker agent
(`worker_agent.py`) replaces `BinaryContent` images in messages
older than the last 4 turns with a placeholder. Without this, every
panel image gets replayed on every subsequent turn — token cost grows
quadratically. The most recent 4 turns' images are kept intact so the
worker can still see the panel it's currently reasoning about.

## Crash safety

`run_agent` flushes `pdf_info.json` after Phase 1 and dumps a
`partial_state.json` via `runtime.dump_partial_state` if Phase 2
raises anything other than the expected `UnexpectedModelBehavior` /
`UsageLimitExceeded`. The benchmark runner uses these to keep partial
results from cases that crashed mid-loop.
