# tools/agent/

The agentic layer of the pipeline. Three pydantic-ai agents (reader,
worker, locate sub-agent) plus an optional critic. Schemas are
enforced by pydantic-ai; structured outputs guarantee that no agent
returns a free-form string.

## Module layout

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | Public entry point — `run_agent(pdf_path, models_state, model_name, …)` |
| [`runtime.py`](runtime.py) | Phase helpers: reader call, pre-render, worker invoke, message-log + stats extraction, return-dict assembly |
| [`reader_agent.py`](reader_agent.py) | Phase 1 Agent (`output_type=PDFInfo`) |
| [`worker_agent.py`](worker_agent.py) | Phase 2 Agent (`output_type=BoundaryOutcome`) + output validator + history processor (strips stale images to bound token cost) |
| [`locate_agent.py`](locate_agent.py) | Locate sub-agent: factory `make_locate_agent(disabled_tools=…)`, 6 geocoder tool implementations, dynamic prompt builder, L2 output validator, emergency LA-centroid fallback |
| [`critic_agent.py`](critic_agent.py) | Optional Phase 3 LLM critic — pairwise judgement across stored `match_attempts`, returns a `CriticDirective` (`approve` / `switch` / `retry_locate`) |
| [`prompts.py`](prompts.py) | `READER_SYSTEM_PROMPT`, `WORKER_SYSTEM_PROMPT`, and the surgically-composed `FOLDED_SYSTEM_PROMPT` for the `--no-reader` ablation |
| [`schemas.py`](schemas.py) | `PDFInfo` (reader output), `MapPageMeta` (per-page categorisation), `BoundaryOutcome` (worker output) |
| [`state.py`](state.py) | `AgentState` (per-case mutable state passed as deps); page-of-interest helpers (`primary_match_page`, `committed_primary_page`) |
| [`_model.py`](_model.py) | `MODEL_ALIASES` table + `resolve_model_name` / `resolve_model` |
| [`_helpers.py`](_helpers.py) | Pure helpers (BGR → `BinaryContent`, dedup tracking, mask overlays, GeoJSON-on-tile drawing) |
| [`_retry.py`](_retry.py) | Wraps `Agent.run_sync` with transient-HTTP-error retries (400/408/425/429/500/502/503/504) |
| [`tools/`](tools/) | Worker tool implementations: `locate.propose_centers`, `match.match_at + commit_match`, `verify.lookup_district`, `submit.submit_pdf_info` (folded only) |

## Pipeline

```
                       ┌─────────────────────────────────────────┐
 PDF binary ─────────► │   reader_agent (READER_SYSTEM_PROMPT)   │
                       │   output_type = PDFInfo                 │
                       └─────────┬───────────────────────────────┘
                                 │ PDFInfo
                                 ▼
                       ┌─────────────────────────────────────────┐
                       │   prepare_worker_state                  │
                       │   • render every map_pages[i] at DPI    │
                       │     (auto-rotation via ResNet50 + TTA)  │
                       │   • build worker user_parts             │
                       └─────────┬───────────────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────────────────────────┐
                       │   worker_agent (WORKER_SYSTEM_PROMPT)   │
                       │   output_type = BoundaryOutcome         │
                       │   tools: propose_centers (→ locate_agent)│
                       │          match_at  (MINIMA + SAM3)      │
                       │          commit_match                   │
                       │          lookup_district                │
                       └─────────┬───────────────────────────────┘
                                 │ BoundaryOutcome + state.current_result
                                 ▼
              ┌───────── optional critic loop (enable_critic=True) ────────┐
              │ critic_agent: pairwise review of stored match_attempts     │
              │   action=approve → done                                    │
              │   action=switch → _direct_switch_commit (no LLM round-trip)│
              │   action=retry_locate → re-invoke worker with instruction  │
              └─────────┬──────────────────────────────────────────────────┘
                        │
                        ▼
            run_agent return dict (geojson, mask, affine_H, tile_info,
            agent_stats, message_log, [worker_first_geojson])
```

## Folded ablation (`folded=True` / `--no-reader`)

`run_agent` skips Phase 1 entirely. Instead, `prepare_folded_state`
attaches the PDF binary to the worker's first user message and
constructs the worker with `FOLDED_SYSTEM_PROMPT`. The
`submit_pdf_info` tool — invisible to the LLM in the standard path
via a `prepare` callback — becomes the required first tool call.
After it runs, `state.pdf_info` is populated and `state.rendered_pages`
is filled by the same render loop `prepare_worker_state` uses, so
the downstream tool surface is identical.

The folded prompt is composed by `_build_folded_system_prompt` in
[`prompts.py`](prompts.py): it slices `READER_SYSTEM_PROMPT`'s FIELD
GUIDANCE block, `WORKER_SYSTEM_PROMPT`'s body, and applies six
surgical edits to remove sentences that assume a separate reader
phase. Any edit to the source prompts propagates automatically.

## Locate sub-agent

`run_locate(pdf_info, map_img_bytes, model_name, …)` builds (cached
via `lru_cache`) the agent, runs it with an oversize-image shrink
helper (PNG → JPEG-90 above 25 MB to avoid HTTP 413), prints the
trajectory, and returns `(LocatePick, all_messages)`. The worker
keeps the message history on `state.locate_message_history` and
replays it on the next `propose_centers` call so the sub-agent can
refine without re-reading pdf_info.

**Production ships `place` only.** The other five geocoders
(`postcode`, `grid_ref`, `road`, `intersect`, `la_check`) are gated
by the factory's `disabled_tools` kwarg. The factory rebuilds the
system prompt dynamically: bulleted tool descriptions, signal-priority
bullets, the LETTERHEAD step, the CLUSTER tier list, and the VALIDATE
step are all included/dropped per the enabled-tool set, so an LOO
variant's agent does not even know a disabled tool ever existed.

## Critic

`run_critic_loop(state, worker_result, model_name, max_iters)`
snapshots the worker's first commit, then iterates up to `max_iters`
critic LLM calls. Each call:

1. Ranks stored `match_attempts` by `n_inliers`, keeps the top 3,
   and adds every committed candidate (so the worker's choice is
   always visible).
2. Builds one LEFT|RIGHT panel per shown candidate (planning map +
   SAM mask | OS tile + projected polygon outline). Each panel is
   sent to the LLM as a separate image so the VLM sees each at full
   resolution.
3. Includes a metrics text block (cand id, group, page, `n_inliers`,
   `road_name_agreement` + verdict, `scale_consistency`, `[COMMITTED]`
   tag), with the same tier definitions used by the worker.
4. Gets a `CriticDirective` (`approve` / `switch` / `retry_locate` +
   `chosen_candidate_id` + reasoning). The critic's own message
   history is carried across iterations so iter 2 sees its prior
   directive.

A `switch` skips the worker entirely: `_direct_switch_commit`
re-points `state.committed_groups[group_id]` to the chosen id and
re-unions every committed group's polygon — no LLM round-trip. A
`retry_locate` re-invokes the worker via a templated instruction
(after refilling its `match_at_budget` and clearing the dedup set so
it can actually call propose_centers again). Critic LLM crashes are
treated as `approve` for loop-exit, but with `llm_error` set so
downstream can tell them apart from genuine approvals.

## Output validator (worker)

In [`worker_agent.py`](worker_agent.py):

- `final_n_inliers` and `rotation_checked` are auto-corrected to the
  truth from state (worker can't lie about them).
- `status="district_lookup"` requires a successful `lookup_district`
  call (state.current_result["geojson"] must exist).
- `status="accepted"` requires at least one `commit_match` call to
  have produced a geojson. If nothing is committed, the validator
  raises `ModelRetry` instructing the worker to commit its
  highest-inlier attempt — the pipeline always produces a polygon.
- In folded mode, the validator also refuses to accept until
  `submit_pdf_info` has populated `state.pdf_info`.

## Token-cost guards

- `_strip_old_images` in [`worker_agent.py`](worker_agent.py) drops
  PDF + image binaries from messages older than `KEEP_RECENT=4`
  before each model call (rebinds `part.content` rather than mutating
  in place, so the on-disk message_log stays intact).
- The locate sub-agent shrinks oversize images (>25 MB PNG → JPEG-90,
  preserving resolution) before the first attempt and detects the
  resulting media type by magic bytes.
- The pydantic-ai run is wrapped by `_run_sync_with_retry` so a single
  transient OpenRouter hiccup doesn't kill a case mid-run.
