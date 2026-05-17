# Production walkthrough — read this first

Paste this entire file as your first prompt in a new Claude Code session
in this repo. Your job is to investigate the production pipeline
**thoroughly** and then **explain to me what actually happens for one
example PDF**, both with and without the critic.

**Read-only. No edits. No API calls. No benchmark runs.**

## Project in one paragraph

`GeoMapAgent_autonomous` extracts UK planning-permission boundaries
from planning-document PDFs into WGS84 GeoJSON. Pipeline: three LLM
agents (reader → worker → optional critic) plus two sub-agents (locate,
reader-refine). The worker drives a tool loop: `propose_centers`
(delegates to the LLM-locate sub-agent which has 6 offline geocoder
tools) → `match_at` (MINIMA LoFTR matcher against OS OpenData tiles) →
`commit_match` (smart-commit gate) → `extract_boundary` (SAM3 LoRA
semantic segmentation) → `project_boundary` (mask → GeoJSON via affine)
→ `verify_position` (visual check) → submit `BoundaryOutcome`. The
critic is opt-in (`enable_critic=True`) and runs a Phase 3 visual review
that can rehand a structured retry directive to the worker.

## Entry points

- CLI: `uv run benchmark_runner.py --model gemini-flash --max-iterations 12 --output-dir results/my_run --force`
- Library: `tools.agent.run_agent(pdf_path, models_state, model_name, enable_critic=False)`
- Phase orchestration helpers: `tools.agent.runtime` (10 named phase fns)

## Read-only investigation

1. **Read `tools/README.md`** (recently rewritten — describes the
   actual current state).
2. **Read `tools/agent/__init__.py`** — the thin `run_agent` orchestrator.
3. **Read `tools/agent/runtime.py`** — each phase as a named function.
4. **Read `tools/agent/reader_agent.py`** + the reader's system prompt
   in `tools/agent/prompts.py`.
5. **Read `tools/agent/worker_agent.py`** + the worker's system prompt.
   Pay attention to the output validator (`validate_boundary_outcome`).
6. **Read `tools/agent/locate_agent.py`** — the locate sub-agent's
   6 tools + protocol.
7. **Read `tools/agent/critic_agent.py`** — both the LLM critic agent
   definition and the worker-rehand mechanism (`_rehand_to_worker`).
8. **Read each worker-tool module** under `tools/agent/tools/`:
   `render.py`, `locate.py` (worker's `geocode` + `propose_centers`),
   `match.py` (`match_at` + `commit_match`), `extract.py`
   (`extract_boundary` + `project_boundary`), `verify.py`
   (`verify_position` + `lookup_district` + `visualize`), `refine.py`
   (`reader_refine`).
9. **Pick ONE representative case** from `results/benchmark_v3/gemini-flash/`
   that succeeded with a non-trivial IoU (between 0.7 and 0.95). Read
   its full artefacts:
   - `pdf_info.json` (what the reader produced)
   - `message_log.json` (the worker's full tool-call trace)
   - `metrics.json` (final IoU + match info)
   - `predicted.geojson` (the output)
   - the original PDF under `evaluation_data/<case>/*.pdf`

## What to deliver

Write `/Users/fabiandegen/Documents/VSCODE/GeoMapAgent_autonomous/PRODUCTION_WALKTHROUGH.md`
with these sections:

### Section A — End-to-end pipeline (architecture summary, ≤300 words)

Brief recap of the 3+2 agent design, what `AgentState` carries between
phases, and how `benchmark_runner` wires up the call.

### Section B — Walkthrough of one specific case (without critic, the default)

Pick the case you chose above. For each of these steps, write
2-4 sentences and quote SPECIFIC values from the artefacts (not
hypothetical):

1. **Reader (Phase 1)**: what `tools/agent/runtime.read_pdf_phase` sent
   to the reader agent, what it returned. Quote 3-5 fields from
   `pdf_info.json` (e.g. `site_address`, `postcodes`, `map_pages`,
   `boundary_color`).
2. **State preparation**: which map page got pre-rendered, what
   `MapPageMeta` roles the reader assigned.
3. **Worker initialization**: what the worker saw in its first user
   message (PDFInfo JSON + active map image + the map-page-roles line).
4. **Worker tool sequence**: walk through the message_log call by
   call. For each tool call: tool name, key args, return summary.
5. **Locate sub-agent call**: when `propose_centers` fired, what the
   locate sub-agent did internally (postcode? grid_ref? place lookup?
   intersect?). Quote the picked `LocatePick` (lat/lon/σ/confidence/
   source/evidence) from message_log.
6. **match_at flow**: which (lat, lon) was tried, what σ was used
   (post-R14 fix — should be the locate sub-agent's σ), what
   `n_inliers` and `overall_score` came back. Did the 2× σ retry
   inside match_at fire?
7. **commit_match**: smart-commit gate behaviour (did it accept
   the LLM's pick or redirect to a different candidate?). Strict-commit
   floors (`MIN_INLIERS_COMMIT=18`, `MIN_MASK_FRAC_COMMIT=0.002`).
8. **extract_boundary**: SAM3 semantic call, mask area %, any bbox
   retry?
9. **project_boundary**: mask → GeoJSON via affine.
10. **verify_position** (if 25 ≤ n_inliers ≤ 100 the validator forces
    this): what visual check the worker did.
11. **BoundaryOutcome submission**: final status, n_inliers,
    visual_check_notes, reasoning.
12. **What benchmark_runner writes** to the case dir at the end.

### Section C — Same case under `enable_critic=True`

Run mentally through what would happen if the worker submitted the
same `BoundaryOutcome` with `enable_critic=True`. Walk through:

1. **When does `runtime.apply_critic_loop` skip vs run?** Quote the
   skip conditions (no `last_output`, status not `"accepted"`, geojson
   None, mask None).
2. **What `build_critic_panel` produces** — describe the 2-panel image
   the critic sees.
3. **What `format_metrics_text` puts in the metrics block** — list the
   keys the critic gets.
4. **Critic Agent's decision** — what `CriticDirective` fields it
   emits (`diagnosis`, `action ∈ {approve, retry_extract_bbox,
   retry_match_at}`, optional `bbox`/`center_idx`, `reason`).
5. **Rehand mechanism** — when the directive is a retry, what
   instruction string `_rehand_to_worker` sends to the worker (with
   `CRITIC DIRECTIVE — you MUST comply.` prefix).
6. **Outer loop budget** — `max_iters=2` outer iterations. What
   determines the loop exits early.
7. **State changes** — what fields of `AgentState` the critic
   loop updates (`critic_iterations`, `critic_final_decision`,
   `critic_changed_mask`, `critic_worker_reentered`).
8. **`flag_low_confidence`** — when does this fire? What it does to
   `state.accepted` / `state.accept_reason`.

For your chosen case, **predict whether the critic would approve or
re-hand** based on the n_inliers + IoU + match panel quality. Don't
actually run it — just reason from the case's metrics.

### Section D — Critic-on vs critic-off diff

A short table or bullet list contrasting:
- Total LLM calls
- Wall-clock cost (rough estimate)
- Failure modes that change behaviour (low-IoU cases, borderline
  inliers, etc.)
- What's written to the case dir (`critic_log.json`,
  `critic_panel.png` only exist with critic on)

### Section E — Anything that's NOT obvious from reading the code

Surface anything that a reader of the code might miss:
- Pre-existing contract subtleties (e.g. σ flowing through
  `state.proposed_centers` → `matched_candidate` in match_at — recently
  wired in R14)
- The `_strip_old_images` history processor that keeps token cost flat
- The pydantic-ai message_history pattern used to persist the locate
  sub-agent's conversation across worker re-calls
- Smart-commit gate redirect behaviour
- Output-validator preconditions (the worker is forced to call
  `verify_position` when 25 ≤ inliers ≤ 100)

## Constraints

- No edits. No API calls. No benchmark runs. Pure read.
- Pick ONE concrete case from `results/benchmark_v3/gemini-flash/` and
  quote real values throughout — no generic "imagine a case" prose.
- Keep total output under 2,500 words.
- Where you reference code, use `file_path:line` markdown links so the
  user can click through.
- If anything you read disagrees with what's in `tools/README.md`,
  flag the inconsistency — the README was rewritten recently and may
  have gaps.
