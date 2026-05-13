# `tools/critic.py`

**902 lines.** Phase-3 of the pipeline: a separate VLM agent that reviews
the worker's final output and decides whether to (a) accept, (b) flag as
low-confidence, or (c) trigger one of three retry strategies (`retry_sam`,
`retry_projection`, `retry_rotation`). Lives between `extract_boundary` and
the final accept/reject decision.

## Public API

| Symbol | Purpose |
|---|---|
| `CriticDecision` (Pydantic) | the structured output schema |
| `_critic_agent` (PydanticAI Agent) | the LLM agent definition |
| `run_critic_loop(state, worker_agent, worker_result, ...)` | top-level driver |
| `build_critic_panel(state)` | side-by-side image for the critic to view |
| `build_context_text(...)` | text context passed to the critic |

## `CriticDecision` schema (line 36)

The critic must return a Pydantic-validated dict shaped like:

```python
{
  "verdict": Literal["accept", "flag_low_confidence",
                     "retry_sam", "retry_projection", "retry_rotation"],
  "confidence": float in [0, 1],
  "reasoning": str (≤500 chars),
  "fix": Optional[str]   # interpreted differently per verdict
}
```

The `fix` field is the only verdict-specific knob:
- For `retry_sam`: which SAM3 candidate index or new bbox to try.
- For `retry_projection`: a projection-axis to nudge.
- For `retry_rotation`: which rotation (degrees) to apply.

This keeps the schema flat while supporting verdict-specific repair info.

## Agent definition (~line 198)

```python
_critic_agent = Agent(
    "test",  # placeholder, overridden at runtime
    output_type=CriticDecision,
    retries=2,
    output_retries=2,
    model_settings={"temperature": 0},
    instructions=_CRITIC_SYSTEM_PROMPT,
)
```

Standard PydanticAI Agent. Temperature=0 because critic decisions should
be deterministic (same input → same verdict).

The `_CRITIC_SYSTEM_PROMPT` (lines 36-196) is a long block describing
when to use each verdict. Excerpts:

- "Use `accept` only if the boundary covers ≥80% of the actual planning
   area visible on the map AND the projection sits in a sensible
   geographic location."
- "Use `flag_low_confidence` for cases where the boundary looks roughly
   right but you're <60% sure — labels the case as 'rejected' but keeps
   the geojson."
- "Use `retry_sam` when the mask covers the wrong region (e.g. grabbed
   the legend, or only a sub-area of the actual boundary)."
- etc.

## Panel rendering

### `_resize_height(img, target_h)` (line 210)

Aspect-ratio-preserving resize to a fixed height. Used to align panels
side-by-side regardless of source aspect ratio.

### `_add_label_bar(img, label)` (line 218)

Adds a 60px black bar at the top with a white text label. Used to
caption panels ("Worker's prediction", "GT polygon overlay", etc.).

### `build_critic_panel(state)` (line 227)

Construct the multi-pane visual the critic sees:

1. **Left**: worker's predicted-boundary overlay on the planning map.
2. **Middle**: predicted boundary projected onto the OS basemap.
3. **Right**: title-block-cropped planning page (so the critic can read
   any text labels).

Returns a single horizontally-concatenated BGR image. The critic is
shown this alongside the text context to make its judgment.

## Text context

### `_summarize_tool_calls(worker_result)` (line 295)

Pulls the worker's tool-call sequence out of the agent's message log and
summarises it ("called geocode → propose_centers → match_at(×3) →
commit_match → extract_boundary → project_boundary"). Lets the critic
see HOW the worker arrived at its answer, not just the final state.

### `build_context_text(state, worker_result)` (line 310)

The big text bundle handed to the critic alongside the panel:
- PDF info summary (postcodes, road names, scale)
- Worker's reasoning (`agent_accepted`, `agent_reason`)
- Match info (n_inliers, score, center)
- Reward axes breakdown (from `reward.py`)
- Tool-call summary
- Verifier score (if available)

Truncated to fit context window (~6KB).

## Retry handlers

When the critic returns `retry_*`, the loop applies the appropriate fix:

### `_apply_retry_sam(state, decision, ...)` (line 410)

Interpret `decision.fix`:
- An int → use that candidate index from `state.instance_masks`.
- A bbox `"[x1,y1,x2,y2]"` → re-extract with that bbox prompt.
- `"none"` → just re-run extract_boundary with the default prompt and
  pick the new top mask.

After applying, re-projects via `_reproject` and updates `state.current_result`.

### `_apply_retry_projection(state, decision)` (line 455)

Re-runs `mask_to_geojson_affine` with the current mask and current
affine — useful when the agent had a stale projection from before
extract_boundary changed the mask. No retry to the LLM, just
re-projection.

### `_apply_retry_rotation(state, decision, ...)` (line 487)

Apply a rotation correction to the planning map (90/180/270 CW), then
re-run extract_boundary + position_boundary in sequence. Used when the
critic spots an obvious 90° miss in the current rotation.

### `_reproject(state, mask)` (line 400)

Helper: given a state with affine + tile_info, project a new mask. Used
by all retry handlers.

## Worker re-entry

### `_build_worker_feedback_prompt(decision)` (line 576)

Constructs a feedback message to the worker agent: "the critic flagged
your output for X reason — please reconsider with this hint." Used when
the critic determines the worker should be re-invoked rather than the
critic doing the fix itself.

### `_worker_reentry(worker_agent, worker_result, state, ...)` (line 600)

Re-run the worker agent with the new feedback. Returns the new result
which the critic loop then re-evaluates.

## Main loop

### `run_critic_loop(state, worker_agent, worker_result, model, sam3, minima_matcher, max_iterations=2, verbose=False)` (line 637)

The driver:

1. **Build panel + context** for the critic.
2. **Call `_critic_agent.run_sync`** with the panel + context.
3. **Parse the verdict**:
   - `accept` → break, mark accepted.
   - `flag_low_confidence` → break, mark not-accepted (geojson kept).
   - `retry_*` → apply the corresponding handler.
4. **Re-evaluate** if a retry handler ran.
5. **Repeat** up to `max_iterations` (default 2 — beyond that you get
   loops where the critic keeps suggesting fixes).
6. **Finalise** via `_finalize` (line 862) — set `state.accepted`,
   `state.accept_reason`, `state.critic_iterations`.

Each iteration is logged (`critic_log.json` written in the case dir) so
post-hoc you can see the critic's reasoning.

## Why this design

**Why a separate critic agent instead of the worker self-checking?** A
worker that accepts its own output is biased — humans do this too. A
separate agent with its own context (the visual panel + tool-call summary)
catches different mistakes than the worker would catch on its own.

**Why structured `CriticDecision` instead of free text?** The retry
handlers need machine-readable verdicts. PydanticAI's structured-output
mode gives the LLM a clear schema to fill, making the verdict reliable.

**Why max_iterations=2?** Tested empirically. With 1 iteration you miss
some easy fixes; with 3+ the critic starts second-guessing earlier
correct decisions. 2 is the sweet spot.

**Why retry handlers in critic.py instead of as worker tools?** The
retry handlers need critic-specific context (the structured fix field,
the iteration count). Putting them as worker tools would either
duplicate logic or pollute the worker's tool set.
