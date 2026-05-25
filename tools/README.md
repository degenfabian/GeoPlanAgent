# tools/

Core pipeline. Two LLM agents (reader → worker) plus an LLM-locate
sub-agent invoked from the worker's `propose_centers` tool. Matching
is MINIMA (LoFTR-based); segmentation is SAM3 + LoRA (k-fold);
geocoding is fully offline (Code-Point Open + OS Open Names + OS
OpenMap Local + OS BoundaryLine + OS Open Zoomstack).

## Entry point

```python
from tools.agent import run_agent
result = run_agent(pdf_path, models_state, model_name="gemini-flash")
# result["geojson"], result["match_info"], result["agent_stats"], ...
```

`run_agent` takes keyword args for `enable_critic`, `critic_max_iters`,
`locate_model`, `locate_disabled_tools`, and `folded`. See
[`tools/agent/__init__.py`](agent/__init__.py).

## Package map

| Subpackage | What's in it | More |
|---|---|---|
| [`agent/`](agent/) | Reader, worker, locate sub-agent, critic, worker tools | [`agent/README.md`](agent/README.md) |
| [`matching/`](matching/) | MINIMA matcher, sliding-window search, RANSAC affine, road verification | [`matching/README.md`](matching/README.md) |
| [`extraction/`](extraction/) | SAM3 + LoRA k-fold loader (single module, no mask cleanup) | [`extraction/README.md`](extraction/README.md) |
| [`geo/`](geo/) | Code-Point Open / OS Open Names / BNG ↔ WGS84 / grid ref parser | [`geo/README.md`](geo/README.md) |
| [`io/`](io/) | PDF render, OS-tile composition, rotation classifier | — |
| [`metrics/`](metrics/) | IoU / F1 / positioning error, MINIMA per-axis reward, viz | — |
| [`core/`](core/) | Shared k-fold case→fold routing (SAM3 + rotation classifier) | — |

Top-level helpers:

| File | Purpose |
|---|---|
| `scoring.py` | `composite_window_score(vanilla_metric, quadrant_coverage)` — single source of truth for the sliding-window reranker (`V × Q/4`). |
| `verification_checks.py` | OS BoundaryLine LA-polygon resolver (`_resolve_la`, `_load_la_polygons`). Used by `lookup_district`, the locate sub-agent's `la_check`, and the emergency LA-centroid fallback. |
| `build_oml_road_index.py` | One-off script to regenerate `oml_road_index.json` + `oml_road_geom_subset.json` from OS OpenMap Local zip files. Consumed by the locate sub-agent's `road` / `intersect` tools (off by default). |

## Worker tools

Tool modules under `tools/agent/tools/` register against the shared
`_agent` via `@_agent.tool` at import time. The surface seen by the
worker:

| Tool | Module | Visibility | Purpose |
|---|---|---|---|
| `propose_centers(extra_terms?, match_context?)` | [`agent/tools/locate.py`](agent/tools/locate.py) | always | Calls the locate sub-agent. Returns ONE picked centre per call (lat, lon, σ, confidence, source, evidence). |
| `match_at(page, name, lat, lon, sigma_m?, scale_ratio?)` | [`agent/tools/match.py`](agent/tools/match.py) | always | Runs MINIMA + SAM3 on ONE page (= one `area_group`) at the supplied centre. Returns one candidate with `n_inliers`, `scale_consistency`, `road_name_agreement`, `area_group`, `page`, `candidate_id`, `budget_remaining`, `committed_groups`. |
| `commit_match(candidate_id)` | [`agent/tools/match.py`](agent/tools/match.py) | always | Commits ONE candidate (one `area_group`). Each call unions its group's polygon into the running final result. Strict gate rejects commits with no valid affine. |
| `lookup_district(district_name)` | [`agent/tools/verify.py`](agent/tools/verify.py) | always | OS BoundaryLine offline lookup for documents whose boundary IS an admin region. Accepts `\|`-separated alternates. On success the worker submits `status="district_lookup"`. |
| `submit_pdf_info(info)` | [`agent/tools/submit.py`](agent/tools/submit.py) | folded only | Hidden via `prepare` callback unless `folded_mode=True` (i.e. `--no-reader` ablation). Required first tool call when shown; populates state.pdf_info from a PDFInfo schema instance. |

## Locate sub-agent (called from `propose_centers`)

A separate pydantic-ai agent with `output_type=LocatePick`. Sees the
rendered map image plus the reader's pdf_info JSON, runs a small
protocol (view map → scan pdf_info → build pool via tool calls →
cluster & pick → emit), and returns ONE centre with σ + confidence.

**Production ships with `place` only.** Five additional offline
geocoders are implemented in [`tools/agent/locate_agent.py`](agent/locate_agent.py)
and remain reachable via the factory's `disabled_tools` kwarg (and
via `benchmark_runner.py --locate-disabled-tools`); they exist only
for the paper-ablation harness (`ablations/locate_only_eval/`). On
the 11 cross-1km regression-risk cases the locate ablation showed
1-tool ≈ 6-tool in IoU (Δmean = +0.001), so the place-only kit ships
to keep the prompt + tool schema sent to the LLM minimal.

| Tool | Source | Production? | Note |
|---|---|---|---|
| `place(q, la?)` | OS Open Names | ✅ | Villages, churches, schools, named roads/buildings. Sigma 200–1500 m by feature type. |
| `postcode(pc)` | Code-Point Open | off | Sub-100 m per full UK postcode (~1.6 M GB units). |
| `grid_ref(gr)` | OS BNG parser | off | Accepts many formats (`TL 150 067`, `TR3559`, `485700 148600`, etc.). |
| `road(q, la?)` | OS OpenMap Local | off | Road-instance centroid, LA-bbox-filtered. Needs `oml_road_index.json`. |
| `intersect(road_a, road_b, la?, road_c?)` | OS OpenMap Local | off | Geometric junction, sub-100 m. Needs `oml_road_geom_subset.json`. |
| `la_check(lat, lon, la)` | OS BoundaryLine | off | LA-polygon containment + distance. |

Budget: 8 geocode calls per case. The agent emits a `LocatePick`
directly via pydantic-ai structured output (no separate "submit"
tool). An `output_validator` cross-checks the emitted pick against
the min distance to every coord-returning tool call in the trajectory
and raises `ModelRetry` when the pick is > 5 km from every tool
return — catches the sign-flip / lat-lon-swap clerical-error class.
On agent-loop failure (validation retries exhausted, HTTP error,
budget exceeded), `run_locate` falls back to an LA-centroid pick so
the worker is guaranteed at least one candidate per call.

On re-invocation (worker calls `propose_centers` again after a weak
`match_at`, optionally with `match_context="..."`), the sub-agent's
full prior conversation is replayed via `prior_messages` so it sees
its own previous reasoning + tool calls + pick + the new feedback,
and is told to pick from a DIFFERENT signal type.

## Empirically-tuned constants (single source of truth)

| Constant | Home | Value | Note |
|---|---|---|---|
| `WINDOW_STRIDE_TARGET` | `tools/matching/_core.py` | 100 | Sliding-window stride target (px) |
| `MAX_CANDIDATES` / `PER_BUCKET` | `tools/matching/_core.py` | 5 / 1 | Diversity-capped top-K within a single sliding-window pass |
| `match_at_budget` | `tools/agent/state.py` | 5 | Cap on `match_at` calls per case |
| `_FALLBACK_SIGMA_M` | `tools/matching/source_priorities.py` | 5000 | Sigma floor used when the worker omits σ (locate sub-agent always supplies one) |
| `_DEFAULT_CONFIDENCE_THRESHOLD` | `tools/io/rotation_classifier.py` | 0.50 | Rotation classifier abstains below this top-class probability |
| `_SAM3_QUERY` | `tools/agent/tools/match.py` | `"planning boundary"` | The literal phrase the SAM3 LoRA was trained against; do not paraphrase |

## Notes on what isn't here

- **Optional critic.** An independent LLM critic
  (`tools/agent/critic_agent.py`) can run after the worker submits, gated
  by `enable_critic=True` (default off). When enabled, it sees the
  visual panels for the top-3 stored candidates plus every committed
  candidate, with per-candidate `n_inliers / scale_consistency /
  road_name_agreement`, makes a pairwise judgement, and can direct
  the worker to switch (handled in Python — no LLM round-trip) or
  re-locate (worker re-invoked). Default-off path is bit-identical
  to a no-critic pipeline. The worker's first-commit polygon is
  snapshotted so a single run produces paired no-critic / with-critic
  IoUs for the ablation. Max 2 rejection iterations per case.
- **No analytical short-circuit.** Removed in R28.
- **No OSM / Nominatim runtime calls.** Every geocoder and the
  district lookup are offline (Code-Point Open, OS Open Names, OS
  OpenMap Local, OS BoundaryLine, OS Open Zoomstack).
- **No `reader_refine` / OCR pipeline.** The reader is a single
  pydantic-ai call over the full PDF binary. No on-disk OCR cache,
  no `text_extraction.py`.
- **No smart-commit gate.** Removed 2026-05-22 after an audit showed
  it fired in 2/208 cases and saved ≤0.04 IoU per firing.
- **No per-iteration `extract_boundary` / `project_boundary` tools.**
  SAM3 segmentation and GeoJSON projection are automatic inside
  `match_at` (semantic head only, mask cached per page on first
  need, projected via the committed affine inside `_match_single_page`).
- **No 6-DOF / Delaunay fallbacks in `estimate_affine`.** Both were
  removed in May 2026 after a 25-case ablation showed each contributed
  ≤ 0.01 mean IoU. The estimator is 4-DOF similarity only.
