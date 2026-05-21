# tools/

Core pipeline. Two LLM agents (reader → worker) plus a live LLM-locate
sub-agent invoked from the worker's `propose_centers` tool. Matching is
MINIMA (LoFTR-based); segmentation is SAM3 + LoRA (k-fold); geocoding
is fully offline (Code-Point Open + OS Open Names + OS OpenMap Local +
OS BoundaryLine).

## Entry point

```python
from tools.agent import run_agent
result = run_agent(pdf_path, models_state, model_name="gemini-flash")
# result["geojson"], result["match_info"], result["agent_stats"], ...
```

## Package map

| Subpackage | What's in it | More |
|---|---|---|
| [`agent/`](agent/) | Reader, worker, locate sub-agent, worker tools | [`agent/README.md`](agent/README.md) |
| [`matching/`](matching/) | MINIMA matcher, sliding-window search, RANSAC affine, road verification | [`matching/README.md`](matching/README.md) |
| [`extraction/`](extraction/) | SAM3 + LoRA k-fold loader, mask cleanup primitives | [`extraction/README.md`](extraction/README.md) |
| [`geo/`](geo/) | Code-Point Open / OS Open Names / BNG ↔ WGS84 / grid ref parser | [`geo/README.md`](geo/README.md) |
| [`io/`](io/) | PDF render, OS-tile composition, OCR, rotation classifier | [`io/README.md`](io/README.md) |
| [`metrics/`](metrics/) | IoU / F1 / positioning error, MINIMA multi-axis reward, viz | — |

Top-level helpers:

| File | Purpose |
|---|---|
| `scoring.py` | `commit_attempt_score(n_inliers, inside_la)` and `composite_window_score` — the single source of truth for match-stage ranking. |
| `verification_checks.py` | OS BoundaryLine LA-polygon resolver (`_resolve_la`, `_load_la_polygons`). Used by `lookup_district`, the locate sub-agent's `la_check`, and the smart-commit LA filter. |
| `build_oml_road_index.py` | One-off script to regenerate `oml_road_index.json` + `oml_road_geom_subset.json` from OS OpenMap Local. Consumed by the locate sub-agent's `road` / `intersect` tools. |

## Worker tools

Each tool module under `tools/agent/tools/` registers its tool against
the shared `_agent` via `@_agent.tool` at import time. The full surface
seen by the worker:

| Tool | Module | Purpose |
|---|---|---|
| `propose_centers(extra_terms?, match_context?)` | [`agent/tools/locate.py`](agent/tools/locate.py) | Calls the locate sub-agent. Returns ONE picked centre per call (lat, lon, σ, confidence, source, evidence). |
| `match_at(page, name, lat, lon, sigma_m?, scale_ratio?)` | [`agent/tools/match.py`](agent/tools/match.py) | MINIMA at the supplied centre. For multi-area-group docs internally matches every group and unions the resulting polygons. Returns multi-axis per-group reward (numbers only). |
| `commit_match(candidate_id)` | [`agent/tools/match.py`](agent/tools/match.py) | Picks one stored match_at attempt as the active result and projects the SAM mask through its affine to a WGS84 GeoJSON. Smart-commit gate redirects to better candidate when there are ≥2 attempts. |
| `lookup_district(district_name)` | [`agent/tools/verify.py`](agent/tools/verify.py) | OS BoundaryLine offline lookup for documents whose boundary IS the entire admin region (Article 4 directions, conservation-area-wide planning, etc.). On success the worker submits `status="district_lookup"`. |
| `reader_refine(question, page_hint?)` | [`agent/tools/refine.py`](agent/tools/refine.py) | Fresh small-context Gemini Flash call on the PDF binary + the cached per-page OCR text block, for one focused question. Budget 3 per case. |

## Locate sub-agent (called from `propose_centers`)

A separate pydantic-ai agent with `output_type=LocatePick`. Sees the
rendered map image plus the reader's pdf_info JSON and chooses ONE
centre with sigma + confidence. Six offline geocoder tools:

| Tool | Source | Note |
|---|---|---|
| `postcode(pc)` | Code-Point Open | Sub-100 m per full UK postcode |
| `grid_ref(gr)` | OS BNG parser | Accepts many formats (`TL 150 067`, `TR3559`, etc.) |
| `place(q, la?)` | OS Open Names | Villages / churches / schools / named buildings |
| `road(q, la?)` | OS OpenMap Local | Road-instance centroid, LA-bbox-filtered |
| `intersect(road_a, road_b, la?, road_c?)` | OS OpenMap Local | Geometric junction, sub-100 m |
| `la_check(lat, lon, la)` | OS BoundaryLine | LA-polygon containment + distance |

Budget: 8 geocode calls per case. The agent emits a `LocatePick` directly
via pydantic-ai structured output (no separate "submit" tool). On
agent-loop failure `run_locate` falls back to an LA-centroid pick so
the worker is guaranteed at least one candidate.

On re-invocation (worker calls `propose_centers` again after a weak
`match_at`), the sub-agent's full prior conversation is replayed via
`prior_messages` so it sees its own previous reasoning + tool calls +
pick + the new `match_context` feedback and chooses a DIFFERENT signal
type.

## Empirically-tuned constants (single source of truth)

| Constant | Home | Value | Note |
|---|---|---|---|
| `WINDOW_STRIDE_TARGET` | `tools/matching/_core.py` | 100 | Sliding-window stride target (px) |
| `OUTSIDE_LA_PENALTY` | `tools/scoring.py` | 0.3 | Smart-commit penalty for picks outside the LA polygon |
| `REFINE_BUDGET_PER_CASE` | `tools/agent/tools/refine.py` | 3 | Cap on `reader_refine` calls per case |
| `match_at_budget` | `tools/agent/state.py` | 5 | Cap on `match_at` calls per case |
| `OCR_FALLBACK_THRESHOLD` | `tools/io/text_extraction.py` | 50 | fitz char count below which we switch to Vision OCR |

## Notes on what isn't here

- **Optional critic.** An independent LLM critic
  (`tools/agent/critic_agent.py`) can run after the worker submits, gated
  by `enable_critic=True` (default False). When enabled, it sees the
  visual panels for ALL stored match candidates plus per-candidate
  `n_inliers / scale_consistency`, makes a
  pairwise judgement, and can direct the worker to switch candidate or
  re-locate (max 2 rejections). Default-off path is bit-identical to a
  no-critic pipeline; the worker is opaque to the critic during initial
  exploration. The worker's first-commit polygon is snapshotted so a
  single run produces paired no-critic / with-critic IoUs for the
  ablation.
- **No analytical short-circuit.** Removed in R28.
- **No OSM / Nominatim runtime calls.** District lookup uses offline OS
  BoundaryLine (R19).
- **No per-iteration `extract_boundary` / `project_boundary` tools.**
  SAM3 segmentation and GeoJSON projection are now automatic inside
  `match_at` / `commit_match`.
