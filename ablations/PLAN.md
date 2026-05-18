# Ablation plan

Captured from planning conversation 2026-05-18. Paper submission target:
~8 days from creation (≈ 2026-05-26).

## Constraints

- Full benchmark = 208 cases × `gemini-flash` ≈ 9 h wall + ~$15-20 API
- Realistic ceiling: ~10-12 full-run-equivalents over 8 days, counting
  re-runs and breakage
- Subset runs: 50-case stratified sample = ~2.2 h, statistically meaningful
  for paired comparisons
- API: paid OpenRouter tier; no platform-level RPM cap on paid Gemini Flash;
  watch credit balance, not RPM

## Tier A — supervisor-requested, must do

These came out of Jialin's most recent meeting notes.

| # | Ablation | Scope | Est. cost | Notes |
|---|---|---|---|---|
| A1 | Prompt: detailed (current) vs generic | 1 full run on generic prompt | 9 h, ~$15-20 | Demonstrate prompt engineering matters |
| A2 | Locate-API subsets | 3 × 50-case (postcode-only, place+road only, no-la_check) | 3 × 2.2 h, ~$6 | Most informative slice of 2⁶ combinations |
| A3 | Matching-step decomposition | 3 × 50-case (no multi-zoom sweep, no road-name verifier, no smart-commit gate) | 3 × 2.2 h, ~$6 | Decompose "step 3" per Jialin |
| A4 | Frontier-model reasoning on 5 best + 5 worst | 10 cases × 3 models (Gemini Pro / Claude Opus 4.7 / GPT-5.4) | ~30 min × 3, ~$10 | Heuristic-driven Flash vs reasoning-heavy frontier |
| A5 | Three-measurement breakdown report | 0 h compute — analytical pass over existing per-case data | $0 | Instruction-following / contour-IoU / position-error |

## Tier B — paper-headline must-haves

| # | Ablation | Scope | Est. cost | Notes |
|---|---|---|---|---|
| B1 | VLM-direct, Flash | full 208 cases | ~$0.50, ~30-60 min | Script already written: `ablations/vlm_segmentation.py`. Compare pixel-IoU against `training/dataset/boundary_masks/`. Pure VLM segmentation. |
| B2 | VLM-direct, Pro | full 208 cases | ~$60, ~9 h | Higher token cost. Skip if budget tight. |
| B3 | Vanilla SAM prompt search | ~20-case stratified subset, 6-8 prompts | ~3 h local, $0 | Find best prompt for non-fine-tuned SAM. Candidates: "planning boundary", "site boundary", "red outlined region", "hatched area", "red polygon", "site plan", "highlighted boundary", "application site". |
| B4 | Vanilla SAM (best prompt) | full 208 cases with no-LoRA SAM + chosen prompt | ~9 h, ~$15-20 | Headline "the fine-tune matters" comparison |

## Tier C — design-justification, run if time permits

| # | Ablation | Scope | Est. cost | Notes |
|---|---|---|---|---|
| C1 | No `verify_position` | 50-case stratified subset | ~2.2 h | Test the recovery-via-verify path; after R32 it's only mandatory for borderline 25-100 inlier `accepted` |
| C2 | No `lookup_district` | district-flagged subset (~13 cases) | ~30 min | Cheap, isolates district fallback contribution |
| C3 | No `reader_refine` | 50-case subset | ~2.2 h | Tests the reader-retry loop |
| C4 | No `propose_centers` sub-agent | full 208 cases | ~$15 | Replace live LLM-locate with deterministic geocoder-only candidates |
| C5 | No rotation classifier | full 208 cases | ~9 h | Skip auto-rotation, feed raw page |
| C6 | No match_at panel (no `ToolReturn` image) | 50-case subset | ~2.2 h | Force commits based on numbers + verify_position alone. Measures the early-rejection signal's IoU contribution. One-line patch in `match_at` (return `summary` instead of `ToolReturn(...)`). |

## Tier D — skip unless time spare

| # | Ablation | Scope | Notes |
|---|---|---|---|
| D1 | LLM swap (Sonnet, Opus) | full 208 cases each | Cross-LLM transferability claim |
| D2 | Per-axis reward ablations | 50-case subset per axis | Post-hoc on cached candidates — no agent re-run needed |
| D3 | Single-iteration agent | `max_iterations=1` on 50-case subset | Worst-case "one-shot" baseline |
| D4 | OUTSIDE_LA_PENALTY sweep | post-hoc on cached `match_attempts` | No agent re-run; tune 0.3 vs 0.5 vs 0.1 |

## Time budget (suggested sequence)

Assuming main R16+ run finishes ~6 h after creation:

| Day | Plan |
|---|---|
| 1 | Main run finishes. Kick off **B3** (vanilla SAM prompt search, local, no API) in parallel. |
| 2 | **B1** VLM-direct Flash full (overnight). **A5** three-measurement re-analysis on main results. |
| 3 | **B4** Vanilla SAM full (overnight). **A4** Frontier-model 10-case contrast in evening. |
| 4 | **A1** Prompt-ablation full. Start **A2** locate-subset runs (3 × 50-case). |
| 5 | Finish **A2**. **A3** matching decomposition (3 × 50-case). |
| 6 | **B2** VLM-direct Pro (if budget OK). **C1** + **C2** subsets. |
| 7 | Buffer for re-runs. **C3** + **C4** + **C5** if time. Start drafting tables. |
| 8 | Final sanity reruns + write-up. |

Total full-run equivalents: ~10-12. Fits in 8 days with ~1 day slack.

## Already-staged code

- `ablations/vlm_segmentation.py` — written, byte-compiles, ready for **B1/B2**.
  Reuses `tools.agent._model.resolve_model`. Compares pixel-IoU vs
  `training/dataset/boundary_masks/<filename>.png` (same GT as `scripts/eval_sam_kfold_v2.py`).
  Reports identical metric shape (mean / median / %≥0.50/0.70/0.80/0.90).

## Open design questions for paper iteration 2

These came up during the cleanup-chain conversations and are queued for
after the main benchmark + Tier A/B ablations.

1. **`retry_mask(page, bbox=...)` tool** — give the agent a way to actually
   fix bad SAM3 masks. Pre-R21 critic had `retry_extract_bbox`; capability
   removed when critic was deleted. Probably the single biggest remaining
   IoU lever (~50 LOC). Tier-2-priority addition.

2. **Make `verify_position` panel optional** for all paths (not just
   district_lookup) — keep tool callable but drop the MANDATORY rule
   in worker_agent output validator. Lets the agent skip the extra
   iteration on borderline cases. Run as **C1** before deciding.

3. **Reader prompt postcode disambiguation** — split `postcodes` into
   `site_postcode: Optional[str]` + `other_postcodes: List[str]`, OR
   reorder so SITE postcode is `postcodes[0]`. Currently the locate
   sub-agent does letterhead-check via la_check; works for most cases
   but misses letterhead in same-LA-as-site (council in same town as
   application).

4. **Verify_position panel zoom for district_lookup** — currently
   irrelevant since district_lookup no longer requires verify_position
   (R32). If you ever re-enable for any reason, render at z=10 or z=11
   to actually see the polygon.

5. **OUTSIDE_LA_PENALTY tuning** — currently 0.3 by reasoning, no fit.
   Tier D2 sweep can find the optimum post-hoc against cached candidate
   data; no agent re-run needed.

6. **Reward DEFAULT_WEIGHTS** — currently inlier 0.35, scale 0.25, road
   0.30, spread 0.10 (heuristic). Post-hoc sweep against cached
   per-candidate axis scores possible without re-running agent.

## Notes on what NOT to touch mid-paper

- Cache layer: tile cache (~219 GB) stays. Deleting would add ~1-2 h to
  any cold-cache run.
- Main benchmark methodology: stick with `gemini-flash --max-iterations 12`
  per Jialin's memo.
- Critic: gone (R21-R32). Don't reintroduce.
- Analytical short-circuit: gone (R28). Don't reintroduce.
- OSM/Nominatim: replaced by OS BoundaryLine for district lookup (R19).
  Stays out.
