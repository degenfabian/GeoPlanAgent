# Outstanding fixes for v13 agent — single-run mean IoU recovery

These four fixes would close most of the gap between v13 single-run (0.7086 mean
IoU) and the cross-version ensemble (0.7569). Validated via failure analysis
of the 6 cases where ZD4 ensemble fell back from v13 to v10/v11/v12.

## 1. Anchor priority (`tools/agent.py`)

v13 prefers `multi_road_consensus`, `road_intersection`, and `gpkg:*Suburban Area*`
over specific named anchors. Every failure case showed v10/v11 winning with a
specific anchor (`nominatim:road:Bigwood Road`, parish names) that v13 didn't try.

**Edit**: in the anchor-priority order, raise:
- `pdf_text:EN` (already top — keep)
- `pdf_text:postcode` (keep)
- **`nominatim:road:*`** (raise — Photon/Nominatim-geocoded specific streets)
- **`pdf_text:addr:*`** (raise — extracted addresses)
- `nominatim:place:*`
- (everything else)
- `multi_road_consensus`, `road_intersection`, `gpkg:*Suburban Area*` (lower)

Test case: A4D4A1 has "Bigwood Road" in `pdf_info.road_names` but v13 never tried it.

## 2. Acceptance threshold (`tools/positioning.py:sliding_window_position`)

Currently accepts matches with `score ≈ 10` and `n_inliers ≈ 20`. In every
failure case, v13's accepted match scored < 21 while the correct version's
match scored > 33.

**Edit**: require `score ≥ 30` AND `n_inliers ≥ 50` to mark `accepted=True`.
Below that, the agent keeps iterating anchors / rotations.

Test case: A4D4A1 v13 accepted score=10.95, n_inliers=29 → IoU 0.115. v11 had
score=33.83, n_inliers=89 → IoU 0.851.

## 3. A097S regression (real code bug)

Same anchor (`gpkg:Barnack(Village)`), same match_info (n_inliers ≈ 53,
score ≈ 22.5) → v10/v11 IoU=0.633, v13 IoU=0.000. Code regression somewhere
between v11 and v13 commits.

**Investigation**: `git bisect` over `tools/positioning.py` and
`tools/os_opendata_tiles.py` between v11 and v13. The bug is likely in
tile-fetching coordinate conversion or affine construction.

## 4. Compactness-preservation reward multiplier (`tools/positioning.py`)

The existing reward `n_inliers × aspect × scale_penalty × avg_scale_penalty`
correlates positively with IoU but weakly. Case A018S: v13 has higher
n_inliers (81 vs 75) but lower IoU (0.032 vs 0.080).

**Edit**: after RANSAC, project the mask through the candidate affine and
compute `compact_match = min(in_compact, proj_compact) / max(...)`. Multiply
into the metric:

```python
metric = (n_inliers / rot_penalty) * aspect * scale_penalty * avg_scale_penalty * max(compact_match, 0.5)
```

The `max(compact_match, 0.5)` floor prevents over-aggressive demotion of
correct candidates with slightly distorted projections. Earlier I tried this
without the floor and it regressed (correct candidates with compact_match < 0.5
got disqualified).

Test case: A018S v13 had compact_match = 0.16 (severely distorted), v10 had 0.84.

## Estimated impact

If all four fixed: v13 single-run mean IoU should rise from 0.7086 → ~0.7400+
(matching v10/v11 on the 6 fallback cases plus general improvement from the
reward multiplier). To validate, requires a small benchmark rerun on the
~50 affected cases.

## How this was discovered

Cross-comparison of v10/v11/v13 cached metrics on the 6 cases where ZD4
ensemble preferred v10/v11 over v13. See:
- `overnight/V13_FAILURE_ANALYSIS.md` — full analysis
- `overnight/phaseZB4_results.json` — fallback case picks
- `overnight/phaseZB6_validate_plausibility.py` — compactness validation
