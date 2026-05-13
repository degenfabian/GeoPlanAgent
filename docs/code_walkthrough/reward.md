# `tools/reward.py`

**369 lines.** Multi-axis reward scoring for MINIMA matches. Returns a
score in [0,1] across several independent quality axes (inlier strength,
scale consistency, road-name agreement, keypoint spread) and aggregates
them via geometric mean. The agent uses these scores to decide whether a
match is good enough to commit, or whether to retry with a different
center.

## Public API

| Symbol | Purpose |
|---|---|
| `AxisResult` (dataclass) | one axis's score + verdict + evidence |
| `RewardResult` (dataclass) | full multi-axis result + overall score + summary |
| `compute_match_reward(...)` | top-level entry — runs all axes, aggregates |
| `format_for_agent(...)` | formats axis verdicts as multi-line text for the LLM |

## Data classes

### `AxisResult` (line 40)

```python
@dataclass
class AxisResult:
    score: float       # in [0, 1]
    verdict: str       # human-readable one-liner
    evidence: dict     # raw numbers used to compute the score
```

`verdict` is what the LLM sees ("strong inlier signal: 87 inliers, score 21.3");
`evidence` is the raw stats for debugging.

### `RewardResult` (line 47)

Holds a dict of named `AxisResult`s, the overall score (geometric mean),
and a pre-formatted multi-line summary for the LLM. Has `to_dict()` for
JSON serialisation in `metrics.json`.

## The 4 reward axes

### `axis_inlier_strength(n_inliers, score)` (lines 66-86)

Measures how confident the MINIMA match was.
- **n_inliers** — number of point correspondences that fit the affine.
- **score** — MINIMA's internal score (a function of inliers + spread).

Score is a logistic curve:
- 0–25 inliers: 0.1–0.4 (weak)
- 25–50 inliers: 0.4–0.7 (decent)
- 50–100 inliers: 0.7–0.9 (good)
- 100+: 0.9+

This is the most-trusted axis — a high inlier count with low spread is
hard to fake.

### `axis_scale_consistency(scale_ratio, recovered_avg_scale, ...)` (lines 88-145)

Compares the PDF's stated scale (e.g. 1:2500) against what MINIMA
recovered (the avg_scale field of match_info).

Math:
1. Convert PDF scale + DPI → expected meters per page-pixel.
2. Convert MINIMA's avg_scale → recovered meters per page-pixel.
3. Penalise the absolute log-ratio: `|log(rec/expected)|`.
4. Translate that into a [0,1] score with a soft tolerance.

Catches cases where MINIMA matched at the right location but at a totally
wrong scale (e.g. matched a shop sign instead of the planning area).

### `axis_road_name_agreement(...)` (lines 147-203)

For cases where the PDF mentioned road names, checks if the OS map at the
matched location actually has those roads.

1. Pull the OSM road network around the matched center.
2. For each road name in `road_names`, fuzzy-match against the OSM names.
3. Score = fraction of input roads found within 500m of the match.

Imperfect because:
- OSM might use a different name (e.g. "Marsham St" vs "Marsham Street").
- The road might exist but not be tagged with a name.
- Generic names ("Main Street") match too easily.

The fuzzy matching tries to handle the first two; the third is just noise
in the score.

### `axis_keypoint_spread(...)` (lines 205-263)

Inliers should be **spread across the map**, not clustered. A cluster of
inliers in one corner is a sign that MINIMA matched a small distinctive
feature (e.g. a logo) that happened to repeat. A well-spread set covers
the actual map area.

Score:
1. Take the convex hull of the inlier points.
2. Hull area / map area = coverage ratio.
3. Coverage 50%+ → score 0.9; 10% → 0.4; 1% → 0.1.

This catches the wrong-window-but-high-inliers failure mode.

## Aggregation

### `aggregate(axes, weights=None)` (lines 265-282)

Geometric mean of the axis scores: `(s1 × s2 × ... × sn) ^ (1/n)`.

Why geometric, not arithmetic: it **penalises any axis that fails**. If
inliers are 0.9 but scale is 0.1, geometric gives 0.3 (correctly skeptical);
arithmetic would give 0.5 (unwarranted optimism).

### `compute_match_reward(match_info, pdf_info, inlier_pts_in_map, map_shape_hw)` (lines 284-330)

Top-level: runs each axis, aggregates, returns a `RewardResult`.

Each axis gets the data it needs from `match_info` + `pdf_info`. Returns
a `RewardResult` with the per-axis breakdown and the aggregate score.

The `summary` field is built by `format_for_agent` — a multi-line text
block that the agent sees in its conversation. It looks like:

```
overall_score: 0.74
- inlier_strength    0.83  strong: 87 inliers, score 21.3
- scale_consistency  0.91  scale ratio matches (1:2500 expected)
- road_name_agreement 0.50  found 2/4 roads within 500m
- keypoint_spread    0.71  inliers cover 38% of the map area
```

### `format_for_agent(axes, overall)` (lines 332+)

Formats the breakdown as the multi-line text shown above. Keeps decimals
to 2dp, sorts axes alphabetically, prefixes the overall score.

## Why this design

**Multi-axis instead of one number?** A single MINIMA score can't
distinguish "matched the right place poorly" from "matched the wrong place
well". Splitting into independent axes makes each failure mode visible.

**Why geometric mean?** Catches "1 axis is broken" cases that arithmetic
mean would smooth over. Recommended in calibration literature for
combining independent quality signals.

**Why does the agent see verdicts as text instead of just numbers?** The
LLM is much better at reading "scale ratio matches" than mentally
computing whether a score of 0.91 is good. The verdict text is the
calibration: it tells the LLM what the score means in plain language.
