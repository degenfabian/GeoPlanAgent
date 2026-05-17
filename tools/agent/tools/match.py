"""Match-stage worker tools: match_at + commit_match.

Extracted from ``tools/agent.py`` (stage-2 split, 2026-05-11). Registers
``match_at`` and ``commit_match`` against the shared ``_agent`` instance
at import time. Also defines two private helpers used only by these
tools: ``_build_match_at_panel`` and ``_try_analytical_match_at``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext, ToolReturn

from tools.agent.state import (
    _agent,
    AgentState,
    _dedup_check,
    _img_to_binary,
)


# Strict commit-gate thresholds. Statistical miner finding 2026-05-07:
# accepted matches below these levels had mean IoU 0.439 vs 0.748 overall
# (per-case stats in results/benchmark_v3/gemini-flash/*/metrics.json).
# Analytical-affine matches (no n_inliers by design) are exempt — they're
# fully determined by E/N + scale + DPI + mask centroid.
MIN_INLIERS_COMMIT = 18
MIN_MASK_FRAC_COMMIT = 0.002


def _build_match_at_panel(map_img: np.ndarray,
                            tile_info: Dict[str, Any],
                            mi: Dict[str, Any]) -> Optional[np.ndarray]:
    """Visual panel for one match_at attempt: planning map | OS tiles at match.

    Returned to the LLM in match_at's ToolReturn so that across multiple
    match_at calls the agent has the visual evidence to compare candidates,
    not just textual reward scores. The OS tile region inside the matched
    window (the part the affine actually fits) is highlighted with a red
    rectangle.

    None if required state is missing.
    """
    if map_img is None or not isinstance(tile_info, dict) or "image" not in tile_info:
        return None

    target_h = 500

    def _label(img, text):
        bar = np.full((28, img.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    # Left: planning map
    h, w = map_img.shape[:2]
    map_resized = cv2.resize(map_img, (max(1, int(w * target_h / h)), target_h))

    # Right: OS tiles at the candidate's match, with the matched window rect.
    tile_img = tile_info["image"]
    if tile_img.shape[2] == 3 and tile_info.get("_was_rgb", True):
        tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
    else:
        tile_bgr = tile_img.copy()
    th, tw = tile_bgr.shape[:2]
    wx, wy = mi.get("window") or (0, 0)
    # The matched window's size on the tile canvas equals the resized map
    # size MINIMA ran against. We don't store it explicitly, so estimate
    # from sf+map shape: tile pixels per map pixel ≈ sf, but sf is the
    # resize factor applied to the map BEFORE matching, so the window in
    # tile space is map.shape * sf. If sf missing, fall back to map shape.
    sf = mi.get("scale_factor") or 1.0
    win_w = max(1, int(map_img.shape[1] * sf))
    win_h = max(1, int(map_img.shape[0] * sf))
    rect_img = tile_bgr.copy()
    cv2.rectangle(rect_img, (int(wx), int(wy)),
                  (int(wx + win_w), int(wy + win_h)),
                  (0, 0, 255), max(2, th // 200))
    tile_resized = cv2.resize(rect_img,
                              (max(1, int(tw * target_h / th)), target_h))

    left = _label(map_resized, "PLANNING MAP")
    right = _label(tile_resized,
                   f"OS TILES @ z={mi.get('zoom')} "
                   f"({mi.get('center_latlon', ['?', '?'])[0]:.4f}, "
                   f"{mi.get('center_latlon', ['?', '?'])[1]:.4f}) "
                   f"— red box = matched window")

    panel = np.hstack([left, right])
    if panel.shape[1] > 1800:
        s = 1800 / panel.shape[1]
        panel = cv2.resize(panel, (1800, int(panel.shape[0] * s)))
    return panel


@_agent.tool
def match_at(
    ctx: RunContext[AgentState],
    name: str,
    lat: float,
    lon: float,
    sigma_m: Optional[float] = None,
    scale_ratio: Optional[float] = None,
    rotation: int = 0,
) -> ToolReturn:
    """Run MINIMA at ONE candidate center, score it, return reward + panel.

    Per-candidate probe. Stores the result under an integer candidate_id
    (returned in the response) — pass that id to commit_match when you decide.
    See the system prompt (step 3) for decision rules on overall_score.

    Args:
        name: Short label (e.g. "gpkg:Hampstead Heath").
        lat / lon: The center's latitude / longitude (must come from
            propose_centers — fabricated coordinates are rejected).
        sigma_m: Search radius in metres (default: scale-aware from PDFInfo).
        scale_ratio: Map scale denominator (default: parsed from PDFInfo.scale).
        rotation: Map rotation in degrees (default 0; auto-rotation has
            already run at render_page time).

    Returns:
        {"success": True, "candidate_id": int, "overall_score": float,
         "reward": <formatted summary>, "match_summary": {...}}
        Plus a planning-map | OS-tiles visual panel.
    """
    state = ctx.deps
    if state.map_img is None:
        raise ModelRetry("No map image available. Call render_page first.")
    if state.match_at_budget <= 0:
        raise ModelRetry(
            "match_at budget exhausted (5 attempts). Pick the best stored "
            "candidate via commit_match and proceed — the pipeline always "
            "produces a polygon, even if the best score is low."
        )

    # Dedup: refuse identical match_at(name, lat, lon, sigma_m, scale_ratio,
    # rotation) calls. Prevents Gemini-Flash from burning 5-15s of MINIMA
    # work on a candidate it already evaluated. Mirrors the dedup pattern
    # already wired into position_boundary, extract_boundary, lookup_district.
    # Note: this fires BEFORE the budget decrement so a duplicate doesn't
    # spend a budget unit.
    _dedup_check(state, "match_at", {
        "name": name, "lat": round(float(lat), 5), "lon": round(float(lon), 5),
        "sigma_m": sigma_m, "scale_ratio": scale_ratio, "rotation": rotation,
    })

    # ── STRICT TOOL: reject invented coordinates ────────────────────────
    # The agent must call match_at with a (lat, lon) that came from
    # propose_centers. If the coordinate is not within 100m of any entry
    # in state.proposed_centers, we refuse and tell the agent the actual
    # candidate IDs available. This prevents Gemini Flash from
    # hallucinating "Wheathampstead Village Center" coordinates etc.
    if state.proposed_centers:
        from tools.geo.coords import haversine_m as _distance_m
        nearest = min(
            (_distance_m(lat, lon, c["lat"], c["lon"]), c)
            for c in state.proposed_centers
        )
        if nearest[0] > 100.0:
            avail = ", ".join(
                f"id={c['id']} ({c['source'][:30]})"
                for c in state.proposed_centers[:8]
            )
            raise ModelRetry(
                f"match_at refuses fabricated coordinates "
                f"({lat:.5f}, {lon:.5f}) — nearest propose_centers entry "
                f"is {nearest[0]:.0f}m away ({nearest[1]['source']}). "
                f"Use a (lat, lon) from a propose_centers candidate "
                f"directly. Available: {avail}. If none look right, call "
                f"propose_centers(extra_terms=[...]) to add more, do NOT "
                f"invent coordinates."
            )

    state.match_at_budget -= 1

    from tools.matching import sliding_window_position, sigma_from_scale

    def _parse_scale(s: Any) -> Optional[int]:
        if not s:
            return None
        import re
        m = re.search(r"1\s*[:/]\s*([\d,]+)", str(s))
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None

    if sigma_m is None:
        sr = scale_ratio
        if sr is None and state.pdf_info:
            sr = _parse_scale(state.pdf_info.get("scale"))
        sigma_m = sigma_from_scale(sr)
    if scale_ratio is None and state.pdf_info:
        scale_ratio = _parse_scale(state.pdf_info.get("scale"))

    # Analytical short-circuit: when this probe anchor sits within ~50m of
    # an exact OS easting/northing parsed from the PDF AND scale is known
    # AND a SAM mask is set, skip MINIMA entirely. The affine is fully
    # determined by E/N + scale + DPI + mask centroid (see Phase 34 in
    # /tmp/recovery for the original experiment, e.g. A4KTRa1 0→0.71).
    analytical = _try_analytical_match_at(
        state, name, lat, lon, scale_ratio)
    if analytical is not None:
        cid = state._match_attempt_counter
        state._match_attempt_counter += 1
        state.match_attempts[cid] = analytical
        state.match_attempts[cid]["candidate_id"] = cid
        print(f"  match_at: analytical short-circuit for {name!r} "
              f"(skipped MINIMA)")
        return {
            "success": True,
            "candidate_id": cid,
            "overall_score": float(analytical["overall_score"]),
            "reward": analytical["reward"]["summary"],
            "match_summary": {
                "n_inliers": "n/a (analytical)",
                "score": "n/a (analytical)",
                "method": "analytical_affine",
                "chosen_zoom": analytical["tile_info"]["zoom"],
                "chosen_center_latlon": [lat, lon],
            },
            "budget_remaining": state.match_at_budget,
        }

    def _run_minima(sigma_used):
        centers_local = [(name, float(lat), float(lon), float(sigma_used))]
        return sliding_window_position(
            matcher=state.minima_matcher, map_img=state.map_img,
            sam3_mask=state.current_mask, centers=centers_local,
            scale_ratio=scale_ratio, dpi=state.dpi,
            rotations=[rotation] if rotation else None,
            road_names=state.pdf_info.get("road_names") or [],
            grayscale=False, return_candidates=False,
            directional_modifier=state.pdf_info.get("directional_modifier"),
        )

    from tools.metrics.reward import compute_match_reward

    def _evaluate(res):
        if not res or res.get("affine_H") is None:
            return None, None
        mi_local = res.get("match_info") or {}
        rw = compute_match_reward(
            match_info=mi_local,
            pdf_info=state.pdf_info,
            inlier_pts_in_map=None,
            map_shape_hw=tuple(state.map_img.shape[:2]) if state.map_img is not None else None,
        )
        return mi_local, rw

    try:
        result = _run_minima(sigma_m)
    except Exception as e:
        return {"success": False, "error": f"sliding_window_position: {e}"}

    if not result or result.get("affine_H") is None:
        return {"success": False,
                "error": "MINIMA returned no usable match at this center"}

    mi = result.get("match_info") or {}
    _, reward = _evaluate(result)

    # 2× sigma retry on weak round-1: if n_inliers < 25 or overall_score
    # < 0.4, rerun with sigma×2 on the same center. Targets SSA*-style
    # rural cottages where the geocoded village center is 800-1900m from
    # GT — bigger search radius gives MINIMA a chance to actually see GT.
    # Only the better of the two results is kept.
    weak = (int(mi.get("n_inliers", 0) or 0) < 25
            or float(reward.overall_score) < 0.4)
    if weak:
        retry_sigma = float(sigma_m) * 2.0
        try:
            retry_result = _run_minima(retry_sigma)
        except Exception:
            retry_result = None
        if retry_result and retry_result.get("affine_H") is not None:
            retry_mi, retry_reward = _evaluate(retry_result)
            if (retry_reward is not None
                    and retry_reward.overall_score > reward.overall_score):
                print(f"  match_at: 2× sigma retry won "
                      f"({reward.overall_score:.2f} → "
                      f"{retry_reward.overall_score:.2f})")
                result = retry_result
                mi = retry_mi
                reward = retry_reward
                sigma_m = retry_sigma

    affine_H = result.get("affine_H")
    tile_info = result.get("tile_info")

    # Store the attempt for later commit
    cid = state._match_attempt_counter
    state._match_attempt_counter += 1
    state.match_attempts[cid] = {
        "candidate_id": cid,
        "name": name, "lat": lat, "lon": lon, "sigma_m": sigma_m,
        "scale_ratio": scale_ratio, "rotation": rotation,
        "affine_H": affine_H, "tile_info": tile_info,
        "match_info": mi, "geojson": result.get("geojson"),
        "reward": reward.to_dict(),
        "overall_score": reward.overall_score,
    }

    summary = {
        "success": True,
        "candidate_id": cid,
        "overall_score": float(reward.overall_score),
        "reward": reward.summary,  # multi-line text for the LLM
        "match_summary": {
            "n_inliers": int(mi.get("n_inliers", 0)),
            "score": float(mi.get("score", 0)),
            "aspect": float(mi.get("aspect", 0)),
            "avg_scale": float(mi.get("avg_scale", 0)),
            "chosen_zoom": mi.get("zoom"),
            "chosen_center_latlon": mi.get("center_latlon"),
        },
        "budget_remaining": state.match_at_budget,
    }

    # Attach a visual panel: planning map | OS tiles at this match. Across
    # multiple match_at calls the agent accumulates these panels in its
    # conversation history so it can compare candidates VISUALLY (not just
    # by reward scores) before commit_match. Targets the v10 wrong-area
    # accepts where MINIMA's textual scores were marginally beaten by a
    # wrong-area window — visually obvious, scalar-invisible.
    panel = None
    try:
        panel = _build_match_at_panel(state.map_img, tile_info, mi)
    except Exception as e:
        print(f"  match_at: panel build failed: {e}")
    if panel is None:
        return summary
    return ToolReturn(
        return_value=summary,
        content=[
            f"match_at id={cid} (overall_score={reward.overall_score:.2f}). "
            f"Visual: planning map (left) vs OS tiles at this match (right, "
            f"red rectangle = matched window). Compare road patterns: do the "
            f"streets inside the red box look like the planning map's drawn "
            f"streets? If you call match_at again on a different center, you "
            f"will see another panel and can pick the one that visually agrees.",
            _img_to_binary(panel),
        ],
    )


@_agent.tool
def commit_match(ctx: RunContext[AgentState], candidate_id: int) -> dict:
    """Mark a stored match attempt as the active result.

    After commit_match, extract_boundary / project_boundary / verify_position
    operate on this candidate's affine + tile_info. The smart-commit gate
    rejects commits with low evidence (n_inliers < 18 or mask_frac < 0.002)
    and redirects to a better candidate when one is available (combines
    n_inliers with an inside-LA-polygon weight). Analytical short-circuit
    matches are exempt. You may call commit_match again to change your mind.

    Args:
        candidate_id: ID returned from a prior match_at call.

    Returns:
        {"success": True, "committed": {n_inliers, overall_score, ...}}
    """
    state = ctx.deps
    cand = state.match_attempts.get(int(candidate_id))
    if cand is None:
        raise ModelRetry(
            f"candidate_id={candidate_id} not found. Available IDs: "
            f"{sorted(state.match_attempts.keys())}"
        )

    # Smart commit gate: pick the BEST match_attempt to commit, not just
    # argmax(n_inliers). Combines n_inliers with inside-LA-polygon (catches
    # wrong-town homonyms — 0.3x penalty for outside-LA). Exempts analytical
    # short-circuit matches and skips when the agent has tried <2 candidates.
    cand_mi = (cand.get("match_info") or {})
    cand_method = str(cand_mi.get("method", ""))
    if cand_method not in ("analytical", "analytical_affine"):
        from tools.matching import candidate_passes_la_filter
        from tools.scoring import commit_attempt_score
        admin_region = (state.pdf_info or {}).get("admin_region") if state.pdf_info else None

        def _attempt_score(c):
            """Score a match_attempt for commit-priority ranking."""
            mi = c.get("match_info") or {}
            method = str(mi.get("method", ""))
            if method in ("analytical", "analytical_affine"):
                return float("inf")  # always wins
            try:
                n = int(mi.get("n_inliers", 0))
            except (TypeError, ValueError):
                return -1
            ll = mi.get("center_latlon") or mi.get("chosen_center_latlon")
            inside_la = True
            if admin_region and ll:
                try:
                    inside_la = candidate_passes_la_filter(
                        "feature_cluster", ll[0], ll[1], admin_region
                    )
                except Exception:
                    inside_la = True  # fail-open
            return commit_attempt_score(n, inside_la)

        best_id = None
        best_score = _attempt_score(cand)
        for cid, c in state.match_attempts.items():
            if cid == int(candidate_id):
                continue
            cscore = _attempt_score(c)
            if cscore > best_score:
                best_score = cscore
                best_id = cid

        if (best_id is not None
            and len(state.match_attempts) >= 2):
            best_mi = (state.match_attempts[best_id].get("match_info") or {})
            best_n = best_mi.get("n_inliers", "?")
            raise ModelRetry(
                f"commit_match REJECTED candidate_id={candidate_id} "
                f"(score={best_score - 0.01:.1f}). Candidate_id={best_id} "
                f"has a better commit-score (n_inliers={best_n}, "
                f"inside-LA-weighted). Commit candidate_id={best_id} "
                f"instead. (Smart-commit gate combines inliers + "
                f"LA-polygon containment.)"
            )

    # Strict commit gate (see MIN_INLIERS_COMMIT / MIN_MASK_FRAC_COMMIT
    # at top of file for empirical derivation). Analytical short-circuit
    # matches are exempt.
    mi = (cand.get("match_info") or {})
    method = str(mi.get("method", ""))
    if method not in ("analytical", "analytical_affine"):
        n_in_raw = mi.get("n_inliers", 0)
        try:
            n_in = int(n_in_raw)
        except (TypeError, ValueError):
            # Non-numeric n_inliers (e.g. "n/a (analytical)") — treat as
            # analytical and exempt rather than rejecting on parse.
            n_in = -1
        mask_frac = 0.0
        if state.current_mask is not None and state.current_mask.size > 0:
            mask_frac = float(np.sum(state.current_mask > 0)) / float(state.current_mask.size)
        if n_in >= 0 and (n_in < MIN_INLIERS_COMMIT or mask_frac < MIN_MASK_FRAC_COMMIT):
            avail_ids = sorted(state.match_attempts.keys())
            raise ModelRetry(
                f"commit_match REJECTED candidate_id={candidate_id}: "
                f"low evidence (n_inliers={n_in}<{MIN_INLIERS_COMMIT} OR "
                f"mask_frac={mask_frac:.4f}<{MIN_MASK_FRAC_COMMIT}). "
                f"Try a different center via match_at, or call propose_centers"
                f"(extra_terms=[...]) to add more candidates. "
                f"Available IDs: {avail_ids}."
            )

    state.current_result = {
        "affine_H": cand["affine_H"],
        "tile_info": cand["tile_info"],
        "match_info": cand["match_info"],
        "geojson": cand.get("geojson"),
        # Stashed so the critic can read reward axes + know which candidate
        # is the live one (and therefore which match_attempts entries are
        # unpicked alternates worth offering as retry_at_center targets).
        "candidate_id": int(candidate_id),
        "reward": cand.get("reward"),
    }
    # Surface for benchmark visibility
    state.position_calls += 1

    return {
        "success": True,
        "committed": {
            "candidate_id": candidate_id,
            "name": cand["name"],
            "n_inliers": int((cand["match_info"] or {}).get("n_inliers", 0)),
            "overall_score": cand["overall_score"],
        }
    }


def _try_analytical_match_at(state: AgentState, name, lat, lon, scale_ratio,
                                tolerance_m=50.0):
    """Analytical-affine variant for the v2 match_at flow.

    Triggered when the probe anchor `(lat, lon)` is within `tolerance_m` of
    an OS easting/northing parsed from `pdf_info.grid_refs`. Returns a
    match-attempt dict (same shape as MINIMA writes into match_attempts)
    or None to fall through to MINIMA.
    """
    from tools.geo.grid_ref import parse_easting_northing
    from tools.geo.coords import haversine_m as _distance_m
    from tools.matching import (analytical_affine_from_anchor,
                                       mask_to_geojson_affine)
    from tools.metrics.reward import RewardResult, AxisResult

    if state.current_mask is None or state.map_img is None or scale_ratio is None:
        return None

    en_anchor = None
    for gr in (state.pdf_info or {}).get("grid_refs") or []:
        ll = parse_easting_northing(gr)
        if ll is None:
            continue
        if _distance_m(lat, lon, ll[0], ll[1]) <= tolerance_m:
            en_anchor = (ll[0], ll[1], gr)
            break
    if en_anchor is None:
        return None

    bin_m = (state.current_mask > 0).astype(np.uint8)
    M = cv2.moments(bin_m)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    a_lat, a_lon, gr_text = en_anchor
    affine_H, tile_info = analytical_affine_from_anchor(
        plan_shape=state.map_img.shape[:2],
        mask_centroid_xy=(cx, cy),
        anchor_lat=a_lat, anchor_lon=a_lon,
        scale_ratio=int(scale_ratio), dpi=int(state.dpi),
    )
    geojson = mask_to_geojson_affine(state.current_mask, affine_H, tile_info)
    match_info = {
        "center": name, "center_latlon": [a_lat, a_lon],
        "zoom": tile_info["zoom"], "rotation": 0,
        "method": "analytical", "anchor_grid_ref": gr_text,
    }
    # Synthetic reward — analytical bypasses the multi-axis consistency
    # check (it's not a search, it's a construction). Score conveys
    # "trust this strongly" so commit_match prefers it over MINIMA probes.
    reward = RewardResult(
        axes={"analytical": AxisResult(
            score=1.0,
            verdict=f"affine constructed from {gr_text} + scale 1:{int(scale_ratio)}",
        )},
        overall_score=0.95,
        summary=(f"Analytical affine from {gr_text} + scale 1:{int(scale_ratio)} "
                 f"@ {state.dpi}dpi (no MINIMA)"),
    )
    return {
        "name": name, "lat": float(a_lat), "lon": float(a_lon),
        "sigma_m": 0.0, "scale_ratio": float(scale_ratio), "rotation": 0,
        "affine_H": affine_H, "tile_info": tile_info,
        "match_info": match_info, "geojson": geojson,
        "reward": reward.to_dict(),
        "overall_score": float(reward.overall_score),
    }
