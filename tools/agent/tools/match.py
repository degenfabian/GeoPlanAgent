"""Match-stage worker tools: match_at + commit_match.

Each match_at call matches ONE page (and therefore one area_group). The
worker calls match_at + commit_match separately for each area_group of
a multi-area document; commit_match incrementally unions the committed
groups into state.current_result["geojson"].

For the typical single-area document (99% of cases) this is just:
  propose_centers → match_at(page=N) → commit_match → done.

SAM3 masks are cached on state.sam_masks_by_page keyed by 1-based page
number, so re-calling match_at on a page that's been segmented is fast
(MINIMA only).
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext

from tools.agent.state import (
    _agent,
    AgentState,
    _dedup_check,
)


# No inlier-count gate. A group "passes" iff MINIMA produced a valid
# affine_H (and therefore a geojson) for it. The mathematical floor is
# 3 inlier point pairs (6 equations, 6 unknowns), but MINIMA's internal
# RANSAC already enforces that — if we got a geojson back, we trust it.

# Fixed query for SAM3 semantic segmentation. The LoRA was trained against
# this literal phrase.
_SAM3_QUERY = "planning boundary"


def _axis_field(reward_dict: Optional[Dict[str, Any]], axis_name: str,
                  field: str) -> Any:
    """Safe extract of an axis's score/verdict from a reward.to_dict() dump.
    Returns None if the reward, axes table, or axis entry is missing."""
    if not reward_dict:
        return None
    axes = reward_dict.get("axes") or {}
    axis = axes.get(axis_name) or {}
    return axis.get(field)


# Per-page render + segmentation helpers

def _get_or_render_page(state: AgentState, page: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return (map_img, map_crop_path) for `page`. Cache on first need."""
    cached = state.rendered_pages.get(page)
    cached_path = state.rendered_page_paths.get(page)
    if cached is not None and cached_path is not None:
        return cached, cached_path

    from tools.io.map_page import render_map_page
    rendered = render_map_page(state.pdf_path, page, dpi=state.dpi,
                                  verbose=False, case_name=state.case_name)
    if rendered is None:
        return None, None
    map_img, rot_info = rendered
    if rot_info.get("applied"):
        state.rotation_checked = True
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    cv2.imwrite(path, map_img)
    state.rendered_pages[page] = map_img
    state.rendered_page_paths[page] = path
    return map_img, path


def _get_or_compute_mask(state: AgentState, page: int,
                          map_crop_path: str) -> Optional[np.ndarray]:
    """Return SAM3 mask for `page`. Compute + cache on first need."""
    cached = state.sam_masks_by_page.get(page)
    if cached is not None:
        return cached
    from tools.extraction.sam3 import (extract_boundary_sam3_semantic,
                                        set_fold_for_case)
    set_fold_for_case(state.sam3_state, state.case_name)
    mask = extract_boundary_sam3_semantic(
        map_crop_path, state.sam3_processor, state.sam3_model,
        state.device, query=_SAM3_QUERY,
    )
    if mask is not None:
        state.sam_masks_by_page[page] = mask
    return mask


def _resolve_area_group(state: AgentState, page: int) -> int:
    """Return the area_group of `page` from pdf_info.map_page_details.

    Raises ModelRetry if `page` isn't a category='match' page. Falls back
    to 0 when pdf_info is empty (the legacy "no metadata" path).
    """
    details = (state.pdf_info or {}).get("map_page_details") or []
    if not details:
        return 0  # legacy path with no metadata — treat as single group 0
    by_page = {int(d["page"]): d for d in details
               if d.get("category") == "match"}
    meta = by_page.get(int(page))
    if meta is None:
        raise ModelRetry(
            f"page={page} is not a category='match' page. Valid match "
            f"pages: {sorted(by_page.keys())}. Pick one from "
            f"pdf_info.map_pages."
        )
    return int(meta.get("area_group", 0))


# match_at

@_agent.tool
def match_at(
    ctx: RunContext[AgentState],
    page: int,
    name: str,
    lat: float,
    lon: float,
    sigma_m: Optional[float] = None,
    scale_ratio: Optional[float] = None,
) -> dict:
    """Run MINIMA at (lat, lon) on ONE page (one area_group).

    Each match_at covers exactly the page you pass and its area_group.
    For multi-area documents, call match_at + commit_match separately
    for each area_group's primary page.

    This tool returns numbers only — judge the match from n_inliers,
    scale_consistency, road_name_agreement + verdict.

    Args:
        page: 1-based page number. Must be a category='match' page from
            the reader's map_pages list.
        name: Short label — pass the ``source`` field of the candidate
            returned by propose_centers (e.g.
            ``"live_locate:postcode:AL1 3JE"``,
            ``"live_locate:intersect:Manor x Linden"``).
        lat / lon: Centre latitude / longitude (must come from
            propose_centers — fabricated coordinates are rejected).
        sigma_m: Search radius in metres (default: scale-aware).
        scale_ratio: Map scale denominator (default: parsed from PDFInfo.scale).

    Returns:
        {"success": True, "candidate_id": int, "area_group": int,
         "page": int, "n_inliers": int, "road_name_agreement": float,
         "road_name_verdict": str, "scale_consistency": float,
         "budget_remaining": int}
    """
    state = ctx.deps
    if state.match_at_budget <= 0:
        raise ModelRetry(
            "match_at budget exhausted. Pick the best stored candidate via "
            "commit_match and proceed — the pipeline always produces a "
            "polygon, even if the best score is low."
        )

    _dedup_check(state, "match_at", {
        "page": int(page), "name": name,
        "lat": round(float(lat), 5), "lon": round(float(lon), 5),
        "sigma_m": sigma_m, "scale_ratio": scale_ratio,
    })

    # Reject invented coordinates.
    matched_candidate = None
    if state.proposed_centers:
        from tools.geo.coords import haversine_km
        nearest = min(
            (haversine_km(lat, lon, c["lat"], c["lon"]) * 1000.0, c)
            for c in state.proposed_centers
        )
        # 100 m tolerance: covers rounding noise on candidate lat/lons
        # (sub-metre postcode centroids round to ~10 m, place-name
        # centroids to ~50 m). Anything beyond that means the LLM
        # produced a coordinate that wasn't in propose_centers — most
        # commonly a hallucinated centre from the map image itself.
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
                f"propose_centers(extra_terms=[...]) — do NOT invent "
                f"coordinates."
            )
        matched_candidate = nearest[1]

    state.match_at_budget -= 1

    # σ resolution.
    from tools.matching import sigma_from_scale

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

    if sigma_m is None and matched_candidate is not None:
        cand_sigma = matched_candidate.get("sigma_m")
        if cand_sigma is not None and float(cand_sigma) > 0:
            sigma_m = float(cand_sigma)
    if sigma_m is None:
        sr = scale_ratio
        if sr is None and state.pdf_info:
            sr = _parse_scale(state.pdf_info.get("scale"))
        sigma_m = sigma_from_scale(sr)
    if scale_ratio is None and state.pdf_info:
        scale_ratio = _parse_scale(state.pdf_info.get("scale"))

    # Resolve the area_group of this page.
    area_group = _resolve_area_group(state, int(page))

    # Match the single requested page.
    single = _match_single_page(state, int(page), name, float(lat), float(lon),
                                  float(sigma_m), scale_ratio, matched_candidate)
    single["area_group"] = area_group
    single["page"] = int(page)
    valid = single.get("affine_H") is not None and not single.get("error")
    n_inliers = int((single.get("match_info") or {}).get("n_inliers") or 0) if valid else 0
    geojson = single.get("geojson") if valid else None

    # Store the attempt. per_group is a 1-element list (kept for shape
    # parity with the rest of the pipeline — critic_agent, output
    # validator, and the saved metrics.json all read per_group).
    cid = state._match_attempt_counter
    state._match_attempt_counter += 1
    state.match_attempts[cid] = {
        "candidate_id": cid,
        "name": name, "lat": float(lat), "lon": float(lon),
        "sigma_m": float(sigma_m), "scale_ratio": scale_ratio,
        "per_group": [single],
        "geojson": geojson,
        "n_groups": 1,
        "n_groups_committed": 1 if valid else 0,
        "requested_page": int(page),
        "requested_group": area_group,
    }

    return {
        "success": True,
        "candidate_id": cid,
        "area_group": area_group,
        "page": int(page),
        "n_inliers": n_inliers,
        "road_name_agreement": _axis_field(
            single.get("reward"), "road_name_agreement", "score"),
        "road_name_verdict": _axis_field(
            single.get("reward"), "road_name_agreement", "verdict"),
        "scale_consistency": _axis_field(
            single.get("reward"), "scale_consistency", "score"),
        "budget_remaining": state.match_at_budget,
        "committed_groups": sorted(state.committed_groups.keys()),
    }


# Per-page MINIMA driver (called once per group inside match_at)

def _match_single_page(state: AgentState, page: int, name: str,
                        lat: float, lon: float, sigma_m: float,
                        scale_ratio: Optional[int],
                        matched_candidate: Optional[dict]) -> Dict[str, Any]:
    """Render+segment+MINIMA on a single page at (lat, lon). Returns a dict
    with affine_H / tile_info / match_info / geojson / mask_frac / reward;
    or error."""
    map_img, map_crop_path = _get_or_render_page(state, page)
    if map_img is None or map_crop_path is None:
        return {"error": f"render failed for page {page}"}
    mask = _get_or_compute_mask(state, page, map_crop_path)
    if mask is None:
        return {"error": f"SAM3 returned no mask for page {page}"}
    mask_frac = float(np.sum(mask > 0)) / float(mask.size)

    from tools.matching import sliding_window_position, mask_to_geojson_affine
    from tools.metrics.reward import compute_match_reward

    road_names = (state.pdf_info or {}).get("road_names") or []

    def _run_minima(sigma_used):
        return sliding_window_position(
            matcher=state.minima_matcher, map_img=map_img,
            sam3_mask=mask, centers=[(name, lat, lon, sigma_used)],
            scale_ratio=scale_ratio, dpi=state.dpi,
            rotations=None,
            road_names=road_names,
            grayscale=False, return_candidates=False,
        )

    def _evaluate(res):
        if not res or res.get("affine_H") is None:
            return None, None
        mi_local = res.get("match_info") or {}
        rw = compute_match_reward(
            match_info=mi_local, pdf_info=state.pdf_info,
        )
        return mi_local, rw

    try:
        result = _run_minima(sigma_m)
    except Exception as e:
        return {"error": f"sliding_window_position: {e!s:.140}"}
    if not result or result.get("affine_H") is None:
        return {"error": "MINIMA returned no usable match",
                "mask_frac": mask_frac}

    mi, reward = _evaluate(result)

    affine_H = result.get("affine_H")
    tile_info = result.get("tile_info")
    geojson = result.get("geojson")
    if geojson is None and affine_H is not None and tile_info is not None:
        geojson = mask_to_geojson_affine(mask, affine_H, tile_info)

    return {
        "affine_H": affine_H, "tile_info": tile_info,
        "match_info": mi, "geojson": geojson,
        "reward": reward.to_dict() if reward is not None else None,
        "mask_frac": mask_frac,
    }


# Polygon union helper

def _union_geojsons(geojsons: List[dict]) -> Optional[dict]:
    """shapely-union per-group GeoJSON Features → one combined Feature.

    Single input → return as-is. Empty → None. Multiple → MultiPolygon.
    """
    if not geojsons:
        return None
    if len(geojsons) == 1:
        return geojsons[0]

    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union

    geoms = []
    properties: Dict[str, Any] = {}
    for g in geojsons:
        if not isinstance(g, dict):
            continue
        geom = g.get("geometry") or g
        try:
            geoms.append(shape(geom))
        except Exception:
            continue
        if not properties:
            properties = dict(g.get("properties") or {})
    if not geoms:
        return None
    union = unary_union(geoms)
    if union.is_empty or union.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    out = {
        "type": "Feature",
        "geometry": mapping(union),
        "properties": {**properties, "source": "match_at_union"},
    }
    # Normalise to MultiPolygon for downstream.
    if out["geometry"].get("type") == "Polygon":
        out["geometry"] = {
            "type": "MultiPolygon",
            "coordinates": [out["geometry"]["coordinates"]],
        }
    return out


# commit_match

def _recompute_current_result(state: AgentState) -> None:
    """Rebuild ``state.current_result`` from every entry in
    ``state.committed_groups``.

    The geojson field is the shapely-union of every committed group's
    geojson. The other fields (affine_H, tile_info, match_info, reward)
    come from the "primary" committed group — the one with the highest
    n_inliers — since they're single-page values that downstream
    visualisations only render against one page.

    For single-area docs (one entry in committed_groups) this matches
    the pre-refactor behavior exactly.
    """
    cands = [state.match_attempts[cid]
             for cid in state.committed_groups.values()]
    if not cands:
        state.current_result = {}
        return

    # Union every group's geojson.
    geojsons = [c.get("geojson") for c in cands if c.get("geojson")]
    if len(geojsons) == 1:
        unioned = geojsons[0]
    elif len(geojsons) > 1:
        unioned = _union_geojsons(geojsons)
    else:
        unioned = None

    # Primary = highest-inlier committed group. Its affine/tile/mask
    # are the ones we render for human visualisations and what
    # downstream `affine_H.npy` / `boundary_mask.png` get saved from.
    # n_inliers per candidate lives at per_group[0].match_info.n_inliers
    # since each candidate covers exactly one area_group.
    def _cand_n_inliers(c) -> int:
        pg = (c.get("per_group") or [{}])[0]
        return int((pg.get("match_info") or {}).get("n_inliers") or 0)

    primary = max(cands, key=_cand_n_inliers)
    primary_pg = (primary.get("per_group") or [{}])[0]

    # Sum n_inliers across committed groups — exposed below as
    # state.current_result["total_inliers"] for the output validator
    # (which surfaces it as BoundaryOutcome.final_n_inliers).
    total = sum(_cand_n_inliers(c) for c in cands)

    state.current_result = {
        "affine_H": primary_pg.get("affine_H"),
        "tile_info": primary_pg.get("tile_info"),
        "match_info": primary_pg.get("match_info"),
        "geojson": unioned,
        "candidate_id": primary.get("candidate_id"),
        "reward": primary_pg.get("reward"),
        # per_group on current_result lists ONE entry per committed
        # group (the first/only per_group entry from each candidate).
        "per_group": [(c.get("per_group") or [{}])[0] for c in cands],
        "requested_group": primary.get("requested_group"),
        "requested_page": primary.get("requested_page"),
        "n_groups_committed": len(cands),
        "total_inliers": total,
    }


@_agent.tool
def commit_match(ctx: RunContext[AgentState], candidate_id: int) -> dict:
    """Commit a stored match_at attempt for its area_group.

    Each commit_match call adds (or replaces) the commit for ONE
    area_group — the one this candidate's match_at was called on. For
    multi-area documents the worker calls commit_match once per group;
    each call unions its new geojson into the running result.

    Calling commit_match a second time with a candidate covering an
    already-committed group OVERWRITES that group's commit; other
    groups stay. To change your mind, just call commit_match with a
    different id whose area_group matches.

    The only precondition is the strict gate: this candidate's match
    must have produced a valid affine.

    Args:
        candidate_id: ID returned from a prior match_at call.
    """
    state = ctx.deps
    cand = state.match_attempts.get(int(candidate_id))
    if cand is None:
        raise ModelRetry(
            f"candidate_id={candidate_id} not found. Available IDs: "
            f"{sorted(state.match_attempts.keys())}"
        )

    # Strict gate: this candidate's match must have produced a valid affine.
    n_committed = int(cand.get("n_groups_committed") or 0)
    if n_committed == 0:
        avail_ids = sorted(state.match_attempts.keys())
        raise ModelRetry(
            f"commit_match REJECTED candidate_id={candidate_id}: MINIMA "
            f"produced no usable affine for this attempt (missing "
            f"affine_H/geojson). Try a different page or a different "
            f"centre via match_at; or call propose_centers"
            f"(extra_terms=[...]) to add more candidates. "
            f"Available IDs: {avail_ids}."
        )

    # Update the per-group commit registry and rebuild current_result.
    group_id = int(cand.get("requested_group", 0))
    state.committed_groups[group_id] = int(candidate_id)
    _recompute_current_result(state)
    state.position_calls += 1

    # Count the number of polygons in the now-unioned final geojson.
    geojson = state.current_result.get("geojson")
    n_polys = 0
    if isinstance(geojson, dict):
        geom = geojson.get("geometry") or {}
        if geom.get("type") == "MultiPolygon":
            n_polys = len(geom.get("coordinates") or [])
        elif geom.get("type") == "Polygon":
            n_polys = 1

    # n_inliers lives in this candidate's only per_group entry.
    cand_pg = (cand.get("per_group") or [{}])[0]
    cand_n_inliers = int((cand_pg.get("match_info") or {}).get("n_inliers") or 0)

    return {
        "success": True,
        "committed": {
            "candidate_id": int(candidate_id),
            "area_group": group_id,
            "name": cand["name"],
            "n_inliers": cand_n_inliers,
            "n_polygons": n_polys,
        },
        # Across-call state so the worker can see which groups are
        # still uncommitted on multi-area documents.
        "all_committed_groups": sorted(state.committed_groups.keys()),
    }
