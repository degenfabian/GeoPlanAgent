"""The worker agent's positioning tool surface: propose_centers, match_at, commit_match, submit_pdf_info, and lookup_district."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition

from geoplanagent.agents.worker import _agent
from geoplanagent.schemas import PDFInfo
from geoplanagent.utils import AgentState, _dedup_check


# propose_centers


@_agent.tool
def propose_centers(
    ctx: RunContext[AgentState],
    extra_terms: Optional[List[str]] = None,
    match_context: Optional[str] = None,
) -> dict:
    """Run the live LLM-locate sub-agent to pick ONE center for positioning.

    Returns EXACTLY ONE candidate per call. To try a different anchor,
    call propose_centers AGAIN — optionally with match_context="..."
    feedback telling the sub-agent why the previous pick was wrong, so
    it picks from a DIFFERENT signal type next time.

    In production the sub-agent has one offline geocoder tool — place
    (OS Open Names) — and views the rendered map image to compose 2-4
    queries before returning one picked (lat, lon, sigma_m, confidence,
    source). The five other geocoders implemented in
    ``geoplanagent.agents.locate`` (postcode, grid_ref, road, intersect,
    la_check) are used only by the locate ablation's all-tools agent.

    If the sub-agent loop fails entirely (validation retries exhausted,
    HTTP error, budget exceeded), run_locate emits an emergency
    LA-centroid LocatePick — so propose_centers always returns one
    candidate, never zero.

    Args:
        extra_terms: extra place-name strings to add to the locate sub-agent's
            inputs (e.g. a landmark visible on the map that the reader missed).
        match_context: feedback to give the locate sub-agent after a prior
            poor match_at result. Describe what went wrong in plain English,
            e.g. "Prior pick at (51.51, -2.63) gave only 12 inliers; OS tile
            showed farmland but planning map shows dense urban streets, so
            the LA centroid was probably wrong — try a road-based pick
            instead." The sub-agent gets this in its user message and is
            told to pick from a DIFFERENT signal type.

    Returns:
        {"success": True, "n_candidates": 1, "candidates": [{...}],
         "engine": "live_llm_locate", "evidence": str}
        — "candidates" is always a one-element list (this call returns
        exactly one pick).
    """
    state = ctx.deps
    if not state.pdf_info:
        if getattr(state, "folded_mode", False):
            return {
                "success": False,
                "error": (
                    "PDFInfo missing — you must call submit_pdf_info first. "
                    "Read the PDF binary attached to your first user "
                    "message, populate the PDFInfo schema, and submit it "
                    "before any positioning tool."
                ),
            }
        return {"success": False, "error": "PDFInfo missing — reader hasn't run"}

    from geoplanagent.agents.locate import run_locate

    # Model is configured at run_agent time via the CLI --locate-model
    # flag, threaded through AgentState. Default in AgentState is
    # google/gemini-3-flash-preview (matches the previous hardcode).
    locate_model_name = state.locate_model_name

    # Locate sub-agent always sees the primary match page (the
    # reader's top-ranked one). That single image is sent on EVERY
    # invocation — including re-invocations made while the worker is
    # positioning a different area_group's page on multi-area docs.
    from geoplanagent.utils import primary_match_page

    primary_page = primary_match_page(state)
    map_img = state.rendered_pages.get(primary_page) if primary_page else None
    map_bytes = None
    if map_img is not None:
        try:
            _, encoded = cv2.imencode(".png", map_img)
            map_bytes = encoded.tobytes()
        except Exception:
            map_bytes = None

    pdf_info = dict(state.pdf_info or {})
    if extra_terms:
        merged_places = list(pdf_info.get("place_names") or [])
        merged_labels = list(pdf_info.get("visible_map_labels") or [])
        for term in extra_terms:
            if not isinstance(term, str) or not term.strip():
                continue
            term = term.strip()
            if term not in merged_places:
                merged_places.insert(0, term)
            if term not in merged_labels:
                merged_labels.insert(0, term)
        pdf_info["place_names"] = merged_places
        pdf_info["visible_map_labels"] = merged_labels

    # ``extra_terms`` is also forwarded explicitly: on the first call,
    # the merge above already surfaces them via the pdf_info JSON, but
    # on a continuation call run_locate does NOT re-send pdf_info, so
    # the new terms have to be spliced into the follow-up user message
    # to actually reach the sub-agent.
    pick, new_history = run_locate(
        pdf_info=pdf_info,
        map_img_bytes=map_bytes,
        model_name=locate_model_name,
        match_context=match_context,
        prior_messages=state.locate_message_history or None,
        extra_terms=extra_terms,
        # Telemetry sink: one dict per invocation appended to state.locate_calls.
        # Aggregated in run.collect_agent_stats so each metrics.json carries
        # locate_request_tokens / locate_response_tokens / locate_n_calls plus
        # the per-call dicts in agent_stats["locate_calls"] (whose
        # generation_ids scripts/compute_costs.py turns into
        # locate_generation_ids) alongside the reader + worker stats.
        usage_sink=state.locate_calls,
    )
    state.locate_message_history = new_history

    confidence = pick.confidence
    specificity = 5 if confidence == "high" else 3 if confidence == "med" else 1
    candidate = {
        "id": 0,
        "source": f"live_locate:{pick.picked_source[:40]}",
        "lat": float(pick.top_lat),
        "lon": float(pick.top_lon),
        "sigma_m": float(pick.sigma_m),
        "specificity": specificity,
    }
    state.proposed_centers = [candidate]
    return {
        "success": True,
        "n_candidates": 1,
        "candidates": [candidate],
        "engine": "live_llm_locate",
        "evidence": pick.evidence,
    }


# No inlier-count gate. A group "passes" iff MINIMA produced a valid
# affine_H (and therefore a geojson) for it. The mathematical floor is
# 3 inlier point pairs (6 equations, 6 unknowns), but MINIMA's internal
# RANSAC already enforces that — if we got a geojson back, we trust it.


def _axis_field(reward_dict: Optional[Dict[str, Any]], axis_name: str, field: str) -> Any:
    """Safe extract of an axis's score/verdict from a reward.to_dict() dump.
    Returns None if the reward, axes table, or axis entry is missing."""
    if not reward_dict:
        return None
    axes = reward_dict.get("axes") or {}
    axis = axes.get(axis_name) or {}
    return axis.get(field)


# Per-page render + segmentation helpers


def _get_or_render_page(state: AgentState, page: int) -> Optional[np.ndarray]:
    """Return the rendered map image for `page`. Cache on first need."""
    cached_img = state.rendered_pages.get(page)
    if cached_img is not None:
        return cached_img

    from geoplanagent.tools.pdf import render_map_page

    rendered = render_map_page(
        state.pdf_path, page, dpi=state.dpi, verbose=False, case_name=state.case_name
    )
    if rendered is None:
        return None
    map_img, rotation = rendered
    if rotation.get("applied"):
        state.rotation_checked = True
    state.rendered_pages[page] = map_img
    return map_img


def _get_or_compute_mask(state: AgentState, page: int, image: np.ndarray) -> Optional[np.ndarray]:
    """Return SAM3 mask for `page`. Compute + cache on first need."""
    cached_mask = state.sam_masks_by_page.get(page)
    if cached_mask is not None:
        return cached_mask
    from geoplanagent.tools.segment import extract_boundary_sam3_semantic, set_fold_for_case

    set_fold_for_case(state.sam3_state, state.case_name)
    mask = extract_boundary_sam3_semantic(
        image,
        state.sam3_state["processor"],
        state.sam3_state["model"],
        state.sam3_state["device"],
    )
    if mask is not None:
        state.sam_masks_by_page[page] = mask
    return mask


def _resolve_area_group(state: AgentState, page: int) -> int:
    """Return the area_group of `page` from pdf_info.map_page_details.

    Raises ModelRetry if `page` isn't a category='match' page. Falls back
    to 0 when pdf_info is empty (the legacy "no metadata" path).
    """
    page_details = (state.pdf_info or {}).get("map_page_details") or []
    if not page_details:
        return 0  # legacy path with no metadata — treat as single group 0
    match_pages_by_number = {
        int(detail["page"]): detail
        for detail in page_details
        if detail.get("category") == "match"
    }
    page_meta = match_pages_by_number.get(int(page))
    if page_meta is None:
        raise ModelRetry(
            f"page={page} is not a category='match' page. Valid match "
            f"pages: {sorted(match_pages_by_number.keys())}. Pick one from "
            f"pdf_info.map_pages."
        )
    return int(page_meta.get("area_group", 0))


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
         "budget_remaining": int, "committed_groups": [int]}
    """
    state = ctx.deps
    if state.match_at_budget <= 0:
        raise ModelRetry(
            "match_at budget exhausted. Pick the best stored candidate via "
            "commit_match and proceed — the pipeline always produces a "
            "polygon, even if the best score is low."
        )

    _dedup_check(
        state,
        "match_at",
        {
            "page": int(page),
            "name": name,
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "sigma_m": sigma_m,
            "scale_ratio": scale_ratio,
        },
    )

    # Reject invented coordinates.
    matched_candidate = None
    if state.proposed_centers:
        from geoplanagent.utils import haversine_m

        nearest_distance_m, nearest_candidate = min(
            (haversine_m(lat, lon, c["lat"], c["lon"]), c) for c in state.proposed_centers
        )
        # 100 m tolerance: covers rounding noise on candidate lat/lons
        # (sub-metre postcode centroids round to ~10 m, place-name
        # centroids to ~50 m). Anything beyond that means the LLM
        # produced a coordinate that wasn't in propose_centers — most
        # commonly a hallucinated centre from the map image itself.
        if nearest_distance_m > 100.0:
            available = ", ".join(
                f"id={c['id']} ({c['source'][:30]})" for c in state.proposed_centers[:8]
            )
            raise ModelRetry(
                f"match_at refuses fabricated coordinates "
                f"({lat:.5f}, {lon:.5f}) — nearest propose_centers entry "
                f"is {nearest_distance_m:.0f}m away ({nearest_candidate['source']}). "
                f"Use a (lat, lon) from a propose_centers candidate "
                f"directly. Available: {available}. If none look right, call "
                f"propose_centers(extra_terms=[...]) — do NOT invent "
                f"coordinates."
            )
        matched_candidate = nearest_candidate

    state.match_at_budget -= 1

    # σ resolution.
    from geoplanagent.tools.matching import sigma_from_scale

    def _parse_scale_denominator(scale: Any) -> Optional[int]:
        if not scale:
            return None
        import re

        match = re.search(r"1\s*[:/]\s*([\d,]+)", str(scale))
        if not match:
            return None
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None

    if sigma_m is None and matched_candidate is not None:
        candidate_sigma_m = matched_candidate.get("sigma_m")
        if candidate_sigma_m is not None and float(candidate_sigma_m) > 0:
            sigma_m = float(candidate_sigma_m)
    if sigma_m is None:
        resolved_scale = scale_ratio
        if resolved_scale is None and state.pdf_info:
            resolved_scale = _parse_scale_denominator(state.pdf_info.get("scale"))
        sigma_m = sigma_from_scale(resolved_scale)
    if scale_ratio is None and state.pdf_info:
        scale_ratio = _parse_scale_denominator(state.pdf_info.get("scale"))

    # Resolve the area_group of this page.
    area_group = _resolve_area_group(state, int(page))

    # Match the single requested page.
    page_result = _match_single_page(
        state, int(page), name, float(lat), float(lon), float(sigma_m), scale_ratio
    )
    page_result["area_group"] = area_group
    page_result["page"] = int(page)
    is_valid = page_result.get("affine_H") is not None and not page_result.get("error")
    n_inliers = int((page_result.get("match_info") or {}).get("n_inliers") or 0) if is_valid else 0
    geojson = page_result.get("geojson") if is_valid else None

    # Store the attempt. per_group is a 1-element list (kept for shape
    # parity with the rest of the pipeline — the critic,
    # _recompute_current_result, and build_error_stats all read per_group).
    candidate_id = state._match_attempt_counter
    state._match_attempt_counter += 1
    state.match_attempts[candidate_id] = {
        "candidate_id": candidate_id,
        "name": name,
        "lat": float(lat),
        "lon": float(lon),
        "per_group": [page_result],
        "geojson": geojson,
        "n_groups_committed": 1 if is_valid else 0,
        "requested_page": int(page),
        "requested_group": area_group,
    }

    return {
        "success": True,
        "candidate_id": candidate_id,
        "area_group": area_group,
        "page": int(page),
        "n_inliers": n_inliers,
        "road_name_agreement": _axis_field(page_result.get("reward"), "road_name_agreement", "score"),
        "road_name_verdict": _axis_field(page_result.get("reward"), "road_name_agreement", "verdict"),
        "scale_consistency": _axis_field(page_result.get("reward"), "scale_consistency", "score"),
        "budget_remaining": state.match_at_budget,
        "committed_groups": sorted(state.committed_groups.keys()),
    }


# Per-page MINIMA driver (called once per group inside match_at)


def _segment_boundary(state: AgentState, page: int):
    """match_at step 1 — render the page and segment the drawn boundary
    with the fold-routed SAM3 (paper §4.2 step 2). Returns
    (map_img, mask, error_dict)."""
    map_img = _get_or_render_page(state, page)
    if map_img is None:
        return None, None, {"error": f"render failed for page {page}"}
    mask = _get_or_compute_mask(state, page, map_img)
    if mask is None:
        return map_img, None, {"error": f"SAM3 returned no mask for page {page}"}
    return map_img, mask, None


def _search_window(
    state: AgentState,
    map_img,
    mask,
    name: str,
    lat: float,
    lon: float,
    sigma_m: float,
    scale_ratio: Optional[int],
) -> Dict[str, Any]:
    """match_at step 2 — sliding-window MINIMA search of the map against
    OS tiles around (lat, lon) (paper §4.2 step 1). Returns the matcher
    result, or a dict with only an "error" key."""
    from geoplanagent.tools.matching import sliding_window_position

    road_names = (state.pdf_info or {}).get("road_names") or []
    try:
        result = sliding_window_position(
            matcher=state.minima_matcher,
            map_img=map_img,
            sam3_mask=mask,
            centers=[(name, lat, lon, sigma_m)],
            scale_ratio=scale_ratio,
            dpi=state.dpi,
            road_names=road_names,
        )
    except Exception as e:
        return {"error": f"sliding_window_position: {e!s:.140}"}
    if not result or result.get("affine_H") is None:
        return {"error": "MINIMA returned no usable match"}
    return result


def _project_candidate(state: AgentState, mask, result) -> Dict[str, Any]:
    """match_at step 3 — score the recovered affine and project the SAM3
    mask through it to WGS84 (paper §4.2 step 3)."""
    from geoplanagent.tools.matching import compute_match_reward, mask_to_geojson_affine

    match_info = result.get("match_info") or {}
    reward = compute_match_reward(match_info=match_info, pdf_info=state.pdf_info)

    affine_H = result.get("affine_H")
    tile_info = result.get("tile_info")
    geojson = result.get("geojson")
    if geojson is None and affine_H is not None and tile_info is not None:
        geojson = mask_to_geojson_affine(mask, affine_H, tile_info)

    return {
        "affine_H": affine_H,
        "tile_info": tile_info,
        "match_info": match_info,
        "geojson": geojson,
        "reward": reward.to_dict() if reward is not None else None,
    }


def _match_single_page(
    state: AgentState,
    page: int,
    name: str,
    lat: float,
    lon: float,
    sigma_m: float,
    scale_ratio: Optional[int],
) -> Dict[str, Any]:
    """One match_at attempt = segment → search → project on a single page.
    Returns a dict with affine_H / tile_info / match_info / geojson /
    reward; or error."""
    map_img, mask, error = _segment_boundary(state, page)
    if error is not None:
        return error

    result = _search_window(state, map_img, mask, name, lat, lon, sigma_m, scale_ratio)
    if result.get("error"):
        return result

    return _project_candidate(state, mask, result)


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

    geometries = []
    properties: Dict[str, Any] = {}
    for feature in geojsons:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or feature
        try:
            geometries.append(shape(geometry))
        except Exception:
            continue
        if not properties:
            properties = dict(feature.get("properties") or {})
    if not geometries:
        return None
    union = unary_union(geometries)
    if union.is_empty or union.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    feature = {
        "type": "Feature",
        "geometry": mapping(union),
        "properties": {**properties, "source": "match_at_union"},
    }
    # Normalise to MultiPolygon for downstream.
    if feature["geometry"].get("type") == "Polygon":
        feature["geometry"] = {
            "type": "MultiPolygon",
            "coordinates": [feature["geometry"]["coordinates"]],
        }
    return feature


# commit_match


def _recompute_current_result(state: AgentState) -> None:
    """Rebuild ``state.current_result`` from every entry in
    ``state.committed_groups``.

    The geojson field is the shapely-union of every committed group's
    geojson. The other fields (affine_H, tile_info, match_info)
    come from the "primary" committed group — the one with the highest
    n_inliers — since they're single-page values that downstream
    visualisations only render against one page.

    For single-area docs (one entry in committed_groups) this matches
    the pre-refactor behavior exactly.
    """
    candidates = [state.match_attempts[cid] for cid in state.committed_groups.values()]
    if not candidates:
        state.current_result = {}
        return

    # Union every group's geojson (helper handles the empty and
    # single-input cases: None / as-is).
    geojsons = [c.get("geojson") for c in candidates if c.get("geojson")]
    unioned_geojson = _union_geojsons(geojsons)

    # Primary = highest-inlier committed group. Its affine/tile/mask
    # are the ones we render for human visualisations and what
    # downstream `affine_H.npy` / `boundary_mask.png` get saved from.
    # n_inliers per candidate lives at per_group[0].match_info.n_inliers
    # since each candidate covers exactly one area_group.
    def _candidate_n_inliers(candidate) -> int:
        per_group = (candidate.get("per_group") or [{}])[0]
        return int((per_group.get("match_info") or {}).get("n_inliers") or 0)

    primary_candidate = max(candidates, key=_candidate_n_inliers)
    primary_per_group = (primary_candidate.get("per_group") or [{}])[0]

    # Sum n_inliers across committed groups — exposed below as
    # state.current_result["total_inliers"] for the output validator
    # (which surfaces it as BoundaryOutcome.final_n_inliers).
    total_inliers = sum(_candidate_n_inliers(c) for c in candidates)

    state.current_result = {
        "affine_H": primary_per_group.get("affine_H"),
        "tile_info": primary_per_group.get("tile_info"),
        "match_info": primary_per_group.get("match_info"),
        "geojson": unioned_geojson,
        "candidate_id": primary_candidate.get("candidate_id"),
        # per_group on current_result lists ONE entry per committed
        # group (the first/only per_group entry from each candidate).
        "per_group": [(c.get("per_group") or [{}])[0] for c in candidates],
        "requested_group": primary_candidate.get("requested_group"),
        "total_inliers": total_inliers,
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
    candidate = state.match_attempts.get(int(candidate_id))
    if candidate is None:
        raise ModelRetry(
            f"candidate_id={candidate_id} not found. Available IDs: "
            f"{sorted(state.match_attempts.keys())}"
        )

    # Strict gate: this candidate's match must have produced a valid affine.
    n_groups_committed = int(candidate.get("n_groups_committed") or 0)
    if n_groups_committed == 0:
        available_ids = sorted(state.match_attempts.keys())
        raise ModelRetry(
            f"commit_match REJECTED candidate_id={candidate_id}: MINIMA "
            f"produced no usable affine for this attempt (missing "
            f"affine_H/geojson). Try a different page or a different "
            f"centre via match_at; or call propose_centers"
            f"(extra_terms=[...]) to add more candidates. "
            f"Available IDs: {available_ids}."
        )

    # Update the per-group commit registry and rebuild current_result.
    group_id = int(candidate.get("requested_group", 0))
    state.committed_groups[group_id] = int(candidate_id)
    _recompute_current_result(state)
    state.n_commits += 1

    # Count the number of polygons in the now-unioned final geojson.
    geojson = state.current_result.get("geojson")
    n_polygons = 0
    if isinstance(geojson, dict):
        geometry = geojson.get("geometry") or {}
        if geometry.get("type") == "MultiPolygon":
            n_polygons = len(geometry.get("coordinates") or [])
        elif geometry.get("type") == "Polygon":
            n_polygons = 1

    # n_inliers lives in this candidate's only per_group entry.
    candidate_per_group = (candidate.get("per_group") or [{}])[0]
    candidate_n_inliers = int((candidate_per_group.get("match_info") or {}).get("n_inliers") or 0)

    return {
        "success": True,
        "committed": {
            "candidate_id": int(candidate_id),
            "area_group": group_id,
            "name": candidate["name"],
            "n_inliers": candidate_n_inliers,
            "n_polygons": n_polygons,
        },
        # Across-call state so the worker can see which groups are
        # still uncommitted on multi-area documents.
        "all_committed_groups": sorted(state.committed_groups.keys()),
    }


async def _hide_unless_folded(
    ctx: RunContext[AgentState], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Make submit_pdf_info invisible to the LLM unless folded_mode is set.

    pydantic-ai calls this before each model request and uses the returned
    ToolDefinition (or None) to decide what tools to expose. Returning None
    in standard mode means the standard worker sees the same 4-tool surface
    it had before the folded ablation was added — full bit-exact parity.
    """
    if getattr(ctx.deps, "folded_mode", False):
        return tool_def
    return None


def _is_empty_pdfinfo(info: PDFInfo) -> bool:
    """True iff every PDFInfo field is at its default — i.e. the agent
    submitted essentially `PDFInfo()` without actually reading anything.

    Used as a folded-mode "did you actually look at the PDF?" gate. A
    legitimate UK planning doc always yields at least one non-default
    field (an address, postcode, road name, place name, district name,
    or map_page_details entry); an all-default submission is the
    agent punting.
    """
    return (
        not info.site_address
        and not info.postcodes
        and not info.grid_refs
        and not info.scale
        and not info.map_pages
        and not info.map_page_details
        and not info.road_names
        and not info.place_names
        and not info.is_district_wide
        and not info.district_name
        and not info.house_number_road_pairs
        and not info.parish_names
        and not info.admin_region
        and not info.likely_town_or_city
        and not info.visible_map_labels
        and not info.adjacency_hints
    )


@_agent.tool(prepare=_hide_unless_folded)
def submit_pdf_info(ctx: RunContext[AgentState], info: PDFInfo) -> dict:
    """Initialise PDFInfo for this case. One-shot per case — this tool
    populates the PDFInfo that the positioning tools (propose_centers,
    match_at, commit_match, lookup_district) read from. It is the
    required first action whenever PDFInfo is not yet populated.

    The PDF binary is attached to your first user message. Read every
    page, populate the PDFInfo schema (the full schema, including
    field descriptions and validators, is sent to you as this tool's
    parameter spec), and submit. The system validates against the
    schema, stores the result on case state, and pre-renders the
    map_pages you identified.

    If PDFInfo is already populated for this case, this tool errors —
    use the positioning tools directly. Submitting a PDFInfo with
    every field at its default also errors (it means you did not
    actually read the PDF).

    Args:
        info: PDFInfo instance with every applicable field populated by
            reading the attached PDF. See the schema for field
            semantics — postcodes, grid_refs, road_names, place_names,
            map_page_details, etc. are all required to be filled when
            present in the document.

    Returns:
        {"success": True, "map_pages_rendered": [page numbers],
         "next_step": short instruction string}
    """
    state = ctx.deps
    if state.pdf_info:
        raise ModelRetry(
            "PDFInfo is already populated for this case — do not call "
            "submit_pdf_info again. Proceed with propose_centers → "
            "match_at → commit_match."
        )

    # pydantic-ai has already validated `info` against the PDFInfo schema
    # by the time we get here (typed parameter). The remaining gate is
    # the "did you actually read the PDF?" check.
    if _is_empty_pdfinfo(info):
        raise ModelRetry(
            "You submitted a PDFInfo with every field at its default — "
            "no address, postcodes, road names, place names, district, "
            "map_page_details, or anything else. That means you did not "
            "actually read the PDF binary attached to your first user "
            "message. Open the PDF, look at every page, and extract: "
            "(a) map_page_details for EVERY page that contains map-like "
            "content (category 'match' or 'discard'), (b) the site "
            "address / road names / place names / postcodes visible in "
            "the text and on the maps, (c) is_district_wide + "
            "district_name if the document covers an entire borough. "
            "Then call submit_pdf_info again with the populated PDFInfo."
        )

    state.pdf_info = info.model_dump()

    # Mirror prepare_worker_state's render loop. We can't import
    # prepare_worker_state here without a cycle, so the render code is
    # duplicated (small, stable). It also duplicates this module's
    # _get_or_render_page helper, with one deliberate difference: it
    # sets rotation_checked only when the FIRST map page was rotated,
    # whereas the helper sets it for any rotated page — collapsing onto
    # the helper would change that telemetry field.
    from geoplanagent.tools.pdf import render_map_page

    map_pages = state.pdf_info.get("map_pages") or []
    rendered_page_numbers: list[int] = []
    for page_1based in map_pages:
        rendered = render_map_page(
            str(state.pdf_path),
            int(page_1based),
            dpi=state.dpi,
            verbose=False,
            case_name=state.case_name,
        )
        if rendered is None:
            continue
        page_img, rotation = rendered
        if rotation.get("applied") and page_1based == map_pages[0]:
            state.rotation_checked = True
        state.rendered_pages[int(page_1based)] = page_img
        rendered_page_numbers.append(int(page_1based))

    if not rendered_page_numbers:
        return {
            "success": True,
            "map_pages_rendered": [],
            "next_step": (
                "No map_pages identified. If you took status='district_lookup' "
                "path, call lookup_district(district_name=...). Otherwise "
                "re-examine the PDF — at least one page must be category='match'."
            ),
        }

    return {
        "success": True,
        "map_pages_rendered": rendered_page_numbers,
        "next_step": (
            f"Primary match page is {rendered_page_numbers[0]}. Now run "
            f"propose_centers → match_at(page={rendered_page_numbers[0]}, ...) → "
            f"commit_match → return BoundaryOutcome. The locate sub-agent "
            f"reads the rendered map image directly from state."
        ),
    }


# Tool: lookup_district


@_agent.tool
def lookup_district(
    ctx: RunContext[AgentState],
    district_name: str,
) -> dict:
    """Look up the boundary of a UK administrative district from
    OS BoundaryLine (offline, OS Open Data).

    Use whenever PDFInfo.is_district_wide=True, or when the document
    explicitly covers an entire administrative area (borough, district,
    ward, parish, named conservation area). On success, the district
    polygon is committed to internal state and you should submit
    BoundaryOutcome with status="district_lookup" next.

    Naming conventions (be specific to avoid ambiguous matches; the
    downstream resolver normalises "London Borough of X" → "X" and
    strips trailing "District"/"Borough"/"Council"):
      - "Camden, UK"
      - "Royal Borough of Kensington and Chelsea, UK"
      - "City of Westminster, UK"
      - "Broadland District, Norfolk, UK"

    Args:
        district_name: UK admin name with "UK" suffix. May contain
            "|"-separated alternates (e.g.
            "City of Westminster, UK | Westminster, UK") — each is
            tried in order until one resolves.

    Returns:
        {"success": true, "matched_variant": str, "instruction": str}
            — district polygon committed to internal state; submit
            BoundaryOutcome(status="district_lookup") next.
        {"success": false, "error": str} — name not in OS BoundaryLine.
    """
    state = ctx.deps
    _dedup_check(state, "lookup_district", {"district_name": district_name})

    from geoplanagent.tools.geocode import lookup_district_boundary

    # Support '|' alternates: try each variant in order until one works.
    variants = [v.strip() for v in district_name.split("|") if v.strip()]
    for variant in variants:
        result = lookup_district_boundary(variant)
        if result.get("success"):
            geojson = result["geojson"]
            # Normalize to MultiPolygon
            geometry = geojson.get("geometry", {})
            if geometry.get("type") == "Polygon":
                geojson["geometry"] = {
                    "type": "MultiPolygon",
                    "coordinates": [geometry["coordinates"]],
                }
            geojson["properties"]["source"] = "os_boundaryline_district_lookup"
            state.current_result = {"geojson": geojson, "match_info": {}}
            return {
                "success": True,
                "matched_variant": variant,
                "instruction": "District lookup succeeded. Submit your final "
                "result with status='district_lookup' and a "
                "brief reasoning.",
            }
    return {
        "success": False,
        "error": f"None of the variants {variants} matched in OS BoundaryLine",
    }
