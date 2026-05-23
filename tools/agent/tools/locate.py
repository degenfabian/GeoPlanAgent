"""Locate-stage worker tool: propose_centers.

`propose_centers` always delegates to the live LLM-locate sub-agent
(`tools.agent.locate_agent.run_locate`). The sub-agent reads pdf_info,
views the rendered map image, and calls 6 offline geocoders
(postcode/grid_ref/place/road/intersect/la_check) to return ONE
picked (lat, lon, sigma_m, confidence, source). Pydantic enforces
the LocatePick shape; if the agent loop fails entirely run_locate
returns an emergency LA-centroid LocatePick. The worker is
guaranteed at least one candidate.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
from pydantic_ai import RunContext

from tools.agent.state import _agent, AgentState


# ── propose_centers ─────────────────────────────────────────────────────────

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

    The sub-agent has 6 offline geocoder tools (postcode, grid_ref, place,
    road, intersect, la_check), views the rendered map image, and returns
    one picked (lat, lon, sigma_m, confidence, source).

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
         "engine": "live_llm_locate", "evidence": str,
         "la_check_passed": bool}
        — "candidates" is always a one-element list (this call returns
        exactly one pick).
    """
    state = ctx.deps
    if not state.pdf_info:
        return {"success": False, "error": "PDFInfo missing — reader hasn't run"}

    from tools.agent.locate_agent import run_locate

    # Model is configured at run_agent time via the CLI --locate-model
    # flag, threaded through AgentState. Default in AgentState is
    # google/gemini-3-flash-preview (matches the previous hardcode).
    model_name = state.locate_model

    # Locate sub-agent always sees the primary match page (the
    # reader's top-ranked one). Single image is sufficient — locate
    # picks one centre per worker run regardless of how many
    # area_groups the document has.
    from tools.agent.state import primary_match_page
    primary_page = primary_match_page(state)
    map_img = state.rendered_pages.get(primary_page) if primary_page else None
    map_bytes = None
    if map_img is not None:
        try:
            _, buf = cv2.imencode(".png", map_img)
            map_bytes = buf.tobytes()
        except Exception:
            map_bytes = None

    pdf_info = dict(state.pdf_info or {})
    if extra_terms:
        merged_places = list(pdf_info.get("place_names") or [])
        merged_labels = list(pdf_info.get("visible_map_labels") or [])
        for t in extra_terms:
            if not isinstance(t, str) or not t.strip():
                continue
            t = t.strip()
            if t not in merged_places:
                merged_places.insert(0, t)
            if t not in merged_labels:
                merged_labels.insert(0, t)
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
        model_name=model_name,
        match_context=match_context,
        prior_messages=state.locate_message_history or None,
        extra_terms=extra_terms,
    )
    state.locate_message_history = new_history

    conf = pick.confidence
    specificity = (5 if conf == "high" else 3 if conf == "med" else 1)
    cand = {
        "id": 0,
        "source": f"live_locate:{pick.picked_source[:40]}",
        "lat": float(pick.top_lat),
        "lon": float(pick.top_lon),
        "sigma_m": float(pick.sigma_m),
        "specificity": specificity,
    }
    state.proposed_centers = [cand]
    return {
        "success": True,
        "n_candidates": 1,
        "candidates": [cand],
        "engine": "live_llm_locate",
        "evidence": pick.evidence,
        "la_check_passed": pick.la_check_passed,
    }
