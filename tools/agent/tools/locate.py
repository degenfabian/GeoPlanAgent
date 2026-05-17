"""Locate-stage worker tools: geocode + propose_centers.

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
from pydantic_ai import ModelRetry, RunContext

from tools.agent.state import _agent, AgentState


# ── Tool 2: geocode ────────────────────────────────────────────────────────

@_agent.tool_plain
def geocode(
    type: str,
    postcode: Optional[str] = None,
    grid_ref: Optional[str] = None,
) -> dict:
    """Geocode a UK postcode or OS grid reference.

    USE THIS ONLY for postcodes or grid references YOU SEE on the map image
    that PDFInfo did NOT already extract. Place-name geocoding (villages,
    farms, conservation areas, named buildings, addresses) is handled by
    propose_centers automatically.

    Types:
      - "postcode": UK postcode (e.g. "AL1 1BY").
      - "grid_ref": OS grid reference (e.g. "TL 1507 0672" or "TL 15 07")
        or full easting/northing like "528942 E 184544 N".

    Returns:
        {"success": true, "lat": float, "lon": float, ...} or
        {"success": false, "error": str}.
    """
    if type == "postcode":
        import requests as req
        if not postcode:
            raise ModelRetry("postcode is required for type='postcode'")
        pc = postcode.strip()
        try:
            r = req.get(f"https://api.postcodes.io/postcodes/{pc}", timeout=10)
            data = r.json()
            if data.get("status") == 200 and data.get("result"):
                res = data["result"]
                return {"success": True, "lat": res["latitude"],
                        "lon": res["longitude"], "type": "postcode",
                        "admin_district": res.get("admin_district", "")}
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {"success": False, "error": f"Postcode '{pc}' not found"}

    elif type == "grid_ref":
        from tools.geo.grid_ref import os_grid_ref_to_latlon
        if not grid_ref:
            raise ModelRetry("grid_ref is required for type='grid_ref'")
        result = os_grid_ref_to_latlon(grid_ref)
        if result:
            return {"success": True, "lat": result[0], "lon": result[1],
                    "type": "grid_ref", "grid_ref": grid_ref}
        return {"success": False,
                "error": f"Could not parse grid reference '{grid_ref}'"}

    raise ModelRetry(
        f"Invalid type '{type}'. Use 'postcode' or 'grid_ref' only — "
        f"place names are handled by propose_centers automatically."
    )


# ── propose_centers ─────────────────────────────────────────────────────────

@_agent.tool
def propose_centers(
    ctx: RunContext[AgentState],
    extra_terms: Optional[List[str]] = None,
    match_context: Optional[str] = None,
) -> dict:
    """Run the live LLM-locate sub-agent to pick ONE center for positioning.

    The sub-agent has 6 offline geocoder tools (postcode, grid_ref, place,
    road, intersect, la_check), views the rendered map image, and returns
    one picked (lat, lon, sigma_m, confidence, source).

    If the sub-agent loop fails (validation retries exhausted, HTTP error,
    budget exceeded), run_locate emits an emergency LA-centroid LocatePick
    — the worker always gets at least one candidate.

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
        {"success": True, "n_candidates": 1, "candidates": [{...}], ...}
    """
    state = ctx.deps
    if not state.pdf_info:
        return {"success": False, "error": "PDFInfo missing — reader hasn't run"}

    from tools.agent.locate_agent import run_locate

    model_name = "google/gemini-3-flash-preview"

    map_bytes = None
    if state.map_img is not None:
        try:
            _, buf = cv2.imencode(".png", state.map_img)
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

    pick, new_history = run_locate(
        pdf_info=pdf_info,
        map_img_bytes=map_bytes,
        model_name=model_name,
        match_context=match_context,
        prior_messages=state.locate_message_history or None,
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
