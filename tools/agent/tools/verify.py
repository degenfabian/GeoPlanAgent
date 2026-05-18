"""Verify-stage worker tools: verify_position + lookup_district.

``verify_position`` now shows both panels the worker used to need two
tools for: (left) the SAM mask overlaid on the planning map, (right) the
projected polygon on a fresh OS-tile render at the committed centre.
The standalone ``visualize`` tool was removed since it was a subset of
this.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext, ToolReturn

from tools.agent.state import (
    _agent,
    AgentState,
    _dedup_check,
    _img_to_binary,
    _create_boundary_overlay,
    _draw_geojson_on_tiles,
)


# ── Tool 7: verify_position ─────────────────────────────────────────────────

@_agent.tool
def verify_position(
    ctx: RunContext[AgentState],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> ToolReturn:
    """Visually inspect the committed result.

    For SINGLE-group commits: one row — left panel = the planning map
    with the SAM3 mask overlaid (red); right panel = an OS-tile render
    at the committed centre with the projected polygon drawn (red).

    For MULTI-group commits: N rows of (planning-map + mask) overlays
    stacked vertically — one per area_group — followed by a single
    wider OS-tile panel showing the UNIONED polygon across all groups.

    Required by the output validator when the committed result is
    borderline (25 ≤ n_inliers ≤ 100). NOT required (or useful) for
    status='district_lookup' — the district polygon comes from
    OS BoundaryLine and cannot be refined visually.

    Args:
        lat: Latitude to render the OS tiles at (default: committed
            match's centre).
        lon: Longitude to render at (default: committed match's centre).
    """
    state = ctx.deps
    from tools.io.os_tiles import fetch_os_opendata_grid

    cr = state.current_result or {}
    if lat is None or lon is None:
        center_ll = (cr.get("match_info") or {}).get("center_latlon")
        if center_ll:
            lat, lon = center_ll
        else:
            raise ModelRetry(
                "No position available. Run match_at → commit_match first, "
                "or provide lat/lon explicitly."
            )

    def _label(img, text):
        bar = np.full((28, img.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    # One (page + SAM mask) overlay per committed group.
    per_group = cr.get("per_group") or []
    plan_panels: list = []
    for g in per_group:
        page = int(g.get("page") or 0)
        page_img = state.rendered_pages.get(page)
        mask = state.sam_masks_by_page.get(page)
        if page_img is None:
            continue
        overlay = (_create_boundary_overlay(page_img, mask)
                   if mask is not None else page_img)
        target_h = 360 if len(per_group) > 1 else 500
        oh, ow = overlay.shape[:2]
        ov = cv2.resize(overlay, (max(1, int(ow * target_h / oh)), target_h))
        plan_panels.append(_label(
            ov, f"page {page} (group {g.get('area_group')}) + SAM mask"))

    # OS tiles + projected polygon (union for multi-group).
    tile_info = fetch_os_opendata_grid(lat, lon, 17, 5, 5)
    tile_bgr = cv2.cvtColor(tile_info["image"], cv2.COLOR_RGB2BGR)
    geojson = cr.get("geojson")
    if geojson:
        tile_bgr = _draw_geojson_on_tiles(tile_bgr, geojson, tile_info)
    # Use the SAME pre-label target_h as plan_panels so heights match after
    # _label() adds 28 px. (Plan panels are 360 if multi-group else 500.)
    tile_target_h = 360 if len(per_group) > 1 else 500
    tile_panel = _label(
        cv2.resize(tile_bgr,
                   (max(1, int(tile_bgr.shape[1] * tile_target_h / tile_bgr.shape[0])),
                    tile_target_h)),
        f"OS tiles @ ({lat:.4f},{lon:.4f}) + polygon"
        + (" (union)" if len(per_group) > 1 else "")
    )

    # Layout. Multi-group: horizontal strip of planning panels on top,
    # OS-tile panel below (full width). Single-group: side-by-side.
    if len(plan_panels) > 1:
        top = np.hstack(plan_panels)
        if tile_panel.shape[1] != top.shape[1]:
            new_h = int(tile_panel.shape[0] * top.shape[1] / tile_panel.shape[1])
            tile_panel = cv2.resize(tile_panel, (top.shape[1], new_h))
        panel = np.vstack([top, tile_panel])
    elif plan_panels:
        panel = np.hstack([plan_panels[0], tile_panel])
    else:
        panel = tile_panel
    if panel.shape[1] > 1800:
        s = 1800 / panel.shape[1]
        panel = cv2.resize(panel, (1800, int(panel.shape[0] * s)))

    state.verify_position_called = True
    n_groups = len(per_group)
    return ToolReturn(
        return_value={"success": True, "lat": lat, "lon": lon,
                       "n_groups": n_groups},
        content=[
            f"verify_position @ ({lat:.4f}, {lon:.4f}) — "
            f"{n_groups} group(s):",
            _img_to_binary(panel),
            "For each planning-map panel, does the SAM mask (red) cover "
            "the intended boundary and miss legends/title blocks? On the "
            "OS-tile panel, does the (possibly unioned) red polygon land "
            "where the planning map shows? If a SPECIFIC group's mask is "
            "off but the rest look fine, call match_at again with "
            "page=<another page in that group> to retry just that group. "
            "Otherwise submit status='accepted' and capture concerns in "
            "visual_check_notes — the pipeline always produces a polygon.",
        ],
    )


# ── Tool 8: lookup_district ──────────────────────────────────────────────

@_agent.tool
def lookup_district(
    ctx: RunContext[AgentState],
    district_name: str,
) -> dict:
    """Look up the boundary of a UK administrative district from
    OS BoundaryLine (offline, OS Open Data).

    Use this when the planning document covers an ENTIRE district,
    borough, unitary authority, ward, or parish — not a specific site
    within one.

    Returns the official boundary polygon directly. If this succeeds,
    submit BoundaryOutcome with status="district_lookup".

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
        {"success": true, "geojson": <GeoJSON Feature>, ...} — boundary
        {"success": false, "error": str} — name not in OS BoundaryLine
    """
    state = ctx.deps
    _dedup_check(state, "lookup_district", {"district_name": district_name})

    from tools.geo.grid_ref import lookup_district_boundary

    # Support '|' alternates: try each variant in order until one works.
    variants = [v.strip() for v in district_name.split("|") if v.strip()]
    for variant in variants:
        result = lookup_district_boundary(variant)
        if result.get("success"):
            geojson = result["geojson"]
            # Normalize to MultiPolygon
            geom = geojson.get("geometry", {})
            if geom.get("type") == "Polygon":
                geojson["geometry"] = {
                    "type": "MultiPolygon",
                    "coordinates": [geom["coordinates"]],
                }
            geojson["properties"]["source"] = "os_boundaryline_district_lookup"
            state.current_result = {"geojson": geojson, "match_info": {}}
            return {
                "success": True,
                "matched_variant": variant,
                "instruction": "District lookup succeeded. Submit your final "
                               "result with status='district_lookup' and a brief "
                               "reasoning. No positioning or verify_position needed.",
            }
    return {"success": False,
            "error": f"None of the variants {variants} matched in OS BoundaryLine"}


