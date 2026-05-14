"""Verify-stage worker tools: verify_position + lookup_district + visualize.

Extracted from ``tools/agent.py`` (stage-2 split, 2026-05-11). Registers
``verify_position``, ``lookup_district`` and ``visualize`` against the
shared ``_agent`` instance at import time.

The cross-tool helpers ``_create_boundary_overlay`` and
``_draw_geojson_on_tiles`` live in ``tools.agent_core`` so other tool
modules can reuse them. They are imported (not redefined) here.
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
    """Visually inspect the committed result on Ordnance Survey tiles.

    Renders the OS OpenData map at the committed (or supplied) location with
    the predicted boundary drawn in red, so you can confirm the positioning
    visually — roads, buildings, and named features should match the planning
    map. If no lat/lon is supplied, uses the committed match's center.

    Required by the output validator when 25 ≤ n_inliers ≤ 100 (borderline)
    OR when submitting status="district_lookup".

    Args:
        lat: Latitude to inspect (default: committed match's center)
        lon: Longitude to inspect (default: committed match's center)

    Returns:
        Image of OS tiles with boundary overlay (shown to you), plus
        {"success": true, "lat": float, "lon": float}.
    """
    state = ctx.deps
    from tools.io.os_tiles import fetch_os_opendata_grid

    if lat is None or lon is None:
        center_ll = state.current_result.get("match_info", {}).get("center_latlon")
        if center_ll:
            lat, lon = center_ll
        else:
            raise ModelRetry(
                "No position available. Run match_at → commit_match first, "
                "or provide lat/lon explicitly."
            )

    tile_info = fetch_os_opendata_grid(lat, lon, 17, 5, 5)
    tile_bgr = cv2.cvtColor(tile_info["image"], cv2.COLOR_RGB2BGR)

    geojson = state.current_result.get("geojson")
    if geojson:
        tile_bgr = _draw_geojson_on_tiles(tile_bgr, geojson, tile_info)

    # Mark that verify_position ran — the output_validator checks this flag.
    state.verify_position_called = True

    return ToolReturn(
        return_value={"success": True, "lat": lat, "lon": lon},
        content=[
            f"OS tiles at ({lat:.4f}, {lon:.4f}):",
            _img_to_binary(tile_bgr),
            "Visual verification complete. Compare road patterns, settlement "
            "shape, and named roads against the planning map. Either way, "
            "submit with status='accepted' and fill visual_check_notes. If "
            "features look weak or mismatched, capture your concerns in the "
            "notes — the pipeline always produces a polygon, never refuses.",
        ],
    )


# ── Tool 8: lookup_district ──────────────────────────────────────────────

@_agent.tool
def lookup_district(
    ctx: RunContext[AgentState],
    district_name: str,
) -> dict:
    """Look up the full boundary of an administrative district from OpenStreetMap.

    Use this when the planning document covers an ENTIRE district, borough, ward,
    or parish — not a specific site within one.

    This returns the official boundary polygon directly from OSM, no positioning
    or SAM3 extraction needed. If this succeeds, you're done — respond with DONE.

    Naming conventions (be specific to avoid ambiguous matches):
      - "London Borough of Barnet, London, UK"
      - "Royal Borough of Kensington and Chelsea, London, UK"
      - "City of Westminster, London, UK"
      - "Rowley Green, London Borough of Barnet, London, UK" (for wards)

    Args:
        district_name: Full name of the district/borough/ward, including parent
            areas for disambiguation and "UK" suffix.

    Returns:
        {"success": true, "geojson": <GeoJSON Feature>} — the complete boundary
        {"success": false, "error": str} — if the district wasn't found in OSM
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
            geojson["properties"]["source"] = "osm_district_lookup"
            state.current_result = {"geojson": geojson, "match_info": {}}
            return {
                "success": True,
                "matched_variant": variant,
                "instruction": "District lookup succeeded. Submit your final "
                               "result with status='district_lookup' and a brief "
                               "reasoning. No positioning or verify_position needed.",
            }
    return {"success": False,
            "error": f"None of the variants {variants} matched in OSM"}


# ── Tool 9: visualize ────────────────────────────────────────────────────

@_agent.tool
def visualize(ctx: RunContext[AgentState]) -> ToolReturn:
    """Show the current state: boundary mask overlay on the map image, instance mask
    candidates if available, and the positioned boundary on OS OpenData tiles.

    Call this to inspect your work before finishing. Check that:
      - The boundary mask correctly outlines the planning site (not too much, not too little)
      - The positioned boundary on OS tiles is in the right real-world location
      - Road names on the OS tiles match roads visible on the planning map

    Returns:
        Images of current state (shown to you), plus:
        {"success": true, "images_available": ["boundary_overlay", "instance_overlay", "positioned_on_os_tiles"]}
    """
    state = ctx.deps
    content_parts: list = []
    images_available = []

    # 1. Boundary overlay
    if state.current_mask is not None and state.map_img is not None:
        overlay = _create_boundary_overlay(state.map_img, state.current_mask)
        content_parts.append("Boundary mask overlay (red):")
        content_parts.append(_img_to_binary(overlay))
        images_available.append("boundary_overlay")

    # 2. Instance masks
    if state.instance_masks and state.map_img is not None:
        inst_viz = state.map_img.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (255, 255, 0), (0, 255, 255)]
        for i, inst in enumerate(state.instance_masks[:5]):
            color = colors[i % len(colors)]
            inst_viz[inst > 0] = color
        inst_overlay = cv2.addWeighted(state.map_img, 0.5, inst_viz, 0.5, 0)
        content_parts.append("Instance masks (red=0, green=1, blue=2, yellow=3, cyan=4):")
        content_parts.append(_img_to_binary(inst_overlay))
        images_available.append("instance_overlay")

    # 3. Positioned GeoJSON on OS tiles
    geojson = state.current_result.get("geojson")
    tile_info = state.current_result.get("tile_info")
    if geojson and tile_info and tile_info.get("image") is not None:
        tile_bgr = cv2.cvtColor(tile_info["image"], cv2.COLOR_RGB2BGR)
        tile_bgr = _draw_geojson_on_tiles(tile_bgr, geojson, tile_info)
        content_parts.append("Positioned boundary on OS OpenData tiles:")
        content_parts.append(_img_to_binary(tile_bgr))
        images_available.append("positioned_on_os_tiles")

    return ToolReturn(
        return_value={"success": True, "images_available": images_available},
        content=content_parts if content_parts else None,
    )
