"""Extract-stage worker tools: extract_boundary + project_boundary."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext

from tools.agent.state import (
    _agent,
    AgentState,
    _dedup_check,
)


# ── Tool 4: extract_boundary ──────────────────────────────────────────────

_FIXED_QUERY = "planning boundary"


@_agent.tool
def extract_boundary(
    ctx: RunContext[AgentState],
    bbox: Optional[List[float]] = None,
) -> dict:
    """Extract the planning boundary from the rendered map using SAM3 semantic.

    Args:
        bbox: Optional [x1, y1, x2, y2] in pixels. FALLBACK only — don't
            pass on the first call. Tighten the bbox to focus SAM3 on a
            specific region of the map.

    Returns:
        {"success": True, "mask_area_pct": float}
    """
    state = ctx.deps
    _dedup_check(state, "extract_boundary", {"bbox": bbox})

    if state.map_img is None or state.map_crop_path is None:
        raise ModelRetry("No map image available. Call render_page first.")

    if bbox is not None:
        if len(bbox) != 4:
            raise ModelRetry(
                f"bbox must be [x1, y1, x2, y2] (4 numbers), got {bbox}")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ModelRetry(
                f"bbox is degenerate (x1={x1} y1={y1} x2={x2} y2={y2}); "
                f"need x2 > x1 and y2 > y1")

    from tools.extraction.sam3 import (extract_boundary_sam3_semantic,
                                        set_fold_for_case)
    set_fold_for_case(state.sam3_state, state.case_name)
    mask = extract_boundary_sam3_semantic(
        state.map_crop_path, state.sam3_processor,
        state.sam3_model, state.device, query=_FIXED_QUERY, bbox=bbox,
    )
    if mask is None:
        return {"success": False, "error": "SAM3 semantic returned no mask"}
    area_pct = float(np.sum(mask > 0)) / mask.size * 100
    state.current_mask = mask
    state.selected_indices = None
    if state.map_img is not None:
        sel_overlay = state.map_img.copy()
        sel_overlay[mask > 0] = [0, 255, 0]
        state.selected_overlay = cv2.addWeighted(
            state.map_img, 0.5, sel_overlay, 0.5, 0)
    return {"success": True, "mask_area_pct": round(area_pct, 2)}


# ── Tool 5: project_boundary ──────────────────────────────────────────────

@_agent.tool
def project_boundary(ctx: RunContext[AgentState]) -> dict:
    """Project the current boundary mask to real-world coordinates (GeoJSON).

    Uses the committed match's affine transform to convert the pixel mask into
    a GeoJSON polygon with lat/lon coordinates. Requires a prior commit_match
    (for the affine) and extract_boundary (for the mask).

    Returns:
        {"success": true, "n_polygons": int}
        The GeoJSON is stored internally and used by verify_position / visualize.
    """
    state = ctx.deps

    if state.current_mask is None:
        raise ModelRetry("No boundary mask available. Call extract_boundary first.")

    affine_H = state.current_result.get("affine_H")
    tile_info = state.current_result.get("tile_info")
    if affine_H is None or tile_info is None:
        raise ModelRetry(
            "No positioning result available. Run match_at → commit_match first."
        )

    from tools.matching import mask_to_geojson_affine
    geojson = mask_to_geojson_affine(state.current_mask, affine_H, tile_info)

    if geojson is None:
        return {"success": False, "error": "Mask projection produced no polygons"}

    state.current_result["geojson"] = geojson

    geom = geojson.get("geometry", {})
    if geom.get("type") == "MultiPolygon":
        n_polys = len(geom.get("coordinates", []))
    elif geom.get("type") == "Polygon":
        n_polys = 1
    else:
        n_polys = 0

    return {"success": True, "n_polygons": n_polys}
