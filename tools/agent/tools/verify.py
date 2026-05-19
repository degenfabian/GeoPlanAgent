"""Verify-stage worker tool: lookup_district.

This module previously also defined ``verify_position`` (a worker-facing
post-commit visual check). That tool was removed once the independent
LLM critic (``tools/agent/critic_agent.py``, enable_critic=True) took
over the post-commit visual-review role. The critic is more reliable
than worker self-verification (Huang et al. ICLR 2024; Voyager NeurIPS
2023; Tong et al. CVPR 2024).

``lookup_district`` stays here — it's a deterministic OS-BoundaryLine
lookup with no visual component.
"""

from __future__ import annotations

import os

from pydantic_ai import RunContext

from tools.agent.state import _agent, AgentState, _dedup_check


# ── Tool: lookup_district ────────────────────────────────────────────────

def _maybe_register_lookup_district(fn):
    """Conditional @_agent.tool registration. When
    GEOMAP_DISABLE_LOOKUP_DISTRICT=1, the function is NOT registered with
    the worker agent — the tool is invisible to the model, no
    ModelRetry path. Clean for ablation.
    """
    if os.environ.get("GEOMAP_DISABLE_LOOKUP_DISTRICT") == "1":
        return fn
    return _agent.tool(fn)


@_maybe_register_lookup_district
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
                               "result with status='district_lookup' and a "
                               "brief reasoning.",
            }
    return {"success": False,
            "error": f"None of the variants {variants} matched in OS BoundaryLine"}
