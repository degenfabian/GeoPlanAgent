"""Locate-stage worker tools: geocode + propose_centers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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

    You don't need to call geocode at all in most cases. Only use it when:
      - You spot a postcode on the map (small text near a building or
        the title block) that PDFInfo.postcodes is missing.
      - You see a grid reference at a map corner (e.g. "TG 21" or "TR 2638")
        that PDFInfo.grid_refs doesn't include.

    Types:
      - "postcode": UK postcode (e.g. "AL1 1BY").
      - "grid_ref": OS grid reference (e.g. "TL 1507 0672" or "TL 15 07")
        or full easting/northing like "528942 E 184544 N".

    Args:
        type: "postcode" or "grid_ref"
        postcode: For type="postcode" — UK postcode
        grid_ref: For type="grid_ref" — OS grid reference or easting/northing

    Returns:
        {"success": true, "lat": float, "lon": float, ...} or
        {"success": false, "error": str}.

        Pass the (lat, lon) directly to match_at:
          match_at(lat=..., lon=..., name="<your label>", scale_ratio=...)
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


# ── Positioning tools ──────────────────────────────────────────────────────
#
# Three tools form the per-candidate positioning loop:
#
#   propose_centers  — generates ranked candidate locations from
#                      tools.candidates (multi-road consensus, triangulation,
#                      gpkg/Wikidata/Photon) UNIONED with positioning.py's
#                      internal geocoders. Returns the unified pool.
#   match_at         — runs MINIMA at ONE center. Stores the result by
#                      integer candidate_id, computes the multi-axis
#                      consistency reward, returns the formatted reward
#                      summary plus a visual panel.
#   commit_match     — selects a stored match as the active state. The
#                      smart-commit gate redirects to a better candidate
#                      when one exists (inliers × inside-LA weight) and
#                      rejects low-evidence commits.
#
# Decision pattern (baked into the system prompt): propose_centers → try
# the top 1-3 with match_at → commit_match on the winner → extract_boundary
# → project_boundary. Reject if no match scores ≥ 0.40 (subject to the
# rural override and the visual-mismatch veto).


@_agent.tool
def propose_centers(
    ctx: RunContext[AgentState],
    extra_terms: Optional[List[str]] = None,
) -> dict:
    """Generate ranked candidate centers for positioning the planning map.

    Fuses multi-road consensus + triangulation + parish/admin/region parsers
    + gpkg/Wikidata/Photon/postcodes.io/Nominatim/OS Open Names into a single
    deduplicated, specificity-sorted pool. Try the top 1-3 with match_at.

    Args:
        extra_terms: extra place-name strings to also geocode (e.g. a landmark
            visible on the map that the reader missed).

    Returns:
        {"success": True, "n_candidates": int,
         "candidates": [{"id": int, "source": str, "lat": float,
                          "lon": float, "sigma_m": float,
                          "specificity": int}, ...]}
    """
    state = ctx.deps
    if not state.pdf_info:
        return {"success": False, "error": "PDFInfo missing — reader hasn't run"}

    # ── LIVE LLM-LOCATE OVERRIDE ─────────────────────────────────────────
    # When GEOMAP_USE_V3_LLM_LOCATE=1, run a LIVE sub-agent that:
    #   - reads the FRESH pdf_info (from this run's reader phase)
    #   - views the rendered planning map image
    #   - calls 6 offline geocoder tools (postcode/grid_ref/place/road/intersect/la_check)
    #   - returns ONE picked (lat, lon, sigma_m, confidence, source)
    # Skips the entire 2200-line heuristic propose_centers_v2 cascade.
    # The worker then calls match_at on this one center, MINIMA runs once,
    # agent visually commits or rejects. No cascade fallback.
    #
    # Model: defaults to env GEOMAP_LOCATE_MODEL, falls back to
    # "google/gemini-3-flash-preview" (the gemini-flash alias).
    if os.environ.get("GEOMAP_USE_V3_LLM_LOCATE") == "1":
        try:
            from tools.agent.locate_agent import run_locate
            import cv2 as _cv2
            # Resolve model
            model_name = os.environ.get(
                "GEOMAP_LOCATE_MODEL",
                "google/gemini-3-flash-preview",  # gemini-flash default
            )
            # Encode map image to PNG bytes (if available)
            map_bytes = None
            if state.map_img is not None:
                try:
                    _, buf = _cv2.imencode(".png", state.map_img)
                    map_bytes = buf.tobytes()
                except Exception:
                    map_bytes = None
            # Call live locate agent with FRESH pdf_info + map image
            pick = run_locate(
                pdf_info=state.pdf_info or {},
                map_img_bytes=map_bytes,
                model_name=model_name,
            )
            if pick is not None:
                conf = pick.confidence
                specificity = (5 if conf == "high"
                               else 3 if conf == "med"
                               else 1)
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
            # Fall through to cascade if live locate failed for some reason
            print("  [propose_centers] live locate returned None — falling "
                  "back to cascade")
        except Exception as e:
            print(f"  [propose_centers] live locate raised: {e!s:.160} — "
                  f"falling back to cascade")
        # Fall through to normal cascade below

    # Unified locate_v2 cascade. Validated 2026-05-08 against 214 v13 cached
    # cases: 212/214 (99.1%) GT inside sigma for at least one candidate.
    # Pulls postcode + grid_ref + parish/landmark/road-inside-LA + feature_cluster
    # + la_centroid + multi_road_consensus + road_intersection + district, ranked
    # by a single feature-cluster scorer.
    try:
        from tools.candidates import propose_centers_v2, rank_candidates
        from tools.matching import (effective_sigma,
                                        candidate_passes_la_filter,
                                        sigma_from_scale)
        import re as _re
        pi = state.pdf_info
        scale_text = pi.get("scale_text") or pi.get("scale") or ""
        scale_ratio_v2 = None
        _m = _re.search(r"1\s*:?\s*([\d,]+)", str(scale_text).lower())
        if _m:
            try: scale_ratio_v2 = int(_m.group(1).replace(",", ""))
            except Exception: scale_ratio_v2 = None
        v2_cands = propose_centers_v2(
            pi, websearch_fn=None, extra_terms=extra_terms,
            # Pass pdf_path so locate_v2 can call v13's road-graph
            # generators (multi_road_consensus, road_intersection)
            # which need OCR access to the rendered map.
            pdf_path=state.pdf_path,
        )
        v2_cands = rank_candidates(v2_cands, pi)
        admin = pi.get("admin_region")
        # cap=6: three parallel diagnostics on v16 found cap=3 truncated
        # winning v13 candidates (Ar4.20, 12:00127's Lundy Green,
        # 12:00126's road geocode, 69's multi_road_consensus:4, etc.).
        # The agent already filters by match_at score so more is additive.
        cap = int(os.environ.get("GEOMAP_LOCATE_V2_TOP_N", "6"))
        out = []
        seen = set()
        for c in v2_cands:
            if not candidate_passes_la_filter(c.source, c.lat, c.lon, admin):
                continue
            key = (round(c.lat, 3), round(c.lon, 3))
            if key in seen: continue
            seen.add(key)
            sigma_use = max(int(c.sigma_m or 0),
                             effective_sigma(c.source, scale_ratio_v2))
            spec = int(getattr(c, "specificity", 3))
            out.append({
                "id": len(out),
                "source": c.source,
                "lat": float(c.lat),
                "lon": float(c.lon),
                "sigma_m": float(sigma_use),
                "specificity": spec,
            })
            if len(out) >= cap: break
        state.proposed_centers = out
        return {
            "success": True,
            "n_candidates": len(out),
            "scale_ratio_inferred": scale_ratio_v2,
            "default_sigma_m": sigma_from_scale(scale_ratio_v2),
            "candidates": out,
            "engine": "locate_v2",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"locate_v2 raised: {e!s:.200}",
            "engine": "locate_v2",
        }
