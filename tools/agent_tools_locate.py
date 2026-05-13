"""Locate-stage worker tools: geocode + propose_centers.

Extracted from ``tools/agent.py`` (stage-2 split, 2026-05-11). Registers
``geocode`` and ``propose_centers`` against the shared ``_agent``
instance at import time. The module also exposes the
``_council_postcodes``, ``_is_council_postcode`` and
``_geocode_os_open_names`` helpers — the latter is re-exported from
``tools.agent`` for the overnight reproducibility scripts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from pydantic_ai import ModelRetry, RunContext

from tools.agent_core import _agent, AgentState


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

# Council-postcode blacklist — postcodes that recur across multiple cases
# in a single benchmark output dir are letterhead artefacts, not site
# postcodes. Built lazily once per process from sibling pdf_info.json
# files. Opt-in via GEOMAP_FILTER_COUNCIL_PC=1 (default off).
_COUNCIL_PC_CACHE: dict = {}


def _council_postcodes(out_dir=None):
    """Return postcodes appearing in pdf_info.postcodes for >=2 cases in
    out_dir's parent (the per-model benchmark dir). Memoised by parent path."""
    import glob, collections
    if not out_dir:
        return set()
    parent = os.path.dirname(os.path.abspath(out_dir))
    if parent in _COUNCIL_PC_CACHE:
        return _COUNCIL_PC_CACHE[parent]
    counter = collections.Counter()
    for f in glob.glob(os.path.join(parent, "*", "pdf_info.json")):
        try:
            d = json.load(open(f))
            for pc in (d.get("postcodes") or []):
                counter[str(pc).replace(" ", "").upper()] += 1
        except Exception:
            continue
    blacklist = {pc for pc, n in counter.items() if n >= 2}
    _COUNCIL_PC_CACHE[parent] = blacklist
    return blacklist


def _is_council_postcode(pc, pdf_info, out_dir=None):
    """Detect council letterhead postcodes by frequency + context.

    Three signals (any one trips it):
      1. Frequency: appears in >=2 pdf_info.postcodes lists in the same
         benchmark dir (built lazily from siblings).
      2. Letterhead context: site_address mentions Council Offices/Town Hall.
    """
    pc_norm = str(pc).replace(" ", "").upper()
    if pc_norm in _council_postcodes(out_dir):
        return True
    site = (pdf_info.get("site_address") or "").lower()
    bad_phrases = ("council offices", "town hall", "civic centre",
                   "civic center", "council house", "guildhall")
    if any(p in site for p in bad_phrases):
        return True
    return False


def _geocode_os_open_names(pdf_info, sigma_default, out_dir=None):
    """Offline OS Open Names lookups with bbox-anchor disambiguation.

    Returns a list of dicts {source, lat, lon, sigma_m, specificity} for
    propose_centers to fold into the candidate pool via _add(). Mirrors
    overnight/phaseZO_creative_anchors.find_bbox_anchor + pdf_info_to_centers
    but emits dicts directly. No network calls; uses
    os_opendata/open_names/csv (3M GB places, OGL v3 free).
    """
    try:
        from tools.os_names import (
            search as _os_search, lookup as _os_lookup,
            lookup_postcode as _os_lookup_pc,
        )
        from tools.geo.grid_ref import os_grid_ref_to_latlon
    except Exception:
        return []

    pi = pdf_info or {}
    ctx_str = (pi.get("admin_region") or "").strip() or None
    out = []

    # Bbox anchor priority: grid_ref → likely_town_or_city → place_name → postcode.
    # Grid refs always reference the SITE (not the council); postcodes are the
    # weakest because they're often the council mailing address.
    anchor = None  # (lat, lon, radius_km, src_label)
    for gr in (pi.get("grid_refs") or []):
        coords = os_grid_ref_to_latlon(str(gr))
        if coords:
            digits = sum(1 for c in str(gr).replace(" ", "") if c.isdigit())
            r = 0.5 if digits >= 6 else (1.5 if digits >= 4 else 5.0)
            anchor = (coords[0], coords[1], r, f"grid_ref:{gr}")
            break
    if anchor is None:
        town = (pi.get("likely_town_or_city") or "").strip()
        if town:
            h = _os_lookup(town, context=ctx_str)
            if h: anchor = (h["lat"], h["lon"], 5.0, f"town:{town}")
    if anchor is None:
        for pn in (pi.get("place_names") or [])[:3]:
            pn = (pn or "").strip()
            if len(pn) < 3: continue
            h = _os_lookup(pn, context=ctx_str)
            if h:
                anchor = (h["lat"], h["lon"], 5.0, f"place:{pn}")
                break
    if anchor is None:
        for pc in (pi.get("postcodes") or []):
            if _is_council_postcode(pc, pi, out_dir):
                continue  # skip council-letterhead postcodes
            # Prefer Code-Point Open (sub-metre BNG) when available;
            # fall back to OS Open Names postcode-district lookup.
            h_cp = None
            try:
                from tools.code_point import lookup_postcode as _cp_lookup
                h_cp = _cp_lookup(str(pc))
            except Exception:
                h_cp = None
            if h_cp:
                # Sub-metre precision → very tight prior radius
                anchor = (h_cp["lat"], h_cp["lon"], 0.3, f"code_point:{pc}")
                break
            h = _os_lookup_pc(str(pc))
            if h:
                r = 1.0 if "outward" not in h.get("type", "") else 3.0
                anchor = (h["lat"], h["lon"], r, f"postcode:{pc}")
                break
    if anchor is None:
        return []
    a_lat, a_lon, a_rad_km, a_src = anchor

    # Always emit the bbox anchor as a broad fallback candidate.
    # Validator: postcode-anchored bbox is risky because pdf_info.postcodes
    # often refers to council mailing address, not site. Demote to
    # specificity=4 so it's a tie-breaker fallback, not a primary pick.
    bbox_specificity = 4 if a_src.startswith("postcode:") else 3
    out.append({
        "source": f"os_names:bbox_anchor:{a_src}",
        "lat": float(a_lat), "lon": float(a_lon),
        "sigma_m": float(a_rad_km * 1000), "specificity": bbox_specificity,
    })

    def _emit(query, kind, spec):
        try:
            hits = _os_search(query, max_results=4, context=ctx_str,
                                bbox_center=(a_lat, a_lon),
                                bbox_radius_km=a_rad_km * 2)
        except Exception:
            hits = []
        for h in hits:
            out.append({
                "source": f"os_names:{kind}:{h.get('type','')}:{query[:25]}",
                "lat": float(h["lat"]), "lon": float(h["lon"]),
                "sigma_m": float(h.get("sigma_m") or sigma_default),
                "specificity": spec,
            })

    # ALWAYS emit Code-Point Open postcode candidates FIRST (after bbox anchor)
    # so they survive the cap. Sub-metre BNG precision. Spec=2 because postcode
    # might be council not site (per validator); but tight enough to be useful
    # when site-aligned. Demote council-letterhead PCs to spec=5 so they're
    # last-resort fallbacks rather than primary picks.
    try:
        from tools.code_point import lookup_postcode as _cp_lookup
        for pc in (pi.get("postcodes") or [])[:3]:
            h_cp = _cp_lookup(str(pc))
            if not h_cp: continue
            is_council = _is_council_postcode(pc, pi, out_dir)
            out.append({
                "source": f"os_names:code_point:{pc}{'(council?)' if is_council else ''}",
                "lat": float(h_cp["lat"]), "lon": float(h_cp["lon"]),
                "sigma_m": 5000.0 if is_council else 50.0,
                "specificity": 5 if is_council else 2,
            })
    except Exception:
        pass

    for r in (pi.get("road_names") or [])[:6]:
        if r and len(str(r).strip()) >= 3: _emit(str(r).strip(), "road", 1)
    for p in (pi.get("place_names") or [])[:5]:
        if p and len(str(p).strip()) >= 3: _emit(str(p).strip(), "place", 2)
    for lab in (pi.get("visible_map_labels") or [])[:8]:
        lab = (lab or "").strip()
        if len(lab) >= 4: _emit(lab, "label", 2)

    # Validator: cap at 6 to avoid LLM-confusion from too many candidates.
    # propose_centers already has ~5-15 candidates from existing cascade;
    # adding 6 (not 12) keeps total LLM-visible options reasonable.
    return out[:6]


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
