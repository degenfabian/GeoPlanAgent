"""LIVE LLM-locate agent.

A pydantic_ai Agent that runs at runtime (inside the worker's propose_centers
call) to produce ONE high-quality center (lat, lon, sigma, confidence) using:
  - pdf_info (from the live reader phase — FRESH per case, not cached)
  - the rendered planning map image (state.map_img)
  - 6 offline geocoder tools (postcode / grid_ref / place / road / intersect /
    la_check)

Called from the worker's propose_centers tool. Pydantic-ai enforces the
LocatePick schema; on agent-loop failure run_locate emits an emergency
LA-centroid LocatePick rather than returning None — the pipeline is
guaranteed at least one candidate.

Model: defaults to Gemini Flash via OpenRouter (matches worker default).
Override with GEOMAP_LOCATE_MODEL env var.
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.usage import UsageLimits

from tools.agent._model import resolve_model

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


# ── Output schema ──────────────────────────────────────────────────────────

class LocatePick(BaseModel):
    """Final locate output: one center coord + uncertainty + provenance."""
    top_lat: float = Field(description="Final picked latitude (WGS84).")
    top_lon: float = Field(description="Final picked longitude (WGS84).")
    sigma_m: int = Field(
        description="Search radius in meters reflecting uncertainty. "
                    "200 = tight (multi-source agreement). "
                    "300-500 = clean single signal (SITE postcode, grid_ref). "
                    "800-1500 = single ambiguous signal (road, place name). "
                    "2500+ = wide (LA centroid only, or empty pdf_info).",
        ge=100, le=50000,
    )
    confidence: str = Field(
        description="One of: 'high', 'med', 'low'.",
        pattern="^(high|med|low)$",
    )
    picked_source: str = Field(
        description="Short label of the winning signal "
                    "(e.g. 'postcode:AL1 3JE', 'intersect:Manor x Linden', "
                    "'place:Weybourne', 'la_centroid').",
    )
    evidence: str = Field(
        description="1-2 sentence explanation of WHY this pick. "
                    "Mention letterhead/LA-consistency checks done.",
    )
    la_check_passed: bool = Field(
        description="True if la_check confirms the pick is inside (or near) "
                    "the named admin_region; False otherwise.",
    )


# ── State (deps) ───────────────────────────────────────────────────────────

class LocateState:
    """Per-case state passed to the locate agent's tools as deps."""
    def __init__(self, pdf_info: dict, admin_region_hint: Optional[str] = None):
        self.pdf_info = pdf_info or {}
        self.admin_region_hint = (
            admin_region_hint or pdf_info.get("admin_region") or None
        )


# ── The Agent ──────────────────────────────────────────────────────────────

LOCATE_SYSTEM_PROMPT = """You are the LOCATE STAGE for a UK planning permission boundary extraction pipeline.

Your job: given planning-document metadata (pdf_info text fields) AND the rendered planning map image, produce ONE center coordinate (lat, lon) + an uncertainty radius σ + confidence, so that downstream MINIMA can refine it visually.

You have 6 offline geocoder tools:
- postcode(pc) — UK postcode → coord (Code-Point Open, sub-100m)
- grid_ref(gr) — OS BNG grid reference → coord
- place(q, la=None) — OS Open Names search (villages, schools, churches, named buildings)
- road(q, la=None) — OML road centroid in LA bbox
- intersect(road_a, road_b, la=None, road_c=None) — geometric junction of 2-3 roads
- la_check(lat, lon, la) — verify coord falls inside LA polygon

PROTOCOL (every case):

1. **VIEW the map image carefully.** Look for labels, landmarks, distinctive
   features, road junctions, named buildings, hatched site polygon, neighbouring
   features. Note ANYTHING that's on the map but missing from pdf_info_text.

2. **SCAN pdf_info_text.** Priority of signals (most specific first):
   - Full postcode IN site_address (= SITE postcode, trust)
   - OS grid_ref (any precision)
   - house_number + named road in site_address
   - Named place / landmark from pdf_info OR from the map image
   - Road name (when LA-filtered)
   - Parish name
   - LA centroid (last resort)

3. **LETTERHEAD CHECK postcodes:** for each postcode in pdf_info.postcodes,
   if it's NOT in site_address, treat as POSSIBLE letterhead. Run la_check
   to verify it's inside admin_region; if it's >5 km from admin_region, drop.

4. **BUILD POOL via tool calls.** Aim for 2-4 candidates from different signal
   types. Augment with terms FROM THE MAP IMAGE (don't limit yourself to
   pdf_info_text).

5. **CLUSTER & PICK:**
   - 2+ candidates within 500m → tight consensus, σ=200m, confidence='high'
   - Clean single confident signal (SITE postcode, grid_ref, intersect) → σ=300-500m, 'high'
   - Single ambiguous (road name, common place) → σ=800-1500m, 'med'
   - LA-only fallback → σ from tool, 'low'

6. **VALIDATE with la_check.** Final pick must be inside admin_region polygon
   OR within 5 km of its boundary. Set la_check_passed accordingly.

7. **CALL submit_pick to terminate.** Emit your final pick via the
   submit_pick tool. Do NOT call any other tool after submit_pick.

BUDGET: ≤ 8 geocode tool calls per case. If you've made 8 calls, commit your
best current guess with confidence='low'.

EDGE CASES:
- Empty pdf_info_text → look hardest at the map image for any labels, then
  fall back to LA centroid with wide σ and confidence='low'.
- "District-wide" cases (whole-borough policy zone) → LA centroid with σ=LA_radius_m.
- Multi-parish sites → midpoint of named parishes/villages with wide σ.
"""


_locate_agent = Agent(
    "test",  # placeholder, overridden at runtime
    deps_type=LocateState,
    output_type=LocatePick,
    retries=5,
    output_retries=5,
    model_settings={"temperature": 0},
    instructions=LOCATE_SYSTEM_PROMPT,
)


# ── Tools ─────────────────────────────────────────────────────────────────

@_locate_agent.tool_plain
def postcode(pc: str) -> dict:
    """Lookup a UK postcode via Code-Point Open (offline, sub-100m).

    Args:
        pc: UK postcode (e.g. "AL1 3JE").

    Returns:
        {"success": bool, "lat": float, "lon": float, "admin_district": str}
        or {"success": False, "error": str} on not-found.
    """
    try:
        from tools.geo.code_point import lookup_postcode
        h = lookup_postcode(pc)
        if not h:
            return {"success": False,
                    "error": f"Postcode '{pc}' not found in Code-Point Open"}
        return {"success": True, "postcode": pc,
                "lat": h["lat"], "lon": h["lon"],
                "admin_district": h.get("admin_district")}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


@_locate_agent.tool_plain
def grid_ref(gr: str) -> dict:
    """Parse an OS British National Grid reference (offline).

    Accepts many formats: 'TL 150 067', 'TR3559', '485700 148600', etc.

    Args:
        gr: OS grid reference string.

    Returns:
        {"success": bool, "lat": float, "lon": float} or error.
    """
    try:
        from tools.geo.grid_ref import os_grid_ref_to_latlon
        pt = os_grid_ref_to_latlon(gr)
        if not pt:
            return {"success": False,
                    "error": f"Could not parse grid_ref '{gr}'"}
        return {"success": True, "grid_ref": gr,
                "lat": pt[0], "lon": pt[1]}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


@_locate_agent.tool_plain
def place(query: str, la: Optional[str] = None, limit: int = 5) -> dict:
    """Search OS Open Names for places, landmarks, named features.

    Covers: villages, hamlets, suburbs, named roads, churches, schools,
    hospitals, recreation grounds, allotments, named buildings, tourist
    attractions, etc.

    Args:
        query: name to search (case-insensitive).
        la: optional admin district / county to disambiguate.
        limit: max hits to return (default 5).

    Returns:
        {"success": bool, "n_hits": int, "hits": [{"name", "type",
        "lat", "lon", "admin_district", "county"}, ...]}
    """
    try:
        from tools.geo.os_names import search as os_search
        hits = os_search(query, max_results=limit * 3, context=la) or []
        hits = hits[:limit]
        out = []
        for h in hits:
            out.append({
                "name": h.get("name") or h.get("NAME1"),
                "type": (h.get("local_type") or h.get("LOCAL_TYPE")
                          or h.get("TYPE") or h.get("type")),
                "lat": h.get("lat") or h.get("LATITUDE"),
                "lon": h.get("lon") or h.get("LONGITUDE"),
                "admin_district": (h.get("admin_district")
                                    or h.get("DISTRICT_BOROUGH")),
                "county": (h.get("county") or h.get("COUNTY_UNITARY")
                            or h.get("REGION")),
            })
        return {"success": True, "query": query, "la_filter": la,
                "n_hits": len(out), "hits": out}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


@_locate_agent.tool_plain
def road(query: str, la: Optional[str] = None, limit: int = 5) -> dict:
    """Find road instances by name (OML index, LA-bbox-filtered).

    Returns the centroid of each road instance that matches the name AND
    falls inside the named LA's bounding box. Useful when you have a road
    name and want all its instances in a specific local authority.

    Args:
        query: road name.
        la: admin district name to filter to.
        limit: max hits.
    """
    try:
        from pathlib import Path as _Path
        from pyproj import Transformer
        idx_p = REPO / "tools" / "oml_road_index.json"
        if not idx_p.exists():
            return {"success": False, "error": "OML road index missing"}
        idx = json.loads(idx_p.read_text())
        q_key = query.lower().strip()
        instances = idx.get(q_key, []) + idx.get(q_key + " road", [])
        from tools.verification_checks import _resolve_la
        la_poly = None
        if la:
            try:
                la_poly = _resolve_la(la)
            except Exception:
                la_poly = None
        rev = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        from shapely.geometry import Point
        out = []
        for inst in instances:
            try:
                cx = (inst["minx"] + inst["maxx"]) / 2
                cy = (inst["miny"] + inst["maxy"]) / 2
                lon, lat = rev.transform(cx, cy)
            except Exception:
                continue
            if la_poly is not None:
                if not la_poly.contains(Point(lon, lat)):
                    continue
            out.append({"name": inst.get("name"), "lat": lat, "lon": lon,
                        "in_la": la})
            if len(out) >= limit: break
        return {"success": True, "query": query, "la_filter": la,
                "n_hits": len(out), "hits": out}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


@_locate_agent.tool_plain
def intersect(road_a: str, road_b: str, la: Optional[str] = None,
              road_c: Optional[str] = None, limit: int = 10) -> dict:
    """Find geometric intersection point(s) of 2-3 named road LineStrings.

    Uses OML road geometry (offline) to compute where the named roads cross.
    Pinpoints junctions to sub-100m precision. Falls back to error if a
    road isn't in the OML subset.

    Args:
        road_a, road_b: road names to intersect.
        la: admin district name (filters roads to LA bbox).
        road_c: optional third road.
        limit: max intersection points.
    """
    try:
        from pathlib import Path as _Path
        from pyproj import Transformer
        from shapely.geometry import LineString, Point
        from tools.verification_checks import _resolve_la
        geom_p = REPO / "tools" / "oml_road_geom_subset.json"
        if not geom_p.exists():
            return {"success": False, "error": "OML road geom missing"}
        geom = json.loads(geom_p.read_text())
        fwd = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        rev = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        la_bbox_bng = None
        if la:
            try:
                la_poly = _resolve_la(la)
                if la_poly is not None:
                    mn_lon, mn_lat, mx_lon, mx_lat = la_poly.bounds
                    x1, y1 = fwd.transform(mn_lon, mn_lat)
                    x2, y2 = fwd.transform(mx_lon, mx_lat)
                    la_bbox_bng = (min(x1, x2), min(y1, y2),
                                    max(x1, x2), max(y1, y2))
            except Exception: pass

        def get_instances(rd):
            key = rd.lower().strip()
            instances = geom.get(key, []) + geom.get(key + " road", [])
            if la_bbox_bng:
                instances = [h for h in instances
                             if not (h.get("maxx", 0) < la_bbox_bng[0]
                                      or h.get("minx", 0) > la_bbox_bng[2]
                                      or h.get("maxy", 0) < la_bbox_bng[1]
                                      or h.get("miny", 0) > la_bbox_bng[3])]
            return instances

        roads = [road_a, road_b] + ([road_c] if road_c else [])
        road_lines = []
        for rd in roads:
            insts = get_instances(rd)
            lines = []
            for inst in insts:
                pts = inst.get("points") or []
                if len(pts) >= 2:
                    try: lines.append(LineString(pts))
                    except Exception: continue
            road_lines.append((rd, lines))
        missing = [rd for rd, lines in road_lines if not lines]
        if missing:
            return {"success": False,
                    "error": f"No road geometry in {la or 'UK'} for: {missing}"}
        intersections = []
        seen = set()
        for i in range(len(road_lines)):
            for j in range(i + 1, len(road_lines)):
                rd_a, lines_a = road_lines[i]
                rd_b, lines_b = road_lines[j]
                for line_a in lines_a:
                    for line_b in lines_b:
                        try: inter = line_a.intersection(line_b)
                        except Exception: continue
                        if inter.is_empty: continue
                        pts = []
                        if inter.geom_type == "Point":
                            pts.append((inter.x, inter.y))
                        elif inter.geom_type == "MultiPoint":
                            pts.extend([(p.x, p.y) for p in inter.geoms])
                        elif inter.geom_type in ("LineString", "MultiLineString"):
                            c = inter.centroid
                            pts.append((c.x, c.y))
                        for x, y in pts:
                            key = (round(x, 1), round(y, 1))
                            if key in seen: continue
                            seen.add(key)
                            lon, lat = rev.transform(x, y)
                            intersections.append({
                                "lat": round(lat, 6), "lon": round(lon, 6),
                                "roads": [rd_a, rd_b],
                            })
        return {"success": True, "roads": roads, "la_filter": la,
                "n_intersections": len(intersections),
                "intersections": intersections[:limit]}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


@_locate_agent.tool_plain
def la_check(lat: float, lon: float, la: str) -> dict:
    """Verify a coord falls inside a Local Authority polygon.

    Args:
        lat, lon: coord to check.
        la: admin district / borough / unitary authority name.

    Returns:
        {"success": bool, "inside_la": bool, "distance_km_approx": float,
        "la_centroid_lat": float, "la_centroid_lon": float}
    """
    try:
        from tools.verification_checks import _resolve_la
        from shapely.geometry import Point
        poly = _resolve_la(la)
        if poly is None:
            return {"success": False,
                    "error": f"No polygon for LA '{la}'"}
        p = Point(lon, lat)
        inside = poly.contains(p)
        if inside: d_km = 0.0
        else:
            d_deg = p.distance(poly.boundary)
            d_km = d_deg * 111.0
        centroid = poly.centroid
        return {"success": True, "lat": lat, "lon": lon, "la": la,
                "inside_la": inside,
                "distance_km_approx": round(d_km, 2),
                "la_centroid_lat": centroid.y,
                "la_centroid_lon": centroid.x}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


# ── Entry point ───────────────────────────────────────────────────────────

def _emergency_la_centroid_pick(pdf_info: dict, reason: str) -> LocatePick:
    """Emergency fallback: build a LocatePick at the LA centroid.

    Used when the agent loop fails entirely (validation retries exhausted,
    HTTP error, etc.). Guarantees run_locate never returns None — the
    pipeline always has at least one candidate to feed to MINIMA.
    """
    admin = (pdf_info.get("admin_region")
             or pdf_info.get("likely_town_or_city")
             or pdf_info.get("district_name") or "").strip()
    try:
        from tools.verification_checks import _resolve_la
        poly = _resolve_la(admin) if admin else None
    except Exception:
        poly = None
    if poly is not None:
        c = poly.centroid
        minx, miny, maxx, maxy = poly.bounds
        radius_m = int(max(maxx - minx, maxy - miny) * 111_000 / 2)
        sigma = max(2000, min(radius_m, 50_000))
        return LocatePick(
            top_lat=float(c.y), top_lon=float(c.x),
            sigma_m=sigma, confidence="low",
            picked_source=f"emergency_la_centroid:{admin[:30]}",
            evidence=f"LA centroid fallback ({reason[:80]})",
            la_check_passed=True,
        )
    return LocatePick(
        top_lat=54.0, top_lon=-2.0,
        sigma_m=50_000, confidence="low",
        picked_source="emergency_uk_centroid",
        evidence=f"UK centroid fallback (no admin_region; {reason[:60]})",
        la_check_passed=False,
    )


def run_locate(
    pdf_info: dict,
    map_img_bytes: Optional[bytes],
    model_name: str,
    match_context: Optional[str] = None,
    prior_messages: Optional[list] = None,
) -> tuple:
    """Run the live LLM-locate agent for one case.

    Pydantic-ai enforces the LocatePick schema; if the agent loop fails
    entirely (validation retries exhausted, HTTP error, budget exceeded),
    we emit an emergency LA-centroid LocatePick rather than returning None.
    The pipeline ALWAYS gets a valid pick.

    When called a second time on the same case (after a poor match_at),
    pass the prior call's message history via `prior_messages` so the
    locate agent SEES its previous reasoning + tool calls + pick, and
    `match_context` so it knows what went wrong. The agent then refines
    rather than re-deriving from scratch.

    Args:
        pdf_info: live reader output (pdf_info dict).
        map_img_bytes: PNG bytes of the rendered planning map page.
        model_name: OpenRouter model identifier (or alias).
        match_context: feedback from a prior pick. The worker passes
            this in when re-calling propose_centers.
        prior_messages: result.all_messages() from the previous run_locate
            call on the same case. When set, only the new user message
            (with match_context) is appended.

    Returns:
        (LocatePick, list_of_all_messages). The caller saves
        list_of_all_messages on AgentState so a subsequent re-call can
        pass it back as `prior_messages`.
    """
    model = resolve_model(model_name)
    deps = LocateState(pdf_info=pdf_info)

    if prior_messages:
        # Continuation: the agent already has pdf_info + map image in its
        # history. Just send a new user message with the feedback.
        ctx = (match_context or "").strip()
        if ctx:
            user_parts: List[object] = [
                "Re-pick based on prior-match feedback (you already have "
                "pdf_info + map image in this conversation):\n\n"
                f"PRIOR MATCH FEEDBACK:\n{ctx[:1200]}\n\n"
                "Avoid sources that produced your prior pick; prefer a "
                "different signal type (e.g. switch from postcode to "
                "road/intersection, or from likely_town to a parish/"
                "landmark). Apply the protocol again and call submit_pick."
            ]
        else:
            user_parts = [
                "Re-pick: the worker re-invoked you. Apply the protocol "
                "again, preferring a DIFFERENT signal type than your last "
                "pick, and call submit_pick."
            ]
    else:
        # First call: full pdf_info JSON + (optional) match_context.
        pi_summary = {
            "site_address": pdf_info.get("site_address"),
            "postcodes": pdf_info.get("postcodes") or [],
            "grid_refs": pdf_info.get("grid_refs") or [],
            "road_names": pdf_info.get("road_names") or [],
            "place_names": (pdf_info.get("place_names") or [])[:8],
            "admin_region": pdf_info.get("admin_region"),
            "likely_town": (pdf_info.get("likely_town")
                             or pdf_info.get("likely_town_or_city")),
            "parish_names": (pdf_info.get("parish_names") or [])[:5],
            "adjacency_hints": (pdf_info.get("adjacency_hints") or [])[:5],
            "house_number_road_pairs": (
                pdf_info.get("house_number_road_pairs") or [])[:3],
            "visible_map_labels": (pdf_info.get("visible_map_labels") or [])[:15],
            "is_district_wide": pdf_info.get("is_district_wide", False),
        }
        ctx_block = ""
        if match_context and match_context.strip():
            ctx_block = (
                "\n\nPRIOR MATCH FEEDBACK (the worker tried a previous pick "
                "and reported back — use this to choose a DIFFERENT pick):\n"
                f"{match_context.strip()[:1200]}\n"
                "Avoid sources that produced the prior pick; prefer a "
                "different signal type."
            )
        user_parts = [
            f"PDF_INFO:\n{json.dumps(pi_summary, indent=2)}{ctx_block}\n\n"
            "Apply the protocol: view the map, scan pdf_info, "
            "letterhead-check postcodes, build pool via tool calls, "
            "cluster & pick, validate with la_check, then call submit_pick. "
            "Budget: 8 geocode calls max.",
        ]
        if map_img_bytes:
            user_parts.insert(
                0, BinaryContent(data=map_img_bytes, media_type="image/png"))

    admin = pdf_info.get("admin_region") or "?"
    pcs = pdf_info.get("postcodes") or []
    grs = pdf_info.get("grid_refs") or []
    history_tag = (f"prior_msgs={len(prior_messages)}" if prior_messages
                   else "first_call")
    print(f"  [locate] start: admin_region={admin!r}, postcodes={pcs[:2]}, "
          f"grid_refs={grs[:2]}, match_context={'yes' if match_context else 'no'}, "
          f"{history_tag}")

    try:
        result = _locate_agent.run_sync(
            user_parts,
            deps=deps,
            model=model,
            usage_limits=UsageLimits(request_limit=15),
            message_history=prior_messages,
        )
    except Exception as e:
        print(f"  [locate] FAILED: {e!s:.200}")
        pick = _emergency_la_centroid_pick(
            pdf_info, reason=f"agent failed: {e!s:.60}")
        return pick, (prior_messages or [])

    _print_locate_trajectory(result)
    pick = result.output
    print(f"  [locate] picked: {pick.picked_source[:50]} → "
          f"({pick.top_lat:.5f}, {pick.top_lon:.5f}) σ={pick.sigma_m}m "
          f"conf={pick.confidence} la_ok={pick.la_check_passed}")
    print(f"  [locate] evidence: {pick.evidence[:200]}")

    try:
        all_msgs = list(result.all_messages())
    except Exception:
        all_msgs = prior_messages or []
    return pick, all_msgs


def _print_locate_trajectory(result) -> None:
    """Print each tool call + summarised result from a pydantic-ai run."""
    try:
        msgs = result.all_messages()
    except Exception:
        return
    for msg in msgs:
        parts = getattr(msg, "parts", None)
        if not parts:
            continue
        for part in parts:
            kind = (getattr(part, "kind", type(part).__name__) or "").lower()
            if "toolcall" in kind:
                name = getattr(part, "tool_name", "?")
                args = getattr(part, "args", None)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                args_str = _fmt_args(args) if isinstance(args, dict) else str(args)[:100]
                print(f"    [locate→tool] {name}({args_str})")
            elif "toolreturn" in kind:
                content = getattr(part, "content", None)
                summary = _fmt_tool_return(content)
                if summary:
                    print(f"    [locate←ret ] {summary}")
            elif "retry" in kind:
                rc = getattr(part, "content", "") or ""
                print(f"    [locate retry] {str(rc)[:160]}")


def _fmt_args(args: dict) -> str:
    if not isinstance(args, dict):
        return str(args)[:100]
    pieces = []
    for k, v in args.items():
        if isinstance(v, (list, tuple)):
            v_str = f"[{', '.join(str(x)[:20] for x in v[:3])}{'...' if len(v) > 3 else ''}]"
        elif isinstance(v, str):
            v_str = f"{v[:40]!r}"
        elif isinstance(v, float):
            v_str = f"{v:.5f}"
        else:
            v_str = str(v)
        pieces.append(f"{k}={v_str}")
    return ", ".join(pieces)


def _fmt_tool_return(content) -> str:
    if isinstance(content, dict):
        if not content.get("success", True):
            return f"error: {str(content.get('error',''))[:80]}"
        # Highlight high-value fields per tool
        out = []
        for k in ("postcode", "grid_ref", "query", "roads", "la"):
            if k in content and content[k] is not None:
                out.append(f"{k}={str(content[k])[:50]}")
        if "lat" in content and "lon" in content:
            out.append(f"lat={content['lat']:.5f}, lon={content['lon']:.5f}")
        if "n_hits" in content:
            out.append(f"n_hits={content['n_hits']}")
        if "n_intersections" in content:
            out.append(f"n_intersections={content['n_intersections']}")
        if "inside_la" in content:
            out.append(f"inside_la={content['inside_la']} "
                       f"d={content.get('distance_km_approx', '?')}km")
        return "  ".join(out) if out else str(content)[:100]
    if isinstance(content, str):
        return content[:120]
    return ""
