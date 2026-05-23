"""LIVE LLM-locate agent.

A pydantic_ai Agent that runs at runtime (inside the worker's propose_centers
call) to produce ONE high-quality center (lat, lon, sigma, confidence) using:
  - pdf_info (from the live reader phase — FRESH per case, not cached)
  - the rendered primary match page (state.rendered_pages[map_pages[0]])
  - up to 6 offline geocoder tools (postcode / grid_ref / place / road /
    intersect / la_check)

Called from the worker's propose_centers tool. Pydantic-ai enforces the
LocatePick schema; on agent-loop failure run_locate emits an emergency
LA-centroid LocatePick rather than returning None — the pipeline is
guaranteed at least one candidate.

The agent is built by ``make_locate_agent(disabled_tools)`` — a factory
that returns an Agent with the given tools removed from BOTH the
registered toolset AND the system prompt. Production callers pass
``disabled_tools=frozenset()`` (or omit the kwarg) and get the cached
default agent. The locate LOO ablation passes a non-empty frozenset to
build a variant that genuinely doesn't know the disabled tool exists.

Model: chosen at runtime via ``run_locate(model_name=...)``. The worker
threads its CLI-configured ``locate_model`` through ``AgentState``.
"""
from __future__ import annotations
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.usage import UsageLimits

from tools.agent._model import resolve_model

# Repo root path used by the `road` / `intersect` tools below to locate
# the OML index files at runtime. We no longer mutate sys.path on import
# — the codebase is always launched from the repo root (so cwd is
# already on sys.path) and import-time side effects are surprising.
REPO = Path(__file__).resolve().parent.parent.parent


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
    verified_inside_admin_region: bool = Field(
        default=False,
        description="True if the pick has been verified to fall inside (or "
                    "near) the polygon for the named admin_region "
                    "(pdf_info.admin_region — the Local Authority polygon "
                    "the pick is supposed to belong to). Set False if not "
                    "verified, if verification returned outside-or-far, or "
                    "if no verification tool is available.",
    )


# ── State (deps) ───────────────────────────────────────────────────────────

class LocateState:
    """Per-case state passed to the locate agent's tools as deps."""
    def __init__(self, pdf_info: dict, admin_region_hint: Optional[str] = None):
        self.pdf_info = pdf_info or {}
        self.admin_region_hint = (
            admin_region_hint or pdf_info.get("admin_region") or None
        )


# ── Tool implementations (registered conditionally by make_locate_agent) ──
#
# Each tool is a standalone module-level function — no @agent.tool_plain
# decorator. ``make_locate_agent`` decides which ones to register on a
# given agent instance via the ``_TOOL_IMPLS`` mapping below.


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


def grid_ref(gr: str) -> dict:
    """Parse an OS British National Grid reference (offline).

    Accepts many formats: 'TL 150 067', 'TR3559', '485700 148600', etc.

    Args:
        gr: OS grid reference string.

    Returns:
        {"success": bool, "lat": float, "lon": float} or error.
    """
    try:
        from tools.geo.grid_ref import (
            os_grid_ref_to_latlon, parse_easting_northing,
        )
        # Try the pure-numeric easting/northing format first — the
        # docstring promises support for it (e.g. "485700 148600") and
        # the reader can emit raw E/N strings extracted from "528942 E
        # 184544 N" patterns. ``os_grid_ref_to_latlon`` requires the
        # two-letter prefix so it returns None on those.
        pt = parse_easting_northing(gr) or os_grid_ref_to_latlon(gr)
        if not pt:
            return {"success": False,
                    "error": f"Could not parse grid_ref '{gr}'"}
        return {"success": True, "grid_ref": gr,
                "lat": pt[0], "lon": pt[1]}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


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
            # Use explicit-None fallbacks for lat/lon — the `or` chain
            # treats coordinate 0 as falsy, so lon=0.0 (Greenwich
            # meridian, which crosses real UK places: Royal Observatory,
            # parts of Greenwich/Bexley/Lewisham) would silently fall
            # through to the always-None LATITUDE/LONGITUDE alias and
            # return lon=None to the LLM.
            lat_v = h.get("lat") if "lat" in h else h.get("LATITUDE")
            lon_v = h.get("lon") if "lon" in h else h.get("LONGITUDE")
            out.append({
                "name": h.get("name") or h.get("NAME1"),
                "type": (h.get("local_type") or h.get("LOCAL_TYPE")
                          or h.get("TYPE") or h.get("type")),
                "lat": lat_v,
                "lon": lon_v,
                "admin_district": (h.get("admin_district")
                                    or h.get("DISTRICT_BOROUGH")),
                "county": (h.get("county") or h.get("COUNTY_UNITARY")
                            or h.get("REGION")),
            })
        return {"success": True, "query": query, "la_filter": la,
                "n_hits": len(out), "hits": out}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


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
        if inside:
            d_km = 0.0
        else:
            # Find the nearest point on the LA boundary, then haversine
            # to get true ground distance in km. The previous code used
            # `d_deg * 111` which treats lon-degrees as 111 km — that
            # over-states E-W distances by ~60% at UK lats (real
            # 1°-lon ≈ 68 km at 52°N) and the la_check tool's "is this
            # anchor near the LA" verdict was warped by it. Haversine
            # handles the cos(lat) factor correctly regardless of
            # bearing.
            from shapely.ops import nearest_points
            from tools.geo.coords import haversine_km
            _, q = nearest_points(p, poly.boundary)
            d_km = haversine_km(lat, lon, q.y, q.x)
        centroid = poly.centroid
        return {"success": True, "lat": lat, "lon": lon, "la": la,
                "inside_la": inside,
                "distance_km_approx": round(d_km, 2),
                "la_centroid_lat": centroid.y,
                "la_centroid_lon": centroid.x}
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


# Tool registry. Each entry is (name, impl, advertised_name). The
# advertised name is what the agent calls; we want callers to invoke
# ``postcode(...)`` not ``_tool_postcode(...)``, so we wrap each impl
# in a lambda that pydantic-ai introspects by argument signature.
#
# pydantic-ai derives the tool name from the function's ``__name__``,
# so we attach a clean public name to each wrapper function (set below
# via the public-named alias) before registration.
_TOOL_IMPLS: dict[str, callable] = {
    "postcode":  postcode,
    "grid_ref":  grid_ref,
    "place":     place,
    "road":      road,
    "intersect": intersect,
    "la_check":  la_check,
}

_LOCATE_TOOL_NAMES: frozenset[str] = frozenset(_TOOL_IMPLS.keys())


# ── System-prompt builder ──────────────────────────────────────────────────
#
# The locate sub-agent system prompt is composed of named sections. The
# tool list, the priority-signals list inside step 2, the CLUSTER step's
# example signals, and the LETTERHEAD / VALIDATE steps are all
# tool-dependent and get filtered out when a tool is in ``disabled``.
#
# Building the prompt this way means an LOO ablation's agent literally
# does not know the disabled tool ever existed — no advertised name in
# the bulleted list, no protocol step referencing it, no signal-priority
# bullet that depends on its output. That avoids the confound of "tool
# stubbed but still listed in the prompt → agent calls it anyway and
# gets confused by the error".

_LOCATE_HEADER = (
    "You are the LOCATE STAGE for a UK planning permission boundary extraction pipeline.\n"
    "\n"
    "Your job: given planning-document metadata (pdf_info text fields) AND the rendered "
    "planning map image, produce ONE center coordinate (lat, lon) + an uncertainty "
    "radius σ + confidence, so that downstream MINIMA can refine it visually."
)

_LOCATE_TOOL_DESCS: dict[str, str] = {
    "postcode":  "- postcode(pc) — UK postcode → coord (Code-Point Open, sub-100m)",
    "grid_ref":  "- grid_ref(gr) — OS BNG grid reference → coord",
    "place":     "- place(q, la=None) — OS Open Names search (villages, schools, churches, named buildings)",
    "road":      "- road(q, la=None) — OML road centroid in LA bbox",
    "intersect": "- intersect(road_a, road_b, la=None, road_c=None) — geometric junction of 2-3 roads",
    "la_check":  "- la_check(lat, lon, la) — verify coord falls inside LA polygon",
}

# Step 2 priority list — each line is (gating_tool, text). gating_tool=None
# means the bullet is tool-independent (text-only signal, not a tool call).
# A bullet is included only if its gating_tool is enabled.
_LOCATE_SIGNAL_PRIORITIES: list[tuple[Optional[str], str]] = [
    ("postcode",  "   - Full postcode IN site_address (= SITE postcode, trust)"),
    ("grid_ref",  "   - OS grid_ref (any precision)"),
    (None,        "   - house_number + named road in site_address"),
    ("place",     "   - Named place / landmark from pdf_info OR from the map image"),
    ("road",      "   - Road name (when LA-filtered)"),
    ("place",     "   - Parish name"),
    ("la_check",  "   - LA centroid (last resort)"),
]

_STEP_VIEW_MAP_BODY = (
    "Look for labels, landmarks, distinctive features, road junctions, "
    "named buildings, hatched site polygon, neighbouring features. Note "
    "ANYTHING that's on the map but missing from pdf_info."
)

_STEP_LETTERHEAD_BODY = (
    "for each postcode in pdf_info.postcodes, if it's NOT in site_address, "
    "treat as POSSIBLE letterhead. Run la_check to verify it's inside "
    "admin_region; if it falls outside admin_region, drop unless no other "
    "signal is available."
)

_STEP_BUILD_POOL_BODY = (
    "Aim for 2-4 candidates from different signal types. Augment with terms "
    "FROM THE MAP IMAGE (don't limit yourself to pdf_info)."
)

_STEP_VALIDATE_BODY = (
    "Final pick should be inside the admin_region polygon. Set "
    "verified_inside_admin_region=True if la_check confirms inside; "
    "leave at default False when admin_region is unknown or every "
    "candidate falls outside."
)

_STEP_EMIT_BODY = (
    "Once you have your pick, output the LocatePick directly as your "
    "final response — do NOT make further tool calls. Pydantic-ai parses "
    "your final structured output as the LocatePick schema."
)

_LOCATE_BUDGET = (
    "BUDGET: ≤ 8 geocode tool calls per case. If you've made 8 calls, "
    "commit your best current guess with confidence='low'."
)

_LOCATE_EDGE_CASES = (
    "EDGE CASES:\n"
    "- Empty pdf_info → look hardest at the map image for any labels, then\n"
    "  fall back to LA centroid with wide σ and confidence='low'.\n"
    "- \"District-wide\" cases (whole-borough policy zone) → LA centroid with σ=LA_radius_m.\n"
    "- Multi-parish sites → midpoint of named parishes/villages with wide σ."
)


def _cluster_step_body(enabled: frozenset[str]) -> str:
    """Step 5 body, listing only confident-signal examples whose tools
    are still enabled. Falls back to a generic phrasing when all the
    sub-500m-precision tools are disabled."""
    confident: list[str] = []
    if "postcode" in enabled:
        confident.append("SITE postcode")
    if "grid_ref" in enabled:
        confident.append("grid_ref")
    if "intersect" in enabled:
        confident.append("intersect")
    examples = (", ".join(confident)
                if confident else "any sub-500m-precision tool")
    return (
        "\n"
        "   - 2+ candidates within 500m → tight consensus, σ=200m, confidence='high'\n"
        f"   - Clean single confident signal ({examples}) → σ=300-500m, 'high'\n"
        "   - Single ambiguous (road name, common place) → σ=800-1500m, 'med'\n"
        "   - LA-only fallback → σ from tool, 'low'"
    )


def _build_locate_prompt(disabled: frozenset[str] = frozenset()) -> str:
    """Assemble the locate sub-agent system prompt.

    The prompt tells the agent which tools it has, in what priority,
    when to use each, and how to package the final pick. When some tools
    are disabled (LOO ablation), the prompt is rebuilt to omit any
    mention of them — bulleted descriptions, priority-list entries,
    confidence-signal examples, and the protocol steps that depend
    exclusively on them (e.g. LETTERHEAD CHECK when postcode is gone).

    Step numbers renumber dynamically — the agent sees a coherent
    1..N protocol with no gaps.
    """
    unknown = disabled - _LOCATE_TOOL_NAMES
    if unknown:
        raise ValueError(
            f"Unknown locate tool name(s) in disabled set: {sorted(unknown)}. "
            f"Valid names: {sorted(_LOCATE_TOOL_NAMES)}"
        )

    enabled_set = _LOCATE_TOOL_NAMES - disabled
    # Stable order for the bulleted tool list — same order as the
    # original prompt so unchanged variants produce a byte-identical
    # prompt (modulo the one auto-fixed count word).
    tool_order = ["postcode", "grid_ref", "place", "road", "intersect", "la_check"]
    enabled_tools_ordered = [t for t in tool_order if t in enabled_set]
    n = len(enabled_tools_ordered)

    parts: list[str] = []
    parts.append(_LOCATE_HEADER)
    parts.append("")
    parts.append(
        f"You have {n} offline geocoder tool{'s' if n != 1 else ''}:"
    )
    for t in enabled_tools_ordered:
        parts.append(_LOCATE_TOOL_DESCS[t])
    parts.append("")
    parts.append("PROTOCOL (every case):")
    parts.append("")

    # Build the dynamically-numbered protocol step list.
    # Each entry is (header, body) where body has been pre-shaped for
    # multi-line formatting.
    steps: list[tuple[str, str]] = []
    steps.append(("**VIEW the map image carefully.**", _STEP_VIEW_MAP_BODY))

    # Step: SCAN pdf_info — priority list filtered by enabled tools.
    priority_lines = [
        line for gating_tool, line in _LOCATE_SIGNAL_PRIORITIES
        if gating_tool is None or gating_tool in enabled_set
    ]
    scan_body = (
        "Priority of signals (most specific first):\n"
        + "\n".join(priority_lines)
    )
    steps.append(("**SCAN pdf_info.**", scan_body))

    # LETTERHEAD CHECK requires BOTH postcode and la_check.
    if "postcode" in enabled_set and "la_check" in enabled_set:
        steps.append(("**LETTERHEAD CHECK postcodes:**", _STEP_LETTERHEAD_BODY))

    steps.append(("**BUILD POOL via tool calls.**", _STEP_BUILD_POOL_BODY))
    steps.append(("**CLUSTER & PICK:**", _cluster_step_body(enabled_set)))

    if "la_check" in enabled_set:
        steps.append(("**VALIDATE with la_check.**", _STEP_VALIDATE_BODY))
    # When la_check is disabled, ``verified_inside_admin_region`` simply
    # stays at its schema default (False) — no extra prompt note needed
    # since the field is no longer tool-named. Other LOO variants
    # (no_postcode etc.) are already invisible by the same logic.

    steps.append(("**Emit the LocatePick to terminate.**", _STEP_EMIT_BODY))

    for i, (header, body) in enumerate(steps, start=1):
        parts.append(f"{i}. {header} {body}")
        parts.append("")

    parts.append(_LOCATE_BUDGET)
    parts.append("")
    parts.append(_LOCATE_EDGE_CASES)

    return "\n".join(parts)


# ── Factory ───────────────────────────────────────────────────────────────


def make_locate_agent(disabled_tools=None) -> Agent:
    """Build a locate sub-agent with ``disabled_tools`` removed from BOTH
    the registered toolset AND the system prompt.

    Cached by ``disabled_tools`` key — production calls (disabled=∅) hit
    the cache after the first build; LOO ablation variants each build
    once and stay cached for the duration of the process. ``Agent``
    instances are not pickled across processes, so the cache is
    per-process.

    ``disabled_tools`` is normalised to a ``frozenset`` before the cache
    lookup, so ``make_locate_agent()``, ``make_locate_agent(None)``,
    ``make_locate_agent(set())``, and ``make_locate_agent(frozenset())``
    all share one cache entry.
    """
    return _make_locate_agent_cached(
        frozenset(disabled_tools) if disabled_tools else frozenset()
    )


@lru_cache(maxsize=16)
def _make_locate_agent_cached(disabled_tools: frozenset) -> Agent:
    """Internal cached agent builder. Always called with a normalised
    ``frozenset`` so lru_cache keys collapse predictably."""
    unknown = disabled_tools - _LOCATE_TOOL_NAMES
    if unknown:
        raise ValueError(
            f"Unknown locate tool name(s): {sorted(unknown)}. "
            f"Valid: {sorted(_LOCATE_TOOL_NAMES)}"
        )

    agent = Agent(
        "test",  # placeholder, overridden per-run via model=...
        deps_type=LocateState,
        output_type=LocatePick,
        retries=5,
        output_retries=5,
        model_settings={"temperature": 0},
        instructions=_build_locate_prompt(disabled_tools),
    )

    # Register only the enabled tools. pydantic-ai derives the tool
    # name from the impl's ``__name__`` (already the public name —
    # ``postcode``, ``grid_ref``, etc. at module level) and the
    # argument schema from its annotations.
    for tool_name in [
        "postcode", "grid_ref", "place", "road", "intersect", "la_check"
    ]:
        if tool_name in disabled_tools:
            continue
        agent.tool_plain(_TOOL_IMPLS[tool_name])

    return agent


# Default production singleton — equivalent to the previous module-level
# _locate_agent. Built once on first call (lru_cache), reused thereafter.
_locate_agent = make_locate_agent()


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
        # Convert degree extent to a metres-radius. 1°lat ≈ 111 km but
        # 1°lon ≈ 111·cos(lat) km — at UK lats that's ~68 km, so using
        # 111 unconditionally over-states E-W extent by ~60% and
        # inflates the search σ. Use lat-aware conversion per axis.
        cos_lat = math.cos(math.radians(c.y))
        dx_m = (maxx - minx) * 111_000.0 * cos_lat  # lon extent
        dy_m = (maxy - miny) * 111_000.0            # lat extent
        radius_m = int(max(dx_m, dy_m) / 2)
        sigma = max(2000, min(radius_m, 50_000))
        return LocatePick(
            top_lat=float(c.y), top_lon=float(c.x),
            sigma_m=sigma, confidence="low",
            picked_source=f"emergency_la_centroid:{admin[:30]}",
            evidence=f"LA centroid fallback ({reason[:80]})",
            verified_inside_admin_region=True,
        )
    return LocatePick(
        top_lat=54.0, top_lon=-2.0,
        sigma_m=50_000, confidence="low",
        picked_source="emergency_uk_centroid",
        evidence=f"UK centroid fallback (no admin_region; {reason[:60]})",
        verified_inside_admin_region=False,
    )


def run_locate(
    pdf_info: dict,
    map_img_bytes: Optional[bytes],
    model_name: str,
    match_context: Optional[str] = None,
    prior_messages: Optional[list] = None,
    extra_terms: Optional[List[str]] = None,
    disabled_tools: frozenset[str] = frozenset(),
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
            call on the same case. When set, the agent already has
            pdf_info + map image in its history, so only a new user
            message (with match_context and any extra_terms) is appended.
        extra_terms: additional place / landmark strings the worker
            spotted on the map and wants the locate agent to consider.
            On the first call these are already merged into
            ``pdf_info.place_names`` / ``visible_map_labels`` by the
            caller, so the JSON serialisation surfaces them. On the
            continuation path (prior_messages set), the pdf_info JSON
            is NOT re-sent, so the new terms are spliced into the
            follow-up user message instead.
        disabled_tools: names of locate tools to omit from BOTH the
            agent's registered toolset and its system prompt. Used by
            the locate LOO ablation. Production callers pass
            ``frozenset()`` (or omit the kwarg) and get the same
            behaviour as before the factory refactor.

    Returns:
        (LocatePick, list_of_all_messages). The caller saves
        list_of_all_messages on AgentState so a subsequent re-call can
        pass it back as `prior_messages`.
    """
    model = resolve_model(model_name)
    agent = make_locate_agent(disabled_tools)
    deps = LocateState(pdf_info=pdf_info)

    if prior_messages:
        # Continuation: the agent already has pdf_info + map image in its
        # history. Just send a new user message with the feedback. Any
        # ``extra_terms`` are folded in explicitly here because the
        # follow-up does NOT re-send pdf_info, so a merge into
        # pdf_info.place_names by the caller would be invisible to the
        # model on this turn.
        ctx = (match_context or "").strip()
        new_terms = [t.strip() for t in (extra_terms or [])
                     if isinstance(t, str) and t.strip()]
        extra_block = ""
        if new_terms:
            extra_block = (
                "\n\nADDITIONAL CANDIDATE TERMS (the worker just surfaced "
                "these from the map image; they are NOT in the pdf_info "
                "you saw earlier — treat them as place / landmark anchors "
                "you should try): " + ", ".join(new_terms)
            )
        if ctx or extra_block:
            ctx_block = (f"PRIOR MATCH FEEDBACK:\n{ctx[:1200]}\n\n"
                         if ctx else "")
            user_parts: List[object] = [
                "Re-pick based on prior-match feedback (you already have "
                "pdf_info + map image in this conversation):\n\n"
                f"{ctx_block}"
                "Avoid sources that produced your prior pick; prefer a "
                "different signal type (e.g. switch from postcode to "
                "road/intersection, or from likely_town to a parish/"
                f"landmark).{extra_block}\n\n"
                "Apply the protocol again, then emit your final LocatePick."
            ]
        else:
            user_parts = [
                "Re-pick: the worker re-invoked you. Apply the protocol "
                "again, preferring a DIFFERENT signal type than your last "
                "pick, then emit your final LocatePick."
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
            "likely_town": pdf_info.get("likely_town_or_city"),
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
            "cluster & pick, validate with la_check, then emit your "
            "final LocatePick. Budget: 8 geocode calls max.",
        ]
        if map_img_bytes:
            user_parts.insert(
                0, BinaryContent(data=map_img_bytes, media_type="image/png"))

    admin = pdf_info.get("admin_region") or "?"
    pcs = pdf_info.get("postcodes") or []
    grs = pdf_info.get("grid_refs") or []
    history_tag = (f"prior_msgs={len(prior_messages)}" if prior_messages
                   else "first_call")
    disabled_tag = (f", disabled={sorted(disabled_tools)}"
                    if disabled_tools else "")
    print(f"  [locate] start: admin_region={admin!r}, postcodes={pcs[:2]}, "
          f"grid_refs={grs[:2]}, match_context={'yes' if match_context else 'no'}, "
          f"{history_tag}{disabled_tag}")

    try:
        result = agent.run_sync(
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
          f"conf={pick.confidence} la_ok={pick.verified_inside_admin_region}")
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
