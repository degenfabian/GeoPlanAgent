"""Live locate sub-agent: pdf_info + map page to one (lat, lon, sigma) LocatePick."""

from __future__ import annotations
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext
from pydantic_ai.usage import UsageLimits

from geoplanagent.utils import haversine_km, resolve_model
from geoplanagent.prompts import LOCATE_PROMPT_PRODUCTION, LOCATE_PROMPT_ALL_TOOLS

REPO = Path(__file__).resolve().parent.parent.parent

# Sub-agent temperature: defaults to 0 (production); GEOMAP_TEMPERATURE env
# var overrides for the appendix temperature ablation.
_TEMPERATURE = float(os.environ.get("GEOMAP_TEMPERATURE", "0"))


# Output schema


class LocatePick(BaseModel):
    """Final locate output: one center coord + uncertainty + provenance."""

    top_lat: float = Field(
        description="Final picked latitude (WGS84). UK range: 49.5 to 61.0.",
        ge=49.5,
        le=61.0,
    )
    top_lon: float = Field(
        description="Final picked longitude (WGS84). UK range: -9.0 to 2.0.",
        ge=-9.0,
        le=2.0,
    )
    sigma_m: int = Field(
        description="Search radius in meters reflecting uncertainty. "
        "200 = tight (multi-source agreement). "
        "300-500 = clean single signal (SITE postcode, grid_ref). "
        "800-1500 = single ambiguous signal (road, place name). "
        "2500+ = wide (LA centroid only, or empty pdf_info).",
        ge=100,
        le=50000,
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


# ── Tool implementations (the all-tools agent registers all six; the
# production agent registers only `place`) ──


def postcode(pc: str) -> dict:
    """Lookup a UK postcode via Code-Point Open (offline, sub-100m).

    Args:
        pc: UK postcode (e.g. "AL1 3JE").

    Returns:
        {"success": bool, "lat": float, "lon": float, "admin_district": str}
        or {"success": False, "error": str} on not-found.
    """
    try:
        from geoplanagent.tools.geocode import lookup_postcode

        hit = lookup_postcode(pc)
        if not hit:
            return {"success": False, "error": f"Postcode '{pc}' not found in Code-Point Open"}
        return {
            "success": True,
            "postcode": pc,
            "lat": hit["lat"],
            "lon": hit["lon"],
            "admin_district": hit.get("admin_district"),
        }
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
        from geoplanagent.tools.geocode import (
            os_grid_ref_to_latlon,
            parse_easting_northing,
        )

        # Try the pure-numeric easting/northing format first — the
        # docstring promises support for it (e.g. "485700 148600") and
        # the reader can emit raw E/N strings extracted from "528942 E
        # 184544 N" patterns. ``os_grid_ref_to_latlon`` requires the
        # two-letter prefix so it returns None on those.
        point = parse_easting_northing(gr) or os_grid_ref_to_latlon(gr)
        if not point:
            return {"success": False, "error": f"Could not parse grid_ref '{gr}'"}
        return {"success": True, "grid_ref": gr, "lat": point[0], "lon": point[1]}
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
        from geoplanagent.tools.geocode import search as os_search

        hits = os_search(query, max_results=limit * 3, context=la) or []
        hits = hits[:limit]
        results = []
        for hit in hits:
            # Use explicit-None fallbacks for lat/lon — the `or` chain
            # treats coordinate 0 as falsy, so lon=0.0 (Greenwich
            # meridian, which crosses real UK places: Royal Observatory,
            # parts of Greenwich/Bexley/Lewisham) would silently fall
            # through to the always-None LATITUDE/LONGITUDE alias and
            # return lon=None to the LLM.
            latitude = hit.get("lat") if "lat" in hit else hit.get("LATITUDE")
            longitude = hit.get("lon") if "lon" in hit else hit.get("LONGITUDE")
            results.append(
                {
                    "name": hit.get("name") or hit.get("NAME1"),
                    "type": (
                        hit.get("local_type")
                        or hit.get("LOCAL_TYPE")
                        or hit.get("TYPE")
                        or hit.get("type")
                    ),
                    "lat": latitude,
                    "lon": longitude,
                    "admin_district": (hit.get("admin_district") or hit.get("DISTRICT_BOROUGH")),
                    "county": (hit.get("county") or hit.get("COUNTY_UNITARY") or hit.get("REGION")),
                }
            )
        return {
            "success": True,
            "query": query,
            "la_filter": la,
            "n_hits": len(results),
            "hits": results,
        }
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

        index_path = REPO / "geoplanagent" / "oml_road_index.json"
        if not index_path.exists():
            return {"success": False, "error": "OML road index missing"}
        index = json.loads(index_path.read_text())
        name_key = query.lower().strip()
        instances = index.get(name_key, []) + index.get(name_key + " road", [])
        from geoplanagent.tools.geocode import resolve_la

        la_poly = None
        if la:
            try:
                la_poly = resolve_la(la)
            except Exception:
                la_poly = None
        bng_to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        from shapely.geometry import Point

        results = []
        for instance in instances:
            try:
                easting_centre = (instance["minx"] + instance["maxx"]) / 2
                northing_centre = (instance["miny"] + instance["maxy"]) / 2
                lon, lat = bng_to_wgs84.transform(easting_centre, northing_centre)
            except Exception:
                continue
            if la_poly is not None:
                if not la_poly.contains(Point(lon, lat)):
                    continue
            results.append({"name": instance.get("name"), "lat": lat, "lon": lon, "in_la": la})
            if len(results) >= limit:
                break
        return {
            "success": True,
            "query": query,
            "la_filter": la,
            "n_hits": len(results),
            "hits": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


def intersect(
    road_a: str,
    road_b: str,
    la: Optional[str] = None,
    road_c: Optional[str] = None,
    limit: int = 10,
) -> dict:
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
        from shapely.geometry import LineString
        from geoplanagent.tools.geocode import resolve_la

        geom_path = REPO / "geoplanagent" / "oml_road_geom_subset.json"
        if not geom_path.exists():
            return {"success": False, "error": "OML road geom missing"}
        road_geometry = json.loads(geom_path.read_text())
        wgs84_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        bng_to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        la_bbox_bng = None
        if la:
            try:
                la_poly = resolve_la(la)
                if la_poly is not None:
                    min_lon, min_lat, max_lon, max_lat = la_poly.bounds
                    corner1_x, corner1_y = wgs84_to_bng.transform(min_lon, min_lat)
                    corner2_x, corner2_y = wgs84_to_bng.transform(max_lon, max_lat)
                    la_bbox_bng = (
                        min(corner1_x, corner2_x),
                        min(corner1_y, corner2_y),
                        max(corner1_x, corner2_x),
                        max(corner1_y, corner2_y),
                    )
            except Exception:
                pass

        def instances_for_road(road_name):
            name_key = road_name.lower().strip()
            instances = road_geometry.get(name_key, []) + road_geometry.get(name_key + " road", [])
            if la_bbox_bng:
                instances = [
                    instance
                    for instance in instances
                    if not (
                        instance.get("maxx", 0) < la_bbox_bng[0]
                        or instance.get("minx", 0) > la_bbox_bng[2]
                        or instance.get("maxy", 0) < la_bbox_bng[1]
                        or instance.get("miny", 0) > la_bbox_bng[3]
                    )
                ]
            return instances

        roads = [road_a, road_b] + ([road_c] if road_c else [])
        road_lines = []
        for road_name in roads:
            lines = []
            for instance in instances_for_road(road_name):
                points = instance.get("points") or []
                if len(points) >= 2:
                    try:
                        lines.append(LineString(points))
                    except Exception:
                        continue
            road_lines.append((road_name, lines))
        missing = [road_name for road_name, lines in road_lines if not lines]
        if missing:
            return {"success": False, "error": f"No road geometry in {la or 'UK'} for: {missing}"}
        intersections = []
        seen = set()
        for i in range(len(road_lines)):
            for j in range(i + 1, len(road_lines)):
                name_a, lines_a = road_lines[i]
                name_b, lines_b = road_lines[j]
                for line_a in lines_a:
                    for line_b in lines_b:
                        try:
                            crossing = line_a.intersection(line_b)
                        except Exception:
                            continue
                        if crossing.is_empty:
                            continue
                        points = []
                        if crossing.geom_type == "Point":
                            points.append((crossing.x, crossing.y))
                        elif crossing.geom_type == "MultiPoint":
                            points.extend([(p.x, p.y) for p in crossing.geoms])
                        elif crossing.geom_type in ("LineString", "MultiLineString"):
                            centroid = crossing.centroid
                            points.append((centroid.x, centroid.y))
                        for easting, northing in points:
                            key = (round(easting, 1), round(northing, 1))
                            if key in seen:
                                continue
                            seen.add(key)
                            lon, lat = bng_to_wgs84.transform(easting, northing)
                            intersections.append(
                                {
                                    "lat": round(lat, 6),
                                    "lon": round(lon, 6),
                                    "roads": [name_a, name_b],
                                }
                            )
        return {
            "success": True,
            "roads": roads,
            "la_filter": la,
            "n_intersections": len(intersections),
            "intersections": intersections[:limit],
        }
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
        from geoplanagent.tools.geocode import resolve_la
        from shapely.geometry import Point

        poly = resolve_la(la)
        if poly is None:
            return {"success": False, "error": f"No polygon for LA '{la}'"}
        point = Point(lon, lat)
        inside = poly.contains(point)
        if inside:
            distance_km = 0.0
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

            _, nearest = nearest_points(point, poly.boundary)
            distance_km = haversine_km(lat, lon, nearest.y, nearest.x)
        centroid = poly.centroid
        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "la": la,
            "inside_la": inside,
            "distance_km_approx": round(distance_km, 2),
            "la_centroid_lat": centroid.y,
            "la_centroid_lon": centroid.x,
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:160]}


# Tool registry: advertised name -> implementation. pydantic-ai derives
# the tool name from ``__name__``.
_TOOL_IMPLS: dict[str, callable] = {
    "postcode": postcode,
    "grid_ref": grid_ref,
    "place": place,
    "road": road,
    "intersect": intersect,
    "la_check": la_check,
}


# Factory


def _build_locate_agent(instructions: str, tool_impls: dict) -> Agent:
    """Construct a locate sub-agent with a fixed prompt + tool set.

    The agent model is a placeholder overridden per-run via ``model=``
    in run_locate.
    """
    agent = Agent(
        "test",  # placeholder, overridden per-run via model=...
        output_type=LocatePick,
        retries=5,
        output_retries=5,
        model_settings={"temperature": _TEMPERATURE},
        instructions=instructions,
    )

    for impl in tool_impls.values():
        agent.tool_plain(impl)

    # L2 output validator: cross-check the agent's final LocatePick
    # against the MIN distance to every coord-returning tool call in the
    # trajectory. Catches sign-flip / lat-lon-swap / number-corruption
    # bugs where the agent's reasoning was correct but the JSON output
    # got mangled — e.g. final_result(top_lat=51.51, top_lon=0.33) when
    # every tool returned coords near (51.51, -0.33).
    #
    # Trigger: pick is > L2_THRESHOLD_KM from EVERY tool return. We use
    # MIN-distance-to-ANY (not last-la_check) because the agent may
    # la_check a candidate it later rejects, picking a different coord
    # from a place/road/intersect return. As long as the pick is close
    # to SOMETHING the agent computed, it's a valid pick.
    #
    # Threshold 5 km empirically separates sign-flips (always >20 km
    # from every tool return — flipping a UK lon doubles the distance)
    # from "agent picked a different candidate after la_check" cases
    # (always <5 km from at least one tool return). Safe to register
    # on all configs — no-ops if there are no coord-returning tool
    # calls in the trajectory.
    L2_THRESHOLD_KM = 5.0

    @agent.output_validator
    async def _validate_pick_against_tool_returns(
        ctx: RunContext[None], pick: LocatePick
    ) -> LocatePick:
        distances = []
        for msg in ctx.messages:
            parts = getattr(msg, "parts", None) or []
            for part in parts:
                kind = (getattr(part, "kind", "") or "").lower()
                if "toolreturn" not in kind:
                    continue
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        continue
                if not isinstance(content, dict):
                    continue
                # Single-coord returns (postcode, grid_ref, la_check)
                if "lat" in content and "lon" in content:
                    try:
                        distances.append(
                            haversine_km(
                                pick.top_lat,
                                pick.top_lon,
                                float(content["lat"]),
                                float(content["lon"]),
                            )
                        )
                    except (ValueError, TypeError):
                        pass
                # Multi-hit returns (place, road)
                for hit in content.get("hits") or []:
                    if not isinstance(hit, dict):
                        continue
                    try:
                        distances.append(
                            haversine_km(
                                pick.top_lat, pick.top_lon, float(hit["lat"]), float(hit["lon"])
                            )
                        )
                    except (KeyError, ValueError, TypeError):
                        pass
                # intersect returns
                for hit in content.get("intersections") or []:
                    if not isinstance(hit, dict):
                        continue
                    try:
                        distances.append(
                            haversine_km(
                                pick.top_lat, pick.top_lon, float(hit["lat"]), float(hit["lon"])
                            )
                        )
                    except (KeyError, ValueError, TypeError):
                        pass

        if not distances:
            return pick  # nothing to validate against; accept as-is

        min_distance_km = min(distances)
        if min_distance_km > L2_THRESHOLD_KM:
            raise ModelRetry(
                f"Your final pick ({pick.top_lat:.4f}, {pick.top_lon:.4f}) "
                f"is {min_distance_km:.1f} km from the nearest coord any of your "
                f"tools returned. This usually indicates a sign-flip or "
                f"lat/lon swap on output. Re-emit using a coord from one "
                f"of your tool calls (check the sign of the longitude "
                f"carefully — UK lon is typically negative)."
            )
        return pick

    return agent


@lru_cache(maxsize=1)
def _locate_agent_production() -> Agent:
    """Production locate agent: the single ``place`` geocoder."""
    return _build_locate_agent(LOCATE_PROMPT_PRODUCTION, {"place": place})


@lru_cache(maxsize=1)
def _locate_agent_all_tools() -> Agent:
    """All-six-geocoders locate agent — used only by the locate ablation."""
    return _build_locate_agent(LOCATE_PROMPT_ALL_TOOLS, _TOOL_IMPLS)


# Helpers for transient-error handling


# When the rendered PNG exceeds this threshold, re-encode the same
# image (FULL RESOLUTION) as JPEG-90 to shrink bytes for the HTTP send.
# JPEG-90 takes a 30 MB Dover scan to ~4-6 MB without resizing — the
# agent still sees the original 9354×3306 px, just slightly lossier.
#
# Why JPEG re-encode beats PNG-downscale here:
#   - Preserves resolution → text labels stay legible
#   - Only fires on 3 cases (1.4%) — the truly-oversized A2/A1 scans
#   - JPEG-90 quality is fine for label-reading (already validated in
#     the segmentation ablation)
#
# Other cases stay PNG. We don't switch to JPEG globally because some
# sparse maps (e.g. Ar4.5: 2 MB PNG, 6.3 MB JPEG-90) GROW under JPEG —
# JPEG handles white-space less efficiently than PNG's RLE.
_MAX_IMAGE_BYTES = 25_000_000
_JPEG_FALLBACK_QUALITY = 90


def _shrink_image_if_oversized(img_bytes: bytes) -> bytes:
    """If bytes > 25 MB, re-encode as JPEG-90 (preserves resolution).

    Returns unchanged otherwise. Pixel-area HTTP 400s are left to the
    retry path — they're transient per JPEG segmentation experience.

    The returned bytes may be JPEG or PNG — callers should detect via
    magic bytes (use ``_image_media_type``) and set BinaryContent's
    media_type accordingly.
    """
    if not img_bytes or len(img_bytes) <= _MAX_IMAGE_BYTES:
        return img_bytes
    try:
        import cv2
        import numpy as np

        buffer = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return img_bytes
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_FALLBACK_QUALITY])
        jpeg_bytes = encoded.tobytes()
        if len(jpeg_bytes) <= _MAX_IMAGE_BYTES:
            return jpeg_bytes
        # JPEG-90 still too big — try a lower quality before giving up.
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes()
    except Exception:
        # On any decode/encode failure, return original — let the
        # downstream HTTP error surface naturally.
        return img_bytes


def _image_media_type(img_bytes: bytes) -> str:
    """Detect PNG vs JPEG from the magic bytes."""
    if len(img_bytes) >= 3 and img_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(img_bytes) >= 4 and img_bytes[:4] == b"\x89PNG":
        return "image/png"
    return "image/png"  # default — pydantic-ai expects something


from geoplanagent.utils import is_transient_http_error as _is_transient_error  # noqa: E402


# Entry point


def _emergency_la_centroid_pick(pdf_info: dict, reason: str) -> LocatePick:
    """Fallback LocatePick at the LA centroid when the agent loop fails."""
    admin = (
        pdf_info.get("admin_region")
        or pdf_info.get("likely_town_or_city")
        or pdf_info.get("district_name")
        or ""
    ).strip()
    try:
        from geoplanagent.tools.geocode import resolve_la

        poly = resolve_la(admin) if admin else None
    except Exception:
        poly = None
    if poly is not None:
        centroid = poly.centroid
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        # Lat-aware degree → metres so E-W extent isn't inflated.
        latitude_cosine = math.cos(math.radians(centroid.y))
        width_m = (max_lon - min_lon) * 111_000.0 * latitude_cosine
        height_m = (max_lat - min_lat) * 111_000.0
        radius_m = int(max(width_m, height_m) / 2)
        sigma = max(2000, min(radius_m, 50_000))
        return LocatePick(
            top_lat=float(centroid.y),
            top_lon=float(centroid.x),
            sigma_m=sigma,
            confidence="low",
            picked_source=f"emergency_la_centroid:{admin[:30]}",
            evidence=f"LA centroid fallback ({reason[:80]})",
        )
    return LocatePick(
        top_lat=54.0,
        top_lon=-2.0,
        sigma_m=50_000,
        confidence="low",
        picked_source="emergency_uk_centroid",
        evidence=f"UK centroid fallback (no admin_region; {reason[:60]})",
    )


def run_locate(
    pdf_info: dict,
    map_img_bytes: Optional[bytes],
    model_name: str,
    match_context: Optional[str] = None,
    prior_messages: Optional[list] = None,
    extra_terms: Optional[List[str]] = None,
    all_tools: bool = False,
    usage_sink: Optional[list] = None,
) -> tuple:
    """Run the locate sub-agent for one case; returns (LocatePick, all_messages).

    On a re-pick, pass the prior call's `all_messages` as `prior_messages` and
    feedback in `match_context`; the agent refines instead of starting over.
    `all_tools=True` selects the six-geocoder ablation agent; the default is
    the production agent (the single `place` geocoder).

    `usage_sink`: optional list. If supplied, one dict per invocation is
    appended:
       {request_tokens, response_tokens, generation_id}
    The locate ablation harness doesn't pass it; the worker tool
    (``geoplanagent.tools.positioning.propose_centers``) does, so per-case cost
    telemetry survives in ``state.locate_calls``.
    """
    model = resolve_model(model_name)
    agent = _locate_agent_all_tools() if all_tools else _locate_agent_production()

    if prior_messages:
        # Continuation: pdf_info already in history; just append feedback.
        # extra_terms are spliced here since pdf_info isn't re-sent.
        context = (match_context or "").strip()
        candidate_terms = [
            t.strip() for t in (extra_terms or []) if isinstance(t, str) and t.strip()
        ]
        extra_terms_block = ""
        if candidate_terms:
            extra_terms_block = (
                "\n\nADDITIONAL CANDIDATE TERMS (the worker just surfaced "
                "these from the map image; they are NOT in the pdf_info "
                "you saw earlier — treat them as place / landmark anchors "
                "you should try): " + ", ".join(candidate_terms)
            )
        if context or extra_terms_block:
            feedback_block = f"PRIOR MATCH FEEDBACK:\n{context[:1200]}\n\n" if context else ""
            user_parts: List[object] = [
                "Re-pick based on prior-match feedback (you already have "
                "pdf_info + map image in this conversation):\n\n"
                f"{feedback_block}"
                "Avoid sources that produced your prior pick; prefer a "
                "different signal type (e.g. switch from postcode to "
                "road/intersection, or from likely_town to a parish/"
                f"landmark).{extra_terms_block}\n\n"
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
        pdf_info_summary = {
            "site_address": pdf_info.get("site_address"),
            "postcodes": pdf_info.get("postcodes") or [],
            "grid_refs": pdf_info.get("grid_refs") or [],
            "road_names": pdf_info.get("road_names") or [],
            "place_names": (pdf_info.get("place_names") or [])[:8],
            "admin_region": pdf_info.get("admin_region"),
            "likely_town": pdf_info.get("likely_town_or_city"),
            "parish_names": (pdf_info.get("parish_names") or [])[:5],
            "adjacency_hints": (pdf_info.get("adjacency_hints") or [])[:5],
            "house_number_road_pairs": (pdf_info.get("house_number_road_pairs") or [])[:3],
            "visible_map_labels": (pdf_info.get("visible_map_labels") or [])[:15],
            "is_district_wide": pdf_info.get("is_district_wide", False),
        }
        feedback_block = ""
        if match_context and match_context.strip():
            feedback_block = (
                "\n\nPRIOR MATCH FEEDBACK (the worker tried a previous pick "
                "and reported back — use this to choose a DIFFERENT pick):\n"
                f"{match_context.strip()[:1200]}\n"
                "Avoid sources that produced the prior pick; prefer a "
                "different signal type."
            )
        user_parts = [
            f"PDF_INFO:\n{json.dumps(pdf_info_summary, indent=2)}{feedback_block}\n\n"
            "Apply the protocol: view the map, scan pdf_info, "
            "letterhead-check postcodes, build pool via tool calls, "
            "cluster & pick, validate with la_check, then emit your "
            "final LocatePick. Budget: 8 geocode calls max.",
        ]
        if map_img_bytes:
            user_parts.insert(0, BinaryContent(data=map_img_bytes, media_type="image/png"))

    # Pre-shrink oversized map images so we don't hit HTTP 413 on the
    # first attempt. Catches A3/A2/A0 planning maps rendered at 200 DPI.
    # The shrink may re-encode as JPEG (preserves resolution) — we
    # detect the resulting format and set BinaryContent's media_type
    # accordingly.
    if map_img_bytes is not None:
        original_size = len(map_img_bytes)
        map_img_bytes = _shrink_image_if_oversized(map_img_bytes)
        if len(map_img_bytes) != original_size:
            new_media_type = _image_media_type(map_img_bytes)
            for index, part in enumerate(user_parts):
                if isinstance(part, BinaryContent):
                    user_parts[index] = BinaryContent(data=map_img_bytes, media_type=new_media_type)
                    break

    admin = pdf_info.get("admin_region") or "?"
    postcodes = pdf_info.get("postcodes") or []
    grid_refs = pdf_info.get("grid_refs") or []
    history_tag = f"prior_msgs={len(prior_messages)}" if prior_messages else "first_call"
    tools_tag = ", all_tools" if all_tools else ""
    img_tag = f", img={len(map_img_bytes) // 1024}KB" if map_img_bytes is not None else ""
    print(
        f"  [locate] start: admin_region={admin!r}, postcodes={postcodes[:2]}, "
        f"grid_refs={grid_refs[:2]}, match_context={'yes' if match_context else 'no'}, "
        f"{history_tag}{tools_tag}{img_tag}"
    )

    # Run the agent with up to one retry on transient HTTP errors. The
    # default OpenRouter exception → caught and falls back to emergency,
    # which is too pessimistic for 4xx/5xx errors that succeed on retry.
    MAX_RETRIES = 1
    result = None
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = agent.run_sync(
                user_parts,
                model=model,
                usage_limits=UsageLimits(request_limit=15),
                message_history=prior_messages,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES and _is_transient_error(e):
                wait_seconds = 2**attempt
                print(
                    f"  [locate] transient error (attempt {attempt + 1}/"
                    f"{MAX_RETRIES + 1}): {e!s:.140} — retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
                continue
            break

    if result is None:
        error = last_error if last_error is not None else RuntimeError("unknown locate failure")
        print(f"  [locate] FAILED: {error!s:.200}")
        pick = _emergency_la_centroid_pick(pdf_info, reason=f"agent failed: {error!s:.60}")
        # Record a zero-token entry so the audit script can still count
        # the invocation attempt (and so n_calls is accurate).
        if usage_sink is not None:
            usage_sink.append(
                {
                    "request_tokens": 0,
                    "response_tokens": 0,
                    "generation_id": None,
                    "error": f"{type(error).__name__}: {error!s:.120}",
                }
            )
        return pick, (prior_messages or [])

    _print_locate_trajectory(result)
    pick = result.output
    print(
        f"  [locate] picked: {pick.picked_source[:50]} → "
        f"({pick.top_lat:.5f}, {pick.top_lon:.5f}) σ={pick.sigma_m}m "
        f"conf={pick.confidence}"
    )
    print(f"  [locate] evidence: {pick.evidence[:200]}")

    try:
        all_messages = list(result.all_messages())
    except Exception:
        all_messages = prior_messages or []

    # Telemetry capture (no-op unless caller passed a sink).
    if usage_sink is not None:
        try:
            usage = result.usage()
            request_tokens = getattr(usage, "request_tokens", None) or 0
            response_tokens = getattr(usage, "response_tokens", None) or 0
        except Exception:
            request_tokens, response_tokens = 0, 0
        usage_sink.append(
            {
                "request_tokens": int(request_tokens),
                "response_tokens": int(response_tokens),
                "generation_id": _extract_generation_id(result),
            }
        )

    return pick, all_messages


def _extract_generation_id(result) -> Optional[str]:
    """Best-effort dig the OpenRouter generation id out of a pydantic-ai result.

    pydantic-ai's surfaced attribute name has shifted across versions
    (``vendor_id`` → ``provider_response_id`` → stored inside
    ``vendor_details``). Try the known shapes and return whichever lands
    first; return None if none do — the audit script tolerates missing ids
    and falls back to token-rate cost estimation for those calls.
    """
    try:
        messages = list(result.all_messages())
    except Exception:
        return None
    for msg in reversed(messages):
        for attr in ("vendor_id", "provider_response_id", "model_response_id", "response_id"):
            value = getattr(msg, attr, None)
            if isinstance(value, str) and value:
                return value
        vendor_details = getattr(msg, "vendor_details", None)
        if isinstance(vendor_details, dict):
            for key in ("id", "generation_id", "openrouter_id"):
                value = vendor_details.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _print_locate_trajectory(result) -> None:
    """Print each tool call + summarised result from a pydantic-ai run."""
    try:
        messages = result.all_messages()
    except Exception:
        return
    for msg in messages:
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
                retry_content = getattr(part, "content", "") or ""
                print(f"    [locate retry] {str(retry_content)[:160]}")


def _fmt_args(args: dict) -> str:
    pieces = []
    for key, value in args.items():
        if isinstance(value, (list, tuple)):
            value_str = (
                f"[{', '.join(str(x)[:20] for x in value[:3])}{'...' if len(value) > 3 else ''}]"
            )
        elif isinstance(value, str):
            value_str = f"{value[:40]!r}"
        elif isinstance(value, float):
            value_str = f"{value:.5f}"
        else:
            value_str = str(value)
        pieces.append(f"{key}={value_str}")
    return ", ".join(pieces)


def _fmt_tool_return(content) -> str:
    if isinstance(content, dict):
        if not content.get("success", True):
            return f"error: {str(content.get('error', ''))[:80]}"
        # Highlight high-value fields per tool
        fields = []
        for key in ("postcode", "grid_ref", "query", "roads", "la"):
            if key in content and content[key] is not None:
                fields.append(f"{key}={str(content[key])[:50]}")
        if "lat" in content and "lon" in content:
            fields.append(f"lat={content['lat']:.5f}, lon={content['lon']:.5f}")
        if "n_hits" in content:
            fields.append(f"n_hits={content['n_hits']}")
        if "n_intersections" in content:
            fields.append(f"n_intersections={content['n_intersections']}")
        if "inside_la" in content:
            fields.append(
                f"inside_la={content['inside_la']} d={content.get('distance_km_approx', '?')}km"
            )
        return "  ".join(fields) if fields else str(content)[:100]
    if isinstance(content, str):
        return content[:120]
    return ""
