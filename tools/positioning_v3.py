"""Locate v3 — offline-by-design, smaller cascade, sharper σ.

Design goals (vs production tools.locate.propose_centers_v2):
1. NO network calls. Every external API is replaced with an offline OS
   Open Data source (Code-Point Open, OS Open Names, OS Open Zoomstack,
   OS Boundary-Line) plus the existing on-disk Nominatim cache for
   street-level resolution.
2. Smaller code surface. ~650 LOC instead of ~2700.
3. Tighter, source-calibrated σ. Each source emits σ matched to its
   intrinsic precision rather than a one-size-fits-all 2500–5000m floor.
4. Honest candidate ranking. Specificity & evidence drive ordering, with
   a sanity rule that demotes candidates >5 km from any other candidate
   when at least 2 candidates exist (catches "letterhead-only" failures).

Public entry points:
  propose_centers_v3(pi: dict, *, max_candidates=6, seed_only=False)
      → list[CandidateV3]
  pick_one_v3(cands: list[CandidateV3]) → Optional[CandidateV3]

`seed_only=True` returns ONLY the highest-confidence source (typically
Code-Point Open postcode + grid_ref + parsed-grid-ref) — useful for the
agent_v3 simulator that wants a tiny set.

`pick_one_v3` returns the single best candidate via agreement-based
scoring:
  1. Prefer grid_ref if present (tightest BNG)
  2. Prefer multi_road_consensus if present
  3. Prefer postcode that has a SITE-SPECIFIC corroborator within 2 km
     (site-specific = os_road, multi_road_consensus, grid_ref,
     nominatim_cache)
  4. Otherwise: best co-located non-letterhead candidate

Iterations:
  v3.0 → cascade + LA centroid fallback
  v3.1 → road-name prefix-stripping, town-distance filter, gpkg tighter
  v3.2 → LA-relax pass for road hits (catches cross-LA sites)
  v3.3 → offline multi_road_consensus (≥3 candidates cluster within 1.5km)
  v3.4 → pick_one_v3 single-best with agreement scoring
  v3.5 → rule 3 requires SITE-SPECIFIC corroborator (kills letterhead PCs)
  v3.6 → Nominatim cache integration (street-level offline geocodes,
         3,500+ free hits without API)
  v3.8 → rule 4 drops postcodes when no site-specific corroborator exists
         (catches letterhead PCs hijacking rule 4 via co_locate)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 1. Public dataclass ────────────────────────────────────────────────────

@dataclass
class CandidateV3:
    """One ranked candidate from the v3 cascade.

    sigma_m is calibrated to the source's intrinsic precision:
      postcode (sub-metre Code-Point Open) → 80 m
      grid_ref 8-fig                       → 200 m
      grid_ref 6-fig                       → 600 m
      os_road                              → 600 m
      os_landmark / os_place               → 1200 m
      gpkg_zoomstack                       → 1500 m
      la_centroid                          → la_radius_m
    """
    lat: float
    lon: float
    sigma_m: float
    source: str
    evidence: str
    confidence: str        # "high" | "med" | "low"
    specificity: int       # lower = more precise (1 = postcode, 5 = LA)
    raw_source_type: str   # postcode | grid_ref | os_road | os_place | gpkg | la

    def to_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "sigma_m": self.sigma_m,
                "source": self.source, "evidence": self.evidence,
                "confidence": self.confidence, "specificity": self.specificity,
                "raw_source_type": self.raw_source_type}


# ── 2. Tiny utilities ─────────────────────────────────────────────────────

_FULL_PC_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$", re.IGNORECASE)


def _is_full_postcode(pc: str) -> bool:
    if not pc or not isinstance(pc, str):
        return False
    s = pc.strip().upper().replace("  ", " ")
    return bool(_FULL_PC_RE.match(s.replace(" ", "")) or _FULL_PC_RE.match(s))


def _hkm(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    return float(2 * R * math.asin(math.sqrt(a)))


def _norm_road(name: str) -> str:
    if not name: return ""
    s = re.sub(r"\s+", " ", name.strip().lower())
    for suf in (" road", " street", " lane", " avenue", " way",
                " close", " crescent", " drive", " place"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


# ── 3. LA polygon + region anchor ──────────────────────────────────────────

def _la_polygon_for(pi: Dict[str, Any]):
    """Lookup the LA polygon from OS BoundaryLine for the case's admin_region
    or district_name. Returns shapely Polygon/MultiPolygon or None.
    """
    try:
        from tools.locate._core import _la_polygon_for as _f
        return _f(pi)
    except Exception:
        return None


def _la_radius_m(poly) -> Optional[int]:
    if poly is None: return None
    try:
        from tools.locate._core import _la_radius_m as _f
        return _f(poly)
    except Exception:
        return None


def _town_centroid(pi: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        from tools.locate._core import town_centroid as _f
        return _f(pi)
    except Exception:
        return None


# ── 4. Source emitters ─────────────────────────────────────────────────────

def _town_lookup_relaxed(pi):
    """Town centroid bypassing LA filter — used for letterhead detection only.

    `_town_centroid` filters by LA polygon, so when the planning case is in
    LA X but the named town is in LA Y (border / different-LA cases), the
    function falls back to LA centroid. For letterhead detection we want
    the ACTUAL named town's location even if it's outside the LA.
    """
    try:
        from tools.geocoding.os_names import search
    except Exception:
        return None
    likely = (pi.get("likely_town_or_city") or "").strip()
    if not likely: return None
    try:
        hits = search(likely, max_results=10) or []
    except Exception:
        hits = []
    for h in hits:
        if h.get("lat") is None: continue
        t = (h.get("type") or "").lower()
        if any(k in t for k in ("city", "town", "village", "hamlet", "suburb")):
            return (h["lat"], h["lon"])
    return None


def _emit_postcode(pi, la_poly) -> List[CandidateV3]:
    """Code-Point Open postcode lookup. Sub-metre BNG → σ=80m.

    v3.9 letterhead detection: if the postcode is >5km from the
    likely_town_or_city centroid (when known), demote it heavily — it's
    almost certainly a council mailing-address postcode, not the site.

    v3.11 fix: use _town_lookup_relaxed (bypasses LA filter) so cases like
    SSA410 (admin=South Staffordshire, town=Stafford in DIFFERENT LA)
    correctly detect that the postcode is far from the named town.
    """
    out = []
    try:
        from tools.geocoding.code_point import lookup_postcode
    except Exception:
        return out
    pcs = pi.get("postcodes") or []
    seen = set()
    # Town centroid for letterhead detection (LA-bypass to handle cross-LA cases)
    town = _town_lookup_relaxed(pi) or _town_centroid(pi)
    for pc in pcs[:5]:
        if not _is_full_postcode(pc):
            continue
        h = lookup_postcode(pc)
        if h is None: continue
        if (h["lat"], h["lon"]) in seen: continue
        seen.add((h["lat"], h["lon"]))
        in_la = True
        if la_poly is not None:
            try:
                from shapely.geometry import Point
                in_la = la_poly.contains(Point(h["lon"], h["lat"]))
            except Exception:
                in_la = True
        # Letterhead detection: postcode > 5km from likely_town_or_city
        is_letterhead = False
        if town is not None:
            d_to_town_km = _hkm(h["lat"], h["lon"], town[0], town[1])
            if d_to_town_km > 5.0:
                is_letterhead = True
        # v3.16: DROP letterheads entirely instead of demoting. In multi-cand
        # path they were still being picked by MINIMA when other candidates
        # produced few inliers. Better to fall through to LA centroid + place
        # candidates than have MINIMA lock onto a council mailing address.
        if is_letterhead:
            continue   # don't emit at all
        out.append(CandidateV3(
            lat=h["lat"], lon=h["lon"],
            sigma_m=(80.0 if in_la else 300.0),
            source=f"postcode:{pc}",
            evidence=(f"Code-Point Open {pc}"
                       + ("" if in_la else " (outside LA)")),
            confidence=("high" if in_la else "med"),
            specificity=1,
            raw_source_type="postcode",
        ))
    return out


def _emit_grid_ref(pi) -> List[CandidateV3]:
    """OS grid references parsed from pdf_info."""
    out = []
    try:
        from tools.geo.grid_ref import os_grid_ref_to_latlon
    except Exception:
        return out
    seen = set()
    for g in (pi.get("grid_refs") or [])[:3]:
        pt = os_grid_ref_to_latlon(str(g))
        if pt is None: continue
        if (pt[0], pt[1]) in seen: continue
        seen.add((pt[0], pt[1]))
        digits = sum(1 for c in str(g).replace(" ", "") if c.isdigit())
        sig = 200 if digits >= 8 else (600 if digits >= 6 else 2000)
        out.append(CandidateV3(
            lat=pt[0], lon=pt[1], sigma_m=float(sig),
            source=f"grid_ref:{g}",
            evidence=f"OS BNG {g} ({digits}-digit)",
            confidence="high",
            specificity=1,
            raw_source_type="grid_ref",
        ))
    return out


def _road_variants(name: str) -> List[str]:
    """Return canonical + prefix-stripped variants. OS Open Names indexes
    'Denmark Street' but not 'Lower Denmark Street' / 'New Denmark Street' —
    stripping common modifiers recovers many missing road hits.
    """
    if not name: return []
    nm = name.strip()
    variants = [nm]
    PREFIXES = ("Lower ", "Upper ", "New ", "Old ", "Little ", "Great ",
                 "North ", "South ", "East ", "West ",
                 "lower ", "upper ", "new ", "old ", "little ", "great ",
                 "north ", "south ", "east ", "west ")
    for p in PREFIXES:
        if nm.startswith(p):
            variants.append(nm[len(p):])
            break
    return variants


def _emit_os_names(pi, la_poly, town) -> List[CandidateV3]:
    """OS Open Names lookups for parish/place names and road names.

    Also folds in visible_map_labels — these are often the most precise
    site-specific names (minor settlements, building names) that don't
    appear in the higher-level parish/place_names lists.

    v3.1: adds prefix-strip road variants ("Lower X Street" → also try "X Street")
    and applies a town-distance filter (≤15 km from likely_town_or_city)
    on top of the LA polygon filter, to keep large districts honest.
    """
    out = []
    try:
        from tools.geocoding.os_names import search as os_search
        from shapely.geometry import Point
    except Exception:
        return out
    # Parishes + place names + visible_map_labels — broader coverage of
    # landmark names.
    seen_places = set()
    queries = ((pi.get("parish_names") or []) +
                (pi.get("place_names") or []) +
                (pi.get("visible_map_labels") or []))[:12]
    # Dedup queries case-insensitively while preserving order.
    seen_q = set(); deduped_q = []
    for q in queries:
        ql = (q or "").strip().lower()
        if not ql or ql in seen_q: continue
        seen_q.add(ql); deduped_q.append(q)
    # v3.12: also relax LA filter for places — same pattern as roads.
    # First pass: strict LA filter. Second pass: town-distance only.
    # Catches cross-LA cases like SSA410 ("Stafford" town in Stafford LA
    # but case admin_region is South Staffordshire LA).
    # v3.13: use the *relaxed* town lookup (bypasses LA filter) so the
    # town-distance filter is anchored at the actual named town, not the
    # LA centroid fallback.
    town_relaxed = _town_lookup_relaxed(pi) or town
    place_town_filter_km = 15.0
    for nm in deduped_q:
        nm = (nm or "").strip()
        if len(nm) < 3: continue
        try: hits = os_search(nm, max_results=10) or []
        except Exception: hits = []
        if not hits: continue
        matched = False
        passes = ("strict", "relaxed") if la_poly is not None else ("relaxed",)
        for pass_kind in passes:
            if matched: break
            for h in hits:
                if h.get("lat") is None: continue
                t = (h.get("type") or "").lower()
                if "water" in t or "coastal" in t: continue
                in_region = True
                if pass_kind == "strict" and la_poly is not None:
                    try: in_region = la_poly.contains(Point(h["lon"], h["lat"]))
                    except Exception: in_region = True
                else:
                    # Relaxed pass: use town-distance only (anchored on the
                    # relaxed town lookup, not LA centroid).
                    if town_relaxed is not None:
                        in_region = (_hkm(h["lat"], h["lon"],
                                           town_relaxed[0], town_relaxed[1])
                                      < place_town_filter_km)
                    else:
                        in_region = True
                if not in_region: continue
                key = (round(h["lat"], 4), round(h["lon"], 4))
                if key in seen_places: continue
                seen_places.add(key)
                kind = ("village" if "village" in t
                        else ("hamlet" if "hamlet" in t else "place"))
                out.append(CandidateV3(
                    lat=h["lat"], lon=h["lon"], sigma_m=1200.0,
                    source=f"os_place:{nm[:40]}",
                    evidence=f"OS Open Names {kind} {h.get('name_full','')[:50]}",
                    confidence="med",
                    specificity=2,
                    raw_source_type="os_place",
                ))
                matched = True
                break  # one hit per query
    # Roads — pull from road_names. Try prefix-stripped variants because
    # OS Open Names indexes "Denmark Street" but not "Lower Denmark Street".
    # Region filter: prefer the LA polygon, but RELAX if no LA hits — a road
    # near the LA border may genuinely fall in the neighbouring LA. Always
    # cap by town-distance (10km) when likely_town_or_city is set.
    seen_roads = set()
    town_filter_km = 10.0
    for rd in (pi.get("road_names") or [])[:6]:
        rd = (rd or "").strip()
        if len(rd) < 3: continue
        matched = False
        # First pass: strict LA filter
        # Second pass: relaxed (town-only) if strict found nothing for this road
        passes = ("strict", "relaxed") if la_poly is not None else ("relaxed",)
        for pass_kind in passes:
            if matched: break
            for variant in _road_variants(rd):
                if matched: break
                try: hits = os_search(variant, max_results=10) or []
                except Exception: hits = []
                for h in hits:
                    if h.get("lat") is None: continue
                    t = (h.get("type") or "").lower()
                    if "road" not in t: continue
                    # Region filter
                    in_region = True
                    if pass_kind == "strict" and la_poly is not None:
                        try: in_region = la_poly.contains(Point(h["lon"], h["lat"]))
                        except Exception: in_region = True
                    else:
                        # Relaxed: drop the LA polygon, keep town-distance
                        in_region = True
                    if not in_region: continue
                    # Town-distance filter (always when town is set)
                    if town is not None:
                        d_to_town = _hkm(h["lat"], h["lon"], town[0], town[1])
                        if d_to_town > town_filter_km: continue
                    key = (round(h["lat"], 4), round(h["lon"], 4))
                    if key in seen_roads: continue
                    seen_roads.add(key)
                    out.append(CandidateV3(
                        lat=h["lat"], lon=h["lon"], sigma_m=1000.0,
                        source=f"os_road:{rd[:40]}",
                        evidence=f"OS Open Names road {h.get('name_full','')[:50]}",
                        confidence="med",
                        specificity=1,
                        raw_source_type="os_road",
                    ))
                    matched = True
                    break
    return out


def _emit_gpkg(pi, la_poly, town) -> List[CandidateV3]:
    """OS Open Zoomstack `names` table lookup (offline). gpkg is the noisiest
    source (common place names like 'Riverside' have many UK matches), so
    we apply the strictest distance filter: must be within max_parent_distance_km
    of the LA centroid or town centroid.
    """
    out = []
    try:
        from tools.geocoding.dispatchers import gpkg_place_search
        from shapely.geometry import Point
    except Exception:
        return out
    parent_lat = la_poly.centroid.y if la_poly is not None else (town[0] if town else None)
    parent_lon = la_poly.centroid.x if la_poly is not None else (town[1] if town else None)
    GOOD = ["Settlement", "Small Settlements", "Hamlet", "Village", "Town"]
    seen = set()
    for nm in ((pi.get("place_names") or []) +
                (pi.get("visible_map_labels") or []))[:8]:
        nm = (nm or "").strip()
        if len(nm) < 3: continue
        try:
            # v3.1: tighter cap (15km, was 40km) and limit=2 (was 3)
            hits = gpkg_place_search(nm,
                                     parent_lat=parent_lat, parent_lon=parent_lon,
                                     max_parent_distance_km=15,
                                     type_filter=GOOD, limit=2)
        except Exception:
            hits = []
        for h in hits:
            # Both LA-polygon AND town-distance filter for gpkg (noisiest src)
            if la_poly is not None:
                try:
                    if not la_poly.contains(Point(h["lon"], h["lat"])):
                        continue
                except Exception:
                    continue
            if town is not None:
                if _hkm(h["lat"], h["lon"], town[0], town[1]) > 15:
                    continue
            key = (round(h["lat"], 4), round(h["lon"], 4))
            if key in seen: continue
            seen.add(key)
            out.append(CandidateV3(
                lat=h["lat"], lon=h["lon"], sigma_m=1500.0,
                source=f"gpkg:{nm[:40]}",
                evidence=f"OS Zoomstack {h.get('type','')}: {h.get('name','')}",
                confidence="med",
                specificity=2,
                raw_source_type="gpkg",
            ))
            break
    return out


def _emit_multi_road_consensus(road_cands: List[CandidateV3],
                                 place_cands: List[CandidateV3]) -> List[CandidateV3]:
    """If ≥3 road / place candidates cluster within ~1.5 km of each other,
    emit their centroid as a high-confidence consensus anchor.

    This is the offline analogue of production's `multi_road_consensus`
    — without the OSM road graph it can't do true intersection, but if
    three named roads on the planning map all geocode within a small
    area then their centroid IS a strong site-level signal.
    """
    pts = []
    for c in road_cands + place_cands:
        if c.raw_source_type in ("os_road", "os_place", "oml_road",
                                  "nominatim_cache", "gpkg"):
            pts.append((c.lat, c.lon, c.source))
    if len(pts) < 3: return []
    # Greedy cluster: for each candidate seed, count how many others are
    # within 1.5 km. Pick the densest seed; emit its cluster centroid.
    best_seed = None
    best_count = 0
    best_cluster = []
    for i, (la_i, lo_i, _) in enumerate(pts):
        cluster = [(la_i, lo_i)]
        for j, (la_j, lo_j, _) in enumerate(pts):
            if i == j: continue
            if _hkm(la_i, lo_i, la_j, lo_j) < 1.5:
                cluster.append((la_j, lo_j))
        if len(cluster) > best_count:
            best_count = len(cluster)
            best_seed = i
            best_cluster = cluster
    if best_count < 3: return []
    lat_c = sum(p[0] for p in best_cluster) / len(best_cluster)
    lon_c = sum(p[1] for p in best_cluster) / len(best_cluster)
    return [CandidateV3(
        lat=lat_c, lon=lon_c, sigma_m=500.0,
        source=f"multi_road_consensus:{best_count}",
        evidence=f"Centroid of {best_count} co-clustered OS Open Names hits",
        confidence="high",
        specificity=1,
        raw_source_type="multi_road_consensus",
    )]


_OML_INDEX = None
_BNG_TO_WGS = None


def _load_oml_index():
    global _OML_INDEX
    if _OML_INDEX is not None: return _OML_INDEX
    try:
        import json as _json
        p = Path(__file__).resolve().parent / "oml_road_index.json"
        if not p.exists():
            _OML_INDEX = {}
            return _OML_INDEX
        _OML_INDEX = _json.loads(p.read_text())
    except Exception:
        _OML_INDEX = {}
    return _OML_INDEX


def _bng_to_wgs84(easting, northing):
    """OSGB36 BNG → WGS84 lat/lon."""
    global _BNG_TO_WGS
    if _BNG_TO_WGS is None:
        try:
            from pyproj import Transformer
            _BNG_TO_WGS = Transformer.from_crs("EPSG:27700", "EPSG:4326",
                                                 always_xy=True)
        except Exception:
            return None
    try:
        lon, lat = _BNG_TO_WGS.transform(easting, northing)
        return (lat, lon)
    except Exception:
        return None


def _emit_oml_roads(pi, la_poly, town) -> List[CandidateV3]:
    """v3.16 — look up road names in the OS Open Map Local index.

    Solves the homonym problem: when "Harpenden Road" exists in 5 UK
    locations, we keep ONLY those whose bbox intersects the LA polygon.
    Real disambiguation, fully offline.

    Output: one candidate per (road_name, in-LA road instance).
    """
    out = []
    index = _load_oml_index()
    if not index: return out
    road_names = pi.get("road_names") or []
    if not road_names: return out
    # Build LA bbox in BNG for fast filter
    la_bng_bbox = None   # (minx, miny, maxx, maxy)
    if la_poly is not None:
        try:
            from pyproj import Transformer
            t_inv = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            # LA polygon is in WGS84; transform bounds
            minlon, minlat, maxlon, maxlat = la_poly.bounds
            x1, y1 = t_inv.transform(minlon, minlat)
            x2, y2 = t_inv.transform(maxlon, maxlat)
            la_bng_bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        except Exception:
            la_bng_bbox = None
    # Town centroid in BNG for fallback distance check
    town_bng = None
    if la_bng_bbox is None and town is not None:
        try:
            from pyproj import Transformer
            t_inv = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            tx, ty = t_inv.transform(town[1], town[0])
            town_bng = (tx, ty)
        except Exception:
            town_bng = None
    seen = set()
    for rd in road_names[:8]:
        rd = (rd or "").strip()
        if len(rd) < 3: continue
        # Try variants
        for variant in _road_variants(rd):
            hits = index.get(variant.lower(), [])
            if not hits:
                # case-insensitive substring fallback for spelling variants
                for k in index:
                    if variant.lower() == k:
                        hits = index[k]; break
            if not hits: continue
            # Filter by LA bbox
            in_la_hits = []
            for h in hits:
                in_box = True
                if la_bng_bbox is not None:
                    in_box = (la_bng_bbox[0] <= h["cx"] <= la_bng_bbox[2]
                              and la_bng_bbox[1] <= h["cy"] <= la_bng_bbox[3])
                elif town_bng is not None:
                    # Within 15km of town centroid
                    dx = h["cx"] - town_bng[0]; dy = h["cy"] - town_bng[1]
                    in_box = (dx*dx + dy*dy) < (15000*15000)
                if in_box:
                    in_la_hits.append(h)
            # v3.17 fix: take only the LONGEST in-LA road instance per name
            # (instead of [:3]). This avoids crowding out place/la_centroid
            # candidates and prefers the major road (not a tiny side-street
            # homonym). Among multiple short hits, the longest is most likely
            # the principal road for that name.
            if not in_la_hits:
                continue
            # Sort by extent (longest first)
            in_la_hits.sort(key=lambda h: ((h["maxx"]-h["minx"])**2
                                            + (h["maxy"]-h["miny"])**2),
                            reverse=True)
            for h in in_la_hits[:1]:
                ll = _bng_to_wgs84(h["cx"], h["cy"])
                if ll is None: continue
                key = (round(ll[0], 4), round(ll[1], 4))
                if key in seen: continue
                seen.add(key)
                # σ from road extent — diagonal / 2, capped to keep search tight
                extent_m = ((h["maxx"] - h["minx"]) ** 2
                            + (h["maxy"] - h["miny"]) ** 2) ** 0.5 / 2.0
                sig = max(300.0, min(extent_m, 2000.0))
                out.append(CandidateV3(
                    lat=ll[0], lon=ll[1], sigma_m=sig,
                    source=f"oml_road:{rd[:40]}",
                    evidence=f"OS Open Map Local road '{h['name']}' "
                              f"[{h['cls']}] in tile {h['tile']}",
                    confidence="high",
                    specificity=1,
                    raw_source_type="oml_road",
                ))
            if out:   # got at least one match for this road
                break
    return out


def _emit_nominatim_cached(pi, la_poly, town) -> List[CandidateV3]:
    """Read the on-disk Nominatim cache (cache/nominatim_structured.json) for
    street-level hits. NOT an API call — just a disk lookup of results
    cached from prior runs. Adds ~3,500 UK street geocodes to v3 without
    any network.

    Mirrors production's section 5b9 (nominatim_road_with_city_context) but
    only fires on cache hits.
    """
    out = []
    try:
        import json as _json
        from pathlib import Path as _P
        from shapely.geometry import Point
        cache_p = _P(__file__).resolve().parent.parent / "cache" / "nominatim_structured.json"
        if not cache_p.exists():
            return out
        cache = _json.loads(cache_p.read_text())
    except Exception:
        return out
    if not isinstance(cache, dict) or not cache:
        return out
    likely_town = (pi.get("likely_town_or_city") or "").strip()
    admin = (pi.get("admin_region") or "").strip()
    seen = set()
    # For each road_name, try a few cache-key variants
    for rd in (pi.get("road_names") or [])[:6]:
        rd = (rd or "").strip()
        if len(rd) < 3: continue
        for ctx in (likely_town, admin, ""):
            ctx_n = ctx.lower().strip()
            key = f"{rd.lower()}|{ctx_n}||uk"
            hit = cache.get(key)
            if hit and hit.get("lat") and hit.get("lon"):
                key3 = (round(hit["lat"], 4), round(hit["lon"], 4))
                if key3 in seen: continue
                # Check LA polygon
                in_la = True
                if la_poly is not None:
                    try:
                        in_la = la_poly.contains(Point(hit["lon"], hit["lat"]))
                    except Exception:
                        in_la = True
                if not in_la:
                    # Try town distance fallback
                    if town is not None:
                        if _hkm(hit["lat"], hit["lon"], town[0], town[1]) > 15:
                            continue
                    else:
                        continue
                seen.add(key3)
                out.append(CandidateV3(
                    lat=hit["lat"], lon=hit["lon"], sigma_m=400.0,
                    source=f"nom_cache:road:{rd[:30]}",
                    evidence=f"Nominatim cached hit for '{rd}' in {ctx_n}",
                    confidence="high",
                    specificity=1,
                    raw_source_type="nominatim_cache",
                ))
                break   # one variant per road
    return out


def _emit_la_centroid(la_poly, la_radius, pi) -> List[CandidateV3]:
    if la_poly is None: return []
    c = la_poly.centroid
    name = (pi.get("district_name") or pi.get("admin_region") or "").split("|")[0].strip()
    short = name[:40] if name else "LA"
    return [CandidateV3(
        lat=c.y, lon=c.x, sigma_m=float(max(la_radius or 8000, 8000)),
        source=f"la_centroid:{short}",
        evidence=f"BoundaryLine LA centroid (radius={la_radius}m)",
        confidence="low",
        specificity=4,
        raw_source_type="la",
    )]


# ── 5. Candidate ranker / dedup ────────────────────────────────────────────

def _rank_and_dedup(cands: List[CandidateV3]) -> List[CandidateV3]:
    """Order by (confidence_weight, -specificity, -sigma_m). Dedup by lat/lon."""
    conf_w = {"high": 3, "med": 2, "low": 1}
    cands.sort(key=lambda c: (-conf_w.get(c.confidence, 1),
                                c.specificity,
                                c.sigma_m))
    seen = set()
    out = []
    for c in cands:
        key = (round(c.lat, 3), round(c.lon, 3))
        if key in seen: continue
        seen.add(key)
        out.append(c)
    return out


def _drop_orphan_outliers(cands: List[CandidateV3],
                          town_anchor: Optional[Tuple[float, float]] = None
                          ) -> List[CandidateV3]:
    """Drop candidates that are >10 km from every other non-letterhead
    candidate AND > 15 km from the town anchor (if known).

    v3.14: anchor on the relaxed town lookup so when two equally-isolated
    places exist (e.g. Hyde Lea 3 km from Stafford vs Codsall 22 km from
    Stafford), the one further from the town gets dropped.
    """
    if len(cands) < 3: return cands
    non_la = [c for c in cands if c.raw_source_type != "la"]
    if len(non_la) < 3: return cands
    def _is_letterhead(c):
        return c.raw_source_type == "postcode" and c.confidence == "low"
    others_pool = [c for c in non_la if not _is_letterhead(c)]
    if len(others_pool) < 2:
        return cands
    drop = set()
    for ci in non_la:
        if _is_letterhead(ci): continue
        # NEVER drop grid_ref, multi_road_consensus, or oml_road — they're
        # authoritative site anchors. Picker rule 1/2 prioritises them; orphan-
        # drop must not remove them just because the place candidates are
        # noisy (e.g. homonymous town like Barham picks Cambridgeshire).
        if ci.raw_source_type in ("grid_ref", "multi_road_consensus", "oml_road"):
            continue
        dmin = min((_hkm(ci.lat, ci.lon, cj.lat, cj.lon)
                    for cj in others_pool if cj is not ci), default=0)
        if dmin <= 10.0: continue   # has a near-cluster mate; keep
        # Far from every other candidate. Compare to town anchor as a
        # tie-breaker: if town is known and this candidate is FURTHER
        # from town than its nearest non-letterhead peer, drop it.
        if town_anchor is not None:
            ci_to_town = _hkm(ci.lat, ci.lon, town_anchor[0], town_anchor[1])
            peer_to_town_min = min(
                (_hkm(cj.lat, cj.lon, town_anchor[0], town_anchor[1])
                 for cj in others_pool if cj is not ci), default=float("inf"))
            if ci_to_town > peer_to_town_min + 2.0:
                drop.add(id(ci))
        else:
            same_type = [c for c in non_la
                         if c.raw_source_type == ci.raw_source_type
                         and id(c) not in drop]
            if len(same_type) > 1:
                drop.add(id(ci))
    return [c for c in cands if id(c) not in drop]


def _diversify_topk(ranked: List[CandidateV3], k: int,
                    fallback: List[CandidateV3]) -> List[CandidateV3]:
    """v3.17 — Build a final top-K list with reserved slots for diversity.

    Without this, ranks are dominated by spec=1 / σ=300 OML/nom_cache roads,
    pushing place/landmark (spec=2) and la_centroid (spec=4) off the top-K
    even though v21 frequently wins on those sources.

    Order of priority (each "slot" filled from `ranked` if available):
      slot 1: best postcode      (raw_source_type == "postcode")
      slot 2: best grid_ref      (raw_source_type == "grid_ref")
      slot 3: best multi-road    (raw_source_type == "multi_road_consensus")
      slot 4: best place         (raw_source_type in {"os_place","gpkg"})
      slot 5: best road          (raw_source_type in {"oml_road","os_road",
                                                       "nominatim_cache"})
      slot 6: second-best road OR second-best place
      slot 7+: la_centroid (fallback, always last if not already there)
      then: fill remaining from `ranked` in original order.

    If k is small (<7) the la_centroid still gets a slot because we
    ALWAYS append it at the end after slotting (i.e. final count may be
    k or k+1; production already handles "max_candidates+1" gracefully
    via the cap downstream).
    """
    if not ranked and not fallback:
        return []
    used = set()
    out: List[CandidateV3] = []

    def _take(pred, label):
        """Pop the first candidate from `ranked` matching `pred`."""
        for c in ranked:
            if id(c) in used: continue
            if pred(c):
                used.add(id(c))
                out.append(c)
                return True
        return False

    # Slot priorities — guaranteed source diversity
    _take(lambda c: c.raw_source_type == "postcode", "postcode")
    _take(lambda c: c.raw_source_type == "grid_ref", "grid_ref")
    _take(lambda c: c.raw_source_type == "multi_road_consensus", "multi")
    _take(lambda c: c.raw_source_type in ("os_place", "gpkg"), "place_a")
    _take(lambda c: c.raw_source_type in ("oml_road", "os_road",
                                            "nominatim_cache"), "road_a")
    # If we still have room, alternate place_b then road_b
    if len(out) < k - 1:  # reserve 1 for la_centroid
        _take(lambda c: c.raw_source_type in ("os_place", "gpkg"), "place_b")
    if len(out) < k - 1:
        _take(lambda c: c.raw_source_type in ("oml_road", "os_road",
                                                "nominatim_cache"), "road_b")
    # Top up with whatever's left from ranked (preserving rank order)
    for c in ranked:
        if len(out) >= k - 1: break
        if id(c) in used: continue
        used.add(id(c))
        out.append(c)
    # Always append la_centroid as final slot if not already present.
    have_la = any(c.raw_source_type == "la" for c in out)
    if fallback and not have_la:
        out.append(fallback[0])
    # Cap at k (la_centroid may push us to k+1; that's intentional — drop
    # the LAST road to make room).
    if len(out) > k:
        # Drop the last spec=1 road to keep la_centroid + diversity
        last_road_idx = None
        for i in range(len(out) - 1, -1, -1):
            if out[i].raw_source_type in ("oml_road", "os_road",
                                            "nominatim_cache") and i != len(out)-1:
                last_road_idx = i
        if last_road_idx is not None:
            out.pop(last_road_idx)
        else:
            out = out[:k]
    return out[:k]


# ── 6. Public entry point ──────────────────────────────────────────────────

def propose_centers_v3(pi: Dict[str, Any], *,
                        max_candidates: int = 6,
                        seed_only: bool = False) -> List[CandidateV3]:
    """Run the v3 cascade. No network calls.

    Returns up to ``max_candidates`` ranked CandidateV3 objects.
    """
    if not isinstance(pi, dict): return []
    la_poly = _la_polygon_for(pi)
    la_radius = _la_radius_m(la_poly) if la_poly is not None else None
    town = _town_centroid(pi)

    seed = []
    seed += _emit_postcode(pi, la_poly)
    seed += _emit_grid_ref(pi)

    if seed_only:
        return _rank_and_dedup(seed)[:max_candidates]

    rest = []
    os_cands = _emit_os_names(pi, la_poly, town)
    rest += os_cands
    gpkg_cands = _emit_gpkg(pi, la_poly, town)
    rest += gpkg_cands
    # v3.6: cached Nominatim hits (no API — disk read).
    nom_cache_cands = _emit_nominatim_cached(pi, la_poly, town)
    rest += nom_cache_cands
    # v3.16: OS Open Map Local road network (offline 348K road names, all
    # with BNG geometry). Disambiguates homonyms via LA bbox filtering.
    oml_road_cands = _emit_oml_roads(pi, la_poly, town)
    rest += oml_road_cands
    # v3.3: multi-source consensus
    consensus = _emit_multi_road_consensus(
        [c for c in os_cands if c.raw_source_type == "os_road"]
                + [c for c in nom_cache_cands if c.raw_source_type == "nominatim_cache"]
                + [c for c in oml_road_cands if c.raw_source_type == "oml_road"],
        [c for c in os_cands if c.raw_source_type == "os_place"]
                + [c for c in gpkg_cands if c.raw_source_type == "gpkg"])
    rest += consensus
    fallback = _emit_la_centroid(la_poly, la_radius, pi)

    # Pass relaxed town anchor so orphan-drop can break ties on cross-LA cases.
    town_anchor = _town_lookup_relaxed(pi) or town
    ranked = _drop_orphan_outliers(_rank_and_dedup(seed + rest), town_anchor)
    # v3.17 source diversification: guarantee that the top-`max_candidates`
    # list ALWAYS contains (when available) at least:
    #   - 1 postcode  (sub-metre seed)
    #   - 1 grid_ref  (BNG seed)
    #   - 1 multi_road_consensus (cluster anchor)
    #   - 2 distinct places/landmarks (spec=2 anchors – v21 frequently wins on these)
    #   - 1 la_centroid (final fallback – v21 wins LA centroid for vague cases)
    # The default ranker squashes spec=2 places off the top-K when many spec=1
    # OML roads exist; this layer rescues them.
    cands = _diversify_topk(ranked, max_candidates, fallback)
    return cands


def pick_one_v3(cands: List[CandidateV3]) -> Optional[CandidateV3]:
    """v3.4 — pick the single best candidate via agreement-based scoring.

    Insight from v21 analysis: a tight σ (e.g. postcode at 80m) is
    DECEIVING when the postcode is a council letterhead 3+km from the
    actual site. The "right" candidate is one where multiple INDEPENDENT
    sources agree on location.

    Scoring rules (in priority order):
    1. grid_ref present → always pick that (8-fig BNG is near-perfect).
    2. multi_road_consensus present → always pick that (≥3 roads cluster
       within 1.5km — strong corroboration).
    3. postcode WITH ≥1 nearby (≤2 km) non-postcode candidate → pick it
       (postcode confirmed by independent source).
    4. Otherwise: pick the candidate with the most "co-location" with
       other candidates (max #nearby_other_sources within 2 km), break
       ties by σ (smaller better).

    Returns the chosen CandidateV3 or None.
    """
    if not cands: return None
    non_la = [c for c in cands if c.raw_source_type != "la"]
    if not non_la:
        return cands[0]   # only fallback exists

    # Rule 1: grid_ref present
    grid = [c for c in non_la if c.raw_source_type == "grid_ref"]
    if grid:
        return min(grid, key=lambda c: c.sigma_m)   # tightest grid_ref

    # Rule 2: multi_road_consensus (the offline analogue of v21's gold
    # source — empirically beats raw postcode/single-road on the 17-case
    # validation; produces tighter MINIMA convergence).
    consensus = [c for c in non_la if c.raw_source_type == "multi_road_consensus"]
    if consensus:
        return max(consensus, key=lambda c: c.confidence == "high")

    # Co-location counts: for each candidate, how many OTHER candidates of a
    # DIFFERENT source type are within 2 km.
    def co_locate(c):
        return sum(1 for c2 in non_la
                   if c2 is not c
                   and c2.raw_source_type != c.raw_source_type
                   and _hkm(c.lat, c.lon, c2.lat, c2.lon) < 2.0)

    # Rule 3: postcode + at least 1 SITE-SPECIFIC corroborator within 2 km.
    # "Site-specific" excludes gpkg (broad places) and os_place (settlements)
    # because council postcodes often happen to be near a named place. To
    # accept a postcode we need an os_road, multi_road_consensus, grid_ref,
    # or nominatim_cache nearby — sources that lock to the actual planning site.
    SITE_SPECIFIC = {"os_road", "multi_road_consensus", "grid_ref",
                      "nominatim_cache", "oml_road"}
    def site_co_locate(c):
        return sum(1 for c2 in non_la
                   if c2 is not c
                   and c2.raw_source_type in SITE_SPECIFIC
                   and _hkm(c.lat, c.lon, c2.lat, c2.lon) < 2.0)
    pcs = [c for c in non_la if c.raw_source_type == "postcode"]
    for pc in pcs:
        if site_co_locate(pc) >= 1:
            return pc

    # Rule 4: best co-location count, tie-break by σ.
    # CRITICAL: when a postcode candidate has NO site-specific corroboration
    # AND there exists a non-postcode site-specific candidate, drop the
    # postcode from consideration. Letterhead postcodes otherwise hijack
    # rule 4 by accumulating co-locations with broad place candidates that
    # happen to cluster near the council building.
    SITE_SPECIFIC4 = {"os_road", "multi_road_consensus", "grid_ref", "nominatim_cache"}
    has_site_specific = any(c.raw_source_type in SITE_SPECIFIC4 for c in non_la)
    if has_site_specific:
        candidate_pool = [c for c in non_la
                          if not (c.raw_source_type == "postcode"
                                  and site_co_locate(c) < 1)]
        if not candidate_pool:
            candidate_pool = non_la
    else:
        candidate_pool = non_la
    best = max(candidate_pool,
                key=lambda c: (co_locate(c), -c.sigma_m, -c.specificity))
    return best


__all__ = ["propose_centers_v3", "CandidateV3", "pick_one_v3"]
