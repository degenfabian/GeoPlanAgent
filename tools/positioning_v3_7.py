"""Locate v3 — offline-by-design, smaller cascade, sharper σ.

Design goals (vs production tools.locate.propose_centers_v2):
1. NO network calls. Every external API is replaced with an offline OS
   Open Data source (Code-Point Open, OS Open Names, OS Open Zoomstack,
   OS Boundary-Line).
2. Smaller code surface. ~400 LOC instead of ~2700.
3. Tighter, source-calibrated σ. Each source emits σ matched to its
   intrinsic precision rather than a one-size-fits-all 2500–5000m floor.
4. Honest candidate ranking. Specificity & evidence drive ordering, with
   a sanity rule that demotes candidates >5 km from any other candidate
   when at least 2 candidates exist (catches "letterhead-only" failures).

Public entry point: propose_centers_v3(pi: dict, *, max_candidates=6,
                                       seed_only=False) -> list[Candidate]

`seed_only=True` returns ONLY the highest-confidence source (typically
Code-Point Open postcode + grid_ref + parsed-grid-ref) — useful for the
agent_v3 simulator that wants a tiny set.
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

def _emit_postcode(pi, la_poly) -> List[CandidateV3]:
    """Code-Point Open postcode lookup. Sub-metre BNG → σ=80m."""
    out = []
    try:
        from tools.geocoding.code_point import lookup_postcode
    except Exception:
        return out
    pcs = pi.get("postcodes") or []
    seen = set()
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
        out.append(CandidateV3(
            lat=h["lat"], lon=h["lon"],
            sigma_m=80.0 if in_la else 300.0,
            source=f"postcode:{pc}",
            evidence=f"Code-Point Open {pc}{'' if in_la else ' (outside LA)'}",
            confidence="high" if in_la else "med",
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
    for nm in deduped_q:
        nm = (nm or "").strip()
        if len(nm) < 3: continue
        if any(suf in _norm_road(nm) for suf in ()):
            # _norm_road strips road suffixes; skipping anything that looked
            # like a road keeps the road branch tidy.
            pass
        try: hits = os_search(nm, max_results=10) or []
        except Exception: hits = []
        for h in hits:
            if h.get("lat") is None: continue
            t = (h.get("type") or "").lower()
            if "water" in t or "coastal" in t: continue
            in_region = True
            if la_poly is not None:
                try: in_region = la_poly.contains(Point(h["lon"], h["lat"]))
                except Exception: in_region = True
            elif town is not None:
                in_region = _hkm(h["lat"], h["lon"], town[0], town[1]) < 15
            if not in_region: continue
            key = (round(h["lat"], 4), round(h["lon"], 4))
            if key in seen_places: continue
            seen_places.add(key)
            kind = "village" if "village" in t else ("hamlet" if "hamlet" in t else "place")
            out.append(CandidateV3(
                lat=h["lat"], lon=h["lon"], sigma_m=1200.0,
                source=f"os_place:{nm[:40]}",
                evidence=f"OS Open Names {kind} {h.get('name_full','')[:50]}",
                confidence="med",
                specificity=2,
                raw_source_type="os_place",
            ))
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
        if c.raw_source_type in ("os_road", "os_place"):
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
        lat=lat_c, lon=lon_c, sigma_m=1000.0,
        source=f"multi_road_consensus:{best_count}",
        evidence=f"Centroid of {best_count} co-clustered OS Open Names hits",
        confidence="high",
        specificity=1,
        raw_source_type="multi_road_consensus",
    )]


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


def _drop_orphan_outliers(cands: List[CandidateV3]) -> List[CandidateV3]:
    """If we have ≥3 candidates and 1 is >10 km from ALL others, drop it.
    Likely a letterhead postcode or wrong-named place. Keeps la_centroid
    intact (it's allowed to be the only "far" candidate).
    """
    if len(cands) < 3: return cands
    non_la = [c for c in cands if c.raw_source_type != "la"]
    if len(non_la) < 3: return cands
    drop = set()
    for i, ci in enumerate(non_la):
        dmin = min((_hkm(ci.lat, ci.lon, cj.lat, cj.lon)
                    for j, cj in enumerate(non_la) if i != j), default=0)
        if dmin > 10.0:
            drop.add(id(ci))
    return [c for c in cands if id(c) not in drop]


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
    # v3.6: cached Nominatim hits (no API — disk read). Adds street-level
    # resolution where OS Open Names misses.
    nom_cache_cands = _emit_nominatim_cached(pi, la_poly, town)
    rest += nom_cache_cands
    # v3.3: multi-source consensus (offline analogue of multi_road_consensus)
    consensus = _emit_multi_road_consensus(
        [c for c in os_cands if c.raw_source_type == "os_road"]
                + [c for c in nom_cache_cands if c.raw_source_type == "nominatim_cache"],
        [c for c in os_cands if c.raw_source_type == "os_place"]
                + [c for c in gpkg_cands if c.raw_source_type == "gpkg"])
    rest += consensus
    fallback = _emit_la_centroid(la_poly, la_radius, pi)

    cands = _drop_orphan_outliers(_rank_and_dedup(seed + rest))
    # Always append the fallback last
    if fallback:
        if not any(c.raw_source_type == "la" for c in cands):
            cands.append(fallback[0])
    return cands[:max_candidates]


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
    SITE_SPECIFIC = {"os_road", "multi_road_consensus", "grid_ref", "nominatim_cache"}
    def site_co_locate(c):
        return sum(1 for c2 in non_la
                   if c2 is not c
                   and c2.raw_source_type in SITE_SPECIFIC
                   and _hkm(c.lat, c.lon, c2.lat, c2.lon) < 2.0)
    pcs = [c for c in non_la if c.raw_source_type == "postcode"]
    for pc in pcs:
        if site_co_locate(pc) >= 1:
            return pc

    # Rule 4: best co-location count, tie-break by σ
    best = max(non_la,
                key=lambda c: (co_locate(c), -c.sigma_m, -c.specificity))
    return best


__all__ = ["propose_centers_v3", "CandidateV3", "pick_one_v3"]
