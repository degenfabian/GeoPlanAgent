"""Candidate ranker for the v2 locate cascade.

Given the raw candidates emitted by :func:`tools.locate.pipeline.propose_centers_v2`,
re-rank them by how many of the document's features (road names, place names,
landmarks) appear in OS Open Names within a few kilometres of each candidate.

This is the strongest discriminator we have between a "close but wrong"
candidate (e.g. the council letterhead postcode) and the actual site.
The right candidate typically has 60-100 % of the document's features
nearby; wrong-region candidates have <20 %.

:func:`feature_cluster_locate` is a sibling primitive: it finds the (lat,
lon) where the most pdf_info features mutually cluster (no anchor needed).
Used as a candidate generator inside the v2 cascade when no
high-precision anchor is available.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from tools.locate.schemas import Candidate


_R_V2 = 6_371_000.0


def _hkm(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in km. Hot-path helper used by the clustering loop."""
    if lat1 is None or lat2 is None: return float("inf")
    dy = math.radians(lat2 - lat1)
    dx = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return _R_V2 * math.hypot(dy, dx) / 1000


# ─── Feature-cluster generator ─────────────────────────────────────────────

def feature_cluster_locate(
    pi: Dict[str, Any],
    cluster_radius_km: float = 2.0,
    min_features_match: int = 3,
    la_poly=None,
) -> Optional[Tuple[float, float, int, int]]:
    """Find the (lat, lon) where the most pdf_info features cluster in OS Open Names.

    For each visible_map_label / place_name / adjacency_hint, search OS Open
    Names for ALL hits across UK (no spatial pre-filter). Then find the hit
    around which the most OTHER LABELS have a hit within ``cluster_radius_km``.
    """
    try:
        from tools.os_names import search as os_search
    except Exception:
        return None
    targets = []
    for k in ("visible_map_labels", "place_names", "adjacency_hints"):
        for v in (pi.get(k) or [])[:6]:
            v = (v or "").strip()
            if not v or len(v) < 4: continue
            if v.lower() in ("the site", "site", "north arrow", "scale", "key"): continue
            targets.append(v)
    seen = set(); targets = [t for t in targets if not (t.lower() in seen or seen.add(t.lower()))]
    if len(targets) < min_features_match:
        return None
    label_hits = {}
    for t in targets[:10]:
        try:
            if la_poly is not None:
                minx, miny, maxx, maxy = la_poly.bounds
                hits = os_search(t, max_results=15,
                                  bbox_wgs84=(miny, minx, maxy, maxx)) or []
            else:
                hits = os_search(t, max_results=15) or []
        except Exception:
            hits = []
        valid = []
        for h in hits:
            if h.get("lat") is None: continue
            if h.get("type") in {"inland water", "coastal feature", "other coastal landform"}:
                continue
            valid.append((h["lat"], h["lon"]))
        if valid: label_hits[t.lower()] = valid
    if len(label_hits) < min_features_match:
        return None
    best = None; best_count = 0
    for label_a, hits_a in label_hits.items():
        for hit_a in hits_a:
            count = 0
            for label_b, hits_b in label_hits.items():
                if label_b == label_a: continue
                if any(_hkm(hit_a[0], hit_a[1], hb[0], hb[1]) < cluster_radius_km
                       for hb in hits_b):
                    count += 1
            if count > best_count:
                best = hit_a; best_count = count
    if best_count + 1 < min_features_match:
        return None
    return (best[0], best[1], best_count + 1, len(targets))


# ─── Scoring ───────────────────────────────────────────────────────────────

def feature_match_score(
    candidate: Candidate, pi: Dict[str, Any], radius_km: float = 3.0,
) -> Dict[str, float]:
    """Score a candidate by how many features mentioned in pdf_info appear
    in OS Open Names within ``radius_km`` of the candidate.

    Strong discriminator: the right candidate has 60-100% of the document's
    features nearby; wrong-region candidates have <20%.

    Returns ``{"score": fraction_matched, "n_targets": N, "n_hits": k}``.
    """
    try:
        from tools.os_names import search as os_search
    except Exception:
        return {"score": 0.0, "n_targets": 0, "n_hits": 0}

    targets = []
    for k in ("road_names", "place_names", "visible_map_labels", "parish_names"):
        for v in (pi.get(k) or [])[:6]:
            v = (v or "").strip()
            if not v or len(v) < 3: continue
            if v.lower() in ("the site", "site", "site boundary"): continue
            targets.append(v)
    seen = set(); targets = [t for t in targets if not (t.lower() in seen or seen.add(t.lower()))]
    if not targets:
        return {"score": 0.0, "n_targets": 0, "n_hits": 0}
    n_hits = 0
    for t in targets[:12]:
        try:
            hits = os_search(t, max_results=5,
                              bbox_center=(candidate.lat, candidate.lon),
                              bbox_radius_km=radius_km) or []
        except Exception:
            hits = []
        if hits:
            n_hits += 1
    return {"score": n_hits / max(1, len(targets[:12])),
            "n_targets": len(targets[:12]),
            "n_hits": n_hits}


def rank_candidates(candidates: List[Candidate], pi: Dict[str, Any]) -> List[Candidate]:
    """Re-rank candidates by combined (source confidence + feature-match score).

    Empirical sweep (2026-05-08): pure feature_match outperforms blends with
    source confidence. Source confidence is already encoded in σ (postcode=100m,
    LA=20km), so weighting it again in the ranker double-counts.

    Modifies candidate.evidence to include the feature-match score.
    Returns candidates sorted by combined score (best first).
    """
    if not candidates:
        return candidates
    scored = []
    for c in candidates:
        fm = feature_match_score(c, pi)
        c.evidence = f"{c.evidence}  [feature_match={fm['n_hits']}/{fm['n_targets']}, score={fm['score']:.2f}]"
        conf_w = {"high": 1.0, "med": 0.6, "low": 0.3}
        scored.append((fm["score"], conf_w.get(c.confidence, 0.5), c))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [c for _, _, c in scored]
