"""Road-name verification helper.

A cross-check used by :func:`sliding_window_position` when picking between
candidates that have similar inlier counts but plausibly different anchors:
query the OS Zoomstack GeoPackage for road names near each candidate's
predicted centre, fuzzy-match against the road names the reader extracted
from the PDF, and prefer the candidate with the best road-name overlap.
Overrides the metric-best candidate only when the override has at least
60 % road-name match AND at least 2× the top candidate's ratio AND ≥70 %
of the top candidate's metric.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Set to True the first time we notice the OS Zoomstack file is missing,
# so the verifier prints exactly one warning per process instead of
# spamming every candidate iteration.
_ZOOMSTACK_WARNED = False


# ─── Road-name verifier ────────────────────────────────────────────────────

def _query_gpkg_road_names(lat, lon, radius_m=1500):
    """Query OS GeoPackage for road names near a point. Fully offline."""
    try:
        import geopandas as gpd
        import pyproj

        gpkg_path = BASE_DIR / "os_opendata" / "OS_Open_Zoomstack.gpkg"
        if not gpkg_path.exists():
            # Warn ONCE per process, not per-call (this fires inside the
            # per-candidate verifier loop and the per-call critic axis).
            global _ZOOMSTACK_WARNED
            if not _ZOOMSTACK_WARNED:
                print(f"  road_verify: WARNING — {gpkg_path} not found; "
                      f"road-name verification disabled. Download from "
                      f"OS Open Zoomstack and place at this path.")
                _ZOOMSTACK_WARNED = True
            return []

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:27700", always_xy=True)
        x, y = transformer.transform(lon, lat)

        names = set()
        for layer in ["roads_local", "roads_regional", "roads_national"]:
            try:
                gdf = gpd.read_file(
                    str(gpkg_path), layer=layer,
                    bbox=(x - radius_m, y - radius_m,
                          x + radius_m, y + radius_m))
                for _, row in gdf.iterrows():
                    name = row.get("name")
                    if name and str(name).strip() and str(name) != "None":
                        names.add(str(name).strip())
            except Exception:
                pass
        return list(names)
    except ImportError:
        return []


def _fuzzy_road_match(llm_name, reference_names):
    """Check if an LLM-extracted road name matches any reference name."""
    llm_lower = llm_name.lower().strip()
    for ref in reference_names:
        ref_lower = ref.lower().strip()
        if llm_lower == ref_lower:
            return True
        if llm_lower in ref_lower or ref_lower in llm_lower:
            return True
        # Handle common abbreviations
        llm_norm = (llm_lower
                    .replace(" street", " st").replace(" road", " rd")
                    .replace(" lane", " ln").replace(" avenue", " ave")
                    .replace(" drive", " dr").replace(" close", " cl"))
        ref_norm = (ref_lower
                    .replace(" street", " st").replace(" road", " rd")
                    .replace(" lane", " ln").replace(" avenue", " ave")
                    .replace(" drive", " dr").replace(" close", " cl"))
        if llm_norm == ref_norm:
            return True
    return False


def _verify_candidates_with_road_names(ranked_candidates, road_names):
    """Re-rank candidates by metric × (1 + road_match_ratio) ** 2.

    Each candidate's metric is multiplied by a quadratic boost from its
    road-name overlap ratio: 0 matches → 1× (no change), all matches →
    4× (full boost). The quadratic shape is symmetric with the squared
    scale_consistency penalty — both treat the relevant signal as
    multiplicatively-quadratic in their evidence.

    Single knob (exponent ``p = 2``); replaces the previous triple-gated
    scheme (5 magic numbers: 0.5 / 0.6 / 2× / 0.7 / 0.01) with one
    decisive multiplicative form.

    Candidates with no nearby OS roads (sparse cartography) get a
    neutral boost of 1.0 — neither helped nor penalised — so the metric
    fully decides for them.
    """
    if not road_names:
        return None

    n_road = len(road_names)
    scored = []
    for metric, _seq, candidate in ranked_candidates:
        center_ll = candidate["match_info"].get("center_latlon")
        if not center_ll:
            scored.append((metric, metric, candidate, None))
            continue
        lat, lon = center_ll
        nearby = _query_gpkg_road_names(lat, lon, radius_m=1500)
        if not nearby:
            scored.append((metric, metric, candidate, None))
            continue
        matches = sum(1 for rn in road_names if _fuzzy_road_match(rn, nearby))
        ratio = matches / n_road
        boosted = metric * (1.0 + ratio) ** 2
        scored.append((boosted, metric, candidate, matches))

    if not scored:
        return None

    for boosted, orig, cand, matches in scored:
        cname = cand["match_info"]["center"]
        inliers = cand["match_info"]["n_inliers"]
        m_str = "n/a" if matches is None else f"{matches}/{n_road}"
        print(f"    Road verify: {cname} inl={inliers} "
              f"metric={orig:.1f} boosted={boosted:.1f} roads={m_str}")

    scored.sort(key=lambda r: -r[0])
    top_cand = scored[0][2]
    metric_best_cand = ranked_candidates[0][2]
    if top_cand is metric_best_cand:
        print("    Road verify: top candidate confirmed")
        return None
    cname = top_cand["match_info"]["center"]
    print(f"    Road verify: OVERRIDE → {cname} "
          f"(boosted {scored[0][0]:.1f})")
    return top_cand
