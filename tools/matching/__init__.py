"""MINIMA sliding-window matching of planning maps against OS OpenData tiles.

The pipeline is:

1. Load the MINIMA-LoFTR matcher with :func:`load_minima`.
2. Render OS Zoomstack tiles around candidate centres
   (``tools.os_opendata_tiles.fetch_os_opendata_grid``).
3. :func:`sliding_window_position` slides a planning-map-sized window across
   the rendered canvas, calling :func:`run_minima` at each position and
   keeping the best per-bucket window via the composite reranker.
4. :func:`estimate_affine` recovers the 2×3 page→tile affine.
5. :func:`mask_to_geojson_affine` projects the SAM mask through that affine
   to a WGS84 GeoJSON polygon.

The implementation currently lives in :mod:`tools.matching._core` and will
be carved into themed sub-modules (search, affine_io, road_verify, sigma_la)
over subsequent passes. Re-exports here keep ``from tools.matching import …``
stable across that work.
"""

from tools.matching._core import (
    # MINIMA model management
    load_minima, run_minima, estimate_affine,
    # Scale / zoom / sigma / LA helpers
    compute_map_mpp, best_zoom_for_scale, sigma_from_scale,
    sigma_from_source,
    effective_sigma, candidate_passes_la_filter,
    # Center specificity table (consumed by geocoders.cross_validate_centers)
    _center_specificity,
    # Affine + GeoJSON
    analytical_affine_from_anchor, resize_map_to_match_zoom,
    affine_center_to_latlon, mask_to_geojson_affine,
    _build_scale_H,
    # Center filtering
    filter_centers,
    # Road-name verification (re-exported from tools.matching.road_verify)
    _verify_candidates_with_road_names,
    _query_gpkg_road_names,
    _fuzzy_road_match,
    # Sliding-window search (the master entry point)
    sliding_window_position,
    # Thin-mask helper used by critic.retry_projection
    _expand_thin_mask,
)

__all__ = [
    "load_minima", "run_minima", "estimate_affine",
    "compute_map_mpp", "best_zoom_for_scale", "sigma_from_scale",
    "effective_sigma", "candidate_passes_la_filter",
    "analytical_affine_from_anchor", "resize_map_to_match_zoom",
    "affine_center_to_latlon", "mask_to_geojson_affine",
    "filter_centers",
    "sliding_window_position",
]
