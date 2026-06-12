"""MINIMA sliding-window matching of planning maps against OS OpenData tiles.

The pipeline is:

1. Load the MINIMA-LoFTR matcher with :func:`load_minima`.
2. Render OS Zoomstack tiles around candidate centres
   (``geoplanagent.io.os_tiles.fetch_os_opendata_grid``).
3. :func:`sliding_window_position` slides a planning-map-sized window across
   the rendered canvas, calling :func:`run_minima` at each position and
   keeping the best window via the quadrant-coverage reranker.
4. :func:`estimate_affine` recovers the 2×3 page→tile affine.
5. :func:`mask_to_geojson_affine` projects the SAM mask through that affine
   to a WGS84 GeoJSON polygon.
"""

from geoplanagent.geo.coords import best_zoom_for_scale, compute_map_mpp
from geoplanagent.matching._core import (
    # MINIMA model management
    load_minima, run_minima, estimate_affine,
    # Affine + GeoJSON
    resize_map_to_match_zoom,
    affine_center_to_latlon, mask_to_geojson_affine,
    # Sliding-window search (the master entry point)
    sliding_window_position,
)
from geoplanagent.matching._core import effective_sigma, sigma_from_scale

# Road-name verification (re-exported from geoplanagent.matching.road_verify)
from geoplanagent.matching.road_verify import (
    _verify_candidates_with_road_names,
    _query_gpkg_road_names,
    _fuzzy_road_match,
)

__all__ = [
    "load_minima", "run_minima", "estimate_affine",
    "compute_map_mpp", "best_zoom_for_scale", "sigma_from_scale",
    "effective_sigma",
    "resize_map_to_match_zoom",
    "affine_center_to_latlon", "mask_to_geojson_affine",
    "sliding_window_position",
    # Road-name verification helpers — exposed for the metrics-reward axis.
    "_verify_candidates_with_road_names",
    "_query_gpkg_road_names",
    "_fuzzy_road_match",
]
