"""Inter-candidate geometry helpers used by the matching stage.

Originally a 700-line grab-bag of geocoder dispatchers (Nominatim, Photon,
GPKG, Wikidata, Postcodes.io bulk, etc.). The locate sub-agent now owns
all geocoding directly via its own tools, so the dispatchers are gone.
Only two helpers remained in active use; they live here pending the
geocoding/ → geo/ merge.

Public API:
  - haversine distance (`_distance_m`)
  - cross-validate-centers outlier drop (`cross_validate_centers`)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# Center tuple: (name, lat, lon, sigma_m). Used by the matching pipeline.
Center = Tuple[str, float, float, Optional[float]]


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate haversine distance in metres between two lat/lon points."""
    dlat = (lat2 - lat1) * 111111
    dlon = (lon2 - lon1) * 111111 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def cross_validate_centers(
    centers: List[Center],
    max_outlier_km: float = 10,
) -> List[Center]:
    """Drop centers that are >threshold from the median center.

    Uses adaptive thresholding: when many centers agree tightly (IQR < 2km),
    the threshold tightens to 3x IQR (min 2km). Otherwise uses max_outlier_km.

    Median is computed from the highest-specificity subset when present
    (Nominatim street/addr, grid_refs, postcode, Zoomstack Town/City/Village)
    so a wrong-sense Sites/Greenspace/county hit can't drag the median off
    and cause the correct street-level anchor to be dropped.

    If fewer than 3 centers but a rank-≤1 anchor exists (Nominatim street,
    grid-ref, postcode), still drop centers >max_outlier_km from it —
    otherwise a wrong-sense hit 100+km away slips through.
    """
    # Lazy import to avoid circular dependency at module load.
    from tools.matching import _center_specificity as _spec

    # Fast path for small center lists: anchor-based drop only.
    if len(centers) < 3:
        anchors = [c for c in centers if _spec(c[0]) <= 1]
        if not anchors:
            return centers
        anchors.sort(key=lambda c: _spec(c[0]))
        a_lat, a_lon = anchors[0][1], anchors[0][2]
        kept = []
        for c in centers:
            if _spec(c[0]) <= 1:
                kept.append(c)
                continue
            d = _distance_m(c[1], c[2], a_lat, a_lon)
            if d <= max_outlier_km * 1000:
                kept.append(c)
            else:
                print(f"  Cross-validate: dropped {c[0]} ({d/1000:.1f}km "
                      f"from anchor {anchors[0][0]!r})")
        return kept if kept else centers

    specific = [c for c in centers if _spec(c[0]) <= 2]
    median_source = specific if len(specific) >= 1 else centers

    lats = [c[1] for c in median_source]
    lons = [c[2] for c in median_source]
    med_lat = np.median(lats)
    med_lon = np.median(lons)

    dists = [_distance_m(c[1], c[2], med_lat, med_lon) for c in centers]
    dists_sorted = sorted(dists)
    q1 = dists_sorted[len(dists_sorted) // 4]
    q3 = dists_sorted[3 * len(dists_sorted) // 4]
    iqr = q3 - q1

    if len(centers) >= 5 and iqr < 2000:
        threshold_m = max(2000, 3 * iqr)
        print(f"  Cross-validate: adaptive threshold={threshold_m:.0f}m "
              f"(IQR={iqr:.0f}m, {len(centers)} centers)")
    else:
        threshold_m = max_outlier_km * 1000

    kept = []
    dropped = []
    for c, d in zip(centers, dists):
        if d <= threshold_m:
            kept.append(c)
        else:
            dropped.append((c[0], d / 1000))

    if dropped:
        for name, dist_km in dropped:
            print(f"  Cross-validate: dropped {name} ({dist_km:.1f}km from median)")

    if not kept:
        return centers
    return kept
