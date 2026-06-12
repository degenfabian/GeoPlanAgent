"""OS National Grid reference parsing → WGS84.

Parses two-letter grid references ("TL 150 067", "TR3559") and labelled
easting/northing strings ("485700 E 148600 N"), converting through
EPSG:27700 → EPSG:4326. Used by the locate sub-agent's grid_ref tool.
"""

import re
from typing import Optional, Tuple

from pyproj import Transformer

# Average meters per degree of latitude
METERS_PER_DEGREE_LAT = 111111.0

# OS Grid Reference → WGS84

# OS National Grid: 2-letter prefix → (easting, northing) base in metres.
# Standard formula: each letter is 0-24 (A-Z skipping I).
_OS_GRID_LETTERS = {}
for _c1 in range(26):
    if _c1 == 8: continue  # skip I
    _l1 = _c1 - (1 if _c1 > 8 else 0)  # 0-24 index
    for _c2 in range(26):
        if _c2 == 8: continue
        _l2 = _c2 - (1 if _c2 > 8 else 0)
        e = ((_l1 - 2) % 5) * 5 + (_l2 % 5)
        n = 19 - 5 * (_l1 // 5) - (_l2 // 5)
        if 0 <= e <= 9 and 0 <= n <= 24:  # valid GB range
            _OS_GRID_LETTERS[chr(_c1 + 65) + chr(_c2 + 65)] = (e * 100000, n * 100000)

_OSGB_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


_EN_RE = re.compile(r"(\d{4,7})\s*E\s*(\d{4,7})\s*N", re.IGNORECASE)


def parse_easting_northing(text: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) parsed from explicit OS easting/northing in metres.

    Accepts formats like "528942 E 184544 N" (typical OS grid coords printed
    on UK planning maps). This is the highest-precision anchor we ever get
    from a PDF — site centre to within ~1 m. Used as a high-confidence
    geocoder candidate via the locate sub-agent's `grid_ref` tool.
    """
    if not isinstance(text, str):
        return None
    m = _EN_RE.search(text)
    if not m:
        return None
    east, north = int(m.group(1)), int(m.group(2))
    # Plausible-BNG range check FIRST — the lat/lon bbox guard alone
    # does not catch all bogus matches: e.g. BNG(1234, 5678) lands at
    # (49.82°N, 7.55°W), which IS inside the inflated UK bbox (the
    # bbox spans down to 49°N to include Channel Isles / Scilly
    # approaches). Real GB BNG eastings are ~60_000-700_000 and
    # northings ~5_000-1_280_000 — a 4-digit easting like 1234 is
    # off the GB mainland entirely. Without this guard, a regex hit
    # on stray text like "ref P1234 E 5678 N" anchors a high-
    # confidence locate candidate in the Celtic Sea.
    if not (60_000 <= east <= 700_000 and 5_000 <= north <= 1_280_000):
        return None
    try:
        lon, lat = _OSGB_TO_WGS84.transform(east, north)
    except Exception:
        return None
    # Defence-in-depth: reject anything outside the UK lat/lon bbox
    # too. Matches the guard in os_grid_ref_to_latlon and
    # os_grid_ref_to_latlon_coarse.
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None
    return float(lat), float(lon)


def os_grid_ref_to_latlon_coarse(grid_ref: str) -> Optional[Tuple[float, float]]:
    """Convert a low-resolution OS grid ref (10km or 5km square) to (lat, lon).

    Handles ``TR 34`` (10km tile centre) and ``TR 34 SE`` (south-east 5km
    quadrant centre). These are too imprecise for the strict
    ``os_grid_ref_to_latlon`` (which requires 1km resolution) but still
    make a useful coarse anchor when no better signal exists.

    Returns (lat, lon) at the CENTRE of the referenced tile / quadrant,
    or None if parsing fails.
    """
    if not grid_ref:
        return None
    s = grid_ref.strip().upper().replace(",", "").replace("  ", " ")

    # Optional trailing compass quadrant
    quad = None
    m = re.search(r"\s+(NE|NW|SE|SW)$", s)
    if m:
        quad = m.group(1)
        s = s[:m.start()]

    # Parse letters + 2 digits (10km tile)
    m = re.match(r"^([A-Z]{2})\s*(\d)\s*(\d)$", s) or \
        re.match(r"^([A-Z]{2})\s*(\d)(\d)$", s)
    if not m:
        return None
    letters, e_dig, n_dig = m.group(1), m.group(2), m.group(3)
    if letters not in _OS_GRID_LETTERS:
        return None

    base_e, base_n = _OS_GRID_LETTERS[letters]
    # 10km tile: digit × 10000 metres, centred at +5000 within the tile
    easting = base_e + int(e_dig) * 10_000
    northing = base_n + int(n_dig) * 10_000
    easting += 5_000
    northing += 5_000

    # Quadrant = 5km sub-square; offset from 10km centre by ±2500 m
    if quad == "NE":
        easting += 2_500;  northing += 2_500
    elif quad == "NW":
        easting -= 2_500;  northing += 2_500
    elif quad == "SE":
        easting += 2_500;  northing -= 2_500
    elif quad == "SW":
        easting -= 2_500;  northing -= 2_500

    try:
        lon, lat = _OSGB_TO_WGS84.transform(easting, northing)
    except Exception:
        return None
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None
    return lat, lon


def os_grid_ref_to_latlon(grid_ref: str) -> Optional[Tuple[float, float]]:
    """Convert OS grid reference string to (lat, lon) WGS84.

    Accepts formats like "TG 210 080", "TG 2105 0803", "TG2108",
    "TG 21 08", "TR 206 48" (asymmetric), with or without spaces.
    Strips trailing compass directions like "SE", "NW" etc.

    Returns (lat, lon) or None if parsing fails.
    """
    s = grid_ref.strip().upper().replace(",", "").replace("  ", " ")
    # Strip trailing compass directions (e.g., "TR 34 SE" → "TR 34")
    s = re.sub(r"\s+[NSEW]{1,2}$", "", s)

    # Range-style refs like "TR3559-60" or "TR 2562-63" or "TG 20-22 08-10":
    # replace the hyphenated tail with just the lower bound so we still get
    # a usable (if slightly coarse) anchor. "TR3559-60" → "TR3559".
    if "-" in s:
        s = re.sub(r"(\d+)-\d+", r"\1", s)
        s = s.strip()

    # Try 2-part format: "TG 210 080" or "TG210 080"
    m = re.match(r"([A-Z]{2})\s*(\d+)\s+(\d+)$", s)
    if not m:
        # Try compact form (with or without space): "TG210080" or "TG2108" or "TG 2638"
        m = re.match(r"([A-Z]{2})\s*(\d+)$", s)
        if not m:
            return None
        letters, digits = m.group(1), m.group(2)
        if len(digits) % 2 != 0:
            # Odd number of digits — try dropping last digit
            digits = digits[:-1]
        if len(digits) < 2:
            return None
        half = len(digits) // 2
        east_digits, north_digits = digits[:half], digits[half:]
    else:
        letters = m.group(1)
        east_digits, north_digits = m.group(2), m.group(3)

    if letters not in _OS_GRID_LETTERS:
        return None

    # Reject low-resolution refs: need at least 4 total digits (2+2 = 1km)
    total_digits = len(east_digits) + len(north_digits)
    if total_digits < 4:
        return None

    # Pad each axis to 5 digits (1m resolution) INDEPENDENTLY. The first
    # padding char is "5" so the resolved metres land at the CENTROID of
    # the precision-defined tile, not its SW corner. Without this, a
    # 4-digit ref like "TR 2048" (1km tile) resolves to the SW corner —
    # up to 1414m from a GT elsewhere in the tile. Centroid bounds
    # worst-case error at the half-diagonal (~707m) and gives ~250-400m
    # for typical GT-in-tile.
    #
    # We must NOT first equalise digit-counts across axes (e.g. ljust
    # north="48" to "480" before centroid-pad): that misinterprets a
    # 2-digit north as a 3-digit north, shifting the centroid by ~450m
    # on the shorter axis for asymmetric refs like "TR 206 48".
    def _centroid_pad(d: str) -> str:
        if len(d) >= 5:
            return d
        return d + "5" + "0" * (5 - len(d) - 1)
    east_digits = _centroid_pad(east_digits)
    north_digits = _centroid_pad(north_digits)

    base_e, base_n = _OS_GRID_LETTERS[letters]
    easting = base_e + int(east_digits)
    northing = base_n + int(north_digits)

    # Convert OSGB36 → WGS84
    lon, lat = _OSGB_TO_WGS84.transform(easting, northing)

    # Validate result is within UK bounding box (49-61°N, -8.5-2°E)
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None

    return lat, lon

# Minimum points required to form a valid polygon
MIN_POLYGON_POINTS = 3


