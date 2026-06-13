"""Offline UK geocoding from OS Open Names, Code-Point Open, the National Grid, and OS BoundaryLine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import re
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "os_opendata" / "open_names" / "csv" / "Data"

# Shared BNG (EPSG:27700) → WGS84 transformer. pyproj is a hard dep, so
# the construction cost is paid once, eagerly, at import.
_OSGB_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

_OPEN_NAMES_COLUMNS = [
    "ID",
    "NAMES_URI",
    "NAME1",
    "NAME1_LANG",
    "NAME2",
    "NAME2_LANG",
    "TYPE",
    "LOCAL_TYPE",
    "GEOMETRY_X",
    "GEOMETRY_Y",
    "MOST_DETAIL_VIEW_RES",
    "LEAST_DETAIL_VIEW_RES",
    "MBR_XMIN",
    "MBR_YMIN",
    "MBR_XMAX",
    "MBR_YMAX",
    "POSTCODE_DISTRICT",
    "POSTCODE_DISTRICT_URI",
    "POPULATED_PLACE",
    "POPULATED_PLACE_URI",
    "POPULATED_PLACE_TYPE",
    "DISTRICT_BOROUGH",
    "DISTRICT_BOROUGH_URI",
    "DISTRICT_BOROUGH_TYPE",
    "COUNTY_UNITARY",
    "COUNTY_UNITARY_URI",
    "COUNTY_UNITARY_TYPE",
    "REGION",
    "REGION_URI",
    "COUNTRY",
    "COUNTRY_URI",
    "RELATED_SPATIAL_OBJECT",
    "SAME_AS_DBPEDIA",
    "SAME_AS_GEONAMES",
]

_OPEN_NAMES_KEEP_COLUMNS = [
    "NAME1",
    "LOCAL_TYPE",
    "GEOMETRY_X",
    "GEOMETRY_Y",
    "DISTRICT_BOROUGH",
    "COUNTY_UNITARY",
    "COUNTRY",
    # The context filter in search() reads POPULATED_PLACE; without it here
    # the column is absent and village-level disambiguation silently no-ops.
    "POPULATED_PLACE",
]


_open_names_table: Optional[pd.DataFrame] = None


def _load() -> pd.DataFrame:
    """Lazy-load the full Open Names dataset into a single DataFrame.
    Memory: ~150MB. Loads in ~3-5s on first call. Idempotent."""
    global _open_names_table
    if _open_names_table is not None:
        return _open_names_table
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"OS Open Names not found at {DATA_DIR}. Run:\n"
            f"  curl -L -o opname.zip 'https://api.os.uk/downloads/v1/products"
            f"/OpenNames/downloads?area=GB&format=CSV&redirect' && unzip "
            f"opname.zip -d os_opendata/open_names/csv"
        )
    csv_paths = sorted(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs in {DATA_DIR}")
    frames = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(
                csv_path,
                names=_OPEN_NAMES_COLUMNS,
                header=None,
                usecols=_OPEN_NAMES_KEEP_COLUMNS,
                encoding="utf-8",
                on_bad_lines="skip",
            )
            frames.append(frame)
        except Exception:
            continue
    _open_names_table = pd.concat(frames, ignore_index=True)
    _open_names_table["LOCAL_TYPE"] = _open_names_table["LOCAL_TYPE"].fillna("").str.lower()
    # Lowercase name column for the exact/prefix matching in search()
    _open_names_table["_NAME_LC"] = (
        _open_names_table["NAME1"].fillna("").str.lower().str.strip()
    )
    return _open_names_table


def _safe_str(value) -> str:
    """Coerce CSV-loaded values to str; pd reads empty fields as NaN floats."""
    if value is None:
        return ""
    if isinstance(value, float) and (value != value):  # NaN
        return ""
    return str(value).strip()


def _row_to_hit(row: pd.Series) -> Dict:
    name = _safe_str(row.get("NAME1"))
    district_borough = _safe_str(row.get("DISTRICT_BOROUGH"))
    county_unitary = _safe_str(row.get("COUNTY_UNITARY"))
    lon, lat = _OSGB_TO_WGS84.transform(float(row["GEOMETRY_X"]), float(row["GEOMETRY_Y"]))
    local_type = row.get("LOCAL_TYPE", "") or ""
    # The locate-agent's ``place`` tool reads ``name``, ``type``,
    # ``admin_district`` and ``county`` under exactly these keys.
    return {
        "name": name or None,
        "type": local_type,
        "lat": float(lat),
        "lon": float(lon),
        "admin_district": district_borough or None,
        "county": county_unitary or None,
    }


_QUALIFIER_SUFFIXES = (
    " village",
    " town",
    " city",
    " hamlet",
    " road",
    " street",
    " lane",
    " avenue",
    " way",
    " borough",
    " district",
)


def _normalize_query(query: str) -> List[str]:
    """Return [original, stripped-of-qualifier-suffix] candidates."""
    base = query.strip().lower()
    candidates = [base]
    for suffix in _QUALIFIER_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix) + 2:
            candidates.append(base[: -len(suffix)].strip())
    return candidates


def search(query: str, max_results: int = 10, context: Optional[str] = None) -> List[Dict]:
    """Return up to `max_results` hits ranked by exact > prefix > fuzzy.

    Args:
        query: name to look up (case-insensitive). Common qualifier suffixes
            ('village', 'road', 'street', etc.) are stripped before lookup
            so 'East Langdon village' matches NAME1='East Langdon'.
        max_results: cap on returned hits.
        context: optional UK county/region/district to disambiguate. Pre-FILTERS
            the search to rows whose DISTRICT_BOROUGH/COUNTY_UNITARY/COUNTRY
            contains any context token. Falls back to global if no rows match.
    """
    if not query or not query.strip():
        return []
    table = _load()

    rows_pool = table

    # Pre-filter by context if provided.
    #    Strip "District"/"Borough" suffixes so "South Norfolk District" matches
    #    DISTRICT_BOROUGH="South Norfolk".
    if context:
        context_lc = context.strip().lower()
        for suffix in (" district", " borough", " unitary", " county", " council"):
            if context_lc.endswith(suffix):
                context_lc = context_lc[: -len(suffix)].strip()
        context_tokens = [
            token.strip() for token in context_lc.replace(",", " ").split() if len(token.strip()) > 2
        ]
        if context_tokens:
            mask = pd.Series(False, index=rows_pool.index)
            for column in ("DISTRICT_BOROUGH", "COUNTY_UNITARY", "POPULATED_PLACE"):
                if column not in rows_pool.columns:
                    continue
                column_lc = rows_pool[column].fillna("").astype(str).str.lower()
                for token in context_tokens:
                    mask = mask | column_lc.str.contains(token, na=False, regex=False)
            if mask.any():
                rows_pool = rows_pool[mask]

    pool_name_lc = rows_pool["_NAME_LC"]
    query_candidates = _normalize_query(query)
    matched_indices: List[int] = []
    seen_indices: set = set()

    def _add_indices(candidate_indices) -> None:
        for index in candidate_indices:
            if index not in seen_indices:
                matched_indices.append(int(index))
                seen_indices.add(index)

    for candidate in query_candidates:
        # 1. Exact match within filtered pool
        _add_indices(rows_pool.index[pool_name_lc == candidate].tolist())
        if len(matched_indices) >= max_results:
            break
        # 2. Prefix match within filtered pool
        _add_indices(rows_pool.index[pool_name_lc.str.startswith(candidate, na=False)].tolist())
        if len(matched_indices) >= max_results:
            break

    # 3. Fuzzy fallback within filtered pool (only if nothing exact)
    if not matched_indices:
        try:
            from rapidfuzz import process, fuzz

            pool_names = pool_name_lc.unique().tolist()
            for candidate in query_candidates:
                fuzzy_matches = process.extract(
                    candidate, pool_names, scorer=fuzz.WRatio, limit=max_results, score_cutoff=85
                )
                for matched_name, score, _ in fuzzy_matches:
                    _add_indices(rows_pool.index[pool_name_lc == matched_name].tolist())
                if matched_indices:
                    break
        except Exception:
            pass

    if not matched_indices:
        # Context filter eliminated everything? Retry globally.
        if context and rows_pool is not table:
            return search(query, max_results=max_results, context=None)
        return []

    rows = table.iloc[matched_indices]
    seen_keys = set()
    hits = []
    for _, row in rows.head(max_results * 3).iterrows():
        key = (
            _safe_str(row.get("NAME1")).lower(),
            round(float(row["GEOMETRY_X"]) / 100),
            round(float(row["GEOMETRY_Y"]) / 100),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        hits.append(_row_to_hit(row))
        if len(hits) >= max_results:
            break
    return hits


CSV_DIR = ROOT / "os_opendata" / "code_point_open" / "csv" / "Data" / "CSV"
_CODELIST_XLSX = ROOT / "os_opendata" / "code_point_open" / "csv" / "Doc" / "Codelist.xlsx"

# area_lower -> {full_postcode -> (easting, northing, district_code)} where
# district_code is the GSS code (e.g. 'E07000240') for the resolving
# admin district, or '' when the CSV row omitted it. Used by
# `lookup_postcode` to surface a human-readable admin_district name.
_postcode_area_cache: Dict[str, Dict[str, tuple]] = {}
# GSS code -> name, lazily loaded from the Codelist.xlsx that ships with
# Code-Point Open. Resolves codes from DIS / LBO / MTD / UTA sheets
# (district + borough + metropolitan + unitary). Empty dict if the
# xlsx is missing or unreadable.
_district_names: Optional[Dict[str, str]] = None


def _normalize_postcode(postcode: str) -> str:
    """Standardize postcode to e.g. 'AL1 3JE' (one space between out + in)."""
    if not postcode:
        return ""
    compact = postcode.strip().upper().replace(" ", "")
    if len(compact) < 5:
        return compact  # invalid
    # Last 3 chars are inward, rest is outward
    return f"{compact[:-3]} {compact[-3:]}"


def _area_for_postcode(postcode_norm: str) -> str:
    """Return the lowercase area code (a-z, e.g. 'al' for AL1, 'b' for B1)."""
    if not postcode_norm:
        return ""
    compact = postcode_norm.replace(" ", "")
    # Area is the leading letters (1-2)
    area = ""
    for char in compact:
        if char.isalpha():
            area += char.lower()
        else:
            break
    return area


def _load_area(area: str) -> Dict[str, tuple]:
    """Lazy-load one area's CSV. Returns {postcode: (easting, northing, district_code)}.

    district_code is parts[8] (the GSS Admin_District_Code, e.g.
    'E07000240'). Empty string when missing. ``lookup_postcode``
    resolves it to a human-readable name via ``_load_district_names``."""
    if area in _postcode_area_cache:
        return _postcode_area_cache[area]
    csv_path = CSV_DIR / f"{area}.csv"
    if not csv_path.exists():
        _postcode_area_cache[area] = {}
        return _postcode_area_cache[area]
    postcodes: Dict[str, tuple] = {}
    with open(csv_path) as csv_file:
        for line in csv_file:
            parts = line.rstrip().split(",")
            if len(parts) < 4:
                continue
            postcode = parts[0].strip('"')
            try:
                easting = int(parts[2])
                northing = int(parts[3])
            except (ValueError, IndexError):
                continue
            # Skip "no position available" postcodes — OS encodes these
            # as BNG(0, 0) with PQ=90 (parts[1]). Without this guard,
            # ``lookup_postcode`` returns a high-confidence σ=50m anchor
            # at WGS84(49.77°N, -7.55°W) — the Celtic Sea — for 866 of
            # 1.75M GB postcodes (0.05%). The locate-agent treats this
            # as a sub-metre prior and wastes the search on open water.
            # Same hazard the BNG-range guard in
            # parse_easting_northing (below) was added for.
            if easting == 0 and northing == 0:
                continue
            try:
                if len(parts) > 1 and int(parts[1].strip('"')) == 90:
                    continue
            except (ValueError, IndexError):
                pass
            district_code = parts[8].strip('"') if len(parts) > 8 else ""
            # Postcodes in file are like '"AL1 1AG"' with single space
            postcodes[postcode] = (easting, northing, district_code)
    _postcode_area_cache[area] = postcodes
    return postcodes


def _load_district_names() -> Dict[str, str]:
    """Load GSS code → district name from Codelist.xlsx. Memoised.

    Sheets DIS (district), LBO (London borough), MTD (metropolitan
    district), UTA (unitary authority) cover every admin code that
    appears in Code-Point Open. Each sheet has two columns
    [Name, GSS code]; the header row is stored as the first data row
    in pandas because Excel doesn't mark it as a header — so we read
    raw and treat every row as data."""
    global _district_names
    if _district_names is not None:
        return _district_names
    if not _CODELIST_XLSX.exists():
        _district_names = {}
        return _district_names
    try:
        names: Dict[str, str] = {}
        for sheet in ("DIS", "LBO", "MTD", "UTA"):
            try:
                frame = pd.read_excel(_CODELIST_XLSX, sheet_name=sheet, header=None, dtype=str)
            except Exception:
                continue
            for _, row in frame.iterrows():
                name, code = str(row.iloc[0]).strip(), str(row.iloc[1]).strip()
                if code and code.upper() != "NAN":
                    names[code] = name
        _district_names = names
        return names
    except Exception:
        _district_names = {}
        return _district_names


def lookup_postcode(postcode: str) -> Optional[Dict]:
    """Lookup a full UK postcode (e.g. 'AL1 3JE'). Returns None if not found."""
    postcode_norm = _normalize_postcode(postcode)
    if not postcode_norm:
        return None
    area = _area_for_postcode(postcode_norm)
    if not area:
        return None
    postcodes = _load_area(area)
    coords = postcodes.get(postcode_norm)
    if coords is None:
        return None
    easting, northing, district_code = coords
    lon, lat = _OSGB_TO_WGS84.transform(easting, northing)
    district_name = _load_district_names().get(district_code) if district_code else None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "admin_district": district_name,
    }


# OS Grid Reference → WGS84

# OS National Grid: 2-letter prefix → (easting, northing) base in metres.
# Standard formula: each letter is 0-24 (A-Z skipping I).
_OS_GRID_LETTERS = {}
for _first_char in range(26):
    if _first_char == 8:
        continue  # skip I
    _first_index = _first_char - (1 if _first_char > 8 else 0)  # 0-24 index
    for _second_char in range(26):
        if _second_char == 8:
            continue
        _second_index = _second_char - (1 if _second_char > 8 else 0)
        _grid_east = ((_first_index - 2) % 5) * 5 + (_second_index % 5)
        _grid_north = 19 - 5 * (_first_index // 5) - (_second_index // 5)
        if 0 <= _grid_east <= 9 and 0 <= _grid_north <= 24:  # valid GB range
            _OS_GRID_LETTERS[chr(_first_char + 65) + chr(_second_char + 65)] = (
                _grid_east * 100000,
                _grid_north * 100000,
            )


_EASTING_NORTHING_RE = re.compile(r"(\d{4,7})\s*E\s*(\d{4,7})\s*N", re.IGNORECASE)


def parse_easting_northing(text: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) parsed from explicit OS easting/northing in metres.

    Accepts formats like "528942 E 184544 N" (typical OS grid coords printed
    on UK planning maps). This is the highest-precision anchor we ever get
    from a PDF — site centre to within ~1 m. Used as a high-confidence
    geocoder candidate via the locate sub-agent's `grid_ref` tool.
    """
    if not isinstance(text, str):
        return None
    match = _EASTING_NORTHING_RE.search(text)
    if not match:
        return None
    east, north = int(match.group(1)), int(match.group(2))
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
    # too. Matches the guard in os_grid_ref_to_latlon.
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None
    return float(lat), float(lon)


def os_grid_ref_to_latlon(grid_ref: str) -> Optional[Tuple[float, float]]:
    """Convert OS grid reference string to (lat, lon) WGS84.

    Accepts formats like "TG 210 080", "TG 2105 0803", "TG2108",
    "TG 21 08", "TR 206 48" (asymmetric), with or without spaces.
    Strips trailing compass directions like "SE", "NW" etc.

    Returns (lat, lon) or None if parsing fails.
    """
    normalized = grid_ref.strip().upper().replace(",", "").replace("  ", " ")
    # Strip trailing compass directions (e.g., "TR 34 SE" → "TR 34")
    normalized = re.sub(r"\s+[NSEW]{1,2}$", "", normalized)

    # Range-style refs like "TR3559-60" or "TR 2562-63" or "TG 20-22 08-10":
    # replace the hyphenated tail with just the lower bound so we still get
    # a usable (if slightly coarse) anchor. "TR3559-60" → "TR3559".
    if "-" in normalized:
        normalized = re.sub(r"(\d+)-\d+", r"\1", normalized)
        normalized = normalized.strip()

    # Try 2-part format: "TG 210 080" or "TG210 080"
    match = re.match(r"([A-Z]{2})\s*(\d+)\s+(\d+)$", normalized)
    if not match:
        # Try compact form (with or without space): "TG210080" or "TG2108" or "TG 2638"
        match = re.match(r"([A-Z]{2})\s*(\d+)$", normalized)
        if not match:
            return None
        letters, digits = match.group(1), match.group(2)
        if len(digits) % 2 != 0:
            # Odd number of digits — try dropping last digit
            digits = digits[:-1]
        if len(digits) < 2:
            return None
        half = len(digits) // 2
        east_digits, north_digits = digits[:half], digits[half:]
    else:
        letters = match.group(1)
        east_digits, north_digits = match.group(2), match.group(3)

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
    def _centroid_pad(digits: str) -> str:
        if len(digits) >= 5:
            return digits
        return digits + "5" + "0" * (5 - len(digits) - 1)

    east_digits = _centroid_pad(east_digits)
    north_digits = _centroid_pad(north_digits)

    base_easting, base_northing = _OS_GRID_LETTERS[letters]
    easting = base_easting + int(east_digits)
    northing = base_northing + int(north_digits)

    # Convert OSGB36 → WGS84
    lon, lat = _OSGB_TO_WGS84.transform(easting, northing)

    # Validate result is within UK bounding box (49-61°N, -8.5-2°E)
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None

    return lat, lon


_LA_POLYGONS = None

# District > county > ceremonial: first layer to claim a name wins.
_LAYER_ORDER = (
    "district_borough_unitary_region.shp",
    "county_region.shp",
    "boundary-line-ceremonial-counties_region.shp",
)


def _normalize_la_name(name: str) -> str:
    if not name:
        return ""
    normalized = str(name).lower().strip().replace(".", "")
    normalized = re.sub(r"\s*\(b\)$", "", normalized)
    normalized = re.sub(r"\s*\((?:district|borough|county|unitary|metro)\)$", "", normalized)
    for suffix in (
        " district council",
        " borough council",
        " city council",
        " county council",
        " metropolitan borough council",
        " london borough council",
        " district",
        " borough",
        " london boro",
        " london borough",
        " metropolitan borough",
        " unitary",
        " unitary authority",
        " council",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    for prefix in (
        "city of ",
        "london borough of ",
        "borough of ",
        "the london borough of ",
        "royal borough of ",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    return normalized


def _add_la_variants(polygons: Dict[str, Any], name: str, geometry):
    name_lc = name.lower()
    if name_lc not in polygons:
        polygons[name_lc] = geometry
    normalized = _normalize_la_name(name)
    if normalized and normalized not in polygons:
        polygons[normalized] = geometry
    if " (b)" in name_lc:
        bare = name_lc.replace(" (b)", "")
        if bare not in polygons:
            polygons[bare] = geometry
    for suffix in (" district", " borough", " london boro", " county"):
        if name_lc.endswith(suffix):
            short = name_lc[: -len(suffix)].strip()
            if short and short not in polygons:
                polygons[short] = geometry


def _load_la_polygons() -> Dict[str, Any]:
    global _LA_POLYGONS
    if _LA_POLYGONS is not None:
        return _LA_POLYGONS
    boundary_dir = ROOT / "os_opendata" / "boundary_line"
    if not boundary_dir.exists():
        _LA_POLYGONS = {}
        return _LA_POLYGONS
    try:
        import geopandas as gpd

        polygons: Dict[str, Any] = {}

        # Case-insensitive shapefile lookup — Linux is case-sensitive and the
        # ceremonial-counties file ships with a capital B.
        def _find_layer(filename: str):
            target = filename.lower()
            for path in sorted(boundary_dir.rglob("*.shp")):
                if path.name.lower() == target:
                    return path
            return None

        if any(_find_layer(layer) is None for layer in _LAYER_ORDER):
            zip_path = boundary_dir / "bdline_essh.zip"
            if zip_path.exists():
                import zipfile

                with zipfile.ZipFile(zip_path) as archive:
                    for member in archive.namelist():
                        member_lc = member.lower()
                        if (
                            "county_region" in member_lc
                            or "ceremonial-counties" in member_lc
                            or "district_borough_unitary" in member_lc
                        ):
                            try:
                                archive.extract(member, str(boundary_dir))
                            except Exception:
                                pass

        layer_paths = []
        seen = set()
        for filename in _LAYER_ORDER:
            path = _find_layer(filename)
            if path is not None and path not in seen:
                seen.add(path)
                layer_paths.append(path)
        if not layer_paths:
            print(f"  BoundaryLine: no LA shapefiles under {boundary_dir}")
            _LA_POLYGONS = {}
            return _LA_POLYGONS
        for path in layer_paths:
            try:
                layer = gpd.read_file(str(path)).to_crs(4326)
            except Exception:
                continue
            name_col = next((col for col in layer.columns if col.lower() == "name"), None)
            if name_col is None:
                continue
            for _, row in layer.iterrows():
                name = str(row[name_col]).strip()
                if name and row.geometry is not None and not row.geometry.is_empty:
                    _add_la_variants(polygons, name, row.geometry)
        _LA_POLYGONS = polygons
        return polygons
    except Exception as error:
        print(f"  BoundaryLine load failed: {error!s:.200}")
        _LA_POLYGONS = {}
        return _LA_POLYGONS


def resolve_la(query: str):
    """Resolve a UK admin-area name to its OS BoundaryLine polygon.

    Tries exact match, then suffix/prefix-normalised forms ("Borough of",
    "... District Council", ...), then the longest substring match;
    None when nothing resolves."""
    query_lc = (query or "").strip().lower()
    if not query_lc:
        return None
    polygons = _load_la_polygons()
    if query_lc in polygons:
        return polygons[query_lc]
    query_norm = _normalize_la_name(query)
    if query_norm and query_norm in polygons:
        return polygons[query_norm]
    best = None
    best_len = 0
    for name, geometry in polygons.items():
        if query_lc in name or query_norm in name or name in query_lc or (query_norm and name in query_norm):
            if len(name) > best_len:
                best = geometry
                best_len = len(name)
    return best


def lookup_district_boundary(
    district_name: str,
) -> Dict[str, Any]:
    """
    Look up the boundary polygon of a UK administrative district from
    OS BoundaryLine (offline, OS Open Data).

    Resolves the name through resolve_la, which normalises common UK
    admin variants (strips "District", "Borough", "London Borough of",
    "City of …", "Council", etc.).

    Used by the worker's `lookup_district` tool for district-wide planning
    documents (entire LA / borough / ward covered).

    Args:
        district_name: e.g. "Camden, UK", "Royal Borough of Kensington
            and Chelsea, UK", or "Broadland District, Norfolk, UK".

    Returns:
        Dict with:
          - success (bool)
          - geojson (Feature with MultiPolygon geometry; source =
            "os_boundaryline")
          - error (on failure)
    """
    from shapely.geometry import mapping, MultiPolygon, Polygon

    try:
        poly = resolve_la(district_name)
    except Exception:
        poly = None

    if poly is None:
        return {
            "success": False,
            "error": f"No OS BoundaryLine polygon found for: {district_name}",
        }

    # Normalise to MultiPolygon for downstream consistency
    if isinstance(poly, Polygon):
        poly = MultiPolygon([poly])
    elif not isinstance(poly, MultiPolygon):
        return {
            "success": False,
            "error": f"Unexpected geometry type from BoundaryLine lookup: {type(poly).__name__}",
        }

    return {
        "success": True,
        "geojson": {
            "type": "Feature",
            "properties": {
                "source": "os_boundaryline",
                "query": district_name,
            },
            "geometry": mapping(poly),
        },
    }


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python -m geoplanagent.tools.geocode <query|postcode> [context]")
        sys.exit(1)
    joined = " ".join(sys.argv[1:]).strip()
    start = time.time()
    if re.fullmatch(r"[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}", joined):
        # Postcode-looking arg → Code-Point Open
        hit = lookup_postcode(joined)
        if hit:
            print(f"{joined} -> ({hit['lat']:.6f}, {hit['lon']:.6f})  district={hit['admin_district']}")
        else:
            print(f"{joined} -> not found")
        print(f"(load + lookup: {time.time() - start:.2f}s)")
    else:
        table = _load()
        print(f"Loaded {len(table):,} rows in {time.time() - start:.1f}s")
        query = sys.argv[1]
        context = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Query: {query!r}  context={context!r}")
        for hit in search(query, max_results=5, context=context):
            print(
                f"  {(hit['name'] or ''):40s} {hit['type']:25s} "
                f"({hit['lat']:.5f}, {hit['lon']:.5f})  "
                f"{hit['admin_district'] or hit['county'] or ''}"
            )
