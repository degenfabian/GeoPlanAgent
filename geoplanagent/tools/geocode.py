"""Offline UK geocoding, one section per data source: OS Open Names search,
Code-Point Open postcode centroids, OS National Grid reference parsing, and
OS BoundaryLine administrative-area polygon resolution.
"""

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

_HEADER = [
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

_KEEP_COLS = [
    "NAME1",
    "LOCAL_TYPE",
    "GEOMETRY_X",
    "GEOMETRY_Y",
    "DISTRICT_BOROUGH",
    "COUNTY_UNITARY",
    "COUNTRY",
    # POPULATED_PLACE is the village/hamlet-level context
    # column (filled on ~16% of rows). The context filter
    # in ``search()`` iterates over POPULATED_PLACE, but
    # without it in _KEEP_COLS the column was silently
    # missing → the village-level disambiguation branch was
    # a no-op (e.g. ``place("Manor Road", la="Cullivoe")``
    # could not match by Cullivoe).
    "POPULATED_PLACE",
]


_TABLE: Optional[pd.DataFrame] = None


def _load() -> pd.DataFrame:
    """Lazy-load the full Open Names dataset into a single DataFrame.
    Memory: ~150MB. Loads in ~3-5s on first call. Idempotent."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"OS Open Names not found at {DATA_DIR}. Run:\n"
            f"  curl -L -o opname.zip 'https://api.os.uk/downloads/v1/products"
            f"/OpenNames/downloads?area=GB&format=CSV&redirect' && unzip "
            f"opname.zip -d os_opendata/open_names/csv"
        )
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {DATA_DIR}")
    parts = []
    for f in files:
        try:
            df = pd.read_csv(
                f,
                names=_HEADER,
                header=None,
                usecols=_KEEP_COLS,
                encoding="utf-8",
                on_bad_lines="skip",
            )
            parts.append(df)
        except Exception:
            continue
    _TABLE = pd.concat(parts, ignore_index=True)
    _TABLE["LOCAL_TYPE"] = _TABLE["LOCAL_TYPE"].fillna("").str.lower()
    # Lowercase name column for the exact/prefix matching in search()
    _TABLE["_NAME_LC"] = _TABLE["NAME1"].fillna("").str.lower().str.strip()
    return _TABLE


def _safe_str(v) -> str:
    """Coerce CSV-loaded values to str; pd reads empty fields as NaN floats."""
    if v is None:
        return ""
    if isinstance(v, float) and (v != v):  # NaN
        return ""
    return str(v).strip()


def _row_to_hit(row: pd.Series) -> Dict:
    name = _safe_str(row.get("NAME1"))
    bo = _safe_str(row.get("DISTRICT_BOROUGH"))
    co = _safe_str(row.get("COUNTY_UNITARY"))
    lon, lat = _OSGB_TO_WGS84.transform(float(row["GEOMETRY_X"]), float(row["GEOMETRY_Y"]))
    lt = row.get("LOCAL_TYPE", "") or ""
    # The locate-agent's ``place`` tool reads ``name``, ``type``,
    # ``admin_district`` and ``county`` under exactly these keys.
    return {
        "name": name or None,
        "type": lt,
        "lat": float(lat),
        "lon": float(lon),
        "admin_district": bo or None,
        "county": co or None,
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


def _normalize_query(q: str) -> List[str]:
    """Return [original, stripped-of-qualifier-suffix] candidates."""
    base = q.strip().lower()
    cands = [base]
    for suf in _QUALIFIER_SUFFIXES:
        if base.endswith(suf) and len(base) > len(suf) + 2:
            cands.append(base[: -len(suf)].strip())
    return cands


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
    df = _load()

    rows_pool = df

    # Pre-filter by context if provided.
    #    Strip "District"/"Borough" suffixes so "South Norfolk District" matches
    #    DISTRICT_BOROUGH="South Norfolk".
    if context:
        ctx = context.strip().lower()
        for suf in (" district", " borough", " unitary", " county", " council"):
            if ctx.endswith(suf):
                ctx = ctx[: -len(suf)].strip()
        ctx_tokens = [t.strip() for t in ctx.replace(",", " ").split() if len(t.strip()) > 2]
        if ctx_tokens:
            mask = pd.Series(False, index=rows_pool.index)
            for col in ("DISTRICT_BOROUGH", "COUNTY_UNITARY", "POPULATED_PLACE"):
                if col not in rows_pool.columns:
                    continue
                col_lc = rows_pool[col].fillna("").astype(str).str.lower()
                for tok in ctx_tokens:
                    mask = mask | col_lc.str.contains(tok, na=False, regex=False)
            if mask.any():
                rows_pool = rows_pool[mask]

    pool_name_lc = rows_pool["_NAME_LC"]
    qcands = _normalize_query(query)
    idxs: List[int] = []
    seen_idx: set = set()

    for q in qcands:
        # 1. Exact match within filtered pool
        exact = rows_pool.index[pool_name_lc == q].tolist()
        for i in exact:
            if i not in seen_idx:
                idxs.append(int(i))
                seen_idx.add(i)
        if len(idxs) >= max_results:
            break
        # 2. Prefix match within filtered pool
        prefix = rows_pool.index[pool_name_lc.str.startswith(q, na=False)].tolist()
        for i in prefix:
            if i not in seen_idx:
                idxs.append(int(i))
                seen_idx.add(i)
        if len(idxs) >= max_results:
            break

    # 3. Fuzzy fallback within filtered pool (only if nothing exact)
    if not idxs:
        try:
            from rapidfuzz import process, fuzz

            pool_names = pool_name_lc.unique().tolist()
            for q in qcands:
                cands = process.extract(
                    q, pool_names, scorer=fuzz.WRatio, limit=max_results, score_cutoff=85
                )
                for cn, score, _ in cands:
                    matches = rows_pool.index[pool_name_lc == cn].tolist()
                    for i in matches:
                        if i not in seen_idx:
                            idxs.append(int(i))
                            seen_idx.add(i)
                if idxs:
                    break
        except Exception:
            pass

    if not idxs:
        # Context filter eliminated everything? Retry globally.
        if context and rows_pool is not df:
            return search(query, max_results=max_results, context=None)
        return []

    rows = df.iloc[idxs]
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

# area_lower -> {full_postcode -> (E, N, district_code)} where
# district_code is the GSS code (e.g. 'E07000240') for the resolving
# admin district, or '' when the CSV row omitted it. Used by
# `lookup_postcode` to surface a human-readable admin_district name.
_CACHE: Dict[str, Dict[str, tuple]] = {}
# GSS code -> name, lazily loaded from the Codelist.xlsx that ships with
# Code-Point Open. Resolves codes from DIS / LBO / MTD / UTA sheets
# (district + borough + metropolitan + unitary). Empty dict if the
# xlsx is missing or unreadable.
_DISTRICT_NAMES: Optional[Dict[str, str]] = None


def _normalize_postcode(pc: str) -> str:
    """Standardize postcode to e.g. 'AL1 3JE' (one space between out + in)."""
    if not pc:
        return ""
    s = pc.strip().upper().replace(" ", "")
    if len(s) < 5:
        return s  # invalid
    # Last 3 chars are inward, rest is outward
    return f"{s[:-3]} {s[-3:]}"


def _area_for_postcode(pc_norm: str) -> str:
    """Return the lowercase area code (a-z, e.g. 'al' for AL1, 'b' for B1)."""
    if not pc_norm:
        return ""
    s = pc_norm.replace(" ", "")
    # Area is the leading letters (1-2)
    a = ""
    for ch in s:
        if ch.isalpha():
            a += ch.lower()
        else:
            break
    return a


def _load_area(area: str) -> Dict[str, tuple]:
    """Lazy-load one area's CSV. Returns {postcode: (E, N, district_code)}.

    district_code is parts[8] (the GSS Admin_District_Code, e.g.
    'E07000240'). Empty string when missing. ``lookup_postcode``
    resolves it to a human-readable name via ``_load_district_names``."""
    if area in _CACHE:
        return _CACHE[area]
    f = CSV_DIR / f"{area}.csv"
    if not f.exists():
        _CACHE[area] = {}
        return _CACHE[area]
    out = {}
    with open(f) as fh:
        for line in fh:
            parts = line.rstrip().split(",")
            if len(parts) < 4:
                continue
            pc = parts[0].strip('"')
            try:
                e = int(parts[2])
                n = int(parts[3])
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
            if e == 0 and n == 0:
                continue
            try:
                if len(parts) > 1 and int(parts[1].strip('"')) == 90:
                    continue
            except (ValueError, IndexError):
                pass
            dc = parts[8].strip('"') if len(parts) > 8 else ""
            # Postcodes in file are like '"AL1 1AG"' with single space
            out[pc] = (e, n, dc)
    _CACHE[area] = out
    return out


def _load_district_names() -> Dict[str, str]:
    """Load GSS code → district name from Codelist.xlsx. Memoised.

    Sheets DIS (district), LBO (London borough), MTD (metropolitan
    district), UTA (unitary authority) cover every admin code that
    appears in Code-Point Open. Each sheet has two columns
    [Name, GSS code]; the header row is stored as the first data row
    in pandas because Excel doesn't mark it as a header — so we read
    raw and treat every row as data."""
    global _DISTRICT_NAMES
    if _DISTRICT_NAMES is not None:
        return _DISTRICT_NAMES
    if not _CODELIST_XLSX.exists():
        _DISTRICT_NAMES = {}
        return _DISTRICT_NAMES
    try:
        names: Dict[str, str] = {}
        for sheet in ("DIS", "LBO", "MTD", "UTA"):
            try:
                df = pd.read_excel(_CODELIST_XLSX, sheet_name=sheet, header=None, dtype=str)
            except Exception:
                continue
            for _, row in df.iterrows():
                name, code = str(row.iloc[0]).strip(), str(row.iloc[1]).strip()
                if code and code.upper() != "NAN":
                    names[code] = name
        _DISTRICT_NAMES = names
        return names
    except Exception:
        _DISTRICT_NAMES = {}
        return _DISTRICT_NAMES


def lookup_postcode(postcode: str) -> Optional[Dict]:
    """Lookup a full UK postcode (e.g. 'AL1 3JE'). Returns None if not found."""
    pc_norm = _normalize_postcode(postcode)
    if not pc_norm:
        return None
    area = _area_for_postcode(pc_norm)
    if not area:
        return None
    area_dict = _load_area(area)
    coords = area_dict.get(pc_norm)
    if coords is None:
        return None
    e, n, dcode = coords
    lon, lat = _OSGB_TO_WGS84.transform(e, n)
    district_name = _load_district_names().get(dcode) if dcode else None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "admin_district": district_name,
    }


# OS Grid Reference → WGS84

# OS National Grid: 2-letter prefix → (easting, northing) base in metres.
# Standard formula: each letter is 0-24 (A-Z skipping I).
_OS_GRID_LETTERS = {}
for _c1 in range(26):
    if _c1 == 8:
        continue  # skip I
    _l1 = _c1 - (1 if _c1 > 8 else 0)  # 0-24 index
    for _c2 in range(26):
        if _c2 == 8:
            continue
        _l2 = _c2 - (1 if _c2 > 8 else 0)
        e = ((_l1 - 2) % 5) * 5 + (_l2 % 5)
        n = 19 - 5 * (_l1 // 5) - (_l2 // 5)
        if 0 <= e <= 9 and 0 <= n <= 24:  # valid GB range
            _OS_GRID_LETTERS[chr(_c1 + 65) + chr(_c2 + 65)] = (e * 100000, n * 100000)


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


_LA_POLYGONS = None

# District > county > ceremonial: first layer to claim a name wins.
_LAYER_ORDER = (
    "district_borough_unitary_region.shp",
    "county_region.shp",
    "boundary-line-ceremonial-counties_region.shp",
)


def _normalize_la_name(s: str) -> str:
    if not s:
        return ""
    out = str(s).lower().strip().replace(".", "")
    out = re.sub(r"\s*\(b\)$", "", out)
    out = re.sub(r"\s*\((?:district|borough|county|unitary|metro)\)$", "", out)
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
        if out.endswith(suffix):
            out = out[: -len(suffix)].strip()
    for prefix in (
        "city of ",
        "london borough of ",
        "borough of ",
        "the london borough of ",
        "royal borough of ",
    ):
        if out.startswith(prefix):
            out = out[len(prefix) :].strip()
    return out


def _add_la_variants(out: Dict[str, Any], name: str, geom):
    nm = name.lower()
    if nm not in out:
        out[nm] = geom
    norm = _normalize_la_name(name)
    if norm and norm not in out:
        out[norm] = geom
    if " (b)" in nm:
        bare = nm.replace(" (b)", "")
        if bare not in out:
            out[bare] = geom
    for suffix in (" district", " borough", " london boro", " county"):
        if nm.endswith(suffix):
            short = nm[: -len(suffix)].strip()
            if short and short not in out:
                out[short] = geom


def _load_la_polygons() -> Dict[str, Any]:
    global _LA_POLYGONS
    if _LA_POLYGONS is not None:
        return _LA_POLYGONS
    bdir = ROOT / "os_opendata" / "boundary_line"
    if not bdir.exists():
        _LA_POLYGONS = {}
        return _LA_POLYGONS
    try:
        import geopandas as gpd

        out: Dict[str, Any] = {}

        # Case-insensitive shapefile lookup — Linux is case-sensitive and the
        # ceremonial-counties file ships with a capital B.
        def _find_layer(fname: str):
            lower = fname.lower()
            for p in sorted(bdir.rglob("*.shp")):
                if p.name.lower() == lower:
                    return p
            return None

        if any(_find_layer(f) is None for f in _LAYER_ORDER):
            zp = bdir / "bdline_essh.zip"
            if zp.exists():
                import zipfile

                with zipfile.ZipFile(zp) as z:
                    for member in z.namelist():
                        ml = member.lower()
                        if (
                            "county_region" in ml
                            or "ceremonial-counties" in ml
                            or "district_borough_unitary" in ml
                        ):
                            try:
                                z.extract(member, str(bdir))
                            except Exception:
                                pass

        layer_paths = []
        seen = set()
        for fname in _LAYER_ORDER:
            p = _find_layer(fname)
            if p is not None and p not in seen:
                seen.add(p)
                layer_paths.append(p)
        if not layer_paths:
            print(f"  BoundaryLine: no LA shapefiles under {bdir}")
            _LA_POLYGONS = {}
            return _LA_POLYGONS
        for path in layer_paths:
            try:
                gdf = gpd.read_file(str(path)).to_crs(4326)
            except Exception:
                continue
            name_col = next((c for c in gdf.columns if c.lower() == "name"), None)
            if name_col is None:
                continue
            for _, row in gdf.iterrows():
                nm = str(row[name_col]).strip()
                if nm and row.geometry is not None and not row.geometry.is_empty:
                    _add_la_variants(out, nm, row.geometry)
        _LA_POLYGONS = out
        return out
    except Exception as e:
        print(f"  BoundaryLine load failed: {e!s:.200}")
        _LA_POLYGONS = {}
        return _LA_POLYGONS


def resolve_la(query: str):
    """Resolve a UK admin-area name to its OS BoundaryLine polygon.

    Tries exact match, then suffix/prefix-normalised forms ("Borough of",
    "... District Council", ...), then the longest substring match;
    None when nothing resolves."""
    q = (query or "").strip().lower()
    if not q:
        return None
    polys = _load_la_polygons()
    if q in polys:
        return polys[q]
    qn = _normalize_la_name(query)
    if qn and qn in polys:
        return polys[qn]
    best = None
    best_len = 0
    for k, v in polys.items():
        if q in k or qn in k or k in q or (qn and k in qn):
            if len(k) > best_len:
                best = v
                best_len = len(k)
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
    t0 = time.time()
    if re.fullmatch(r"[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}", joined):
        # Postcode-looking arg → Code-Point Open
        h = lookup_postcode(joined)
        if h:
            print(f"{joined} -> ({h['lat']:.6f}, {h['lon']:.6f})  district={h['admin_district']}")
        else:
            print(f"{joined} -> not found")
        print(f"(load + lookup: {time.time() - t0:.2f}s)")
    else:
        df = _load()
        print(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")
        q = sys.argv[1]
        ctx = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Query: {q!r}  context={ctx!r}")
        for h in search(q, max_results=5, context=ctx):
            print(
                f"  {(h['name'] or ''):40s} {h['type']:25s} "
                f"({h['lat']:.5f}, {h['lon']:.5f})  "
                f"{h['admin_district'] or h['county'] or ''}"
            )
