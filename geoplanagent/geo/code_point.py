"""Code-Point Open — full UK postcode → BNG (sub-metre) lookup, free OGL.

OS Code-Point Open contains 1.6M GB postcode unit centroids at sub-metre
BNG resolution — much tighter than os_names postcode-district lookup
(which only knows outward codes like "AL1") or postcodes.io (~100m).

Used as a high-priority anchor source when pdf_info.postcodes contains a
full postcode (outward + inward, e.g. "AL1 3JE"); hits carry sigma_m ≈ 50 m.

Setup: download once via
    curl -L -o codepo_gb.zip \\
      "https://api.os.uk/downloads/v1/products/CodePointOpen/downloads?area=GB&format=CSV&redirect"
    unzip codepo_gb.zip -d os_opendata/code_point_open/csv

Usage:
    from geoplanagent.geo.code_point import lookup_postcode
    hit = lookup_postcode("AL1 3JE")
    # → {'lat': 51.7534, 'lon': -0.3361, 'easting': 515387, 'northing': 206398,
    #    'sigma_m': 50, 'source': 'code_point_open',
    #    'admin_district': 'St Albans'}

Format: 122 CSVs in csv/Data/CSV/ (one per postcode area, e.g. al.csv).
Columns: Postcode, Positional_Quality_Indicator, Eastings, Northings,
Country_Code, NHS_Regional_HA_Code, NHS_HA_Code, Admin_County_Code,
Admin_District_Code, Admin_Ward_Code

Memory: lazy-loaded per area on first call. Full UK in-memory ~150MB.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = ROOT / "os_opendata" / "code_point_open" / "csv" / "Data" / "CSV"
_CODELIST_XLSX = (ROOT / "os_opendata" / "code_point_open" / "csv" / "Doc"
                  / "Codelist.xlsx")

# area_lower -> {full_postcode -> (E, N, district_code)} where
# district_code is the GSS code (e.g. 'E07000240') for the resolving
# admin district, or '' when the CSV row omitted it. Used by
# `lookup_postcode` to surface a human-readable admin_district name.
_CACHE: Dict[str, Dict[str, tuple]] = {}
_TRANSFORMER = None
# GSS code -> name, lazily loaded from the Codelist.xlsx that ships with
# Code-Point Open. Resolves codes from DIS / LBO / MTD / UTA sheets
# (district + borough + metropolitan + unitary). Empty dict if the
# xlsx is missing or unreadable.
_DISTRICT_NAMES: Optional[Dict[str, str]] = None


def _normalize_postcode(pc: str) -> str:
    """Standardize postcode to e.g. 'AL1 3JE' (one space between out + in)."""
    if not pc: return ""
    s = pc.strip().upper().replace(" ", "")
    if len(s) < 5: return s  # invalid
    # Last 3 chars are inward, rest is outward
    return f"{s[:-3]} {s[-3:]}"


def _area_for_postcode(pc_norm: str) -> str:
    """Return the lowercase area code (a-z, e.g. 'al' for AL1, 'b' for B1)."""
    if not pc_norm: return ""
    s = pc_norm.replace(" ", "")
    # Area is the leading letters (1-2)
    a = ""
    for ch in s:
        if ch.isalpha(): a += ch.lower()
        else: break
    return a


def _load_area(area: str) -> Dict[str, tuple]:
    """Lazy-load one area's CSV. Returns {postcode: (E, N, district_code)}.

    district_code is parts[8] (the GSS Admin_District_Code, e.g.
    'E07000240'). Empty string when missing. ``lookup_postcode``
    resolves it to a human-readable name via ``_load_district_names``."""
    if area in _CACHE: return _CACHE[area]
    f = CSV_DIR / f"{area}.csv"
    if not f.exists():
        _CACHE[area] = {}
        return _CACHE[area]
    out = {}
    with open(f) as fh:
        for line in fh:
            parts = line.rstrip().split(",")
            if len(parts) < 4: continue
            pc = parts[0].strip('"')
            try:
                e = int(parts[2]); n = int(parts[3])
            except (ValueError, IndexError):
                continue
            # Skip "no position available" postcodes — OS encodes these
            # as BNG(0, 0) with PQ=90 (parts[1]). Without this guard,
            # ``lookup_postcode`` returns a high-confidence σ=50m anchor
            # at WGS84(49.77°N, -7.55°W) — the Celtic Sea — for 866 of
            # 1.75M GB postcodes (0.05%). The locate-agent treats this
            # as a sub-metre prior and wastes the search on open water.
            # Same hazard the BNG-range guard in
            # geoplanagent/geo/grid_ref.parse_easting_northing was added for.
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
        import pandas as pd
        names: Dict[str, str] = {}
        for sheet in ("DIS", "LBO", "MTD", "UTA"):
            try:
                df = pd.read_excel(_CODELIST_XLSX, sheet_name=sheet,
                                    header=None, dtype=str)
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


def _bng_to_wgs84(easting: float, northing: float):
    global _TRANSFORMER
    if _TRANSFORMER is None:
        from pyproj import Transformer
        _TRANSFORMER = Transformer.from_crs(27700, 4326, always_xy=True)
    lon, lat = _TRANSFORMER.transform(easting, northing)
    return lat, lon


def lookup_postcode(postcode: str) -> Optional[Dict]:
    """Lookup a full UK postcode (e.g. 'AL1 3JE'). Returns None if not found."""
    pc_norm = _normalize_postcode(postcode)
    if not pc_norm: return None
    area = _area_for_postcode(pc_norm)
    if not area: return None
    area_dict = _load_area(area)
    coords = area_dict.get(pc_norm)
    if coords is None: return None
    e, n, dcode = coords
    lat, lon = _bng_to_wgs84(e, n)
    district_name = _load_district_names().get(dcode) if dcode else None
    return {
        "lat": float(lat), "lon": float(lon),
        "easting": int(e), "northing": int(n),
        "sigma_m": 50,  # Code-Point Open is sub-metre; sigma is positional uncertainty
        "source": "code_point_open",
        "name_full": f"Postcode {pc_norm}",
        "type": "postcode_unit",
        "admin_district": district_name,
        "admin_district_code": dcode or None,
    }


def is_loaded(area: str = None) -> bool:
    if area is None: return bool(_CACHE)
    return area in _CACHE


if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) < 2:
        print("usage: python -m geoplanagent.geo.code_point <postcode>")
        sys.exit(1)
    t0 = time.time()
    pc = " ".join(sys.argv[1:])
    h = lookup_postcode(pc)
    if h:
        print(f"{pc} -> ({h['lat']:.6f}, {h['lon']:.6f})  BNG=({h['easting']}, {h['northing']})  σ={h['sigma_m']}m")
        print(f"(load + lookup: {time.time()-t0:.2f}s)")
    else:
        print(f"{pc} -> not found")
