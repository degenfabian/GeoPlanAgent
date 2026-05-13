"""Code-Point Open — full UK postcode → BNG (sub-metre) lookup, free OGL.

OS Code-Point Open contains 1.6M GB postcode unit centroids at sub-metre
BNG resolution — much tighter than os_names postcode-district lookup
(which only knows outward codes like "AL1") or postcodes.io (~100m).

Use as a high-priority anchor source when pdf_info.postcodes contains a
full postcode (outward + inward, e.g. "AL1 3JE"). Drop sigma_m to ~50m
for these cases vs the 800-2500m floor we currently pay.

Setup: download once via
    curl -L -o codepo_gb.zip \\
      "https://api.os.uk/downloads/v1/products/CodePointOpen/downloads?area=GB&format=CSV&redirect"
    unzip codepo_gb.zip -d os_opendata/code_point_open/csv

Usage:
    from tools.code_point import lookup_postcode
    hit = lookup_postcode("AL1 3JE")
    # → {'lat': 51.7534, 'lon': -0.3361, 'easting': 515387, 'northing': 206398,
    #    'sigma_m': 50, 'source': 'code_point_open'}

Format: 122 CSVs in csv/Data/CSV/ (one per postcode area, e.g. al.csv).
Columns: Postcode, Positional_Quality_Indicator, Eastings, Northings,
Country_Code, NHS_Regional_HA_Code, NHS_HA_Code, Admin_County_Code,
Admin_District_Code, Admin_Ward_Code

Memory: lazy-loaded per area on first call. Full UK in-memory ~150MB.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "os_opendata" / "code_point_open" / "csv" / "Data" / "CSV"

_CACHE: Dict[str, Dict[str, tuple]] = {}  # area_lower -> {full_postcode -> (E, N)}
_TRANSFORMER = None


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
    """Lazy-load one area's CSV into memory. Returns {postcode: (E, N)}."""
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
            # Postcodes in file are like '"AL1 1AG"' with single space
            out[pc] = (e, n)
    _CACHE[area] = out
    return out


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
    if coords is None:
        # Try with no space (some files might be inconsistent)
        coords = area_dict.get(pc_norm.replace(" ", ""))
    if coords is None: return None
    e, n = coords
    lat, lon = _bng_to_wgs84(e, n)
    return {
        "lat": float(lat), "lon": float(lon),
        "easting": int(e), "northing": int(n),
        "sigma_m": 50,  # Code-Point Open is sub-metre; sigma is positional uncertainty
        "source": "code_point_open",
        "name_full": f"Postcode {pc_norm}",
        "type": "postcode_unit",
    }


def is_loaded(area: str = None) -> bool:
    if area is None: return bool(_CACHE)
    return area in _CACHE


if __name__ == "__main__":
    import sys, time
    if len(sys.argv) < 2:
        print("usage: python -m tools.code_point <postcode>")
        sys.exit(1)
    t0 = time.time()
    pc = " ".join(sys.argv[1:])
    h = lookup_postcode(pc)
    if h:
        print(f"{pc} -> ({h['lat']:.6f}, {h['lon']:.6f})  BNG=({h['easting']}, {h['northing']})  σ={h['sigma_m']}m")
        print(f"(load + lookup: {time.time()-t0:.2f}s)")
    else:
        print(f"{pc} -> not found")
