"""Postcode helpers: normalisation, lookup, area-centroid.

Two backends:

* Code-Point Open (offline, sub-metre, used by :func:`tools.code_point.lookup_postcode`)
  is the preferred path and is called directly from the v2 cascade.
* :func:`_lookup_postcode` here goes through postcodes.io with disk-cache +
  retry. It's the legacy v13 fallback, still used by ``locate_map`` and a
  couple of overnight reproducibility scripts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ─── Normalisation ─────────────────────────────────────────────────────────

def _normalize_postcode(pc: str) -> str:
    if not pc: return ""
    s = pc.upper().replace(" ", "").strip()
    if len(s) >= 5:
        return f"{s[:-3]} {s[-3:]}"
    return s


def _is_full_postcode(pc: str) -> bool:
    pc_norm = _normalize_postcode(pc)
    parts = pc_norm.split()
    return len(parts) == 2 and len(parts[0]) >= 2 and len(parts[1]) == 3


def _postcode_area(pc: str) -> Optional[str]:
    """Outward letters only: AL1 3JE → 'AL', SW3 4BA → 'SW'."""
    pc_norm = _normalize_postcode(pc)
    if not pc_norm: return None
    out = pc_norm.split()[0]
    return "".join(c for c in out if c.isalpha())


# ─── postcodes.io lookup with disk cache + retry ───────────────────────────

# Shares cache/postcodes_io.json with tools.geocoders.query_postcodes_io_bulk
# so successful lookups from either path persist for both. Confirmed-misses
# (404) are also cached so we don't re-query non-existent postcodes. Network
# errors are NOT cached so transient failures get retried on the next run.
_POSTCODE_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "cache" / "postcodes_io.json"
_POSTCODE_CACHE: Optional[Dict[str, Any]] = None


def _load_postcode_cache() -> Dict[str, Any]:
    global _POSTCODE_CACHE
    if _POSTCODE_CACHE is not None:
        return _POSTCODE_CACHE
    try:
        if _POSTCODE_CACHE_PATH.exists():
            _POSTCODE_CACHE = json.loads(_POSTCODE_CACHE_PATH.read_text())
        else:
            _POSTCODE_CACHE = {}
    except Exception:
        _POSTCODE_CACHE = {}
    return _POSTCODE_CACHE


def _save_postcode_cache() -> None:
    if _POSTCODE_CACHE is None:
        return
    try:
        _POSTCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _POSTCODE_CACHE_PATH.write_text(json.dumps(_POSTCODE_CACHE, indent=2))
    except Exception:
        pass


def _lookup_postcode(pc: str) -> Optional[Dict[str, float]]:
    """Resolve a UK postcode to {lat, lon} via postcodes.io, with disk cache + retry.

    Caches both successful hits and confirmed-misses (postcodes.io 404).
    Network errors (timeouts, 429/5xx) are NOT cached — they're retried 2x
    with exponential backoff, then return None for this call but leave the
    cache untouched so the next run gets a fresh shot. Without these two
    properties (caching successes, NOT caching transient fails) the previous
    benchmark lost the NR15 2XE postcode for 12:00115:ART4 because the
    single live API call happened to fail and the candidate was silently
    dropped.
    """
    import urllib.request
    import urllib.parse
    import urllib.error

    pc_norm = pc.strip().upper().replace(" ", "")
    if not pc_norm:
        return None
    cache = _load_postcode_cache()
    if pc_norm in cache:
        v = cache[pc_norm]
        return v if v else None

    req = urllib.request.Request(
        f"https://api.postcodes.io/postcodes/{urllib.parse.quote(pc)}",
        headers={"User-Agent": "GeoMapAgent-locate/0.1"},
    )
    data = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 ** attempt)  # 2s, 4s
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                cache[pc_norm] = None
                _save_postcode_cache()
                return None
            if e.code not in (429, 500, 502, 503, 504):
                return None  # 4xx other than 404 — don't cache, don't retry
        except (urllib.error.URLError, TimeoutError, OSError):
            pass  # transient — retry
    if data is None:
        return None  # all retries failed; don't poison the cache

    r = (data.get("result") or {})
    if "latitude" in r and "longitude" in r:
        out = {"lat": r["latitude"], "lon": r["longitude"]}
        cache[pc_norm] = out
        _save_postcode_cache()
        return out
    cache[pc_norm] = None
    _save_postcode_cache()
    return None


# ─── Postcode-area centroid (for town disambiguation) ──────────────────────

# Lazily filled by hitting Code-Point Open with a representative postcode.
_AREA_CENTROID_CACHE: Dict[str, Optional[Tuple[float, float]]] = {}


def _area_centroid(area: str) -> Optional[Tuple[float, float]]:
    """Approximate centroid of a postcode area (e.g. AL → St Albans).
    Computed from the first 200 postcodes in that area's Code-Point Open
    file, cached in-process."""
    if not area: return None
    if area in _AREA_CENTROID_CACHE:
        return _AREA_CENTROID_CACHE[area]
    try:
        from tools.code_point import _load_area
        d = _load_area(area.lower())
        if not d:
            _AREA_CENTROID_CACHE[area] = None
            return None
        es, ns = [], []
        for i, (pc, (e, n)) in enumerate(d.items()):
            es.append(e); ns.append(n)
            if i > 200: break
        if not es:
            _AREA_CENTROID_CACHE[area] = None
            return None
        from pyproj import Transformer
        t = Transformer.from_crs(27700, 4326, always_xy=True)
        e_mid, n_mid = sum(es) / len(es), sum(ns) / len(ns)
        lon, lat = t.transform(e_mid, n_mid)
        _AREA_CENTROID_CACHE[area] = (lat, lon)
        return _AREA_CENTROID_CACHE[area]
    except Exception:
        _AREA_CENTROID_CACHE[area] = None
        return None
