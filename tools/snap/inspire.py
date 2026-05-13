"""INSPIRE Index Polygons freehold-snap post-processor.

Snaps a predicted boundary polygon to the nearest INSPIRE freehold parcel
edges, when those edges are within a small distance threshold (~10-30m).
40-60% of UK conservation/article-4/listed-building boundaries follow
freehold edges, so snapping converts near-miss IoU 0.65-0.80 → 0.85+.

Free under Open Government Licence (OGL v3). Attribution:
  "Contains HM Land Registry data © Crown copyright and database right {year}.
   This data is licensed under the Open Government Licence v3.0."

GT-leak verified safe (researcher 2026-05-06): no shared identifier with
planning.data.gov.uk; only common ancestor is OS MasterMap.

Usage:
    from tools.snap.inspire import InspireSnap
    snap_obj = InspireSnap(['Dover_District_Council'])
    snapped_geom = snap_obj.snap_polygon(predicted_geom, max_dist_m=20)
"""
from __future__ import annotations
import os, time, zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, snap

INSPIRE_DIR = (Path(__file__).resolve().parent.parent
               / "os_opendata" / "inspire")
CACHE_DIR = INSPIRE_DIR / "cache_wgs84"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class InspireSnap:
    """Loads INSPIRE polygons for one or more LAs and offers snap operations.

    Memory: ~30K polygons × ~200 vertices = ~6M points per LA after loading.
    On first call per LA: 60-90s (GML read). Subsequent calls: <1s (parquet cache).
    """

    def __init__(self, las: Iterable[str], inspire_dir: Optional[Path] = None):
        self.dir = Path(inspire_dir) if inspire_dir else INSPIRE_DIR
        self._edges: List[LineString] = []
        self._loaded: set = set()
        for la in las:
            self.add_la(la)

    def add_la(self, la_name: str) -> None:
        if la_name in self._loaded:
            return
        cache_path = CACHE_DIR / f"{la_name}.parquet"
        if cache_path.exists():
            self._load_from_cache(cache_path)
            self._loaded.add(la_name)
            return
        # Slow path: parse GML
        edges = self._parse_la_gml(la_name)
        if edges is None:
            return
        # Save to cache
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(geometry=edges, crs="EPSG:4326")
            gdf.to_parquet(cache_path)
        except Exception:
            pass
        self._edges.extend(edges)
        self._loaded.add(la_name)

    def _load_from_cache(self, cache_path: Path) -> None:
        try:
            import geopandas as gpd
            gdf = gpd.read_parquet(cache_path)
            self._edges.extend(gdf.geometry.tolist())
        except Exception:
            pass

    def _parse_la_gml(self, la_name: str) -> Optional[List[LineString]]:
        zp = self.dir / f"{la_name}.zip"
        if not zp.exists():
            return None
        try:
            import geopandas as gpd
            t0 = time.time()
            gdf = gpd.read_file(f"zip://{zp}!Land_Registry_Cadastral_Parcels.gml")
            print(f"  INSPIRE: read {len(gdf)} parcels for {la_name} "
                  f"in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  INSPIRE: failed to read {la_name}: {e}")
            return None
        # Reproject to WGS84
        try:
            gdf = gdf.to_crs(4326)
        except Exception:
            return None
        # Boundary lines (parcel edges)
        edges = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                b = geom.boundary
                if b.geom_type == "LineString":
                    edges.append(b)
                elif b.geom_type == "MultiLineString":
                    edges.extend(b.geoms)
            except Exception:
                continue
        return edges

    def n_edges(self) -> int:
        return len(self._edges)

    def snap_polygon(self, geom: BaseGeometry,
                       max_dist_m: float = 20.0,
                       loaded_only: bool = True) -> BaseGeometry:
        """Snap polygon vertices to nearest INSPIRE edge points within max_dist_m.

        Per-vertex: each polygon vertex is replaced by the nearest point on a
        nearby INSPIRE edge IF that point is within max_dist_m (using projected
        BNG coords for accurate metres). Vertices with no close edge stay put.
        Avoids shapely.ops.snap which mangles polygons against dense edge networks.
        """
        from pyproj import Transformer
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import transform as _shp_transform
        if geom is None or geom.is_empty or not self._edges:
            return geom
        try:
            c = geom.centroid
            if c.x == 0 and c.y == 0:
                return geom
        except Exception:
            return geom

        # NOTE 2026-05-07: tried area-skip guard at 0.5 km² — didn't help
        # because predicted polygons are mostly small (<0.1 km²). The +2/-2
        # trade-off is inherent to dense urban areas; further tuning of
        # tolerance / area / compactness guards converges to the same +2 net.
        # Accepted as +2 stable lever.

        # Transformers (cached on the instance for speed)
        if not hasattr(self, '_to_bng'):
            self._to_bng = Transformer.from_crs(4326, 27700, always_xy=True)
            self._to_wgs = Transformer.from_crs(27700, 4326, always_xy=True)
        to_bng = lambda x, y: self._to_bng.transform(x, y)
        to_wgs = lambda x, y: self._to_wgs.transform(x, y)

        geom_bng = _shp_transform(to_bng, geom)
        # Filter edges to those near the polygon's BNG bounds
        bbox = geom_bng.envelope.buffer(max_dist_m * 4).bounds
        # Re-project edges to BNG (cache per LA on first call)
        if not hasattr(self, '_edges_bng'):
            self._edges_bng = [_shp_transform(to_bng, e) for e in self._edges]
        nearby = [e for e in self._edges_bng
                  if not (e.bounds[2] < bbox[0] or e.bounds[0] > bbox[2] or
                          e.bounds[3] < bbox[1] or e.bounds[1] > bbox[3])]
        if not nearby:
            return geom

        # Union into one geometry for fast nearest-point queries
        from shapely.ops import unary_union as _uu
        try:
            edge_union_bng = _uu(nearby)
        except Exception:
            return geom

        from shapely.geometry import Point
        # Snap is disabled if FEWER than min_alignment_frac of vertices have a
        # close edge, because that means the polygon doesn't actually lie on
        # freehold parcel lines (it's rural / motorway / open land), and any
        # snap would just teleport vertices to random nearby parcels.
        min_alignment_frac = 0.5
        max_area_drift = 0.25  # reject if snapped area drifts > 25% from original

        def _snap_ring(coords):
            """Snap each coord to nearest edge point if within tol. Returns
            (snapped_coords, n_aligned, n_total)."""
            out_coords = []
            n_aligned = 0
            for x, y in coords:
                p = Point(x, y)
                d = p.distance(edge_union_bng)
                if 0 < d <= max_dist_m:
                    np_pt = edge_union_bng.interpolate(edge_union_bng.project(p))
                    out_coords.append((np_pt.x, np_pt.y))
                    n_aligned += 1
                else:
                    out_coords.append((x, y))
            return out_coords, n_aligned, len(coords)

        def _snap_poly_bng(poly):
            try:
                ext_snapped, n_a, n_t = _snap_ring(list(poly.exterior.coords))
                if n_t == 0 or n_a / n_t < min_alignment_frac:
                    return poly  # not aligned with freeholds → don't snap
                # Interior rings: also gate per-ring by alignment frac
                # (validator caught: previously snapped unconditionally)
                ints_snapped = []
                for r in poly.interiors:
                    snapped, n_a_r, n_t_r = _snap_ring(list(r.coords))
                    if n_t_r > 0 and n_a_r / n_t_r >= min_alignment_frac:
                        ints_snapped.append(snapped)
                    else:
                        ints_snapped.append(list(r.coords))  # keep original
                new_poly = Polygon(ext_snapped, ints_snapped)
                if not new_poly.is_valid:
                    new_poly = new_poly.buffer(0)
                if not new_poly.is_valid or new_poly.is_empty:
                    return poly
                # Reject if area drifted too far (snap distorted the shape)
                old_area = poly.area
                if old_area > 0:
                    drift = abs(new_poly.area - old_area) / old_area
                    if drift > max_area_drift:
                        return poly
                return new_poly
            except Exception:
                return poly

        try:
            if isinstance(geom_bng, Polygon):
                snapped_bng = _snap_poly_bng(geom_bng)
            elif isinstance(geom_bng, MultiPolygon):
                parts = [_snap_poly_bng(p) for p in geom_bng.geoms]
                snapped_bng = MultiPolygon([p for p in parts if not p.is_empty])
            else:
                return geom
            return _shp_transform(to_wgs, snapped_bng)
        except Exception:
            return geom


_HISTORIC_LA_ALIASES = {
    "south bedfordshire": "central bedfordshire",
    "mid bedfordshire": "central bedfordshire",
    "st albans": "st albans city",
    "st. albans": "st albans city",
}

# When admin name is a substring of LARGER unrelated stems, we must prefer
# the EXACT matching council. E.g. "Leicester" → Leicester_City_Council,
# NOT North_West_Leicestershire_District_Council.
_PREFER_EXACT = {
    "leicester": "Leicester_City_Council",
    "york": "City_of_York_Council",
    "lincoln": "City_of_Lincoln_Council",
    "manchester": "Manchester_City_Council",
    "newcastle": "Newcastle_upon_Tyne_City_Council",
    "cambridge": "Cambridge_City_Council",
    "oxford": "Oxford_City_Council",
}


def la_for_admin_region(admin_region: str) -> Optional[str]:
    """Map a pdf_info.admin_region (e.g. 'Dover') to an INSPIRE filename
    stem (e.g. 'Dover_District_Council'). Uses word-boundary matching so
    'Leicester' doesn't accidentally match 'Leicestershire'.
    """
    if not admin_region:
        return None
    # Normalize: strip punctuation, lowercase
    a = "".join(c for c in admin_region.lower() if c.isalnum() or c == " ").strip()
    if not a:
        return None
    # Apply alias map (historic LA renames)
    a = _HISTORIC_LA_ALIASES.get(a, a)
    # Direct override for known-ambiguous names
    if a in _PREFER_EXACT:
        stem = _PREFER_EXACT[a]
        if (INSPIRE_DIR / f"{stem}.zip").exists():
            return stem
    # Word-boundary-ish substring match
    import re
    candidates = []
    if INSPIRE_DIR.is_dir():
        a_re = re.escape(a)
        for f in INSPIRE_DIR.glob("*.zip"):
            stem_lc = f.stem.lower().replace("_", " ")
            # Must match as whole word: " Leicester " not "Leicestershire"
            if re.search(rf"(^|\s){a_re}(\s|$)", stem_lc):
                candidates.append(f.stem)
    # Prefer longer (more specific) where word-bounded matches exist
    candidates.sort(key=lambda s: -len(s))
    return candidates[0] if candidates else None


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python -m tools.snap.inspire <case>")
        sys.exit(1)
    case = sys.argv[1]
    cd = Path("/Users/fabiandegen/Documents/VSCODE/GeoMapAgent_autonomous"
              "/results/benchmark_v13/gemini-flash") / case
    if not (cd / "predicted.geojson").exists():
        print(f"no predicted.geojson for {case}"); sys.exit(2)
    pred = shape(json.load(open(cd/"predicted.geojson"))["geometry"])
    pi = json.load(open(cd/"pdf_info.json"))
    la = la_for_admin_region(pi.get("admin_region") or "")
    print(f"Admin region: {pi.get('admin_region')!r}  -> LA: {la}")
    if not la: sys.exit(3)
    snap_obj = InspireSnap([la])
    print(f"Loaded {snap_obj.n_edges()} INSPIRE edges from {la}")
    snapped = snap_obj.snap_polygon(pred, max_dist_m=20)
    print(f"Original area: {pred.area * 1e10:.4f} sq-deg×1e10  "
          f"snapped: {snapped.area * 1e10:.4f}")
    print(f"Cached IoU: {json.load(open(cd/'metrics.json')).get('iou')}")
