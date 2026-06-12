"""One-time script: build a global UK road-name → (tile, bbox_BNG, centroid_BNG)
index from all 56 OS Open Map Local zip files.

Output: geoplanagent/oml_road_index.json (~50-200 MB).

This index lets us do O(1) lookup by road name with disambiguation by
LA bbox — fixing the homonym problem in v3's locate cascade.

Each entry:
  road_name_lower → [{
      "tile": "TL",
      "minx": E, "miny": N, "maxx": E, "maxy": N,
      "centroid": [E, N],
      "classification": "Local Road" | "Minor Road" | "B Road" | ...
  }, ...]

Run once. Subsequent v3 calls just read the JSON.
"""
from __future__ import annotations
import json
import time
import zipfile
from pathlib import Path
from collections import defaultdict
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OML_DIR = REPO / "os_opendata" / "open_map_local"
OUT = HERE.parent / "geoplanagent" / "oml_road_index.json"


def main():
    if not OML_DIR.exists():
        print(f"OML dir missing: {OML_DIR}"); return
    zips = sorted(OML_DIR.glob("opmplc_essh_*.zip"))
    # Skip the GB-wide aggregate (opmplc_essh_gb.zip) since per-tile zips
    # together cover everything.
    zips = [z for z in zips if z.name != "opmplc_essh_gb.zip"]
    print(f"Found {len(zips)} per-letter tiles")

    import shapefile
    index = defaultdict(list)
    n_total_roads = 0
    n_named = 0
    t0 = time.time()
    for i, zp in enumerate(zips):
        tile_letter = zp.stem.replace("opmplc_essh_", "").upper()
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zp) as zf:
                # Find the Road shapefile members
                shp_members = [n for n in zf.namelist()
                               if n.endswith("_Road.shp")
                               or n.endswith("_Road.dbf")
                               or n.endswith("_Road.shx")
                               or n.endswith("_Road.prj")]
                if not shp_members: continue
                zf.extractall(td, members=shp_members)
            # Find the extracted .shp
            shp_paths = list(Path(td).rglob("*_Road.shp"))
            if not shp_paths: continue
            shp = shp_paths[0]
            try:
                # OS shapefiles use ISO-8859-1 (Latin-1) encoding
                r = shapefile.Reader(str(shp), encoding="latin-1")
            except Exception as e:
                print(f"  {tile_letter}: read fail {e!s:.50}"); continue
            n_in_tile = 0
            for rec, shp_geom in zip(r.iterRecords(), r.iterShapes()):
                d = rec.as_dict()
                name = (d.get("DISTNAME") or "").strip()
                if not name:
                    n_in_tile += 1; continue
                n_named += 1
                pts = shp_geom.points
                if not pts: continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                index[name.lower()].append({
                    "tile": tile_letter,
                    "name": name,
                    "minx": minx, "miny": miny,
                    "maxx": maxx, "maxy": maxy,
                    "cx": cx, "cy": cy,
                    "cls": d.get("CLASSIFICA", "") or "",
                })
                n_in_tile += 1
            n_total_roads += n_in_tile
        print(f"  {i+1}/{len(zips)} {tile_letter}: {n_in_tile} roads "
              f"(named so far: {n_named}, total: {n_total_roads}, "
              f"wall: {time.time()-t0:.0f}s)", flush=True)

    print(f"\nTotal roads scanned: {n_total_roads:,}")
    print(f"Named roads: {n_named:,}")
    print(f"Distinct road names: {len(index):,}")
    # Save
    OUT.write_text(json.dumps(dict(index)))
    sz_mb = OUT.stat().st_size / 1e6
    print(f"Saved {OUT}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
