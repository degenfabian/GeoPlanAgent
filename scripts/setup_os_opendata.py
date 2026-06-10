"""One-command setup: download + extract every OS OpenData product the
pipeline reads into ``os_opendata/``.

    uv run scripts/setup_os_opendata.py              # everything (~19 GB on disk)
    uv run scripts/setup_os_opendata.py --main-only  # just Open Names + Zoomstack
    uv run scripts/setup_os_opendata.py --list       # show what would be fetched

All five products are OS OpenData (Open Government Licence v3): free, no API
key, no rate limit. The script asks the public OS Downloads API for the
current download URLs, so it keeps working even if OS re-hosts the files.
Downloads resume (curl ``-C -``) and every step is idempotent — re-running
skips whatever is already in place, so an interrupted run is safe to repeat.

Required for the main benchmark:  Open Names, Open Zoomstack.
Ablation-only (locate all-tools):  Code-Point Open, BoundaryLine, OpenMap Local.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from geoplanagent.paths import (  # noqa: E402
    OS_OPENDATA_DIR,
    OS_ZOOMSTACK_GPKG,
    OML_ROAD_INDEX,
    OML_ROAD_GEOM,
)

API = "https://api.os.uk/downloads/v1/products/{product}/downloads"
# Persistent staging for in-flight zips so curl -C - can resume across runs.
STAGING = OS_OPENDATA_DIR / "_downloads"


def _has_glob(root: Path, pattern: str) -> bool:
    """True when `root` exists and holds at least one file matching `pattern`."""
    return root.exists() and any(root.glob(pattern))


# Each dataset names the OS Downloads API product + the format/area to pull,
# where its files land, and a "done" predicate that reports whether it's already
# set up (so re-runs skip it). "ablation" marks the three datasets only the
# ablation experiments need — the main benchmark never touches them.
DATASETS = {
    "open_names": {
        "title": "OS Open Names",
        "product": "OpenNames",
        "format": "CSV",
        "ablation": False,
        # Zip has no top-level csv/ wrapper, but the geocoder reads
        # open_names/csv/Data/*.csv — so extract into the csv/ subdir.
        "extract_to": OS_OPENDATA_DIR / "open_names" / "csv",
        "done": lambda: _has_glob(OS_OPENDATA_DIR / "open_names" / "csv" / "Data", "*.csv"),
    },
    "zoomstack": {
        "title": "OS Open Zoomstack",
        "product": "OpenZoomstack",
        "format": "GeoPackage",
        "ablation": False,
        # Ships as a zip containing the .gpkg; we lift the gpkg to this path.
        "gpkg_to": OS_ZOOMSTACK_GPKG,
        "done": lambda: OS_ZOOMSTACK_GPKG.exists(),
    },
    "code_point": {
        "title": "Code-Point Open",
        "product": "CodePointOpen",
        "format": "CSV",
        "ablation": True,
        # Same csv/-wrapper trick: geocoder reads code_point_open/csv/Data/CSV/.
        "extract_to": OS_OPENDATA_DIR / "code_point_open" / "csv",
        "done": lambda: _has_glob(
            OS_OPENDATA_DIR / "code_point_open" / "csv" / "Data" / "CSV", "*.csv"
        ),
    },
    "boundary_line": {
        "title": "BoundaryLine",
        "product": "BoundaryLine",
        "format": "ESRI® Shapefile",
        "ablation": True,
        # Loader rglobs for *.shp, so the wrapper layout doesn't matter.
        "extract_to": OS_OPENDATA_DIR / "boundary_line",
        "done": lambda: _has_glob(OS_OPENDATA_DIR / "boundary_line", "**/*.shp"),
    },
    "open_map_local": {
        "title": "OS OpenMap Local",
        "product": "OpenMapLocal",
        "format": "ESRI® Shapefile",
        "ablation": True,
        # Special: download every per-National-Grid-tile zip (not the GB
        # aggregate), leave them zipped, then build the road lookups.
        "oml": True,
        "done": lambda: OML_ROAD_INDEX.exists() and OML_ROAD_GEOM.exists(),
    },
}


def _api_downloads(product: str):
    """Return the OS Downloads API listing for a product (list of dicts with
    format/area/fileName/url/size)."""
    url = API.format(product=product)
    try:
        out = subprocess.run(
            ["curl", "-fsSL", "--retry", "3", url],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as error:
        # capture_output swallows curl's stderr — resurface the diagnostic.
        raise RuntimeError(
            f"OS Downloads API request for {product} failed: "
            f"{error.stderr.strip() or error}"
        ) from error
    return json.loads(out.stdout)


def _curl_download(url: str, dest: Path) -> None:
    """Stream a URL to dest with resume + retry. curl shows its own progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading -> {dest.name}")
    subprocess.run(
        ["curl", "-fSL", "--retry", "3", "-C", "-", "-o", str(dest), url],
        check=True,
    )


def _extract_into_place(zip_path: Path, final_dir: Path) -> None:
    """Extract so final_dir only ever appears fully populated: unpack into a
    sibling temp dir, then atomically swap it into place. An interrupted
    extract leaves only the temp (cleaned up) — never a half-filled final_dir
    that the dataset's "done" predicate would mistake for a complete dataset."""
    final_dir = Path(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=final_dir.parent, prefix=f".{final_dir.name}.tmp-"))
    print(f"    extracting  -> {final_dir}")
    try:
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(staging)
        if final_dir.exists():
            # Reachable when a previous extract completed but the "done"
            # predicate still failed (layout change upstream): final_dir holds
            # unusable files, and os.replace can't rename onto a non-empty
            # dir — clear it so re-runs recover without manual cleanup.
            shutil.rmtree(final_dir)
        os.replace(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _staged_download(entry: dict) -> Path:
    """Download an API entry into STAGING, skipping if a full copy is already
    there and resuming a partial one. Returns the local zip path."""
    STAGING.mkdir(parents=True, exist_ok=True)
    zip_path = STAGING / entry["fileName"]
    if zip_path.exists() and zip_path.stat().st_size == entry.get("size"):
        print(f"    already downloaded -> {zip_path.name}")
    else:
        _curl_download(entry["url"], zip_path)
    return zip_path


def _setup_standard(spec: dict) -> None:
    """Download one GB-wide product and extract (or lift the gpkg) into place."""
    entry = next(
        (candidate for candidate in _api_downloads(spec["product"])
         if candidate.get("format") == spec["format"] and candidate.get("area") == "GB"),
        None,
    )
    if entry is None:
        raise RuntimeError(
            f"No '{spec['format']}' GB download offered for {spec['product']}"
        )
    zip_path = _staged_download(entry)
    if spec.get("gpkg_to"):
        # Zoomstack: pull the .gpkg out of the zip to its canonical path.
        with tempfile.TemporaryDirectory(dir=str(OS_OPENDATA_DIR)) as temp_dir:
            print(f"    extracting  -> {spec['gpkg_to']}")
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(temp_dir)
            gpkgs = list(Path(temp_dir).rglob("*.gpkg"))
            if len(gpkgs) != 1:
                raise RuntimeError(
                    f"Expected exactly one .gpkg inside {entry['fileName']}, found {len(gpkgs)}"
                )
            spec["gpkg_to"].parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(gpkgs[0]), str(spec["gpkg_to"]))
    else:
        _extract_into_place(zip_path, spec["extract_to"])
    # Confirm the data actually landed where the pipeline (and done()) looks
    # before deleting the only copy — otherwise a layout change upstream would
    # silently loop: extract, delete, re-download, repeat.
    if not spec["done"]():
        raise RuntimeError(
            f"{spec['title']}: extracted {entry['fileName']} but expected files "
            f"are missing — the OS package layout may have changed. Left the zip "
            f"in {STAGING} so a re-run won't re-download."
        )
    zip_path.unlink()  # reclaim space once the data is confirmed in place


def _build_oml_road_index() -> None:
    """Build the UK road-name lookups from the per-tile OpenMap Local zips.

    Reads every ``opmplc_essh_*.zip`` next to the outputs (one per National Grid
    tile) and writes, into ``os_opendata/open_map_local/``:
      oml_road_index.json — name -> bbox/centroid, for road() name lookups.
      oml_road_geom.json  — name -> LineString geometry, for intersect().

    This is only required for the tools in the ablation experiments.
    """
    oml_dir = OML_ROAD_INDEX.parent  # inputs live next to the index we write
    # Skip the GB-wide aggregate; the per-tile zips together cover everything.
    zips = [zip_path for zip_path in sorted(oml_dir.glob("opmplc_essh_*.zip"))
            if zip_path.name != "opmplc_essh_gb.zip"]
    if not zips:
        raise RuntimeError(f"No opmplc_essh_*.zip tiles under {oml_dir} — download step incomplete?")
    print(f"    found {len(zips)} National Grid tiles")

    import shapefile  # pyshp; only needed for this ablation build

    index = defaultdict(list)
    geom = defaultdict(list)
    n_total_roads = n_named = 0
    t0 = time.time()
    for i, zip_path in enumerate(zips):
        tile_code = zip_path.stem.replace("opmplc_essh_", "").upper()
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path) as zip_file:
                shp_members = [
                    member for member in zip_file.namelist()
                    if member.endswith(("_Road.shp", "_Road.dbf", "_Road.shx", "_Road.prj"))
                ]
                if not shp_members:
                    print(f"      {tile_code}: no _Road layer (all-sea square?), skipped")
                    continue
                zip_file.extractall(temp_dir, members=shp_members)
            shp_paths = list(Path(temp_dir).rglob("*_Road.shp"))
            if not shp_paths:
                print(f"      {tile_code}: _Road members extracted but no .shp found, skipped")
                continue
            try:
                # OS shapefiles use ISO-8859-1 (Latin-1) encoding.
                reader = shapefile.Reader(str(shp_paths[0]), encoding="latin-1")
            except Exception as error:
                print(f"      {tile_code}: read fail {error!s:.50}")
                continue
            n_in_tile = 0
            for record, shp_geom in zip(reader.iterRecords(), reader.iterShapes()):
                n_in_tile += 1
                fields = record.as_dict()
                name = (fields.get("DISTNAME") or "").strip()
                if not name:
                    continue  # unnamed road — useless for a name lookup
                points = shp_geom.points
                if not points:
                    continue
                n_named += 1
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                index[name.lower()].append({
                    "tile": tile_code, "name": name,
                    "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                    "cx": cx, "cy": cy, "cls": fields.get("CLASSIFICA", "") or "",
                })
                # Same identity + the LineString itself, for intersect(). Points
                # rounded to 0.1 m (BNG) — far finer than junction precision
                # needs, and it trims the file.
                geom[name.lower()].append({
                    "name": name, "tile": tile_code,
                    "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                    "points": [[round(px, 1), round(py, 1)] for px, py in points],
                })
            n_total_roads += n_in_tile
        print(f"      {i + 1}/{len(zips)} {tile_code}: {n_in_tile} roads "
              f"(named so far: {n_named}, total: {n_total_roads}, "
              f"wall: {time.time() - t0:.0f}s)", flush=True)

    print(f"    named roads: {n_named:,}; distinct road names: {len(index):,}")
    if not n_named:
        raise RuntimeError("No named roads harvested — refusing to write empty road lookups")
    for label, data, path in (("index", index, OML_ROAD_INDEX), ("geom", geom, OML_ROAD_GEOM)):
        # Write-then-rename so a crash mid-write can't leave a truncated file
        # that done() would mistake for a complete lookup.
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(json.dumps(dict(data)))
        os.replace(temp_path, path)
        print(f"    saved {label}: {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def _setup_oml(spec: dict) -> None:
    """OpenMap Local: fetch every per-tile shapefile zip, then build the road
    lookups the road()/intersect() geocoders read.
    """
    oml_dir = OML_ROAD_INDEX.parent  # downloads land next to the index outputs
    tiles = [
        entry for entry in _api_downloads(spec["product"])
        if entry.get("format") == spec["format"] and entry.get("area") != "GB"
    ]
    if not tiles:
        raise RuntimeError(f"No per-tile '{spec['format']}' downloads offered for {spec['product']}")
    print(f"    {len(tiles)} National Grid tiles (skipping the GB aggregate)")
    oml_dir.mkdir(parents=True, exist_ok=True)
    for entry in tiles:
        dest = oml_dir / entry["fileName"]
        if dest.exists() and dest.stat().st_size == entry.get("size"):
            continue  # already have this tile at the expected size
        _curl_download(entry["url"], dest)
    print("    building road index (oml_road_index.json + oml_road_geom.json)…")
    _build_oml_road_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-only", action="store_true",
                        help="only the two products the main benchmark needs")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS),
                        help="explicit subset to fetch (overrides --main-only)")
    parser.add_argument("--list", action="store_true",
                        help="print the plan and exit without downloading")
    args = parser.parse_args()

    if args.datasets:
        names = args.datasets
    elif args.main_only:
        names = [key for key, spec in DATASETS.items() if not spec["ablation"]]
    else:
        names = list(DATASETS)

    if not args.list:  # a dry run shouldn't touch the filesystem
        OS_OPENDATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target: {OS_OPENDATA_DIR}\n")
    for name in names:
        spec = DATASETS[name]
        tier = "ablation" if spec["ablation"] else "required"
        if spec["done"]():
            print(f"[skip] {spec['title']} ({tier}) — already present")
            continue
        if args.list:
            print(f"[plan] {spec['title']} ({tier}) — {spec['product']}/{spec['format']}")
            continue
        print(f"[get ] {spec['title']} ({tier})")
        if spec.get("oml"):
            _setup_oml(spec)
        else:
            _setup_standard(spec)
        print(f"[done] {spec['title']}\n")

    if not args.list:
        print("OS OpenData setup complete.")


if __name__ == "__main__":
    main()
