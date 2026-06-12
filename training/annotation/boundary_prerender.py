"""Pre-render the planning map at production DPI for every eval case.

  1. render_pdf_page(..., dpi=200)
  2. auto_rotate DISABLED — annotate the map as the PDF renders it,
     so we don't depend on the rotation classifier for the eval frame.

Title-block cropping was removed (the heuristic ate real map content);
a since-removed map-crop helper no longer exists. Pages are taken from the
reader's ``pdf_info.map_pages`` directly — manual page overrides used
to live in ``scripts/annotate_page_overrides.json`` but the reader now
produces correct ``map_pages`` for every case that used to need an
override.

The GT polygon is projected onto the rendered map via the cached
affine_H from an earlier benchmark run (AFFINE_SOURCES, newest first).
Because auto_rotate is off here but ON
in the cached benchmarks, the projection lands in the wrong place for
any case the classifier rotated. We detect that by checking whether
the projected polygon is sane (most coordinates inside the image
bounds); if not, we fall back to a centered, scaled-to-40% placement
so you can transform it manually in the UI.

Outputs per case under ``boundary_annotations/<case_id>/``:
  map.png            -- rendered map (production DPI, no crop)
  initial.json       -- polygon(s) in image-pixel coords + affine source tag
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from geoplanagent.tools.pdf import render_pdf_page


OUT_ROOT = REPO / "boundary_annotations"
EVAL_ROOT = REPO / "evaluation_data"

# Cached-affine source priority — pick the freshest run that has an affine_H.
AFFINE_SOURCES = ["benchmark_v20", "benchmark_v17"]


def _list_cases() -> List[str]:
    return sorted(
        c for c in os.listdir(EVAL_ROOT)
        if (EVAL_ROOT / c).is_dir() and not c.endswith(".xlsx")
    )


def _gt_geojson(case_id: str) -> Optional[Dict[str, Any]]:
    base = EVAL_ROOT / case_id
    for fn in os.listdir(base):
        if fn.endswith(".geojson"):
            try:
                return json.loads((base / fn).read_text())
            except Exception:
                pass
    return None


def _pdf_path(case_id: str) -> Optional[Path]:
    """Pick the PDF most likely to contain the planning map.

    Prefer (in order): filename contains 'map' → 'plan' → 'direction' →
    'boundary'; falls back to the largest PDF (text-only notices are
    typically a small fraction of a Plan document's size). A4Da2 was
    the motivating case: a 17 KB confirmation notice was being picked
    over a 420 KB "Article_4_Direction_Plan.pdf" sitting in the same
    folder.
    """
    base = EVAL_ROOT / case_id
    if not base.exists():
        return None
    pdfs = [base / fn for fn in os.listdir(base) if fn.lower().endswith(".pdf")]
    if not pdfs:
        return None
    if len(pdfs) == 1:
        return pdfs[0]
    for kw in ("map", "plan", "direction", "boundary"):
        hits = [p for p in pdfs if kw in p.name.lower()]
        if hits:
            return max(hits, key=lambda p: p.stat().st_size)
    return max(pdfs, key=lambda p: p.stat().st_size)


def _pdf_info(case_id: str) -> Optional[Dict[str, Any]]:
    """Pick up pdf_info from a previous benchmark run. Prefer a source
    whose map_pages is non-empty: an in-flight run has empty pdf_info for
    cases it hasn't reached yet, and falling back to the older run is the
    right call there."""
    best = None
    for src in AFFINE_SOURCES:
        p = REPO / "results" / src / "gemini-flash" / case_id / "pdf_info.json"
        if not p.exists():
            continue
        try: pi = json.loads(p.read_text())
        except Exception: continue
        if pi.get("map_pages"):  # non-empty → win
            return pi
        if best is None:  # fallback only if nothing better seen
            best = pi
    return best


def _cached_affine(case_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any], str]]:
    """Return (affine_H, tile_info, source_name) if any cached run has them."""
    for src in AFFINE_SOURCES:
        d = REPO / "results" / src / "gemini-flash" / case_id
        if (d / "affine_H.npy").exists() and (d / "tile_info.json").exists():
            try:
                H = np.load(d / "affine_H.npy")
                ti = json.loads((d / "tile_info.json").read_text())
                return H, ti, src
            except Exception:
                continue
    return None


def _latlon_to_tile_px(lat: float, lon: float, tile_info: Dict[str, Any]) -> Tuple[float, float]:
    """WGS84 → pixel in the OS tile canvas (used at match time).
    Web-Mercator slippy tile math; matches geoplanagent.tools.tiles convention."""
    import math
    z = tile_info["zoom"]
    tx_min = tile_info["tx_min"]
    ty_min = tile_info["ty_min"]
    tile_size = tile_info.get("tile_size", 256)
    n = 2.0 ** z
    x_tile = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_tile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    cx = (x_tile - tx_min) * tile_size
    cy = (y_tile - ty_min) * tile_size
    return cx, cy


def _project_gt_to_image(
    gt_geojson: Dict[str, Any],
    affine_H: np.ndarray,
    tile_info: Dict[str, Any],
) -> Optional[List[List[List[float]]]]:
    """Project GT polygons through INVERSE(affine_H) into image-pixel coords.

    affine_H maps map_px → tile_canvas_px. We want tile_canvas_px → map_px,
    which is the inverse of the 2×3 affine (treated as a 3×3 matrix).
    Returns a list of rings (each ring = list of [x, y] in image px).
    """
    A = np.vstack([affine_H, [0.0, 0.0, 1.0]])
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None

    rings: List[List[List[float]]] = []
    geom = gt_geojson.get("geometry") or gt_geojson
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    polys = []
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return None

    for poly in polys:
        # poly = list of rings; first is exterior, rest are interior
        for ring in poly:
            r = []
            for lon, lat in ring:
                tx, ty = _latlon_to_tile_px(lat, lon, tile_info)
                v = Ainv @ np.array([tx, ty, 1.0])
                r.append([float(v[0]), float(v[1])])
            rings.append(r)
    return rings if rings else None


def _projection_lands_inside_image(
    rings: List[List[List[float]]], w: int, h: int, min_frac: float = 0.5,
) -> bool:
    """Sanity: at least `min_frac` of the projected polygon's points must
    fall inside [0,w]×[0,h]. When the cached affine was computed in an
    auto-rotated frame but we're rendering raw, the projection will land
    mostly outside the image — detect that and use the centered fallback.
    """
    n_pts = sum(len(r) for r in rings)
    if n_pts == 0:
        return False
    inside = 0
    for ring in rings:
        for x, y in ring:
            if 0 <= x <= w and 0 <= y <= h:
                inside += 1
    return inside / n_pts >= min_frac


def _centered_initial(image_shape: Tuple[int, int],
                      gt_geojson: Dict[str, Any]) -> List[List[List[float]]]:
    """Fallback when no affine is available — place the GT polygon centered,
    scaled to ~40% of the image's smaller dimension, no rotation."""
    h, w = image_shape[:2]
    rings: List[List[List[float]]] = []
    geom = gt_geojson.get("geometry") or gt_geojson
    coords = geom.get("coordinates") or []
    polys = [coords] if geom.get("type") == "Polygon" else coords

    all_pts = []
    for poly in polys:
        for ring in poly:
            for lon, lat in ring:
                all_pts.append((lon, lat))
    if not all_pts:
        return []
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    gx_min, gx_max = min(xs), max(xs)
    gy_min, gy_max = min(ys), max(ys)
    gw = max(gx_max - gx_min, 1e-9)
    gh = max(gy_max - gy_min, 1e-9)
    target = 0.40 * min(w, h)
    scale = target / max(gw, gh)
    # Centre in image
    cx_img, cy_img = w / 2.0, h / 2.0
    gx_cen, gy_cen = (gx_min + gx_max) / 2.0, (gy_min + gy_max) / 2.0

    for poly in polys:
        for ring in poly:
            r = []
            for lon, lat in ring:
                # Note: image-y points DOWN, geographic-y points UP → flip y
                px = cx_img + (lon - gx_cen) * scale
                py = cy_img - (lat - gy_cen) * scale
                r.append([float(px), float(py)])
            rings.append(r)
    return rings


def render_one(case_id: str, force: bool = False) -> Dict[str, Any]:
    out_dir = OUT_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    map_png = out_dir / "map.png"
    init_json = out_dir / "initial.json"
    if map_png.exists() and init_json.exists() and not force:
        return {"case_id": case_id, "status": "cached"}

    pdf = _pdf_path(case_id)
    if pdf is None:
        return {"case_id": case_id, "status": "no_pdf"}
    pi = _pdf_info(case_id) or {}
    pages = pi.get("map_pages") or [1]
    page_idx = int(pages[0]) - 1

    try:
        # 200 DPI matches production benchmark_runner default.
        img = render_pdf_page(str(pdf), page_index=page_idx, dpi=200)
        # auto_rotate and title-block crop are intentionally OFF — annotate
        # the raw PDF frame. The crop heuristic ate too many real map regions
        # historically; user wants every case to render uncropped.
    except Exception as e:
        return {"case_id": case_id, "status": f"render_failed: {e!s:.80}"}

    cv2.imwrite(str(map_png), img)
    h, w = img.shape[:2]

    gt = _gt_geojson(case_id)
    if gt is None:
        init_json.write_text(json.dumps({
            "case_id": case_id,
            "image_size": [w, h],
            "rings": [],
            "affine_source": None,
            "note": "no GT geojson on disk",
        }, indent=2))
        return {"case_id": case_id, "status": "ok_no_gt"}

    cached = _cached_affine(case_id)
    rings: Optional[List[List[List[float]]]] = None
    src_tag = "centered_fallback"
    if cached is not None:
        H, ti, src = cached
        projected = _project_gt_to_image(gt, H, ti)
        if projected and _projection_lands_inside_image(projected, w, h):
            rings = projected
            src_tag = f"affine:{src}"
    if not rings:
        # Cached affine doesn't fit (likely auto_rotate was applied at match
        # time but we're rendering raw here). Drop polygon centered.
        rings = _centered_initial((h, w), gt)
        src_tag = "centered_fallback"

    init_json.write_text(json.dumps({
        "case_id": case_id,
        "image_size": [w, h],
        "rings": rings,
        "affine_source": src_tag,
    }, indent=2))
    return {"case_id": case_id, "status": "ok", "src": src_tag,
            "n_rings": len(rings)}


def main():
    force = "--force" in sys.argv
    cases = _list_cases()
    print(f"Pre-rendering {len(cases)} cases at production DPI=200...", flush=True)
    n_ok = n_skip = n_err = 0
    for i, c in enumerate(cases, 1):
        r = render_one(c, force=force)
        s = r["status"]
        if s.startswith("ok"): n_ok += 1
        elif s == "cached":   n_skip += 1
        else:                 n_err += 1
        if i % 25 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] ok={n_ok} cached={n_skip} err={n_err}",
                  flush=True)
        if not s.startswith(("ok", "cached")):
            print(f"    !! {c}: {s}", flush=True)
    print(f"\nDone. Outputs under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
