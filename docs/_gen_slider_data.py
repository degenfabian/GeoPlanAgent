"""Generate real sliding-window data for the docs/ demo slider.

OFFLINE — no LLM calls, no API credits. Uses MINIMA-LoFTR + the OS tile
cache only. Output lands under docs/assets/slider_data/.

Run from the project root:
    uv run docs/_gen_slider_data.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoplanagent.tools.pdf import render_map_page  # noqa: E402
from geoplanagent.tools.tiles import fetch_os_opendata_grid  # noqa: E402
from geoplanagent.tools.matching import (  # noqa: E402
    WINDOW_STRIDE_TARGET,
    estimate_affine,
    load_minima,
    run_minima,
)

# Case selection
# 12:00116:ART4 — the README's example case. Strong match (834 inliers,
# IoU 0.94 in the v_post_fix benchmark), 7×9 tile canvas at z17.
CASE = "12:00116:ART4"
RESULTS_DIR = ROOT / "results" / "benchmark_std_post_fix" / "gemini-flash" / CASE
# Find the PDF inside the case directory (filename varies per case)
_case_dir = ROOT / "evaluation_data" / CASE
_pdfs = list(_case_dir.glob("*.pdf"))
if not _pdfs:
    raise SystemExit(f"No PDF found in {_case_dir}")
PDF_PATH = _pdfs[0]
OUT_DIR = ROOT / "docs" / "assets" / "slider_data"

# Load cached metadata
metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
pdf_info = json.loads((RESULTS_DIR / "pdf_info.json").read_text())
tile_info_cached = json.loads((RESULTS_DIR / "tile_info.json").read_text())

mi = metrics["match_info"]
anchor_lat, anchor_lon = mi["anchor_latlon"]
zoom = mi["zoom"]
scale_factor_cached = mi["scale_factor"]
print(f"Case {CASE}")
print(f"  Anchor (lat, lon)  = ({anchor_lat:.5f}, {anchor_lon:.5f})")
print(f"  Cached zoom        = {zoom}")
print(f"  Cached scale       = {scale_factor_cached}")
print(f"  Cached best window = ({mi['window'][0]}, {mi['window'][1]})")
print(f"  Cached n_inliers   = {mi['n_inliers']}")

# Render the map page
map_page = pdf_info["map_pages"][0]
print(f"\nRendering page {map_page} of {PDF_PATH.name} at 200 DPI...")
rendered = render_map_page(str(PDF_PATH), int(map_page), dpi=200, case_name=CASE)
if rendered is None:
    raise SystemExit(f"render_map_page returned None for page {map_page}")
map_img, rot_info = rendered
print(f"  raw map: {map_img.shape[1]}x{map_img.shape[0]} (rotation_applied={rot_info.get('applied')})")

# Load MINIMA
print("\nLoading MINIMA-LoFTR weights...")
matcher = load_minima()
print("  loaded.")

# Fetch OS tile canvas
nx, ny = tile_info_cached["nx"], tile_info_cached["ny"]
print(f"\nFetching {nx}x{ny} OS Open Zoomstack tiles at z{zoom}...")
tile_info = fetch_os_opendata_grid(anchor_lat, anchor_lon, zoom, nx, ny)
os_canvas = tile_info["image"]
ch, cw = os_canvas.shape[:2]
print(f"  canvas: {cw}x{ch}")

# Resize map to match the cached scale
# We force the cached scale so the sliding-window grid matches what
# actually ran in production for this case.
new_w = int(round(map_img.shape[1] * scale_factor_cached))
new_h = int(round(map_img.shape[0] * scale_factor_cached))
interp = cv2.INTER_AREA if scale_factor_cached < 1 else cv2.INTER_CUBIC
resized_map = cv2.resize(map_img, (new_w, new_h), interpolation=interp)
rh, rw = resized_map.shape[:2]
print(f"  resized map: {rw}x{rh}  (scale_factor={scale_factor_cached})")

if rh >= ch or rw >= cw:
    raise SystemExit(
        f"Resized map ({rw}x{rh}) does not fit in canvas ({cw}x{ch}); "
        "pick a different case or zoom."
    )

# Compute the stride the same way sliding_window_position does
_area_available = max(1, (ch - rh) * (cw - rw))
_target_stride = int(math.sqrt(_area_available / WINDOW_STRIDE_TARGET))
step_x = max(32, min(_target_stride, max(1, cw - rw)))
step_y = max(32, min(_target_stride, max(1, ch - rh)))
xs = list(range(0, cw - rw + 1, step_x))
ys = list(range(0, ch - rh + 1, step_y))
print(f"  stride: {step_x}x{step_y}  ⇒  {len(xs)}x{len(ys)} = {len(xs) * len(ys)} windows")

# Run sliding window MINIMA
windows: list[dict] = []
best_n = 0
best_meta: dict | None = None
for iy, wy in enumerate(ys):
    for ix, wx in enumerate(xs):
        window_img = os_canvas[wy:wy + rh, wx:wx + rw]
        mkpts0, mkpts1, mconf = run_minima(matcher, resized_map, window_img)
        H, n_inliers, score, inlier_mask = estimate_affine(mkpts0, mkpts1, mconf=mconf)

        avg_scale = None
        if H is not None:
            a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
            sx = math.sqrt(a * a + c * c)
            sy = math.sqrt(b * b + d * d)
            avg_scale = float((sx + sy) / 2.0)

        rec = {
            "x": int(wx), "y": int(wy), "w": int(rw), "h": int(rh),
            "n_inliers": int(n_inliers),
            "avg_scale": round(avg_scale, 4) if avg_scale is not None else None,
        }
        windows.append(rec)

        if n_inliers > best_n:
            best_n = n_inliers
            # For the best window, also stash the keypoints so we can draw
            # MINIMA correspondences in the demo.
            inl = inlier_mask.ravel().astype(bool) if inlier_mask is not None else None
            best_meta = {
                "x": int(wx), "y": int(wy),
                "n_inliers": int(n_inliers),
                # Keep ALL matches (so the UI can show outliers in grey + inliers bright)
                "mkpts0":      mkpts0.tolist() if mkpts0 is not None else [],
                "mkpts1":      mkpts1.tolist() if mkpts1 is not None else [],
                "inlier_mask": [int(v) for v in inl.tolist()] if inl is not None else [],
            }
        print(
            f"  ({wx:4d},{wy:4d}) "
            f"[{ix + 1}/{len(xs)} col, {iy + 1}/{len(ys)} row]  "
            f"n_inliers={n_inliers}",
            flush=True,
        )

# Save outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Resized map: keep as BGR (cv2 writes BGR). The OS canvas is RGB (per
# fetch_os_opendata_grid), so convert to BGR before writing.
cv2.imwrite(str(OUT_DIR / "resized_map.png"), resized_map)
cv2.imwrite(str(OUT_DIR / "tile_canvas.png"), cv2.cvtColor(os_canvas, cv2.COLOR_RGB2BGR))

if best_meta is not None and best_meta["mkpts0"]:
    # Prefer inliers, then top-scoring outliers, capped to 80 total
    inl = best_meta["inlier_mask"]
    inlier_idx = [i for i, v in enumerate(inl) if v == 1]
    outlier_idx = [i for i, v in enumerate(inl) if v == 0]
    keep_idx = (inlier_idx[: min(64, len(inlier_idx))]
                + outlier_idx[: max(0, 80 - min(64, len(inlier_idx)))])
    best_meta["mkpts0"] = [best_meta["mkpts0"][i] for i in keep_idx]
    best_meta["mkpts1"] = [best_meta["mkpts1"][i] for i in keep_idx]
    best_meta["inlier_mask"] = [best_meta["inlier_mask"][i] for i in keep_idx]

payload = {
    "case": CASE,
    "zoom": zoom,
    "scale_factor": scale_factor_cached,
    "canvas_w": cw, "canvas_h": ch,
    "map_w": rw, "map_h": rh,
    "anchor_latlon": [anchor_lat, anchor_lon],
    "windows": windows,
    "best_window": best_meta,
}
(OUT_DIR / "windows.json").write_text(json.dumps(payload, indent=2))

print("\nDone.")
print(f"  Best window: ({best_meta['x']},{best_meta['y']}) "
      f"n_inliers={best_meta['n_inliers']}")
print(f"  Wrote {OUT_DIR}/resized_map.png  ({rw}x{rh})")
print(f"  Wrote {OUT_DIR}/tile_canvas.png  ({cw}x{ch})")
print(f"  Wrote {OUT_DIR}/windows.json     ({len(windows)} windows)")
