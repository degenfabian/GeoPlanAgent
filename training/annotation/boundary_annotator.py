"""Flask backend for the boundary annotation UI.

Run after `training/annotation/boundary_prerender.py` has produced the
per-case map images and initial polygon coordinates.

Endpoints
---------
GET  /                              -> the annotation HTML page
GET  /api/cases                     -> JSON [{case_id, status, has_edit}]
GET  /api/case/<case_id>/map.png    -> rendered map PNG
GET  /api/case/<case_id>/initial    -> initial polygon coords (image px)
GET  /api/case/<case_id>/edited     -> latest edited polygon if any
POST /api/case/<case_id>/save       -> body {rings: [[[x,y],…], …]}
                                       writes edited.json + edited_mask.png

The "edited" geojson is written in IMAGE-PIXEL space, not WGS84. A later
script `annotate_export.py` will project image-pixel rings back to WGS84
via the same affine that was used to project them in.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, abort

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

WORK = REPO / "boundary_annotations"
UI_HTML = Path(__file__).resolve().parent / "boundary_annotator_ui.html"

app = Flask(__name__)


@app.get("/")
def index():
    return send_file(str(UI_HTML))


@app.get("/api/cases")
def list_cases():
    if not WORK.exists():
        return jsonify({"error": "Run training/annotation/boundary_prerender.py first"}), 500
    cases = []
    for d in sorted(WORK.iterdir()):
        if not d.is_dir(): continue
        init_p = d / "initial.json"
        if not init_p.exists(): continue
        cases.append({
            "case_id": d.name,
            "has_edit": (d / "edited.json").exists(),
            "has_map":  (d / "map.png").exists(),
        })
    return jsonify(cases)


def _case_dir(case_id: str) -> Path:
    d = WORK / case_id
    if not d.is_dir():
        abort(404)
    return d


@app.get("/api/case/<path:case_id>/map.png")
def case_map(case_id):
    d = _case_dir(case_id)
    p = d / "map.png"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/png")


@app.get("/api/case/<path:case_id>/initial")
def case_initial(case_id):
    d = _case_dir(case_id)
    p = d / "initial.json"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="application/json")


@app.get("/api/case/<path:case_id>/edited")
def case_edited(case_id):
    d = _case_dir(case_id)
    p = d / "edited.json"
    if not p.exists():
        return jsonify({"rings": None})
    return send_file(str(p), mimetype="application/json")


@app.post("/api/case/<path:case_id>/save")
def save_edit(case_id):
    d = _case_dir(case_id)
    payload = request.get_json(force=True, silent=True) or {}
    rings = payload.get("rings")
    if not isinstance(rings, list):
        return jsonify({"error": "rings must be a list"}), 400

    # Save the JSON
    out_json = d / "edited.json"
    out_json.write_text(json.dumps({
        "case_id": case_id,
        "rings": rings,
    }, indent=2))

    # Rasterise to a mask PNG at the same size as map.png
    map_path = d / "map.png"
    if map_path.exists() and rings:
        img = cv2.imread(str(map_path))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ring in rings:
            if not ring or len(ring) < 3: continue
            pts = np.array([[int(round(x)), int(round(y))] for x, y in ring],
                           dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        cv2.imwrite(str(d / "edited_mask.png"), mask)

    return jsonify({"ok": True, "n_rings": len(rings)})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"Serving annotation UI on http://{args.host}:{args.port}/")
    print(f"Reads/writes under: {WORK}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
