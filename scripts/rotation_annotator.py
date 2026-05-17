"""Quick web app to annotate corrective rotations for boundary_annotations/.

For each case's `map.png`, click the corrective rotation needed to make the
map upright (e.g. if the map appears rotated 90° clockwise, click 270 — the
amount you'd rotate it CW to undo). Annotations are saved to
`rotation_annotations.json` at repo root after every click. The dataset is
read-only: only map.png files are read, nothing is written under
boundary_annotations/.

Run:
    uv run scripts/rotation_annotator.py

Then open http://127.0.0.1:5000 in a browser.

Keyboard shortcuts: 0 / 1 / 2 / 3 for 0° / 90° / 180° / 270°, S to skip,
LeftArrow to go back, RightArrow to skip forward without annotating.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from flask import Flask, abort, redirect, request, send_file, url_for

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "boundary_annotations"
ANNOTATIONS_FILE = REPO / "rotation_annotations.json"

app = Flask(__name__)


def list_cases() -> list[str]:
    """All case directories under boundary_annotations/, sorted."""
    if not DATASET_DIR.exists():
        return []
    cases = []
    for p in sorted(DATASET_DIR.iterdir()):
        if p.is_dir() and (p / "map.png").exists():
            cases.append(p.name)
    return cases


def load_annotations() -> dict:
    if ANNOTATIONS_FILE.exists():
        try:
            return json.loads(ANNOTATIONS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_annotations(data: dict) -> None:
    ANNOTATIONS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))


PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rotation annotator — {case_name}</title>
<style>
  body {{ margin: 0; padding: 0; font-family: -apple-system, system-ui, sans-serif;
          background: #1a1a1a; color: #f0f0f0; display: flex; flex-direction: column;
          min-height: 100vh; }}
  .topbar {{ padding: 10px 16px; background: #2a2a2a; border-bottom: 1px solid #444;
             display: flex; justify-content: space-between; align-items: center;
             flex-wrap: wrap; gap: 10px; }}
  .topbar h1 {{ margin: 0; font-size: 14px; font-weight: 500; }}
  .topbar .meta {{ font-size: 12px; opacity: 0.7; font-family: monospace; }}
  .topbar .progress {{ font-size: 12px; }}
  .topbar .prev {{ color: #ccc; text-decoration: none; padding: 4px 10px;
                   background: #333; border-radius: 3px; }}
  .topbar .prev:hover {{ background: #444; }}
  .img-wrap {{ flex: 1; display: flex; align-items: center; justify-content: center;
               padding: 16px; overflow: hidden; }}
  .img-wrap img {{ max-width: 100%; max-height: 80vh; object-fit: contain;
                   box-shadow: 0 4px 20px rgba(0,0,0,0.5); background: white; }}
  .btnbar {{ display: flex; gap: 8px; padding: 12px 16px; background: #2a2a2a;
             border-top: 1px solid #444; flex-wrap: wrap; justify-content: center; }}
  .btnbar form {{ display: contents; }}
  .btn {{ padding: 14px 22px; font-size: 16px; font-weight: 500;
          background: #3a5a8a; color: white; border: none; border-radius: 4px;
          cursor: pointer; min-width: 100px; }}
  .btn:hover {{ background: #4a6a9a; }}
  .btn.skip {{ background: #6a4a4a; }}
  .btn.skip:hover {{ background: #7a5a5a; }}
  .btn .kbd {{ display: inline-block; opacity: 0.6; font-size: 11px;
               margin-left: 6px; padding: 1px 5px; border: 1px solid #fff5;
               border-radius: 3px; }}
  .legend {{ padding: 8px 16px; font-size: 12px; background: #222; color: #888;
             text-align: center; }}
  .done {{ padding: 60px; text-align: center; }}
  .done h2 {{ color: #6c6; }}
</style>
</head>
<body>
  <div class="topbar">
    <h1>Corrective rotation annotator</h1>
    <span class="meta">{case_name}</span>
    <span class="progress">{cur_idx}/{total}  ({done} annotated, {remaining} left)</span>
    <a class="prev" href="{prev_url}">&laquo; Prev</a>
  </div>
  <div class="img-wrap">
    <img src="{img_url}" alt="map.png for {case_name}">
  </div>
  <div class="btnbar">
    <form method="post" action="/annotate"><input type="hidden" name="case" value="{case_name}"><input type="hidden" name="rot" value="0"><button class="btn" type="submit" title="0 = already upright">0&deg; (upright)<span class="kbd">0</span></button></form>
    <form method="post" action="/annotate"><input type="hidden" name="case" value="{case_name}"><input type="hidden" name="rot" value="90"><button class="btn" type="submit">90&deg; CW<span class="kbd">1</span></button></form>
    <form method="post" action="/annotate"><input type="hidden" name="case" value="{case_name}"><input type="hidden" name="rot" value="180"><button class="btn" type="submit">180&deg;<span class="kbd">2</span></button></form>
    <form method="post" action="/annotate"><input type="hidden" name="case" value="{case_name}"><input type="hidden" name="rot" value="270"><button class="btn" type="submit">270&deg; CW<span class="kbd">3</span></button></form>
    <form method="post" action="/annotate"><input type="hidden" name="case" value="{case_name}"><input type="hidden" name="rot" value="skip"><button class="btn skip" type="submit">Skip<span class="kbd">S</span></button></form>
  </div>
  <div class="legend">
    Click the CORRECTIVE rotation — if the map appears rotated 90&deg; CW, click 270&deg; (to undo). Current annotation: <b>{existing}</b>.
    Shortcuts: 0 1 2 3 = rotations, S = skip, ArrowLeft = prev, ArrowRight = skip next.
  </div>

<script>
  function submitRot(rot) {{
    const f = document.createElement('form');
    f.method = 'POST'; f.action = '/annotate';
    for (const [k,v] of [['case', '{case_name}'], ['rot', rot]]) {{
      const i = document.createElement('input'); i.type='hidden'; i.name=k; i.value=v; f.appendChild(i);
    }}
    document.body.appendChild(f); f.submit();
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.key === '0') submitRot('0');
    else if (e.key === '1') submitRot('90');
    else if (e.key === '2') submitRot('180');
    else if (e.key === '3') submitRot('270');
    else if (e.key === 's' || e.key === 'S') submitRot('skip');
    else if (e.key === 'ArrowLeft') window.location.href = '{prev_url}';
    else if (e.key === 'ArrowRight') window.location.href = '{next_url}';
  }});
</script>
</body>
</html>
"""

DONE_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Done</title></head>
<body style="font-family: sans-serif; padding: 40px; text-align: center;">
  <h2>All cases annotated.</h2>
  <p>{done} annotations saved to <code>rotation_annotations.json</code>.</p>
  <p><a href="/?idx=0">Start over</a> &middot; <a href="/summary">Summary</a></p>
</body></html>
"""


def find_next_unannotated(cases: list[str], anns: dict, start: int = 0) -> Optional[int]:
    for i in range(start, len(cases)):
        if cases[i] not in anns:
            return i
    return None


@app.route("/")
def index():
    cases = list_cases()
    if not cases:
        return "No cases found under boundary_annotations/", 404
    anns = load_annotations()

    # Allow ?idx=N to jump to a specific case; otherwise next unannotated.
    idx_arg = request.args.get("idx")
    if idx_arg is not None:
        try:
            idx = max(0, min(len(cases) - 1, int(idx_arg)))
        except ValueError:
            idx = 0
    else:
        nxt = find_next_unannotated(cases, anns, 0)
        if nxt is None:
            return DONE_PAGE.format(done=len(anns))
        idx = nxt

    case = cases[idx]
    prev_idx = max(0, idx - 1)
    next_idx = min(len(cases) - 1, idx + 1)
    done = len(anns)
    return PAGE.format(
        case_name=case,
        cur_idx=idx + 1,
        total=len(cases),
        done=done,
        remaining=len(cases) - done,
        img_url=url_for("image", case=case),
        prev_url=url_for("index", idx=prev_idx),
        next_url=url_for("index", idx=next_idx),
        existing=str(anns.get(case, "—")),
    )


@app.route("/image/<path:case>")
def image(case: str):
    # Strict path containment — refuse to read anything outside DATASET_DIR.
    target = (DATASET_DIR / case / "map.png").resolve()
    try:
        target.relative_to(DATASET_DIR.resolve())
    except ValueError:
        abort(403)
    if not target.exists():
        abort(404)
    return send_file(str(target), mimetype="image/png")


@app.route("/annotate", methods=["GET"])
def annotate_get():
    # Browser back-button / refresh on the POST URL would otherwise return
    # 405 Method Not Allowed. Bounce to / and resume on the next unannotated.
    return redirect("/", code=303)


@app.route("/annotate", methods=["POST"])
def annotate():
    case = request.form.get("case", "").strip()
    rot = request.form.get("rot", "").strip()
    if not case:
        return "missing case", 400
    cases = list_cases()
    if case not in set(cases):
        return f"unknown case: {case}", 400

    anns = load_annotations()
    if rot == "skip":
        anns[case] = "skip"
    else:
        try:
            r = int(rot)
            if r not in (0, 90, 180, 270):
                return f"bad rot: {rot}", 400
            anns[case] = r
        except ValueError:
            return f"bad rot: {rot}", 400
    anns[f"__updated_{case}__"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_annotations(anns)

    # Move to the next unannotated case (or next index if all done).
    idx = cases.index(case)
    nxt = find_next_unannotated(cases, anns, idx + 1)
    if nxt is None:
        nxt = find_next_unannotated(cases, anns, 0)
    if nxt is None:
        return redirect("/", code=303)  # will show DONE_PAGE
    # 303 See Other forces the browser to do a GET on the new URL even if it
    # is in a strict POST-redirect-GET-ignoring state.
    return redirect(url_for("index", idx=nxt), code=303)


@app.route("/summary")
def summary():
    cases = list_cases()
    anns = load_annotations()
    real = {k: v for k, v in anns.items() if not k.startswith("__")}
    by_rot: dict = {0: 0, 90: 0, 180: 0, 270: 0, "skip": 0}
    for v in real.values():
        if v in by_rot:
            by_rot[v] += 1
    annotated_cases = set(real.keys())
    unannotated = [c for c in cases if c not in annotated_cases]

    lines = [f"<h2>Annotation summary</h2>",
             f"<p>{len(annotated_cases)} / {len(cases)} cases annotated.</p>",
             "<table border=1 cellpadding=6><tr><th>Label</th><th>Count</th></tr>"]
    for k, v in by_rot.items():
        lines.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    lines.append("</table>")
    if unannotated:
        lines.append(f"<p><b>Unannotated:</b> {len(unannotated)} cases. "
                     f"<a href='/?idx={cases.index(unannotated[0])}'>Jump to first unannotated</a></p>")
    return "<html><body style='font-family:sans-serif;padding:24px;'>" + "\n".join(lines) + "</body></html>"


if __name__ == "__main__":
    cases = list_cases()
    anns = load_annotations()
    print(f"Loaded {len(cases)} cases from {DATASET_DIR}")
    print(f"Existing annotations: {len([k for k in anns if not k.startswith('__')])}")
    print(f"Annotations file: {ANNOTATIONS_FILE}")
    print(f"Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
