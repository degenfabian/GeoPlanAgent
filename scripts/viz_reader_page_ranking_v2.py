"""Visualize the reader's ranked + grouped map_pages for the 22 multi-page cases.

Uses the rerun outputs in analysis/multi_map_pages/reader_rerun_22/ which
include the new area_signature + content_groups fields.

One large image, 22 rows. Each row:
  - case label on the left (case name, used page, IoU, group structure)
  - thumbnails of every reader page in rank order (left → right)
  - each thumbnail has:
      * a coloured vertical stripe on its left edge identifying its
        content_group (pages in the same group share a colour)
      * a label bar above with rank, page#, role, group ID, role,
        truncated caption, and area_signature
      * green outline on the page the v_postrot worker actually used

Output: reader_page_ranking/all_22_cases_v2_with_groups.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.io.map_page import render_map_page


CASES_22 = [
    "A4D4A1",
    "2CCA14C4-32BC-443D-9148-22DFE30A3DAC",
    "1D1A9561-7534-409B-9937-29C887F66069",
    "Ar4.9",
    "Ar4.6",
    "12:00126:ART4",
    "A4D-04",
    "Ar4.30",
    "12:00121:ART4",
    "CPA4(2a)",
    "A108P",
    "A4D8A_merged",
    "A4aSHP1",
    "Ar4.4",
    "23:53149:ART4",
    "A4D-24",
    "Ar4.17",
    "Ar4.2",
    "A4D9",
    "85",
    "A4D6A_merged",
    "25",
]

BENCH = Path("results/benchmark_v_postrot/gemini-flash")
EVAL = Path("evaluation_data")
RERUN_DIR = Path("analysis/multi_map_pages/reader_rerun_22")

THUMB_W = 240
THUMB_H = 240
LABEL_H = 90
GROUP_BAR_W = 10      # coloured left stripe for group ID
CASE_LABEL_W = 320
ROW_PAD = 14
COL_PAD = 10

# BGR palette for content_groups. Repeats if a case has more than 8 groups.
GROUP_COLORS = [
    (210, 130, 60),    # blue
    (60, 130, 240),    # orange
    (180, 60, 200),    # purple
    (60, 200, 200),    # yellow
    (200, 80, 180),    # magenta
    (140, 200, 80),    # teal
    (80, 80, 220),     # red
    (180, 140, 60),    # cyan
]


def _find_pdf(case: str):
    safe = case.replace(":", "_").replace("/", "_")
    for variant in (case, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".pdf":
                    return f
    return None


def _used_page_for_case(case: str) -> int | None:
    cdir = BENCH / case
    if not cdir.exists():
        return None
    try:
        pdf_info = json.loads((cdir / "pdf_info.json").read_text())
        log = json.loads((cdir / "message_log.json").read_text())
    except Exception:
        return None
    pages = pdf_info.get("map_pages") or []
    used = pages[0] if pages else None
    for msg in log:
        if msg.get("kind") == "ToolCallPart" and msg.get("tool") == "render_page":
            args = msg.get("args") or {}
            if isinstance(args, dict) and "page" in args:
                used = int(args["page"])
    return used


def _load_rerun(case: str) -> dict | None:
    safe = case.replace(":", "_").replace("/", "_")
    p = RERUN_DIR / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _label_block(text_lines, w, h, bg, fg=(255, 255, 255), highlight_colour=None):
    box = np.full((h, w, 3), bg, dtype=np.uint8)
    if highlight_colour is not None:
        cv2.rectangle(box, (0, 0), (w - 1, h - 1), highlight_colour, 2)
    y = 16
    font_scale = 0.42
    thick = 1
    max_chars = max(1, int(w / 7.2))
    for line in text_lines:
        s = line[:max_chars]
        cv2.putText(box, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, fg, thick, cv2.LINE_AA)
        y += 16
    return box


def _fit_page_thumb(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _build_row(case: str, max_cols: int) -> np.ndarray | None:
    cdir = BENCH / case
    if not cdir.exists():
        return None
    rerun = _load_rerun(case)
    if rerun is None or "error" in rerun:
        # Fall back to the v_postrot pdf_info if rerun failed
        try:
            rerun = json.loads((cdir / "pdf_info.json").read_text())
        except Exception:
            return None

    metrics = json.loads((cdir / "metrics.json").read_text())
    map_pages = rerun.get("map_pages") or []
    details_list = rerun.get("map_page_details") or []
    details = {d["page"]: d for d in details_list}
    groups = rerun.get("content_groups") or [[p] for p in map_pages]

    # Map page → group index
    page_to_group: dict[int, int] = {}
    for gi, grp in enumerate(groups):
        for p in grp:
            page_to_group[int(p)] = gi

    used = _used_page_for_case(case)
    used_iou = metrics.get("iou") or metrics.get("iou_polygon")

    pdf = _find_pdf(case)
    if pdf is None:
        return None

    block_h = LABEL_H + THUMB_H
    block_w = THUMB_W + GROUP_BAR_W
    full_w = CASE_LABEL_W + max_cols * (block_w + COL_PAD)
    row = np.full((block_h, full_w, 3), 232, dtype=np.uint8)

    # Group structure summary for the row label
    n_groups = len(groups)
    n_pages = len(map_pages)
    grp_repr = ", ".join("[" + ",".join(str(p) for p in g) + "]" for g in groups[:3])
    if len(groups) > 3:
        grp_repr += " …"

    case_box = _label_block(
        [case[:36],
         f"used p{used}",
         (f"iou {used_iou:.3f}" if isinstance(used_iou, (int, float))
          else "iou N/A"),
         f"{n_pages} pages, {n_groups} group(s)",
         grp_repr[:36],
         ],
        CASE_LABEL_W, block_h, bg=(40, 40, 40), fg=(255, 255, 255),
    )
    row[:, :CASE_LABEL_W] = case_box

    x = CASE_LABEL_W
    for rank, page in enumerate(map_pages, 1):
        rendered = render_map_page(str(pdf), page, dpi=120, verbose=False,
                                      case_name=case)
        if rendered is None:
            continue
        img, _ = rendered
        thumb = _fit_page_thumb(img, THUMB_W, THUMB_H)

        meta = details.get(page) or {}
        cat = meta.get("category", "?")
        clarity = meta.get("boundary_clarity", "?")
        detail = meta.get("detail_level", "?")
        role = f"{cat}/{clarity}/{detail}"
        caption = meta.get("caption", "") or ""
        sig = meta.get("area_signature", "") or ""
        gi = page_to_group.get(page, 0)
        gcol = GROUP_COLORS[gi % len(GROUP_COLORS)]
        is_used = (page == used)

        bg = (0, 130, 0) if is_used else (60, 60, 60)
        lbl = _label_block(
            [
                f"#{rank}  p{page}  G{gi+1}  role={role}",
                (caption[:34] + ("…" if len(caption) > 34 else "")),
                "sig: " + (sig[:30] + ("…" if len(sig) > 30 else "")),
                ("USED 🟢" if is_used else ""),
            ],
            THUMB_W + GROUP_BAR_W, LABEL_H,
            bg=bg, fg=(255, 255, 255),
            highlight_colour=(0, 220, 0) if is_used else None,
        )

        # Build the block: left colour stripe + thumb
        stripe = np.full((THUMB_H, GROUP_BAR_W, 3), gcol, dtype=np.uint8)
        thumb_with_stripe = np.hstack([stripe, thumb])
        block = np.vstack([lbl, thumb_with_stripe])
        if is_used:
            cv2.rectangle(block, (0, 0),
                          (block.shape[1] - 1, block.shape[0] - 1),
                          (0, 220, 0), 3)
        row[:, x:x + block_w] = block
        x += block_w + COL_PAD

    return row


def main():
    out_dir = REPO / "reader_page_ranking"
    out_dir.mkdir(exist_ok=True)

    # Discover max pages across the 22 cases.
    max_cols = 0
    for case in CASES_22:
        r = _load_rerun(case)
        if r is None:
            cdir = BENCH / case
            if not cdir.exists():
                continue
            try:
                r = json.loads((cdir / "pdf_info.json").read_text())
            except Exception:
                continue
        n = len(r.get("map_pages") or [])
        if n > max_cols:
            max_cols = n
    print(f"max pages across the 22 cases: {max_cols}")

    rows = []
    for case in CASES_22:
        print(f"  building row: {case}")
        row = _build_row(case, max_cols)
        if row is not None:
            rows.append(row)

    if not rows:
        print("no rows built — aborting")
        return

    pad_strip = np.full((ROW_PAD, rows[0].shape[1], 3), 200, dtype=np.uint8)
    stacked = [rows[0]]
    for r in rows[1:]:
        stacked.append(pad_strip)
        stacked.append(r)
    big = np.vstack(stacked)
    out_path = out_dir / "all_22_cases_v2_with_groups.png"
    cv2.imwrite(str(out_path), big)
    print(f"saved {out_path} ({big.shape[1]}×{big.shape[0]} px)")


if __name__ == "__main__":
    main()
