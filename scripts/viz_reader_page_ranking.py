"""Visualize the reader's ranked map_pages for the 22 multi-page cases.

One large image, 22 rows (one per case). Each row shows:
  - the case name on the left
  - thumbnails of every reader-selected map page, in rank order (left → right)
  - each thumbnail labeled with rank, page #, role, and a USED tag for
    whichever page v_postrot's worker committed against

Output: reader_page_ranking/all_22_cases.png
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

THUMB_W = 220        # width of each page thumbnail
THUMB_H = 220        # height of each page thumbnail
LABEL_H = 56         # label bar above each thumbnail
CASE_LABEL_W = 280   # left-column row label width
ROW_PAD = 12         # vertical pad between rows
COL_PAD = 8          # horizontal pad between thumbnails


def _find_pdf(case: str):
    safe = case.replace(":", "_").replace("/", "_")
    for variant in (case, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".pdf":
                    return f
    return None


def _label_block(text_lines, w, h, bg, fg=(255, 255, 255), highlight=False):
    """Render a small label block with multiple lines."""
    box = np.full((h, w, 3), bg, dtype=np.uint8)
    if highlight:
        cv2.rectangle(box, (0, 0), (w - 1, h - 1), (0, 200, 0), 2)
    y = 18
    for line in text_lines:
        # Truncate to fit
        font_scale = 0.42
        thick = 1
        max_chars = max(1, int(w / 7.5))
        s = line[:max_chars]
        cv2.putText(box, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, fg, thick, cv2.LINE_AA)
        y += 16
    return box


def _fit_page_thumb(img, w, h):
    """Resize image to fit (w, h) preserving aspect; pad with white."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


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


def _build_row(case: str, max_cols: int) -> np.ndarray:
    cdir = BENCH / case
    if not cdir.exists():
        return None
    pdf_info = json.loads((cdir / "pdf_info.json").read_text())
    metrics = json.loads((cdir / "metrics.json").read_text())
    map_pages = pdf_info.get("map_pages") or []
    details = {d["page"]: d for d in (pdf_info.get("map_page_details") or [])}
    used = _used_page_for_case(case)
    used_iou = metrics.get("iou") or metrics.get("iou_polygon")

    pdf = _find_pdf(case)
    if pdf is None:
        return None

    # Per-page (label + thumb) blocks
    block_h = LABEL_H + THUMB_H
    block_w = THUMB_W
    full_w = CASE_LABEL_W + max_cols * (block_w + COL_PAD)
    row = np.full((block_h, full_w, 3), 235, dtype=np.uint8)

    # Case label (left)
    case_box = _label_block(
        [case[:32],
         f"used p{used}",
         (f"iou {used_iou:.3f}" if isinstance(used_iou, (int, float))
          else "iou N/A"),
         f"{len(map_pages)} pages"],
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
        caption = meta.get("caption", "")
        is_used = (page == used)
        bg = (0, 130, 0) if is_used else (70, 70, 70)
        lbl = _label_block(
            [f"#{rank}  p{page}  role={role}",
             (caption[:32] + ("…" if len(caption) > 32 else "")),
             ("USED 🟢" if is_used else "")],
            THUMB_W, LABEL_H, bg=bg, fg=(255, 255, 255), highlight=is_used,
        )
        block = np.vstack([lbl, thumb])
        if is_used:
            cv2.rectangle(block, (0, 0), (block.shape[1] - 1, block.shape[0] - 1),
                          (0, 200, 0), 3)
        row[:, x:x + block_w] = block
        x += block_w + COL_PAD

    return row


def main():
    out_dir = REPO / "reader_page_ranking"
    out_dir.mkdir(exist_ok=True)

    # Discover max pages across the 22 cases to size the canvas.
    max_cols = 0
    for case in CASES_22:
        cdir = BENCH / case
        if not cdir.exists():
            continue
        try:
            pi = json.loads((cdir / "pdf_info.json").read_text())
            n = len(pi.get("map_pages") or [])
            if n > max_cols:
                max_cols = n
        except Exception:
            pass
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

    # Stack with padding
    pad_strip = np.full((ROW_PAD, rows[0].shape[1], 3), 200, dtype=np.uint8)
    stacked = [rows[0]]
    for r in rows[1:]:
        stacked.append(pad_strip)
        stacked.append(r)
    big = np.vstack(stacked)
    out_path = out_dir / "all_22_cases.png"
    cv2.imwrite(str(out_path), big)
    print(f"saved {out_path} ({big.shape[1]}×{big.shape[0]} px)")


if __name__ == "__main__":
    main()
