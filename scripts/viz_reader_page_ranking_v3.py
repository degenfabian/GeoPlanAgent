"""V3 visualization: match/discard categorisation + area_groups + clarity/zoom.

Reads analysis/multi_map_pages/reader_rerun_22_v3/<case>.json plus
v_postrot's metrics.json for the used-page IoU.

For each of the 22 cases, the row shows:
  - case label on the left (case name, v_postrot used page, used IoU,
    v3 match count / discard count, area_group structure)
  - thumbnails of every candidate page, in this order:
      * v3 match pages first, in v3 map_pages rank order
      * then pages that v3 demoted from v_postrot's map_pages
        (i.e. pages v_postrot considered but v3 sent to discard)
  - each thumbnail has:
      * coloured left stripe → area_group ID (gray for discard)
      * label bar with rank, page#, v3 category, group, clarity, zoom,
        and area_signature
      * green outline if v_postrot's worker used this page

Pure-text discards that weren't in v_postrot's map_pages are skipped —
they're not informative thumbnails.

Output: reader_page_ranking/all_22_cases_v3_match_discard.png
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
V3_DIR = Path("analysis/multi_map_pages/reader_rerun_22_v3")

THUMB_W = 240
THUMB_H = 240
LABEL_H = 110
GROUP_BAR_W = 12
CASE_LABEL_W = 340
ROW_PAD = 14
COL_PAD = 10

GROUP_COLORS = [
    (210, 130, 60),     # blue
    (60, 130, 240),     # orange
    (180, 60, 200),     # purple
    (60, 200, 200),     # yellow
    (200, 80, 180),     # magenta
    (140, 200, 80),     # teal
    (80, 80, 220),      # red
    (180, 140, 60),     # cyan
]
DISCARD_STRIPE = (120, 120, 120)   # gray for discard pages


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


def _load_v3(case: str) -> dict | None:
    safe = case.replace(":", "_").replace("/", "_")
    p = V3_DIR / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _load_v_postrot_pages(case: str) -> list[int]:
    cdir = BENCH / case
    if not cdir.exists():
        return []
    try:
        return json.loads((cdir / "pdf_info.json").read_text()).get("map_pages") or []
    except Exception:
        return []


def _label_block(text_lines, w, h, bg, fg=(255, 255, 255), highlight=None):
    box = np.full((h, w, 3), bg, dtype=np.uint8)
    if highlight is not None:
        cv2.rectangle(box, (0, 0), (w - 1, h - 1), highlight, 2)
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


def _page_order(case: str, v3: dict) -> list[int]:
    """Return pages to render for this case, in display order:
       1) v3 match pages (in v3 map_pages rank order)
       2) v_postrot-but-v3-demoted pages (discards that v_postrot thought
          were maps), in v_postrot rank order
    """
    match_pages = v3.get("map_pages") or []
    details = v3.get("map_page_details") or []
    cats = {d["page"]: d.get("category") for d in details}
    v_postrot_pages = _load_v_postrot_pages(case)
    demoted = [p for p in v_postrot_pages
                if p not in match_pages and cats.get(p) == "discard"]
    # Some safety: any v_postrot page not in v3 details at all → include too
    untagged_old = [p for p in v_postrot_pages
                    if p not in match_pages and p not in cats]
    return match_pages + demoted + untagged_old


def _build_row(case: str, max_cols: int) -> np.ndarray | None:
    v3 = _load_v3(case)
    if v3 is None or "error" in v3:
        return None

    cdir = BENCH / case
    metrics = {}
    if cdir.exists():
        try:
            metrics = json.loads((cdir / "metrics.json").read_text())
        except Exception:
            metrics = {}

    used = _used_page_for_case(case)
    used_iou = metrics.get("iou") or metrics.get("iou_polygon")
    match_pages = v3.get("map_pages") or []
    details = {d["page"]: d for d in (v3.get("map_page_details") or [])}

    # Build a stable area_group → palette index mapping per case
    match_groups = []
    for p in match_pages:
        g = (details.get(p) or {}).get("area_group")
        if g is not None and g not in match_groups:
            match_groups.append(g)
    group_to_color: dict[int, tuple[int, int, int]] = {}
    for i, g in enumerate(match_groups):
        group_to_color[int(g)] = GROUP_COLORS[i % len(GROUP_COLORS)]

    pdf = _find_pdf(case)
    if pdf is None:
        return None

    page_order = _page_order(case, v3)

    block_h = LABEL_H + THUMB_H
    block_w = THUMB_W + GROUP_BAR_W
    full_w = CASE_LABEL_W + max_cols * (block_w + COL_PAD)
    row = np.full((block_h, full_w, 3), 232, dtype=np.uint8)

    n_match = sum(1 for d in details.values() if d.get("category") == "match")
    n_discard = sum(1 for d in details.values() if d.get("category") == "discard")
    grp_struct = ", ".join(f"G{i+1}=grp{g}" for i, g in enumerate(match_groups[:3]))
    if len(match_groups) > 3:
        grp_struct += " …"

    case_box = _label_block(
        [case[:38],
         f"used p{used}",
         (f"iou {used_iou:.3f}" if isinstance(used_iou, (int, float))
          else "iou N/A"),
         f"v3 match: {match_pages}",
         f"{n_match} match / {n_discard} discard",
         f"{len(match_groups)} area_group(s)",
         ],
        CASE_LABEL_W, block_h, bg=(40, 40, 40), fg=(255, 255, 255),
    )
    row[:, :CASE_LABEL_W] = case_box

    x = CASE_LABEL_W
    for slot_idx, page in enumerate(page_order):
        if slot_idx >= max_cols:
            break
        rendered = render_map_page(str(pdf), page, dpi=120, verbose=False,
                                      case_name=case)
        if rendered is None:
            continue
        img, _ = rendered
        thumb = _fit_page_thumb(img, THUMB_W, THUMB_H)

        meta = details.get(page) or {}
        cat = meta.get("category", "?")
        grp = meta.get("area_group", -1)
        clarity = meta.get("boundary_clarity", "?")
        zoom = meta.get("detail_level", "?")
        sig = meta.get("area_signature", "") or ""
        caption = meta.get("caption", "") or ""
        is_used = (page == used)
        is_match = (cat == "match")

        rank_in_match = (match_pages.index(page) + 1
                          if page in match_pages else None)

        # Stripe colour: group colour if match, gray if discard
        stripe_col = (group_to_color.get(int(grp), DISCARD_STRIPE) if is_match
                      else DISCARD_STRIPE)

        bg = ((0, 130, 0) if is_used else
              (60, 80, 120) if is_match else
              (90, 50, 50))
        rank_str = (f"#{rank_in_match}" if rank_in_match is not None
                    else "demoted")
        clarity_zoom = f"{clarity}/{zoom}" if is_match else "discarded"
        lbl_lines = [
            f"{rank_str}  p{page}  cat={cat}  G{grp}",
            clarity_zoom[:34],
            (caption[:34] + ("…" if len(caption) > 34 else "")),
            "sig: " + (sig[:30] + ("…" if len(sig) > 30 else "")),
            ("USED 🟢" if is_used else ""),
        ]
        lbl = _label_block(
            lbl_lines, THUMB_W + GROUP_BAR_W, LABEL_H,
            bg=bg, fg=(255, 255, 255),
            highlight=(0, 220, 0) if is_used else None,
        )

        stripe = np.full((THUMB_H, GROUP_BAR_W, 3), stripe_col, dtype=np.uint8)
        thumb_with_stripe = np.hstack([stripe, thumb])
        block = np.vstack([lbl, thumb_with_stripe])
        if is_used:
            cv2.rectangle(block, (0, 0),
                          (block.shape[1] - 1, block.shape[0] - 1),
                          (0, 220, 0), 3)
        elif not is_match:
            # Light dim overlay on discards to visually de-emphasise
            overlay = block.copy()
            overlay = (overlay * 0.6).astype(np.uint8)
            block = overlay

        row[:, x:x + block_w] = block
        x += block_w + COL_PAD

    return row


def main():
    out_dir = REPO / "reader_page_ranking"
    out_dir.mkdir(exist_ok=True)

    # Compute max display columns: max over cases of (match + demoted) length
    max_cols = 0
    for case in CASES_22:
        v3 = _load_v3(case)
        if v3 is None:
            continue
        page_order = _page_order(case, v3)
        max_cols = max(max_cols, len(page_order))
    print(f"max display columns across 22 cases: {max_cols}")

    rows = []
    for case in CASES_22:
        print(f"  building row: {case}")
        r = _build_row(case, max_cols)
        if r is not None:
            rows.append(r)

    if not rows:
        print("no rows built — aborting")
        return

    pad_strip = np.full((ROW_PAD, rows[0].shape[1], 3), 200, dtype=np.uint8)
    stacked = [rows[0]]
    for r in rows[1:]:
        stacked.append(pad_strip)
        stacked.append(r)
    big = np.vstack(stacked)
    out_path = out_dir / "all_22_cases_v3_match_discard.png"
    cv2.imwrite(str(out_path), big)
    print(f"saved {out_path} ({big.shape[1]}×{big.shape[0]} px)")


if __name__ == "__main__":
    main()
