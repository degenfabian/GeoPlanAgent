"""Detect and crop the title-block / legend area off a planning-map render.

Background: many UK planning maps occupy only ~50-70% of the rendered page;
the remainder is occupied by an "Article 4 Direction" title block, a council
seal, signature panel, or notice text. When MINIMA matches the full rendered
page against OS tiles, the text region either:
  - generates spurious LoFTR keypoints that match noise features in OS tiles,
    diluting the inlier count and pushing wrong-area windows up the ranking, or
  - simply burns compute on regions that can never match.

Worst observed in v10 case 5B1: GT was 192m from "Broomfield Wood" anchor and
764m from the picked anchor; both candidates returned ~0.41 overall_score
because every match was noisy. After cropping the title block (which had a
clean vertical border at x=2760 on a 4676-wide render), the map portion is
clean and MINIMA should be able to actually distinguish good matches.

This helper:
  1. Detects the title-block via long vertical Hough line(s) in the right half.
  2. Optionally also detects horizontal title-block borders (top-only).
  3. Returns a (cropped_image, x_offset, y_offset) so callers can adjust the
     affine matrix downstream if needed.
  4. NO-OPS (returns the input) when no clear title block is detected, so it
     only kicks in when there's something obvious to remove.

Designed to be conservative: it would rather miss a title block than crop
real map content. False negatives are tolerable; false positives are not.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def detect_title_block_crop(
    bgr: np.ndarray,
    *,
    debug: bool = False,
) -> Tuple[np.ndarray, int, int, dict]:
    """Detect a title-block border and return a cropped image.

    Returns:
        (cropped_bgr, x_offset, y_offset, info) where info has keys:
          - "cropped": bool — did we crop anything?
          - "reason": str — diagnostic
          - "crop_box": (x0, y0, x1, y1) of the kept region in the original

    Only crops along the right and top edges (where title blocks usually live).
    Conservative: requires a vertical line ≥ 40% of page height in the right
    half of the page AND that the region to its right is dominantly empty/text
    (ink density well below the page mean).
    """
    if bgr is None or bgr.size == 0:
        return bgr, 0, 0, {"cropped": False, "reason": "empty input"}
    h, w = bgr.shape[:2]
    if w < 800 or h < 600:
        return bgr, 0, 0, {"cropped": False, "reason": "image too small to crop reliably"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    binary = (gray < 200).astype(np.uint8)

    # Find vertical line SEGMENTS (short threshold) and merge collinear ones.
    # Using a long min-length directly misses title-block borders that are
    # broken by intersecting horizontal lines or text — saw this on case 5B1
    # where the title block's left edge was 1300+ px tall but no single
    # unbroken segment exceeded 360px. The merge step recovers them.
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=80, maxLineGap=10,
    )
    if lines is None:
        return bgr, 0, 0, {"cropped": False, "reason": "no Hough lines"}

    # Bucket vertical line segments by x position (5px buckets), then merge
    # segments within 50px gap into runs. We need ≥40% of page height total
    # vertical coverage to call something a "title-block border".
    from collections import defaultdict
    by_x: dict = defaultdict(list)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 5 or abs(y2 - y1) < 50:
            continue
        mid_x = (x1 + x2) // 2
        if mid_x < int(w * 0.40):
            continue
        by_x[mid_x // 5 * 5].append((min(y1, y2), max(y1, y2)))

    min_v_len = int(h * 0.40)
    qualifying: list = []  # (x_bucket, merged_total_length, span_top, span_bottom)
    for x_bucket, segs in by_x.items():
        segs.sort()
        merged_segs = [segs[0]]
        for s, e in segs[1:]:
            if s <= merged_segs[-1][1] + 50:
                merged_segs[-1] = (merged_segs[-1][0], max(merged_segs[-1][1], e))
            else:
                merged_segs.append((s, e))
        total_len = sum(e - s for s, e in merged_segs)
        if total_len < min_v_len:
            continue
        qualifying.append((x_bucket,
                            total_len,
                            min(s for s, _ in merged_segs),
                            max(e for _, e in merged_segs)))

    if not qualifying:
        return bgr, 0, 0, {
            "cropped": False,
            "reason": (f"no vertical run of ≥{min_v_len}px in right half "
                       f"(checked {len(by_x)} columns)"),
        }

    # Take the LEFTMOST qualifying column — that's the title-block's left edge.
    qualifying.sort(key=lambda t: t[0])
    crop_x = qualifying[0][0]
    cluster_lines = [(crop_x, qualifying[0][2], qualifying[0][3])]

    # Sanity check: the area to the right of crop_x should be ink-sparse
    # (text/whitespace), not dense (real map). Compare its ink density to the
    # mean of the page.
    page_density = float(binary.mean())
    right_density = float(binary[:, crop_x:].mean())
    if right_density > page_density * 0.7:
        # The "removed" region looks like map content, not a title block.
        return bgr, 0, 0, {
            "cropped": False,
            "reason": (f"region right of x={crop_x} too ink-dense "
                       f"({right_density:.3f} vs page {page_density:.3f}) — "
                       f"likely map content, not title block"),
        }

    # Sanity check: don't crop more than 60% of the width away.
    if crop_x < int(w * 0.40):
        return bgr, 0, 0, {
            "cropped": False,
            "reason": f"crop_x={crop_x} would remove >60% of width — too aggressive",
        }

    # Crop. (We could also crop the top-right corner if the title block is
    # only in the corner, but full-height crop is simpler and safe given the
    # ink-density check above.)
    cropped = bgr[:, :crop_x]
    info = {
        "cropped": True,
        "reason": (f"removed title block at x≥{crop_x} "
                   f"(right_density={right_density:.3f} vs "
                   f"page={page_density:.3f})"),
        "crop_box": (0, 0, crop_x, h),
    }
    if debug:
        info["v_cluster"] = cluster_lines
    return cropped, 0, 0, info
