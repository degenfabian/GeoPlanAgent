"""
Prototype: Detect the map region in planning PDFs by finding the transition
from dense map content to sparse admin text/seal area.

Strategy:
1. Convert to grayscale, run Canny edge detection
2. Compute horizontal edge density (sum of edge pixels per row)
3. Use a sliding window to find the vertical extent of the "dense" region
4. Look for a big drop in edge density from top→bottom — that's where the map ends
5. Also check for the top boundary (title text → map transition)
"""

import cv2
import numpy as np
import fitz
from pathlib import Path

OUTPUT_DIR = Path("map_crop_test")
OUTPUT_DIR.mkdir(exist_ok=True)

EVAL_DIR = Path("evaluation_data")

TEST_CASES = [
    {"dir": "C97065B6-03D0-48C4-AE0E-508DB0BE644B", "page": 4, "label": "Shepherdswell_1978"},
    {"dir": "7202D619-4C27-4DA4-857E-B89F78C9D8D5", "page": 4, "label": "West_Stourmouth_1978"},
    {"dir": "43C82C9C-0E1B-4CAE-83F8-E33277D7AC41", "page": 1, "label": "Droveway_Gardens_2010"},
    {"dir": "D9176429-F30F-4638-A67E-3B87E7ED603D", "page": 3, "label": "Moon_Hill_2005"},
    {"dir": "B4BE31D4-36A8-452E-97FF-04A53362B26C", "page": 2, "label": "Coombe_Road_Dover_2007"},
    {"dir": "3DA282A7-E829-47CF-B842-E03E0C704072", "page": 3, "label": "Townsend_Farm_No2_1974"},
    {"dir": "FDBC0FDC-D090-4778-A123-232EB71DF3C6", "page": 3, "label": "Townsend_Farm_No1_1974"},
    {"dir": "8CAFB06E-C92F-41CC-B701-6A38171FFAC2", "page": 3, "label": "Elms_Vale_Dover"},
    {"dir": "FA067403-6115-4489-9ED0-2CF26FC2D299", "page": 1, "label": "Westmarsh_Drove_Farm_2011"},
]


def render_pdf_page(pdf_path, page_num, dpi=200):
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def detect_map_region(pdf_bgr, label="", debug_dir=None):
    """
    Detect the map region in a planning document page.

    Returns (y_top, y_bottom, x_left, x_right) — the bounding box of the map region.
    """
    h, w = pdf_bgr.shape[:2]
    gray = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Compute edge density per row (horizontal strips)
    # Use a window to smooth out individual text lines
    window_size = max(h // 50, 10)  # ~2% of image height
    row_density = np.sum(edges > 0, axis=1).astype(float) / w  # fraction of edge pixels

    # Smooth with a moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(row_density, kernel, mode='same')

    # Also compute column density for left/right bounds
    col_density = np.sum(edges > 0, axis=0).astype(float) / h
    col_smoothed = np.convolve(col_density, np.ones(window_size) / window_size, mode='same')

    # Strategy for finding map bounds:
    # The map region has consistently HIGH edge density.
    # The admin text/seal at bottom has LOW average density with sparse text spikes.
    # The title at top has a brief spike of text then transitions to map.

    # Find the median density of the "map region" — it should be the densest part
    # Use the middle 50% as a reference for "what map density looks like"
    mid_start = h // 4
    mid_end = 3 * h // 4
    map_density_ref = np.median(smoothed[mid_start:mid_end])

    # Threshold: regions below 30% of the map reference density are "not map"
    threshold = map_density_ref * 0.30

    # Find bottom boundary: scan from bottom up, find where density consistently exceeds threshold
    y_bottom = h - 1
    consecutive_high = 0
    min_consecutive = max(h // 20, 20)  # need at least 5% of height above threshold
    for y in range(h - 1, 0, -1):
        if smoothed[y] >= threshold:
            consecutive_high += 1
            if consecutive_high >= min_consecutive:
                y_bottom = y + min_consecutive  # include the transition zone
                break
        else:
            consecutive_high = 0

    # Find top boundary: scan from top down, find where density consistently exceeds threshold
    y_top = 0
    consecutive_high = 0
    for y in range(0, h):
        if smoothed[y] >= threshold:
            consecutive_high += 1
            if consecutive_high >= min_consecutive:
                y_top = max(0, y - min_consecutive)
                break
        else:
            consecutive_high = 0

    # Find left boundary
    x_left = 0
    col_threshold = np.median(col_smoothed[w // 4: 3 * w // 4]) * 0.25
    consecutive_high = 0
    min_consec_x = max(w // 30, 10)
    for x in range(0, w):
        if col_smoothed[x] >= col_threshold:
            consecutive_high += 1
            if consecutive_high >= min_consec_x:
                x_left = max(0, x - min_consec_x)
                break
        else:
            consecutive_high = 0

    # Find right boundary
    x_right = w - 1
    consecutive_high = 0
    for x in range(w - 1, 0, -1):
        if col_smoothed[x] >= col_threshold:
            consecutive_high += 1
            if consecutive_high >= min_consec_x:
                x_right = min(w - 1, x + min_consec_x)
                break
        else:
            consecutive_high = 0

    # Clamp to valid range and ensure minimum map size (at least 40% of page)
    min_map_height = int(h * 0.4)
    if (y_bottom - y_top) < min_map_height:
        # Fallback: use middle 70% of page
        y_top = int(h * 0.05)
        y_bottom = int(h * 0.75)

    print(f"  Map region: y=[{y_top}..{y_bottom}] ({(y_bottom-y_top)/h*100:.0f}%), "
          f"x=[{x_left}..{x_right}] ({(x_right-x_left)/w*100:.0f}%)")
    print(f"  Original size: {w}x{h}, cropped: {x_right-x_left}x{y_bottom-y_top}")
    print(f"  Map density ref: {map_density_ref:.4f}, threshold: {threshold:.4f}")

    # Debug visualization
    if debug_dir:
        # Draw density profile
        viz = pdf_bgr.copy()

        # Draw horizontal lines at top/bottom bounds
        cv2.line(viz, (0, y_top), (w, y_top), (0, 255, 0), 3)
        cv2.line(viz, (0, y_bottom), (w, y_bottom), (0, 0, 255), 3)
        cv2.line(viz, (x_left, 0), (x_left, h), (255, 255, 0), 2)
        cv2.line(viz, (x_right, 0), (x_right, h), (255, 255, 0), 2)

        # Draw density profile on the right side
        profile_w = 100
        for y in range(h):
            bar_len = int(smoothed[y] / max(smoothed) * profile_w)
            color = (0, 200, 0) if smoothed[y] >= threshold else (0, 0, 200)
            cv2.line(viz, (w - profile_w, y), (w - profile_w + bar_len, y), color, 1)

        # Draw threshold line on profile
        thresh_x = int(threshold / max(smoothed) * profile_w)
        cv2.line(viz, (w - profile_w + thresh_x, 0), (w - profile_w + thresh_x, h), (255, 0, 255), 1)

        cv2.putText(viz, f"Map: y={y_top}-{y_bottom}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(str(debug_dir / f"{label}_crop_debug.png"), viz)

        # Also save the cropped map
        cropped = pdf_bgr[y_top:y_bottom, x_left:x_right].copy()
        cv2.imwrite(str(debug_dir / f"{label}_cropped.png"), cropped)

    return y_top, y_bottom, x_left, x_right


def main():
    for i, case in enumerate(TEST_CASES):
        label = case["label"]
        case_dir = EVAL_DIR / case["dir"]

        # Find PDF
        pdfs = list(case_dir.glob("*.pdf"))
        if not pdfs:
            print(f"Case {i}: {label} — No PDF found")
            continue

        pdf_path = pdfs[0]
        print(f"\n{'='*60}")
        print(f"Case {i}: {label}")
        print(f"{'='*60}")

        pdf_bgr = render_pdf_page(pdf_path, case["page"], dpi=200)
        y_top, y_bottom, x_left, x_right = detect_map_region(
            pdf_bgr, label=label, debug_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    main()
