"""
Manually crop test case PDF pages to map-only regions.
Crop coordinates determined by visual inspection — no API calls.
Saves cropped images and verifies they look correct.
"""

import cv2
import numpy as np
import fitz
from pathlib import Path

OUTPUT_DIR = Path("map_crop_test")
OUTPUT_DIR.mkdir(exist_ok=True)

EVAL_DIR = Path("evaluation_data")

# Crop as fraction of page: (y_top_frac, y_bottom_frac, x_left_frac, x_right_frac)
# Determined by visual inspection of each rendered page
TEST_CASES = [
    {
        "dir": "C97065B6-03D0-48C4-AE0E-508DB0BE644B",
        "page": 4,
        "label": "Shepherdswell_1978",
        # Title text at top, "From O.S. sheets" + seal below map
        # Tightened: cut before "From O.S. sheets" text line
        "crop_frac": (0.02, 0.78, 0.02, 0.98),
    },
    {
        "dir": "7202D619-4C27-4DA4-857E-B89F78C9D8D5",
        "page": 4,
        "label": "West_Stourmouth_1978",
        # Title at top, text + seal card at bottom
        # Tightened: cut before "From O.S. sheets" line
        "crop_frac": (0.0, 0.73, 0.0, 1.0),
    },
    {
        "dir": "43C82C9C-0E1B-4CAE-83F8-E33277D7AC41",
        "page": 1,
        "label": "Droveway_Gardens_2010",
        # Wax seal bleeds into bottom-right of map area
        # Tightened: cut higher to remove seal, trim right side
        "crop_frac": (0.02, 0.66, 0.03, 0.93),
    },
    {
        "dir": "D9176429-F30F-4638-A67E-3B87E7ED603D",
        "page": 3,
        "label": "Moon_Hill_2005",
        # Wax seal overlaps top-right corner of map
        # Tightened: trim right to cut seal
        "crop_frac": (0.02, 0.60, 0.02, 0.85),
    },
    {
        "dir": "B4BE31D4-36A8-452E-97FF-04A53362B26C",
        "page": 2,
        "label": "Coombe_Road_Dover_2007",
        # Map fills top portion, clean crop below map content
        "crop_frac": (0.02, 0.60, 0.02, 0.98),
    },
    {
        "dir": "3DA282A7-E829-47CF-B842-E03E0C704072",
        "page": 3,
        "label": "Townsend_Farm_No2_1974",
        # Title block at top, "Scale" text + seal at bottom
        # Tightened: cut before "Scale 1/10560" text
        "crop_frac": (0.07, 0.76, 0.02, 0.98),
    },
    {
        "dir": "FDBC0FDC-D090-4778-A123-232EB71DF3C6",
        "page": 3,
        "label": "Townsend_Farm_No1_1974",
        # Title block at top, "Scale" text + seal at bottom
        # Tightened: cut before "Scale 1/10560" text
        "crop_frac": (0.07, 0.76, 0.02, 0.98),
    },
    {
        "dir": "8CAFB06E-C92F-41CC-B701-6A38171FFAC2",
        "page": 3,
        "label": "Elms_Vale_Dover",
        # Title at top, glued card + seal at bottom
        # Tightened: cut before "From O.S. sheet" text
        "crop_frac": (0.02, 0.58, 0.03, 0.97),
    },
    {
        "dir": "FA067403-6115-4489-9ED0-2CF26FC2D299",
        "page": 1,
        "label": "Westmarsh_Drove_Farm_2011",
        # Map only, "District of Dover" text below
        "crop_frac": (0.02, 0.67, 0.05, 0.95),
    },
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


def main():
    for i, case in enumerate(TEST_CASES):
        label = case["label"]
        case_dir = EVAL_DIR / case["dir"]
        pdfs = list(case_dir.glob("*.pdf"))
        if not pdfs:
            print(f"Case {i}: {label} — No PDF found")
            continue

        pdf_path = pdfs[0]
        pdf_bgr = render_pdf_page(pdf_path, case["page"], dpi=200)
        h, w = pdf_bgr.shape[:2]

        y_top_f, y_bot_f, x_left_f, x_right_f = case["crop_frac"]
        y_top = int(h * y_top_f)
        y_bot = int(h * y_bot_f)
        x_left = int(w * x_left_f)
        x_right = int(w * x_right_f)

        cropped = pdf_bgr[y_top:y_bot, x_left:x_right].copy()
        ch, cw = cropped.shape[:2]

        print(f"Case {i}: {label}")
        print(f"  Original: {w}x{h}, Crop: y=[{y_top}..{y_bot}], x=[{x_left}..{x_right}]")
        print(f"  Cropped: {cw}x{ch} ({cw/w*100:.0f}% x {ch/h*100:.0f}%)")

        # Save cropped image
        cv2.imwrite(str(OUTPUT_DIR / f"case{i:02d}_{label}_cropped.png"), cropped)

        # Also save a debug image showing the crop box on the original
        debug = pdf_bgr.copy()
        cv2.rectangle(debug, (x_left, y_top), (x_right, y_bot), (0, 255, 0), 4)
        # Gray out the non-map areas
        mask = np.ones_like(pdf_bgr) * 128
        mask[y_top:y_bot, x_left:x_right] = pdf_bgr[y_top:y_bot, x_left:x_right]
        cv2.rectangle(mask, (x_left, y_top), (x_right, y_bot), (0, 255, 0), 4)
        cv2.imwrite(str(OUTPUT_DIR / f"case{i:02d}_{label}_crop_debug.png"), mask)

    print(f"\nAll crops saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
