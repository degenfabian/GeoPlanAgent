"""
PDF Processing Tools for Planning Document Digitization

This module provides tools for converting PDF documents to images
for further processing (boundary extraction, analysis, etc.).
"""

import os
import base64
import io
from typing import Dict, Any

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


# ── High-resolution PDF rendering ────────────────────────────────────────────

def render_pdf_page(pdf_path, page_index, dpi=200):
    """Render a single PDF page as a numpy BGR image at full resolution.

    Uses PyMuPDF (fitz) for fast rendering. Falls back to pdf2image.
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        doc.close()
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except ImportError:
        pages = convert_from_path(
            pdf_path, dpi=dpi,
            first_page=page_index + 1, last_page=page_index + 1,
        )
        if not pages:
            return None
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)


def find_pdf_for_case(case_folder, eval_dir=None):
    """Find the PDF file in evaluation_data/<case>/."""
    if eval_dir is None:
        eval_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evaluation_data",
        )
    case_dir = os.path.join(eval_dir, case_folder)
    if not os.path.isdir(case_dir):
        return None
    pdfs = [f for f in os.listdir(case_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        return None
    return os.path.join(case_dir, pdfs[0])


def get_pdf_page_as_image(
    pdf_path: str,
    page_index: int = 0,
    dpi: int = 200,
    max_size: int = 1024,
) -> Dict[str, Any]:
    """
    Convert a specific PDF page to a base64-encoded image.

    This tool extracts a single page from a PDF document and converts it
    to a PNG image encoded as base64, suitable for vision model input
    or boundary extraction tools.

    WHEN TO USE:
    - Before calling boundary extraction tools (extract_color_boundary, extract_region_boundary)
    - When you need to analyze a specific page of a planning document
    - To get the map page for geo-referencing

    Args:
        pdf_path (str):
            Path to the PDF file to process.

        page_index (int):
            Zero-indexed page number (first page = 0, second page = 1, etc.).
            Default: 0

        dpi (int):
            Resolution for rendering. Higher values = more detail but larger files.
            Default: 200 (good balance for most planning documents)

        max_size (int):
            Maximum dimension (width or height) in pixels. Images larger than this
            will be resized while preserving aspect ratio.
            Default: 1024 (suitable for most vision APIs)

    Returns:
        Dict containing:
        - "success" (bool): Whether conversion succeeded
        - "image_base64" (str): Base64-encoded PNG image
        - "image_width" (int): Image width in pixels
        - "image_height" (int): Image height in pixels
        - "page_index" (int): The page that was converted
        - "total_pages" (int): Total number of pages in the PDF
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)

        if not images:
            return {
                "success": False,
                "error": "PDF contains no pages",
            }

        total_pages = len(images)

        if page_index < 0 or page_index >= total_pages:
            return {
                "success": False,
                "error": f"Page index {page_index} out of range. PDF has {total_pages} pages (0-{total_pages - 1})",
            }

        # Get the requested page
        page_image = images[page_index]

        # Resize if larger than max_size while preserving aspect ratio
        if page_image.width > max_size or page_image.height > max_size:
            ratio = min(max_size / page_image.width, max_size / page_image.height)
            page_image = page_image.resize(
                (int(page_image.width * ratio), int(page_image.height * ratio)),
                Image.Resampling.LANCZOS,
            )

        # Convert to base64
        buffer = io.BytesIO()
        page_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()

        return {
            "success": True,
            "image_base64": image_base64,
            "image_width": page_image.width,
            "image_height": page_image.height,
            "page_index": page_index,
            "total_pages": total_pages,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"PDF conversion failed: {str(e)}",
        }


# Tool definitions for LLM function calling
PDF_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pdf_page_as_image",
            "description": """Convert a PDF page to a base64-encoded image.

Use this FIRST to get the map image before calling boundary extraction tools.

WORKFLOW:
1. Call this to get the map page as an image
2. Use the image_base64 with extract_color_boundary or extract_region_boundary
3. The image dimensions are returned for use with geo-transformation

PAGE INDEXING: Zero-indexed (first page = 0, second page = 1)
OUTPUT includes total_pages so you know how many pages exist.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Path to the PDF file",
                    },
                    "page_index": {
                        "type": "integer",
                        "description": "Zero-indexed page number (first page = 0). Default: 0",
                    },
                    "dpi": {
                        "type": "integer",
                        "description": "Resolution for rendering (default: 200). Higher = more detail",
                    },
                },
                "required": ["pdf_path"],
            },
        },
    },
]
