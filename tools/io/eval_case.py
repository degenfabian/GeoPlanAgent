"""Helpers for resolving evaluation-case files on disk.

The evaluation set lives under ``evaluation_data/<case_name>/`` with one
PDF per case (sometimes several — notice + plan + supplementary). These
helpers centralise "given a case folder, what's the canonical PDF" so
benchmark_runner and the ablation harnesses agree on the same answer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# Filename tokens that hint at a dedicated map/plan PDF (vs. notice or
# form documents in the same folder). The first PDF whose lowercase
# name contains any of these tokens wins. "plan" catches cases like
# A4Da2 where one file is "..._Direction_Plan.pdf" and another is a
# notice.
_MAP_TOKENS = ("map", "plan")


def resolve_case_pdf(folder_path: Path) -> Optional[Path]:
    """Pick the canonical PDF for a single evaluation case folder.

    Prefers PDFs whose filename contains 'map' or 'plan' (case-insensitive);
    falls back to the first PDF in the folder if none match.

    Returns ``None`` if the folder doesn't exist or has no PDFs.
    """
    if not folder_path.is_dir():
        return None
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        return None
    map_pdfs = [p for p in pdf_files
                if any(tok in p.name.lower() for tok in _MAP_TOKENS)]
    return map_pdfs[0] if map_pdfs else pdf_files[0]
