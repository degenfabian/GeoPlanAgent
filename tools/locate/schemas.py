"""Locate-stage dataclasses and pydantic models.

Two parallel candidate types exist for historical reasons:

* :class:`LocateCandidate` — pydantic, used by the v13 :func:`locate_map` path.
* :class:`Candidate` — plain dataclass, used by the production
  :func:`propose_centers_v2` cascade.

A :meth:`Candidate.to_locate_candidate` method bridges the two when interop
with the v13 schema is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Pydantic output models (v13 path) ─────────────────────────────────────

class LocateCandidate(BaseModel):
    """One ranked candidate centre. Confidence is a calibrated blend of source
    specificity, cross-source agreement, and VLM corroboration."""
    lat: float
    lon: float
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(description="Machine-readable tag, e.g. 'nominatim:road:High Street'.")
    evidence: str = Field(description="Human-readable one-line justification.")
    specificity: int = Field(description="0=house, 1=street/grid/postcode, 2=settlement, 5=POI, 9=unknown.")


class DirectAffine(BaseModel):
    """Closed-form affine resolved from graticule ticks. Skip MINIMA when present."""
    matrix_2x3: List[List[float]] = Field(
        description="Affine page_pixel → OSGB (easting_m, northing_m) as a 2×3 matrix."
    )
    tick_count: int
    mean_residual_m: float = Field(description="Reprojection residual of the fit, in metres.")
    source: str = "grid_ticks"


class LocateResult(BaseModel):
    """Full output of the locate stage."""
    direct_affine: Optional[DirectAffine] = None
    scale_ratio: Optional[int] = Field(default=None, description="e.g. 2500 for a 1:2500 map.")
    scale_source: Optional[str] = None
    candidates: List[LocateCandidate] = Field(default_factory=list)
    ocr_grid_refs_found: List[str] = Field(default_factory=list)
    ocr_scale_texts: List[str] = Field(default_factory=list)
    vlm_labels: Optional[Any] = None  # legacy field — VLM path retired 2026-05
    timings: Dict[str, float] = Field(default_factory=dict)
    notes: str = ""


# ─── OCR primitive ─────────────────────────────────────────────────────────

@dataclass
class OCRWord:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float  # 0-100, as pytesseract returns it


# ─── locate_v2 candidate (production path) ─────────────────────────────────

@dataclass
class Candidate:
    """Locate v2 candidate — superset of :class:`LocateCandidate`.

    Use :meth:`to_locate_candidate` if interop with the v13 schema is needed.
    """
    lat: float
    lon: float
    sigma_m: float
    confidence: str   # "high" | "med" | "low"
    source: str
    evidence: str
    specificity: int

    def to_locate_candidate(self) -> LocateCandidate:
        conf_map = {"high": 0.9, "med": 0.6, "low": 0.3}
        return LocateCandidate(
            lat=self.lat, lon=self.lon,
            confidence=conf_map.get(self.confidence, 0.5),
            source=self.source, evidence=self.evidence,
            specificity=self.specificity,
        )
