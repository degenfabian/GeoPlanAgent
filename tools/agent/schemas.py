"""Pydantic schemas for the planning-boundary agent pipeline.

- PDFInfo             : output of the reader agent — everything the worker
                        needs to know about a planning PDF.
- BoundaryOutcome     : output of the worker agent (status + checklist +
                        reasoning). The output_validator in agent.py enforces
                        that tool-call preconditions are met before accepting.

The module is intentionally dependency-light: only pydantic. It does NOT
import pydantic_ai or anything from tools.* so it can be loaded without
spinning up SAM3, MINIMA, etc.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Structured Outputs (Pydantic models enforced via pydantic-ai) ─────────

class MapPageMeta(BaseModel):
    """Per-page categorisation. One entry per page that contains any
    map-like or potentially-map content (BOTH the pages we want to
    position — category='match' — and the ones we explicitly discard).
    """
    page: int = Field(description="1-based page number in the PDF.")

    category: Literal["match", "discard"] = Field(
        description="Whether this page is a positionable map. "
                    "'match' = real cartographic page with a drawn "
                    "planning boundary on a recognisable map background "
                    "(OS-style / aerial / hand-drawn over OS). The "
                    "downstream MINIMA matcher runs on this page and "
                    "projects the SAM mask. "
                    "'discard' = NOT positionable. Legends, key tables, "
                    "text-heavy pages, regional context overviews with "
                    "no drawn boundary, bare location pins, indicative "
                    "diagrams without scale, decorative imagery."
    )

    area_group: int = Field(
        description="Equivalence class over the 'match' pages. "
                    "Pages with the SAME area_group show the SAME "
                    "geographic area (duplicate scans, same site at "
                    "different zoom). Pages with DIFFERENT area_groups "
                    "show DIFFERENT geographic areas — these are "
                    "projected separately and the resulting polygons "
                    "UNIONED. Use -1 for category='discard' pages and "
                    "0, 1, 2, … for category='match' pages."
    )

    boundary_clarity: Literal["clear", "ambiguous", "none"] = Field(
        description="'clear' requires BOTH (a) the boundary "
                    "line/hatch/edge is unambiguous to trace AND "
                    "(b) cartographic detail (streets, labels) is "
                    "visible within and around the boundary. "
                    "Otherwise 'ambiguous'. 'none' = no boundary drawn."
    )

    detail_level: Literal["close", "medium", "wide"] = Field(
        description="Zoom / scale of this view. "
                    "'close' ≈ parcel / property level (≈ 1:500-1:2500); "
                    "'medium' ≈ neighbourhood (≈ 1:2500-1:10000); "
                    "'wide' ≈ town / district / regional (≈ 1:10000+)."
    )

    area_signature: str = Field(
        default="",
        description="Short noun phrase (≤8 words) identifying the "
                    "geographic area shown. Pages with the same "
                    "area_group MUST have the identical signature "
                    "(same spelling, same case). For category='discard' "
                    "pages give a short description like 'legend' / "
                    "'location pin' / 'application form text'."
    )

    caption: str = Field(
        default="",
        description="One-line description of the page content so the "
                    "worker can pick wisely without re-rendering "
                    "(≤120 chars). Examples: "
                    "'Detail map at 1:1250 showing red boundary around 4 "
                    "houses'; 'Regional context of South Norfolk with site "
                    "marked'; 'Site plan key — legend for hatching styles'."
    )


class PDFInfo(BaseModel):
    """Structured output for the reader agent. pydantic-ai enforces the schema;
    the model physically cannot return a string — it must fill these fields."""
    site_address: str = Field(
        default="",
        description="The SITE address (location of the planning boundary). "
                    "Prefer 'Site Address', 'Location', or 'Land at...' fields. "
                    "IGNORE council/agent/architect office addresses."
    )
    postcodes: List[str] = Field(
        default_factory=list,
        description="All UK postcodes found (format 'XX1 2YZ')."
    )
    grid_refs: List[str] = Field(
        default_factory=list,
        description="OS grid references (e.g. 'TG 210 080')."
    )
    scale: Optional[str] = Field(
        default=None,
        description="Printed map scale (e.g. '1:2500')."
    )
    map_pages: List[int] = Field(
        default_factory=list,
        description="Pages with category='match' (positionable maps), "
                    "RANKED for the worker. Ordering across different "
                    "area_groups is arbitrary — they will all be "
                    "projected and unioned. WITHIN an area_group, the "
                    "primary (best by boundary_clarity > wider "
                    "detail_level > more cartographic detail) comes "
                    "first; any additional duplicates from the same "
                    "group may follow as fallbacks. "
                    "Do NOT include category='discard' pages here — "
                    "they live only in map_page_details for audit."
    )
    map_page_details: List[MapPageMeta] = Field(
        default_factory=list,
        description="ONE MapPageMeta per page that contains any "
                    "map-like or potentially-map content. SUPERSET of "
                    "map_pages: it includes BOTH category='match' "
                    "pages AND the category='discard' pages the reader "
                    "examined (legends, context overviews, location "
                    "pins, text-heavy pages), so downstream consumers "
                    "can audit discards. Match category='match' "
                    "entries by page number against map_pages."
    )
    n_pages: int = Field(
        default=0,
        description="Total page count of the PDF."
    )
    road_names: List[str] = Field(
        default_factory=list,
        description="Named UK roads that appear in the document text OR on "
                    "the map image. Bare road names with full suffix "
                    "(e.g. 'Norwich Road', 'High Street', 'Pipers Lane'). "
                    "Used downstream for road-based geocoding and for the "
                    "road_name_agreement reward axis, which checks how many "
                    "of these names actually exist at the matched location."
    )
    place_names: List[str] = Field(
        default_factory=list,
        description="Named places (villages, towns, neighbourhoods, "
                    "landmarks, named buildings) that appear in the document "
                    "text OR on the map image. Bare names "
                    "(e.g. 'Hampstead Heath', 'Colney', 'St Mary\\'s Church'). "
                    "Used downstream for place-name geocoding via OS Open Names."
    )
    is_district_wide: bool = Field(
        default=False,
        description="TRUE if the boundary covers an ENTIRE borough/district/ward/"
                    "parish/conservation area. Common trigger phrases include "
                    "'Borough Wide Direction', 'District Wide', 'entire area "
                    "of [admin name]', 'all the land within [admin name]', "
                    "'Various sites across X', 'throughout the District of X', "
                    "'Land in the Urban District of X'."
    )
    district_name: Optional[str] = Field(
        default=None,
        description="If is_district_wide, the UK administrative name with 'UK' "
                    "suffix. Provide '|' alternates if ambiguous (e.g. "
                    "'Dover District, Kent, UK | Dover, Kent, UK'). Downstream "
                    "lookup uses OS BoundaryLine and normalises common variants."
    )
    # ── Fields for the dedicated locate stage (added 2026-04-24) ─────────
    # These mirror things that downstream regex parsers currently extract
    # from site_address / notes. Having the LLM populate them directly is
    # more reliable than regex (handles paraphrasing, typos, mixed formats).

    directional_modifier: Optional[str] = Field(
        default=None,
        description="Directional phrase from site_address in compact form, if "
                    "present: '<direction> of <reference>'. Directions: north, "
                    "south, east, west, NE, NW, SE, SW. Examples: "
                    "'north of 98 Pipers Lane' → 'north of 98 Pipers Lane'; "
                    "'Land rear of 26-64 Manor Road' → 'south of 26-64 Manor Road' "
                    "(interpret 'rear of' as behind/south if no other cue); "
                    "'land between A and B' → null (no unambiguous direction). "
                    "Null if site_address has no clear directional offset."
    )

    house_number_road_pairs: List[str] = Field(
        default_factory=list,
        description="House-numbered addresses from the text, in format "
                    "'<numbers> <road name>'. Preserve ranges and lists. Examples: "
                    "'at no. 41 Linden Grove' → ['41 Linden Grove']; "
                    "'126, 128, 130, 132 and 134 Norwich Road' → "
                    "['126-134 Norwich Road']; "
                    "'4, 8-50, 54-92, 11, 15-37, 41 Chelsea Park Gardens' → "
                    "['4-92 Chelsea Park Gardens']. Collapse lists into one "
                    "range. Only include if a proper house number precedes a "
                    "named road — NOT parcel numbers ('OS parcel 0731')."
    )

    parish_names: List[str] = Field(
        default_factory=list,
        description="Civil/ecclesiastical parishes named in the text. Bare names "
                    "only, with periods preserved. Examples: 'in the parish of "
                    "St. Margaret's at Cliffe' → ['St. Margaret\\'s at Cliffe']; "
                    "'parishes of Caistor St. Edmund and Keswick' → "
                    "['Caistor St. Edmund', 'Keswick']. Do NOT include "
                    "'Parish Council' / 'Parish of X'."
    )

    admin_region: Optional[str] = Field(
        default=None,
        description="Most specific administrative region encompassing the site. "
                    "From 'in the District of X' / 'Borough of Y' / 'various sites "
                    "across Z' / 'Land within the X of Y' patterns. Bare name only "
                    "(e.g. 'South Norfolk', 'Dover', 'Rossendale', 'Southwark'). "
                    "Prefer district/borough over county. Null if none mentioned."
    )

    likely_town_or_city: Optional[str] = Field(
        default=None,
        description="Best single answer for the town/city containing the planning "
                    "site, synthesised from ALL available signals (text, map labels, "
                    "addresses, postcodes, district). Bare name: 'Leicester', "
                    "'Heswall', 'St Albans', 'Harpenden', 'Dover'. This is used to "
                    "disambiguate homonymous road names (every UK city has a "
                    "'Manor Road') so pick the most specific town you can justify. "
                    "If the site spans multiple towns, return the nearest one to "
                    "the drawn boundary."
    )

    visible_map_labels: List[str] = Field(
        default_factory=list,
        description="Named features printed ON THE MAP IMAGE itself (not in the "
                    "text body): road labels shown on roads, named buildings "
                    "(e.g. 'Colney Hall', 'St Mary\\'s Church'), landmarks, "
                    "adjacent labeled places, compass-point labels. Copy verbatim. "
                    "Include even if already in other fields — this is the "
                    "map-readable ground truth, separate from text body content. "
                    "Skip generic labels like 'Scale 1:2500' or 'A4 Direction'."
    )

    adjacency_hints: List[str] = Field(
        default_factory=list,
        description="Named features explicitly adjacent to / bordering the "
                    "planning boundary. From phrases like 'bordered by X', "
                    "'adjoining Y', 'fronting Z', 'land at W', 'abutting V'. "
                    "Include ONLY the named reference (X, Y, Z, W), not the "
                    "preposition. Examples: 'Land adjoining Old Bottom Free Down' "
                    "→ ['Old Bottom Free Down']; 'bounded by Pipers Lane and "
                    "residential properties' → ['Pipers Lane']."
    )

    @field_validator("place_names", "road_names", "parish_names",
                     "house_number_road_pairs", "visible_map_labels",
                     "adjacency_hints",
                     mode="after")
    @classmethod
    def _strict_ascii(cls, v: List[str]) -> List[str]:
        # UK metadata is plain English. Reject CJK / Cyrillic / etc.
        # Allow Latin Extended-A/B (covers accented place names).
        for s in v:
            if any(ord(c) > 0x024F for c in s):
                raise ValueError(
                    f"non-ASCII characters in UK string field: {s!r}. "
                    f"UK metadata must be plain English. Re-extract.")
        return v

    @model_validator(mode="after")
    def _critical_fields_not_all_empty(self):
        # Catches the silent-dropout failure mode where the LLM returns
        # an otherwise-valid PDFInfo with all the actually-useful map
        # fields blank, even though other fields prove this is a real
        # planning doc.
        all_empty = (not self.place_names and not self.road_names
                     and not self.visible_map_labels)
        has_other_signal = (self.site_address or self.house_number_road_pairs
                            or self.parish_names)
        if all_empty and has_other_signal:
            raise ValueError(
                "place_names, road_names, AND visible_map_labels are all "
                "empty but other fields show this is a real planning doc. "
                "Re-extract — likely a partial generation failure.")
        return self


class BoundaryOutcome(BaseModel):
    """Structured output for the worker agent.

    NOTE: rejection was removed from the schema 2026-05-14. The agent always
    submits status="accepted" or status="district_lookup" for the OS
    BoundaryLine district fallback. Refusing a case is no longer a
    supported action — the pipeline always produces a polygon,
    downstream measures IoU on whatever was committed.

    Post-commit visual review is no longer the worker's responsibility.
    The optional independent critic (enable_critic=True) handles that
    role using pairwise comparison across stored match candidates."""
    status: Literal["accepted", "district_lookup"] = Field(
        description="accepted = produce GeoJSON from match_at + commit_match; "
                    "district_lookup = boundary from OS BoundaryLine district fallback."
    )
    final_n_inliers: int = Field(
        default=0,
        description="n_inliers from the committed match_at attempt (0 if none)."
    )
    rotation_checked: bool = Field(
        default=False,
        description="(Auto-set by wrapper.) True when the reader detected and "
                    "applied a rotation. You don't manage this — leave default."
    )
    reasoning: str = Field(
        description="One-paragraph summary of what you did and why the result is correct."
    )


__all__ = [
    "PDFInfo",
    "BoundaryOutcome",
]
