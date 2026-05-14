"""Pydantic schemas for the planning-boundary agent pipeline.

These BaseModel classes are extracted from `tools/agent.py` (Stage 1A of the
agent.py split, 2026-05-11). They define the structured I/O contracts that
pydantic-ai enforces:

- BoundaryConstraint  : one spatial constraint from boundary-description prose
                        (collected by the reader; consumed offline in v19+).
- PDFInfo             : output of the reader agent — everything the worker
                        needs to know about a planning PDF.
- CenterInput         : a geocoded search center; argument shape for match_at.
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

class BoundaryConstraint(BaseModel):
    """One spatial constraint extracted from the boundary description text.

    Idea-A data capture (v18, design at overnight/BOUNDARY_TEXT_CONSTRAINTS_DESIGN.md):
    UK planning documents describe boundaries in natural language ("from the
    southwest corner along Mill Road to the bridge over the River Stour, then
    northeast along the river to OS parcel 4521"). The reader's job here is to
    decompose that prose into structured constraints. v18 does NOT yet apply
    these constraints to the predicted polygon — they're collected as a
    side-output so we can offline-test a constraint refiner against v18's
    cached predictions, then wire it for v19 with clean attribution.
    """
    type: str = Field(
        description="Constraint type. Use one of these tokens: "
                    "'follows_road' (boundary edge runs along a named road); "
                    "'touches_river' (boundary edge meets a named river/stream/canal); "
                    "'abuts_parcel' (boundary aligns with a named OS parcel / plot / field); "
                    "'bounded_by' (a side of the boundary is delimited by a named feature, with optional compass direction); "
                    "'near_landmark' (boundary close to a named landmark e.g. 'next to the church'); "
                    "'along_centerline' (boundary follows the centerline of a feature, not its edge); "
                    "'corner_at' (a vertex of the boundary is at a named point); "
                    "'other' (anything else — describe in description_snippet)."
    )
    target: str = Field(
        description="The NAMED target of the constraint. For 'follows_road' "
                    "use the road name ('Mill Road'). For 'touches_river' "
                    "use the river name ('River Stour'). For 'abuts_parcel' "
                    "use the OS parcel identifier or descriptor ('OS plot 4521', "
                    "'Field Number 0731'). For 'bounded_by' use the named "
                    "feature ('the railway line'). Keep the name verbatim."
    )
    direction: Optional[str] = Field(
        default=None,
        description="Optional compass direction the constraint applies on. "
                    "One of: 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'. "
                    "Use only when the text says it explicitly "
                    "('bounded on the north by ...' → direction='N'). "
                    "Leave null otherwise."
    )
    description_snippet: Optional[str] = Field(
        default=None,
        description="A short verbatim quote (up to ~80 chars) from the boundary "
                    "description that this constraint came from. For debugging "
                    "and offline verification. Optional but encouraged."
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
        description="1-based page numbers containing maps, RANKED by which is "
                    "most likely the canonical site map (best candidate FIRST). "
                    "The first entry is what the worker uses by default. "
                    "Ranking criteria, in order: (1) the page that most "
                    "clearly shows ONE drawn planning boundary (red line, "
                    "stippled area, edged region) at a useful scale; "
                    "(2) prefer pages with visible site labels, road names, "
                    "and OS-style cartography over location-context maps "
                    "(town overviews, regional maps with no boundary drawn); "
                    "(3) for multi-page docs where a small-scale overview "
                    "appears alongside a detailed site map, the DETAILED page "
                    "comes first. Include ALL map pages even if some are "
                    "context-only — the worker can fall back to them."
    )
    n_pages: int = 0
    road_names: List[str] = Field(default_factory=list)
    place_names: List[str] = Field(default_factory=list)
    boundary_color: Optional[str] = Field(
        default=None,
        description="Color of the planning boundary line (red, blue, pink, etc.)."
    )
    boundary_description: str = Field(
        default="",
        description="Verbatim quote of the prose describing the boundary path "
                    "if the document contains one. Examples: 'From the southwest "
                    "corner along Mill Road eastward to the bridge over the River "
                    "Stour, then northeast along the river to OS plot 4521.' "
                    "Leave empty if the doc only has a map without a worded "
                    "description. Used to populate boundary_constraints below."
    )

    boundary_constraints: List[BoundaryConstraint] = Field(
        default_factory=list,
        description="Structured decomposition of boundary_description into "
                    "spatial constraints. ONE entry per constraint. Examples for "
                    "'From the southwest corner along Mill Road eastward to the "
                    "bridge over the River Stour, then northeast along the river "
                    "to OS plot 4521, bounded on the south by the railway line': "
                    "[{type:'follows_road', target:'Mill Road', description_snippet:'along Mill Road eastward'}, "
                    "{type:'touches_river', target:'River Stour', description_snippet:'the bridge over the River Stour'}, "
                    "{type:'follows_road', target:'River Stour', along_centerline-ish — use 'follows_road' if uncertain}, "
                    "{type:'abuts_parcel', target:'OS plot 4521', description_snippet:'to OS plot 4521'}, "
                    "{type:'bounded_by', target:'the railway line', direction:'S', description_snippet:'bounded on the south by the railway line'}]. "
                    "If boundary_description is empty or has no spatial cues, leave this empty. "
                    "Be conservative: if you're not sure what feature a phrase "
                    "refers to, use type='other' and include the full quote in "
                    "description_snippet rather than guessing."
    )
    is_district_wide: bool = Field(
        default=False,
        description="TRUE if the boundary covers an ENTIRE borough/district/ward/"
                    "parish/conservation area. Patterns that trigger TRUE: "
                    "'Land within the X of Y', 'Various sites across X', "
                    "'The X Conservation Area', 'Land in the Urban District of X'. "
                    "When unsure, prefer true — downstream falls through if lookup fails."
    )
    district_name: Optional[str] = Field(
        default=None,
        description="If is_district_wide, the OSM-format name with 'UK' suffix. "
                    "Provide '|' alternates if ambiguous (e.g. 'Dover District, Kent, UK | Dover, Kent, UK')."
    )
    multiple_map_areas: bool = Field(
        default=False,
        description="True if different map pages show different geographic areas. "
                    "Set true whenever map_pages has more than one entry unless all "
                    "pages are zoomed views of the same site."
    )
    map_rotation: int = Field(
        default=0,
        description="Rotation in degrees CLOCKWISE needed to make the map's north "
                    "point UP. Set 0 if the map is already correctly oriented. "
                    "Set 90 if the map is rotated 90° counterclockwise (i.e., "
                    "north points right and you'd rotate it 90° clockwise to fix it). "
                    "Set 180 if upside-down. Set 270 if rotated 90° clockwise "
                    "(north points left). Look at the north arrow if visible, "
                    "or the orientation of place-name labels and the scale bar. "
                    "Most modern maps are 0; planning maps and historic OS sheets "
                    "can be 90, 180, or 270."
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

    coordinate_labels_on_map: List[str] = Field(
        default_factory=list,
        description="OS grid references or coordinate labels printed ON THE MAP "
                    "MARGINS (graticule ticks). Examples: 'TG 210 080', 'TR 34 SE', "
                    "'TL 452 305'. These are hyper-precise anchors for "
                    "georeferencing. If the map shows no visible graticule labels "
                    "(many modern planning maps do not), leave empty. Duplicate of "
                    "grid_refs is OK — grid_refs is text-body; this is map-surface."
    )

    notes: str = ""

    @field_validator("place_names", "road_names", "parish_names",
                     "house_number_road_pairs", "visible_map_labels",
                     "adjacency_hints", "coordinate_labels_on_map",
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


class CenterInput(BaseModel):
    """A geocoded search center for match_at. All three fields required."""
    name: str = Field(description="A short label for this center (e.g. 'postcode NR15 2XE').")
    lat: float = Field(description="Latitude in decimal degrees (e.g. 52.4774).")
    lon: float = Field(description="Longitude in decimal degrees (e.g. 1.3854).")


class BoundaryOutcome(BaseModel):
    """Structured output for the worker agent. Includes mandatory checklist
    fields so the output_validator can enforce that required tools were called.

    NOTE: rejection was removed from the schema 2026-05-14. The agent always
    submits status="accepted" (with concerns captured in visual_check_notes)
    or status="district_lookup" for the OSM-district fallback. Refusing a
    case is no longer a supported action — the pipeline always produces a
    polygon, downstream measures IoU on whatever was committed."""
    status: Literal["accepted", "district_lookup"] = Field(
        description="accepted = produce GeoJSON from match_at + extract_boundary; "
                    "district_lookup = boundary from OSM district fallback."
    )
    final_n_inliers: int = Field(
        default=0,
        description="n_inliers from the committed match_at attempt (0 if none)."
    )
    verify_position_called: bool = Field(
        default=False,
        description="Did you call verify_position for this result? "
                    "MUST be true if final_n_inliers is in 25-100 band."
    )
    visual_check_notes: str = Field(
        default="",
        description="If you called verify_position, describe whether OS tile features "
                    "(roads, buildings, settlement shape) matched the planning map. "
                    "If features looked weak or mismatched, still submit accepted "
                    "and note your concerns here. Required when final_n_inliers "
                    "is 25-100."
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
    "BoundaryConstraint",
    "PDFInfo",
    "CenterInput",
    "BoundaryOutcome",
]
