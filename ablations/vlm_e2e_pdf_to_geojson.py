"""VLM-direct PDF-to-GeoJSON ablation.

Sends the whole PDF binary to a single-shot VLM and parses a strict
GeoJSON ``Feature`` (with a ``Polygon`` or ``MultiPolygon`` geometry)
response. No structured pdf_info, no tool calls, no agent loop. Scored
against the GT GeoJSON with the same :func:`calculate_spatial_metrics`
the main pipeline uses.

    results/ablations/vlm_e2e/
        <model_alias>/
            results.csv
            pred_geojsons/<case>.geojson
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field, ValidationError, model_validator  # noqa: E402
from pydantic_ai import Agent, BinaryContent, NativeOutput  # noqa: E402
from pydantic_ai.usage import UsageLimits  # noqa: E402

from geoplanagent.utils import (  # noqa: E402
    resolve_model,
    resolve_model_name,
    normalise_case_name,
    result_tokens,
)
from geoplanagent.tools.pdf import resolve_case_pdf  # noqa: E402
from geoplanagent.metrics import calculate_spatial_metrics, load_case_ground_truth  # noqa: E402
from geoplanagent.paths import ABL_VLM_E2E, DATA_DIR, VLM_E2E_SUBSET  # noqa: E402

load_dotenv()


DEFAULT_VLM_MODEL = "gemini-flash"


VLM_E2E_PROMPT = """You are a UK planning permission boundary geocoder.
Given a UK planning permission PDF, output a single GeoJSON Feature
whose geometry is a Polygon (single site) or MultiPolygon (multiple
disjoint sites) covering the APPLICATION SITE in WGS84 coordinates.
NOT the council office that issued the document.

Think through these four steps before you write the output.

══════════════════════════════════════════════════════════════════════
STEP 1 — READ
══════════════════════════════════════════════════════════════════════
Scan the PDF for every geographic signal you can find:

  • Site address (the location of the boundary), NOT the council /
    agent / architect office address.
  • UK postcodes inside the site address (format 'XX1 2YZ').
    Ignore postcodes that appear in council letterheads.
  • OS grid references (e.g. 'TG 210 080', 'TR 2648').
  • Named roads — in the text or labelled on the map.
  • Named places — parishes, villages, neighbourhoods, landmarks.
  • Labels printed on the map page itself.
  • Printed map scale (e.g. '1:2500').
  • Whether the boundary covers an entire borough / district / parish
    ('Borough Wide Direction', 'throughout the District of X', etc.).

══════════════════════════════════════════════════════════════════════
STEP 2 — LOCATE
══════════════════════════════════════════════════════════════════════
Convert your evidence into a single WGS84 anchor point for the site.
UK longitudes range roughly -8.2 to 1.9; UK latitudes range 49.8 to
60.9.

══════════════════════════════════════════════════════════════════════
STEP 3 — TRACE
══════════════════════════════════════════════════════════════════════
Segment the drawn boundary on the planning map page. This is what
you will project to WGS84 in STEP 4. Note:

  • Line style (red solid outline, hatched red, dashed blue, filled
    pink, black dot-dash, etc.).
  • Shape (rectangular, L-shaped, multiple disjoint parcels,
    elongated strip along the river, etc.).
  • If a printed scale is available, it can be useful for
    estimating the boundary's real-world size.

══════════════════════════════════════════════════════════════════════
STEP 4 — PROJECT
══════════════════════════════════════════════════════════════════════
Translate the traced boundary into a WGS84 GeoJSON Feature anchored
on the STEP 2 center, shaped and sized per STEP 3.

OUTPUT FORMAT
  • type: 'Feature'
  • properties: free-form dict; may be empty.
  • geometry.type: 'Polygon' for a single site, 'MultiPolygon' for
    multi-area documents (Article 4 directions, conservation areas
    covering multiple disjoint sites).
  • geometry.coordinates: a list of linear rings (Polygon) or a list
    of polygons each with their rings (MultiPolygon).

COORDINATE CONVENTION (do not get this wrong)
  • WGS84.
  • [longitude, latitude] order, NOT [latitude, longitude].
  • UK longitudes range -8.2 to 1.9; UK latitudes range 49.8 to 60.9.
  • Outer ring should close (first vertex == last); auto-closed if not.
  • Use 5 to 50 vertices per ring. Do NOT subdivide straight edges
    into many small segments — a square needs 4 vertices, not 400.

Give your single best prediction. There is no follow-up. Be specific
even when the document is ambiguous; default to your most confident
interpretation."""


# Output schema
# A strict GeoJSON Feature with a Polygon | MultiPolygon geometry. The
# discriminator on ``geometry.type`` forces the VLM to commit to one
# shape — invalid shapes fail pydantic validation, which becomes our
# "schema failure" metric.


class GeoJSONPolygon(BaseModel):
    type: Literal["Polygon"]
    coordinates: List[List[List[float]]] = Field(
        description=(
            "GeoJSON Polygon coordinates: a list of linear rings. First "
            "ring is the outer boundary; any subsequent rings are holes. "
            "Each ring is a list of [longitude, latitude] pairs in WGS84 "
            "(EPSG:4326). UK longitudes are roughly -8.2 to 1.9; UK "
            "latitudes are roughly 49.8 to 60.9. Order: [lon, lat], "
            "NOT [lat, lon]. Rings should close (first vertex equals "
            "last); if they don't, shapely will auto-close."
        )
    )


class GeoJSONMultiPolygon(BaseModel):
    type: Literal["MultiPolygon"]
    coordinates: List[List[List[List[float]]]] = Field(
        description=(
            "GeoJSON MultiPolygon coordinates: a list of polygons; each "
            "polygon is a list of linear rings (outer + optional holes); "
            "each ring is a list of [longitude, latitude] pairs in WGS84. "
            "Use MultiPolygon when the document covers more than one "
            "disjoint site."
        )
    )


GeoJSONGeometry = Annotated[
    Union[GeoJSONPolygon, GeoJSONMultiPolygon],
    Field(discriminator="type"),
]


class GeoJSONFeature(BaseModel):
    """A GeoJSON Feature: type='Feature', a Polygon or MultiPolygon geometry
    (discriminated on geometry.type), and an optional free-form properties dict."""

    type: Literal["Feature"]
    properties: dict = Field(
        default_factory=dict,
        description="GeoJSON properties dict. Free-form; may be left empty.",
    )
    geometry: GeoJSONGeometry

    @model_validator(mode="before")
    @classmethod
    def _parse_json_string_subfields(cls, data):
        # Anthropic Claude (via tool-call structured output) sometimes
        # serialises deeply-nested args as JSON strings instead of nested
        # objects — e.g. geometry='{"type":"Polygon","coordinates":[...]}'
        # instead of geometry={"type":"Polygon","coordinates":[...]}.
        # That's a tool-protocol artifact, not a generation failure. We
        # parse the string back into a dict before validation. Does NOT
        # mask real schema errors: if the JSON is invalid, the parse
        # raises and the case is correctly recorded as a schema failure.
        if isinstance(data, dict):
            for key in ("geometry", "properties"):
                value = data.get(key)
                if isinstance(value, str):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass  # leave as str so pydantic raises a clear error
        return data



# Pydantic-ai agent


def build_agent(temperature: float, resolved_model: str) -> Agent:
    # NativeOutput uses the provider's native JSON-schema response mode.
    # Gemini supports it. Anthropic (Claude) does not — it raises
    # "Native structured output is not supported by this model" and
    # falls back. For non-Gemini providers, use the default ToolOutput
    # (passing the class directly), which pydantic-ai implements via
    # tool calls — slightly more framing overhead but universally
    # supported across Anthropic, OpenAI, and others.
    is_gemini = resolved_model.startswith("google/") or "gemini" in resolved_model
    output_type = NativeOutput(GeoJSONFeature) if is_gemini else GeoJSONFeature
    model_settings: dict = {
        "max_tokens": 32768, # Needs to be high enough due to many reasoning tokens.
        "temperature": temperature,
    }
    return Agent(
        "test",  # model overridden per-call
        output_type=output_type,
        retries=3,
        output_retries=0,
        model_settings=model_settings,
        instructions=VLM_E2E_PROMPT,
    )


# Helpers

UK_LAT_RANGE = (49.8, 60.9)
UK_LON_RANGE = (-8.2, 1.9)


def feature_first_vertex(feature: GeoJSONFeature) -> Optional[tuple[float, float]]:
    """First (lon, lat) of the first ring of the first polygon, or None."""
    geom = feature.geometry
    if isinstance(geom, GeoJSONPolygon):
        if geom.coordinates and geom.coordinates[0]:
            vertex = geom.coordinates[0][0]
            return (vertex[0], vertex[1])
    else:
        if geom.coordinates and geom.coordinates[0] and geom.coordinates[0][0]:
            vertex = geom.coordinates[0][0][0]
            return (vertex[0], vertex[1])
    return None


def latlon_inversion_warning(feature: GeoJSONFeature) -> Optional[str]:
    """Detect the common [lat, lon] swap. Returns a short warning or None."""
    vertex = feature_first_vertex(feature)
    if vertex is None:
        return None
    lon, lat = vertex
    # If the first coord is in UK-lat range and second is in UK-lon
    # range, the model probably swapped them.
    if UK_LAT_RANGE[0] <= lon <= UK_LAT_RANGE[1] and UK_LON_RANGE[0] <= lat <= UK_LON_RANGE[1]:
        return f"first vertex ({lon:.4f}, {lat:.4f}) looks like (lat, lon)"
    return None


def count_polygons(feature: GeoJSONFeature) -> int:
    if isinstance(feature.geometry, GeoJSONPolygon):
        return 1
    return len(feature.geometry.coordinates)


def gt_is_multipolygon(gt_geojson: dict) -> bool:
    geom = (gt_geojson or {}).get("geometry") or {}
    return geom.get("type") == "MultiPolygon"





# CSV / per-case schema

CSV_FIELDNAMES = [
    "case",
    "stratum",
    "iou",
    "precision",
    "recall",
    "centroid_distance_m",
    "valid_pred",
    "schema_failure",
    "validation_error",
    "n_polygons",
    "is_multipolygon_pred",
    "is_multipolygon_gt",
    "latlon_swap_warning",
    "call_seconds",
    "vlm_request_tokens",
    "vlm_response_tokens",
    "error",
]


# Main eval


def load_subset(subset_path: Path) -> list[dict]:
    """Load subset_N.json → list of {folder, stratum, …}."""
    payload = json.loads(subset_path.read_text())
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError(f"subset {subset_path} has no 'cases'")
    return cases


def evaluate(args: argparse.Namespace) -> int:
    """Run the ablation over every case in the subset and write per-case results.

    Loads ``args.subset`` (optionally narrowed by ``--cases`` / ``--max-cases``,
    and ``--resume``-able), then for each case sends the whole PDF to the VLM,
    validates the reply against ``GeoJSONFeature``, scores the polygon against the
    case ground truth with ``calculate_spatial_metrics``, and appends a row to
    ``<out>/<model>/results.csv`` (plus the prediction under ``pred_geojsons/``).
    Schema-validation failures and call errors are recorded in the row rather than
    raised, so they count toward the schema-failure / error rates. Prints the
    end-of-run aggregate and returns 0.
    """
    subset_path = Path(args.subset).resolve()
    cases_meta = load_subset(subset_path)
    try:
        subset_label = str(subset_path.relative_to(REPO_ROOT))
    except ValueError:
        subset_label = str(subset_path)
    print(f"Subset:        {subset_label}  ({len(cases_meta)} cases)", flush=True)

    config_label = normalise_case_name(args.vlm_model)
    out_root = Path(args.out)
    out_dir = out_root / config_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    pred_dir = out_dir / "pred_geojsons"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:        {config_label}", flush=True)
    print(f"VLM model:     {args.vlm_model}", flush=True)
    print(f"Temperature:   {args.temperature}", flush=True)
    print(f"Output CSV:    {out_csv}", flush=True)
    print(f"Pred geojson:  {pred_dir}/<case>.geojson", flush=True)

    eval_root = Path(args.eval_dir)

    # Optional filter / cap.
    if args.cases:
        wanted = set(args.cases)
        cases_meta = [case_meta for case_meta in cases_meta if case_meta["folder"] in wanted]
        not_found = wanted - {case_meta["folder"] for case_meta in cases_meta}
        if not_found:
            print(f"WARNING: --cases not in subset: {sorted(not_found)}", flush=True)
    if args.max_cases:
        cases_meta = cases_meta[: args.max_cases]

    already_done: set[str] = set()
    if args.resume and out_csv.exists():
        with open(out_csv) as csv_file:
            for row in csv.DictReader(csv_file):
                already_done.add(row["case"])
        if already_done:
            print(f"--resume:      {len(already_done)} cases already in CSV", flush=True)

    csv_mode = "a" if (args.resume and already_done) else "w"
    resolved = resolve_model_name(args.vlm_model)
    agent = build_agent(temperature=args.temperature, resolved_model=resolved)
    model = resolve_model(args.vlm_model)
    t0 = time.time()
    n_ok = n_schema_fail = n_err = 0

    with open(out_csv, csv_mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if csv_mode == "w":
            writer.writeheader()

        for i, meta in enumerate(cases_meta, start=1):
            case = meta["folder"]
            if case in already_done:
                continue

            print(f"\n[{i}/{len(cases_meta)}] {case}  [{meta['stratum']}]", flush=True)

            case_dir = eval_root / case
            pdf_path = resolve_case_pdf(case_dir)
            gt_geojson = load_case_ground_truth(case_dir)

            row = {field: "" for field in CSV_FIELDNAMES}
            row["case"] = case
            row["stratum"] = meta["stratum"]
            row["is_multipolygon_gt"] = gt_is_multipolygon(gt_geojson) if gt_geojson else ""

            if pdf_path is None:
                row["error"] = "no PDF"
                writer.writerow(row)
                csv_file.flush()
                n_err += 1
                print("  -> SKIP (no PDF)", flush=True)
                continue
            if gt_geojson is None:
                row["error"] = "no GT geojson"
                writer.writerow(row)
                csv_file.flush()
                n_err += 1
                print("  -> SKIP (no GT)", flush=True)
                continue

            try:
                pdf_bytes = pdf_path.read_bytes()
            except Exception as error:
                row["error"] = f"PDF read failed: {error!s:.140}"
                writer.writerow(row)
                csv_file.flush()
                n_err += 1
                print(f"  -> SKIP (PDF read failed: {error!s:.80})", flush=True)
                continue

            print(f"  -> sending {pdf_path.name} ({len(pdf_bytes) // 1024} KB)", flush=True)
            t_call = time.time()
            feature: Optional[GeoJSONFeature] = None
            schema_failure = False

            try:
                result = agent.run_sync(
                    [
                        BinaryContent(
                            data=pdf_bytes,
                            media_type="application/pdf",
                        ),
                        "Geocode and trace the planning boundary in this UK "
                        "planning permission PDF. Think through the four "
                        "steps from the system prompt (READ, LOCATE, TRACE, "
                        "PROJECT); output a single GeoJSON Feature with "
                        "Polygon or MultiPolygon geometry in WGS84 "
                        "[lon, lat] coordinates.",
                    ],
                    model=model,
                    usage_limits=UsageLimits(request_limit=4),
                )
                feature = result.output
            except ValidationError as error:
                schema_failure = True
                row["schema_failure"] = True
                row["validation_error"] = str(error)[:200]
                n_schema_fail += 1
                print(f"  -> SCHEMA FAIL: {str(error)[:100]}", flush=True)
            except Exception as error:
                traceback.print_exc()
                # Walk the __cause__ chain to surface the real underlying
                # error (pydantic-ai wraps ValidationError -> ToolRetryError
                # -> UnexpectedModelBehavior, which swallows the schema
                # detail under a generic "Exceeded maximum retries" message).
                cause_chain = []
                current_error = error
                seen_ids = set()
                while current_error is not None and id(current_error) not in seen_ids:
                    seen_ids.add(id(current_error))
                    cause_chain.append(f"{type(current_error).__name__}: {current_error!s}")
                    current_error = getattr(current_error, "__cause__", None) or getattr(current_error, "__context__", None)
                full_error = " | ".join(cause_chain)
                row["error"] = f"vlm raised: {full_error[:1200]}"
                # If a ValidationError is in the chain, also drop it
                # into validation_error so it's grep-able alongside
                # schema-failure rows.
                for chain_entry in cause_chain:
                    if chain_entry.startswith("ValidationError"):
                        row["validation_error"] = chain_entry[:800]
                        break
                n_err += 1
                print(f"  -> ERROR ({full_error[:200]})", flush=True)
                writer.writerow(row)
                csv_file.flush()
                continue

            row["call_seconds"] = f"{time.time() - t_call:.2f}"
            row["vlm_request_tokens"], row["vlm_response_tokens"] = result_tokens(result)

            if schema_failure:
                writer.writerow(row)
                csv_file.flush()
                continue

            assert feature is not None
            pred_dict = feature.model_dump()
            row["n_polygons"] = count_polygons(feature)
            row["is_multipolygon_pred"] = isinstance(feature.geometry, GeoJSONMultiPolygon)
            swap_warning = latlon_inversion_warning(feature)
            if swap_warning:
                row["latlon_swap_warning"] = swap_warning
                print(f"  WARN: {swap_warning}", flush=True)

            # Scoring — same metric as benchmark_runner. The scorer raises
            # if a geometry can't be built; record that as an invalid pred.
            try:
                metrics = calculate_spatial_metrics(gt_geojson, pred_dict)
                valid_pred = True
                row["valid_pred"] = True
                row["iou"] = f"{metrics['iou']:.6f}"
                row["precision"] = f"{metrics['precision']:.6f}"
                row["recall"] = f"{metrics['recall']:.6f}"
                row["centroid_distance_m"] = f"{metrics['centroid_distance_m']:.2f}"
            except Exception as error:
                metrics = {}
                valid_pred = False
                row["valid_pred"] = False
                row["validation_error"] = str(error)[:200]

            writer.writerow(row)
            csv_file.flush()
            n_ok += 1

            # Per-case pred geojson (for visual inspection).
            try:
                (pred_dir / f"{normalise_case_name(case)}.geojson").write_text(
                    json.dumps(pred_dict, indent=2)
                )
            except Exception as error:
                print(f"  WARN: pred geojson dump failed: {error!s:.80}", flush=True)

            iou_display = f"{metrics['iou']:.3f}" if valid_pred else "invalid"
            print(
                f"  -> ok | IoU={iou_display} | polys={row['n_polygons']} | {row['call_seconds']}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed / 60:.1f} min. n_ok={n_ok}, schema_fail={n_schema_fail}, err={n_err}.",
        flush=True,
    )
    print(f"Wrote {out_csv}", flush=True)

    # End-of-run aggregate to the console (per-case detail is in results.csv).
    if out_csv.exists():
        with open(out_csv) as csv_file:
            rows = list(csv.DictReader(csv_file))
        print_run_summary(rows)
    return 0


def print_run_summary(rows: list[dict]) -> None:
    """Print the end-of-run aggregate — honest IoU (missing/invalid scored 0,
    matching benchmark_runner) plus valid / schema-failure rates. Per-case
    detail lives in results.csv."""
    n = len(rows)
    n_valid = sum(1 for row in rows if str(row.get("valid_pred", "")).lower() == "true")
    n_schema_fail = sum(1 for row in rows if str(row.get("schema_failure", "")).lower() == "true")

    honest = [float(row["iou"]) if row.get("iou") else 0.0 for row in rows]
    if honest:
        sorted_ious = sorted(honest)
        mean = sum(honest) / len(honest)
        median = sorted_ious[len(sorted_ious) // 2]
        ge_50 = sum(1 for iou in honest if iou >= 0.50) / len(honest)
        ge_80 = sum(1 for iou in honest if iou >= 0.80) / len(honest)
        print(
            f"  honest IoU: n={len(honest)}  mean={mean:.3f}  median={median:.3f}  "
            f">=0.5={ge_50 * 100:.1f}%  >=0.8={ge_80 * 100:.1f}%",
            flush=True,
        )
    print(
        f"  valid_rate={(n_valid / n * 100) if n else 0.0:.1f}%  "
        f"schema_fail_rate={(n_schema_fail / n * 100) if n else 0.0:.1f}%",
        flush=True,
    )


# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subset",
        default=str(VLM_E2E_SUBSET),
        help=f"Path to subset_N.json. Default: {VLM_E2E_SUBSET.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DATA_DIR),
        help=f"Eval data root. Default: {DATA_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help=f"Model alias or OpenRouter identifier. Default: "
        f"{DEFAULT_VLM_MODEL}. For the paper baseline use --vlm-model gemini-pro.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default 1.0). temp=0 caused similar problems as "
        "in the vlm segmentation.",
    )
    parser.add_argument(
        "--out",
        default=str(ABL_VLM_E2E),
        help=f"Output root; a per-model subdir is created under it. "
        f"Default: {ABL_VLM_E2E.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Space-separated case folders; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke limit — evaluate only the first N cases of the subset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already in the output CSV.",
    )
    args = parser.parse_args()

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
