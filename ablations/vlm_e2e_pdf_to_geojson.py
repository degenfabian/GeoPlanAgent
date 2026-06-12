"""VLM-direct PDF-to-GeoJSON ablation.

Sends the whole PDF binary to a single-shot VLM and parses a strict
GeoJSON ``Feature`` (with a ``Polygon`` or ``MultiPolygon`` geometry)
response. No structured pdf_info, no tool calls, no agent loop. Scored
against the GT GeoJSON with the same :func:`calculate_spatial_metrics`
the production benchmark uses, so the VLM and pipeline IoUs are
byte-identical on the same input.

Tests the paper claim "direct frontier-VLM polygon prediction from
PDFs is insufficient for the task" — the VLM has every text and image
signal in the PDF; what it lacks is location lookup, image-to-WGS84
registration, and any access to OS basemap data.

Output sits under the same root that holds the subset definitions, so
the aggregation step can compare VLM-direct against the pipeline
baseline already cached at ``subset_40_pipeline_baseline.json``:

    ablations/vlm_e2e_pdf_to_geojson/
        subset_40.json                    # subset definition (committed)
        subset_40_pipeline_baseline.json  # cached pipeline IoU per case
        <model_alias>/
            results.csv
            summary.json
            pred_geojsons/<case>.geojson
            trajectories/<case>.json

Usage (from repo root):

    uv run python ablations/vlm_e2e_pdf_to_geojson.py --dump-prompt
    uv run python ablations/vlm_e2e_pdf_to_geojson.py --max-cases 2
    uv run python ablations/vlm_e2e_pdf_to_geojson.py
    uv run python ablations/vlm_e2e_pdf_to_geojson.py --vlm-model gemini-pro --temperature 1.0
    uv run python ablations/vlm_e2e_pdf_to_geojson.py --resume
"""
from __future__ import annotations

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

from tools.agent._model import resolve_model, resolve_model_name  # noqa: E402
from tools.agent.runtime import extract_message_log_from_msgs  # noqa: E402
from tools.io.pdf import resolve_case_pdf  # noqa: E402
from tools.metrics.geojson import calculate_spatial_metrics, load_geojson  # noqa: E402

load_dotenv()


DEFAULT_SUBSET = REPO_ROOT / "ablations" / "vlm_e2e_pdf_to_geojson" / "subset_40.json"
DEFAULT_EVAL_DIR = REPO_ROOT / "evaluation_data"
DEFAULT_VLM_MODEL = "gemini-flash"
DEFAULT_OUT_ROOT = REPO_ROOT / "ablations" / "vlm_e2e_pdf_to_geojson"
DEFAULT_PROMPT_DUMP = (
    REPO_ROOT / "ablations" / "prompts" / "vlm_e2e_pdf_to_geojson_prompt.md"
)


# Output schema
# A strict GeoJSON Feature with a Polygon | MultiPolygon geometry. The
# discriminator on ``geometry.type`` forces the VLM to commit to one
# shape — invalid shapes fail pydantic validation, which becomes our
# "schema failure" metric. We do NOT post-process loose decomposed
# coordinate lists into a Feature; that would mask exactly the failure
# mode the ablation is trying to measure.

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
    """Strict GeoJSON Feature output: type='Feature', geometry of type
    Polygon or MultiPolygon (discriminated), optional properties dict.

    This is the only output type the agent returns. The 4-step
    decomposition (Read → Locate → Trace → Project) lives in the
    system prompt as guidance, not as required output fields — keeping
    the schema surface minimal lowers the schema-failure rate and
    avoids the "you confused the model" reviewer critique."""
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
                v = data.get(key)
                if isinstance(v, str):
                    try:
                        data[key] = json.loads(v)
                    except json.JSONDecodeError:
                        pass  # leave as str so pydantic raises a clear error
        return data


# Prompt

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


# Pydantic-ai agent

def build_agent(temperature: Optional[float], resolved_model: str) -> Agent:
    # NativeOutput uses the provider's native JSON-schema response mode.
    # Gemini supports it. Anthropic (Claude) does not — it raises
    # "Native structured output is not supported by this model" and
    # falls back. For non-Gemini providers, use the default ToolOutput
    # (passing the class directly), which pydantic-ai implements via
    # tool calls — slightly more framing overhead but universally
    # supported across Anthropic, OpenAI, and others.
    is_gemini = resolved_model.startswith("google/") or "gemini" in resolved_model
    output_type = NativeOutput(GeoJSONFeature) if is_gemini else GeoJSONFeature
    # Only set temperature when the caller asked for it. OpenAI
    # reasoning models (gpt-5.5-pro etc.) ignore temperature and emit
    # a UserWarning if it is passed; letting each provider use its
    # documented default sidesteps that and keeps the per-provider
    # comparison on each model's recommended sampling profile.
    model_settings: dict = {
        # 32K cap. Gemini 3 Flash needed ~16K to absorb 500+ vertex
        # polygons; OpenAI reasoning models (GPT-5.5 Pro) bill the
        # internal reasoning trace as output tokens and can burn
        # 16K+ purely thinking before emitting structured output. 32K
        # gives reasoning models comfortable headroom; well-behaved
        # responses are unaffected.
        "max_tokens": 32768,
    }
    if temperature is not None:
        model_settings["temperature"] = temperature
    return Agent(
        "test",  # model overridden per-call
        output_type=output_type,
        retries=3,
        # output_retries=0 — pydantic validation failures get reported
        # as schema_failure=True immediately rather than triggering paid
        # retries. "Schema failure rate" is a metric for this ablation;
        # silently rescuing it via retries would defeat the point.
        output_retries=0,
        model_settings=model_settings,
        instructions=VLM_E2E_PROMPT,
    )


# Helpers

UK_LAT_RANGE = (49.8, 60.9)
UK_LON_RANGE = (-8.2, 1.9)


def feature_first_vertex(feature: GeoJSONFeature) -> Optional[tuple[float, float]]:
    """First (lon, lat) of the first ring of the first polygon, or None."""
    g = feature.geometry
    if isinstance(g, GeoJSONPolygon):
        if g.coordinates and g.coordinates[0]:
            v = g.coordinates[0][0]
            return (v[0], v[1])
    else:
        if g.coordinates and g.coordinates[0] and g.coordinates[0][0]:
            v = g.coordinates[0][0][0]
            return (v[0], v[1])
    return None


def latlon_inversion_warning(feature: GeoJSONFeature) -> Optional[str]:
    """Detect the common [lat, lon] swap. Returns a short warning or None."""
    v = feature_first_vertex(feature)
    if v is None:
        return None
    lon, lat = v
    # If the first coord is in UK-lat range and second is in UK-lon
    # range, the model probably swapped them.
    if (UK_LAT_RANGE[0] <= lon <= UK_LAT_RANGE[1]
            and UK_LON_RANGE[0] <= lat <= UK_LON_RANGE[1]):
        return f"first vertex ({lon:.4f}, {lat:.4f}) looks like (lat, lon)"
    return None


def count_polygons(feature: GeoJSONFeature) -> int:
    if isinstance(feature.geometry, GeoJSONPolygon):
        return 1
    return len(feature.geometry.coordinates)


def feature_to_dict(feature: GeoJSONFeature) -> dict:
    return feature.model_dump()


def gt_is_multipolygon(gt_geojson: dict) -> bool:
    geom = (gt_geojson or {}).get("geometry") or {}
    return geom.get("type") == "MultiPolygon"


def _model_label(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


# Prompt dump (no LLM calls)

def dump_prompt(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(VLM_E2E_PROMPT)
    print(f"Wrote prompt to {out_path.relative_to(REPO_ROOT)} "
          f"({len(VLM_E2E_PROMPT)} chars, "
          f"{VLM_E2E_PROMPT.count(chr(10)) + 1} lines)")


# CSV / per-case schema

CSV_FIELDNAMES = [
    "case", "stratum",
    "iou", "precision", "recall", "f1_score", "positioning_error_m",
    "valid_pred", "schema_failure", "validation_error",
    "n_polygons", "is_multipolygon_pred", "is_multipolygon_gt",
    "latlon_swap_warning",
    "call_seconds", "vlm_request_tokens", "vlm_response_tokens",
    "error", "evidence",
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
    subset_path = Path(args.subset).resolve()
    cases_meta = load_subset(subset_path)
    try:
        subset_label = str(subset_path.relative_to(REPO_ROOT))
    except ValueError:
        subset_label = str(subset_path)
    print(f"Subset:        {subset_label}  ({len(cases_meta)} cases)", flush=True)

    config_label = _model_label(args.vlm_model)
    out_root = Path(args.out_root)
    out_dir = out_root / config_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    pred_dir = out_dir / "pred_geojsons"
    pred_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:        {config_label}", flush=True)
    print(f"VLM model:     {args.vlm_model}", flush=True)
    print(f"Temperature:   {args.temperature}", flush=True)
    print(f"Output CSV:    {out_csv.relative_to(REPO_ROOT)}", flush=True)
    print(f"Pred geojson:  {pred_dir.relative_to(REPO_ROOT)}/<case>.geojson",
          flush=True)
    print(f"Trajectories:  {traj_dir.relative_to(REPO_ROOT)}/<case>.json",
          flush=True)

    eval_root = Path(args.eval_dir)

    # Optional filter / cap.
    if args.only_cases:
        wanted = {c.strip() for c in args.only_cases.split(",") if c.strip()}
        cases_meta = [c for c in cases_meta if c["folder"] in wanted]
        not_found = wanted - {c["folder"] for c in cases_meta}
        if not_found:
            print(f"WARNING: --only-cases not in subset: "
                  f"{sorted(not_found)}", flush=True)
    if args.max_cases:
        cases_meta = cases_meta[: args.max_cases]

    already_done: set[str] = set()
    if args.resume and out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                already_done.add(row["case"])
        if already_done:
            print(f"--resume:      {len(already_done)} cases already in CSV",
                  flush=True)

    csv_mode = "a" if (args.resume and already_done) else "w"
    resolved = resolve_model_name(args.vlm_model)
    agent = build_agent(temperature=args.temperature, resolved_model=resolved)
    model = resolve_model(args.vlm_model)
    t0 = time.time()
    n_ok = n_schema_fail = n_err = 0

    with open(out_csv, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if csv_mode == "w":
            writer.writeheader()

        for i, meta in enumerate(cases_meta, start=1):
            case = meta["folder"]
            if case in already_done:
                continue

            print(f"\n[{i}/{len(cases_meta)}] {case}  [{meta['stratum']}]",
                  flush=True)

            case_dir = eval_root / case
            pdf_path = resolve_case_pdf(case_dir)
            gt_relpath = meta.get("gt_geojson_relpath")
            gt_path = (REPO_ROOT / gt_relpath) if gt_relpath else None
            gt_geojson = load_geojson(str(gt_path)) if gt_path else None

            row = {fn: "" for fn in CSV_FIELDNAMES}
            row["case"] = case
            row["stratum"] = meta["stratum"]
            row["is_multipolygon_gt"] = (
                gt_is_multipolygon(gt_geojson) if gt_geojson else ""
            )

            if pdf_path is None:
                row["error"] = "no PDF"
                writer.writerow(row); f.flush()
                n_err += 1
                print("  -> SKIP (no PDF)", flush=True)
                continue
            if gt_geojson is None:
                row["error"] = "no GT geojson"
                writer.writerow(row); f.flush()
                n_err += 1
                print("  -> SKIP (no GT)", flush=True)
                continue

            try:
                pdf_bytes = pdf_path.read_bytes()
            except Exception as e:
                row["error"] = f"PDF read failed: {e!s:.140}"
                writer.writerow(row); f.flush()
                n_err += 1
                print(f"  -> SKIP (PDF read failed: {e!s:.80})", flush=True)
                continue

            print(f"  -> sending {pdf_path.name} ({len(pdf_bytes)//1024} KB)",
                  flush=True)
            t_call = time.time()
            feature: Optional[GeoJSONFeature] = None
            msgs: list = []
            usage = None
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
                msgs = list(result.all_messages())
                usage = result.usage()
            except ValidationError as e:
                schema_failure = True
                row["schema_failure"] = True
                row["validation_error"] = str(e)[:200]
                n_schema_fail += 1
                print(f"  -> SCHEMA FAIL: {str(e)[:100]}", flush=True)
            except Exception as e:
                traceback.print_exc()
                # Walk the __cause__ chain to surface the real underlying
                # error (pydantic-ai wraps ValidationError -> ToolRetryError
                # -> UnexpectedModelBehavior, which swallows the schema
                # detail under a generic "Exceeded maximum retries" message).
                cause_chain = []
                cur = e
                seen = set()
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    cause_chain.append(f"{type(cur).__name__}: {cur!s}")
                    cur = getattr(cur, "__cause__", None) or getattr(
                        cur, "__context__", None
                    )
                full_error = " | ".join(cause_chain)
                row["error"] = f"vlm raised: {full_error[:1200]}"
                # If a ValidationError is in the chain, also drop it
                # into validation_error so it's grep-able alongside
                # schema-failure rows.
                for part in cause_chain:
                    if part.startswith("ValidationError"):
                        row["validation_error"] = part[:800]
                        break
                n_err += 1
                print(f"  -> ERROR ({full_error[:200]})", flush=True)
                writer.writerow(row); f.flush()
                continue

            row["call_seconds"] = f"{time.time() - t_call:.2f}"
            if usage is not None:
                row["vlm_request_tokens"] = (
                    getattr(usage, "input_tokens", None)
                    or getattr(usage, "request_tokens", 0)
                    or 0
                )
                row["vlm_response_tokens"] = (
                    getattr(usage, "output_tokens", None)
                    or getattr(usage, "response_tokens", 0)
                    or 0
                )

            if schema_failure:
                writer.writerow(row); f.flush()
                continue

            assert feature is not None
            pred_dict = feature_to_dict(feature)
            row["n_polygons"] = count_polygons(feature)
            row["is_multipolygon_pred"] = isinstance(
                feature.geometry, GeoJSONMultiPolygon)
            warn = latlon_inversion_warning(feature)
            if warn:
                row["latlon_swap_warning"] = warn
                print(f"  WARN: {warn}", flush=True)

            # Scoring — same metric as benchmark_runner.
            metrics = calculate_spatial_metrics(gt_geojson, pred_dict)
            row["valid_pred"] = bool(metrics.get("valid_prediction"))
            row["iou"] = (f"{metrics['iou']:.6f}"
                          if metrics.get("valid_prediction") else "")
            row["precision"] = (f"{metrics['precision']:.6f}"
                                if metrics.get("valid_prediction") else "")
            row["recall"] = (f"{metrics['recall']:.6f}"
                             if metrics.get("valid_prediction") else "")
            row["f1_score"] = (f"{metrics['f1_score']:.6f}"
                               if metrics.get("valid_prediction") else "")
            pos_err = metrics.get("positioning_error_m")
            row["positioning_error_m"] = (
                f"{pos_err:.2f}" if pos_err is not None else ""
            )
            if metrics.get("validation_error"):
                row["validation_error"] = str(metrics["validation_error"])[:200]

            evidence = (feature.properties or {}).get("reasoning") or ""
            row["evidence"] = str(evidence)[:240]

            writer.writerow(row); f.flush()
            n_ok += 1

            # Per-case pred geojson (for visual inspection).
            try:
                (pred_dir / f"{case.replace('/', '_').replace(':', '_')}.geojson"
                 ).write_text(json.dumps(pred_dict, indent=2))
            except Exception as _e:
                print(f"  WARN: pred geojson dump failed: {_e!s:.80}",
                      flush=True)

            # Per-case trajectory (same shape as locate ablations).
            try:
                trajectory, traj_stats = extract_message_log_from_msgs(msgs)
                traj_payload = {
                    "case": case,
                    "stratum": meta["stratum"],
                    "config": {
                        "approach": "vlm_e2e_pdf_to_geojson",
                        "vlm_model": args.vlm_model,
                        "temperature": args.temperature,
                    },
                    "metrics": {
                        k: v for k, v in metrics.items()
                        if k != "validation_error" or v
                    },
                    "pipeline_baseline_iou": meta.get("pipeline_baseline_iou"),
                    "trajectory_stats": traj_stats,
                    "trajectory": trajectory,
                }
                (traj_dir / f"{case.replace('/', '_').replace(':', '_')}.json"
                 ).write_text(json.dumps(traj_payload, indent=2, default=str))
            except Exception as _e:
                print(f"  WARN: trajectory dump failed: {_e!s:.80}",
                      flush=True)

            iou_val = metrics.get("iou", 0) if metrics.get("valid_prediction") else None
            iou_s = f"{iou_val:.3f}" if iou_val is not None else "invalid"
            print(f"  -> ok | IoU={iou_s} | polys={row['n_polygons']} | "
                  f"{row['call_seconds']}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. "
          f"n_ok={n_ok}, schema_fail={n_schema_fail}, err={n_err}.",
          flush=True)
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}", flush=True)

    # Aggregate summary.json (idempotent end-of-run rewrite).
    if out_csv.exists():
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        write_summary(rows, out_dir, args, elapsed)
    return 0


def write_summary(rows: list[dict], out_dir: Path,
                  args: argparse.Namespace, elapsed: float) -> None:
    ious = [float(r["iou"]) for r in rows if r.get("iou")]
    n = len(rows)
    n_valid = sum(1 for r in rows if str(r.get("valid_pred", "")).lower() == "true")
    n_schema_fail = sum(1 for r in rows
                        if str(r.get("schema_failure", "")).lower() == "true")
    n_err = sum(1 for r in rows if r.get("error"))

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        s = sorted(xs)
        return {
            "n": len(xs),
            "mean": sum(xs) / len(xs),
            "median": s[len(s) // 2],
            "min": s[0],
            "max": s[-1],
            "ge_0.30": sum(1 for x in xs if x >= 0.30) / len(xs),
            "ge_0.50": sum(1 for x in xs if x >= 0.50) / len(xs),
            "ge_0.70": sum(1 for x in xs if x >= 0.70) / len(xs),
            "ge_0.80": sum(1 for x in xs if x >= 0.80) / len(xs),
        }

    # Honest scoring (treat missing/invalid as 0, matching benchmark_runner).
    honest = [float(r["iou"]) if r.get("iou") else 0.0 for r in rows]

    summary = {
        "config": {
            "approach": "vlm_e2e_pdf_to_geojson",
            "vlm_model": args.vlm_model,
            "temperature": args.temperature,
            "subset": (
                str(Path(args.subset).resolve().relative_to(REPO_ROOT))
                if Path(args.subset).resolve().is_relative_to(REPO_ROOT)
                else str(Path(args.subset).resolve())
            ),
        },
        "totals": {
            "n_cases": n,
            "n_valid_pred": n_valid,
            "n_schema_failures": n_schema_fail,
            "n_errors": n_err,
            "valid_rate": n_valid / n if n else 0.0,
            "schema_failure_rate": n_schema_fail / n if n else 0.0,
        },
        "iou_valid_only": stats(ious),
        "iou_honest": stats(honest),
        "elapsed_seconds": round(elapsed, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {(out_dir / 'summary.json').relative_to(REPO_ROOT)}", flush=True)

    s = summary["iou_honest"]
    if s.get("n"):
        print(f"  honest IoU: n={s['n']}  mean={s['mean']:.3f}  "
              f"median={s['median']:.3f}  "
              f">=0.5={s['ge_0.50']*100:.1f}%  "
              f">=0.8={s['ge_0.80']*100:.1f}%", flush=True)
    print(f"  valid_rate={summary['totals']['valid_rate']*100:.1f}%  "
          f"schema_fail_rate="
          f"{summary['totals']['schema_failure_rate']*100:.1f}%",
          flush=True)


# CLI

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subset", default=str(DEFAULT_SUBSET),
        help=f"Path to subset_N.json. Default: "
             f"{DEFAULT_SUBSET.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--eval-dir", default=str(DEFAULT_EVAL_DIR),
        help=f"Eval data root. Default: "
             f"{DEFAULT_EVAL_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--vlm-model", default=DEFAULT_VLM_MODEL,
        help=f"Model alias or OpenRouter identifier. Default: "
             f"{DEFAULT_VLM_MODEL}. For the paper baseline use "
             f"--vlm-model gemini-pro --temperature 1.0.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. If unset, lets each provider use "
             "its documented default (avoids the 'temperature ignored' "
             "warning on OpenAI reasoning models). Pass --temperature "
             "1.0 explicitly for Gemini, where temp=0 triggered an "
             "arithmetic-progression looping bug in pilots.",
    )
    parser.add_argument(
        "--out-root", default=str(DEFAULT_OUT_ROOT),
        help=f"Output root; a per-model subdir is created under it. "
             f"Default: {DEFAULT_OUT_ROOT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--only-cases", default=None,
        help="Comma-separated case folders; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Smoke limit — evaluate only the first N cases of the subset.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cases already in the output CSV.",
    )
    parser.add_argument(
        "--dump-prompt", action="store_true",
        help=f"Write the VLM-E2E prompt to "
             f"{DEFAULT_PROMPT_DUMP.relative_to(REPO_ROOT)} and exit. "
             f"No LLM calls.",
    )
    args = parser.parse_args()

    if args.dump_prompt:
        dump_prompt(DEFAULT_PROMPT_DUMP)
        return 0

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
