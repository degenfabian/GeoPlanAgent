"""VLM-direct geocode ablation.

Sends the whole PDF binary to a single-shot VLM and parses a JSON
``{lat, lon, reasoning}`` response. No structured pdf_info, no tool
calls, no agent loop. Scored against the same nearest GT polygon-part
centroid metric as :mod:`locate_only_eval`.

Tests the paper claim "geographic localization needs tools" — directly
comparing a frontier VLM with full PDF access against our tool-augmented locate
sub-agent. The VLM has every text and image signal in the PDF; what it
lacks is the ability to look up place names, road names, etc.
against the actual OS Open Names data.

Output sits next to the locate configs under the same root so the
aggregation step can pivot all configs into one table:

    results/ablations/locate_only_eval/
        min_1_tool/, full/               # locate baselines
        vlm_direct_<model_alias>/        # this script
            locate_picks.csv             # same column schema
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from pydantic_ai import Agent, BinaryContent  # noqa: E402
from pydantic_ai.usage import UsageLimits  # noqa: E402

from ablations.utils import (  # noqa: E402
    gt_part_centroids,
    nearest_part_err_km,
    print_err_km_summary,
    LOCATE_PICKS_FIELDNAMES,
)
from geoplanagent.utils import resolve_model, normalise_case_name  # noqa: E402
from geoplanagent.tools.pdf import resolve_case_pdf  # noqa: E402
from geoplanagent.metrics import load_case_ground_truth  # noqa: E402
from geoplanagent.paths import ABL_LOCATE_ONLY, DATA_DIR  # noqa: E402

load_dotenv()


DEFAULT_VLM_MODEL = "gemini-flash"

# Per-case CSV schema (shared by both locate harnesses).
CSV_FIELDNAMES = LOCATE_PICKS_FIELDNAMES

VLM_LOCATE_PROMPT = """You are a UK planning permission geocoder. You will be given a UK
planning permission PDF. Your job: output the WGS84 (lat, lon) of the
APPLICATION SITE — i.e. the property/parcel/area the planning
application is about. NOT the council office that issued the document.

Helpful evidence on the document:
- UK postcodes inside the SITE ADDRESS (not the council letterhead)
- Street names labelled on the planning map OR in the body text
- Place names: parishes, villages, neighbourhoods, named landmarks
- OS grid references (e.g. "TG 210 080", "TR 2648")
- Labels visible on the planning map page (named buildings, adjacent
  roads, distinctive features)

Do NOT use as your primary signal:
- Council / borough / district office postcodes from letterheads
  (these are the council's own address, miles from the site)
- Agent or architect contact addresses
- A district-wide admin name if the site is a specific property
  (the district centroid will be miles off the actual site)

Multi-area documents: some Article 4 directions, conservation areas,
and similar documents cover multiple distinct sites. In that case,
geocoding ANY ONE of the covered sites is fine — pick whichever one
you have the strongest evidence for.

Output exactly the JSON structure required, with these fields:
- lat:        WGS84 latitude as a float (UK is roughly 49.8 to 60.9 N)
- lon:        WGS84 longitude as a float (UK is roughly -8.2 to 1.9 E)
- reasoning:  ONE sentence describing how you arrived at this
              coordinate. Mention the specific evidence you used
              (e.g. "site postcode AL1 3JE → 51.752, -0.336" or
              "intersection of Manor Road and Linden Grove on the
              planning map, both visible in central Peckham").

Give your single best guess — there is no follow-up. If the document
is ambiguous, default to your most confident interpretation."""


# Output schema for the single-shot VLM. Mirrors the locate sub-agent's
# LocatePick on the fields downstream scoring needs; omits sigma_m /
# confidence (no tools available).
class VlmGeocodePick(BaseModel):
    """The geocoded location of the UK planning application site."""

    lat: float = Field(
        description="WGS84 latitude of the application site. UK range roughly 49.8 to 60.9.",
        ge=-90,
        le=90,
    )
    lon: float = Field(
        description="WGS84 longitude of the application site. UK range roughly -8.2 to 1.9.",
        ge=-180,
        le=180,
    )
    reasoning: str = Field(
        description="One sentence explaining the evidence used.",
    )


_vlm_agent = Agent(
    "test",  # placeholder, overridden per-run via model=...
    output_type=VlmGeocodePick,
    retries=3,
    output_retries=3,
    model_settings={"temperature": 0},
    instructions=VLM_LOCATE_PROMPT,
)


# Main eval


def evaluate(args: argparse.Namespace) -> int:
    """Send each case's whole PDF to a single-shot VLM and score its geocode.

    For every case under --eval-dir: hand the raw PDF to the VLM, parse its
    {lat, lon, reasoning}, and measure the haversine error (km) from that pick
    to the nearest GT polygon-part centroid. Writes one row per case to
    <out>/vlm_direct_<model>/locate_picks.csv (resume-aware), same schema as
    locate_only_eval. Returns 0.
    """
    config_label = f"vlm_direct_{normalise_case_name(args.vlm_model)}"
    out_dir = Path(args.out) / config_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "locate_picks.csv"

    print(f"Config:        {config_label}", flush=True)
    print(f"VLM model:     {args.vlm_model}", flush=True)
    print(f"Temperature:   {args.temperature}", flush=True)
    print(f"Output CSV:    {out_csv}", flush=True)

    eval_root = Path(args.eval_dir)
    all_cases = sorted(p.name for p in eval_root.iterdir() if p.is_dir())

    if args.cases:
        wanted = set(args.cases)
        cases = [case for case in all_cases if case in wanted]
        not_found = wanted - set(cases)
        if not_found:
            print(f"WARNING: --cases not in eval dir: {sorted(not_found)}", flush=True)
    else:
        cases = all_cases
    if args.max_cases:
        cases = cases[: args.max_cases]

    already_done: set[str] = set()
    if args.resume and out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                already_done.add(row["case"])
        if already_done:
            print(f"--resume:      {len(already_done)} cases already in CSV", flush=True)

    fieldnames = CSV_FIELDNAMES

    csv_mode = "a" if (args.resume and already_done) else "w"
    model = resolve_model(args.vlm_model)
    t0 = time.time()
    n_ok = n_err = 0

    # Override the agent's temperature if the caller asked for non-0.
    # We rebuild the model_settings dict per-run (don't mutate the
    # module-level Agent).
    model_settings = {"temperature": args.temperature}

    with open(out_csv, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        for i, case in enumerate(cases, start=1):
            if case in already_done:
                continue

            print(f"\n[{i}/{len(cases)}] {case}", flush=True)

            case_dir = eval_root / case
            pdf_path = resolve_case_pdf(case_dir)
            gt_geojson = load_case_ground_truth(case_dir)
            centroids = gt_part_centroids(gt_geojson) if gt_geojson else []

            row = {field: "" for field in fieldnames}
            row["case"] = case
            row["n_gt_parts"] = len(centroids)
            row["picked_source"] = "vlm_direct"

            if pdf_path is None:
                row["error"] = "no PDF"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print("  -> SKIP (no PDF)", flush=True)
                continue

            try:
                pdf_bytes = pdf_path.read_bytes()
            except Exception as error:
                row["error"] = f"PDF read failed: {error!s:.140}"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print(f"  -> SKIP (PDF read failed: {error!s:.80})", flush=True)
                continue

            print(f"  -> sending {pdf_path.name} ({len(pdf_bytes) // 1024} KB)", flush=True)

            try:
                result = _vlm_agent.run_sync(
                    [
                        BinaryContent(
                            data=pdf_bytes,
                            media_type="application/pdf",
                        ),
                        "Geocode this UK planning permission PDF. Output "
                        "the JSON described in the system prompt — lat, "
                        "lon, reasoning.",
                    ],
                    model=model,
                    model_settings=model_settings,
                    usage_limits=UsageLimits(request_limit=4),
                )
                pick: VlmGeocodePick = result.output
            except Exception as error:
                traceback.print_exc()
                row["error"] = f"vlm geocode raised: {error!s:.140}"
                writer.writerow(row)
                f.flush()
                n_err += 1
                print(f"  -> ERROR ({error!s:.80})", flush=True)
                continue

            err = nearest_part_err_km(pick.lat, pick.lon, centroids)
            row.update(
                {
                    "err_km": (f"{err:.3f}" if err is not None else ""),
                    "picked_lat": f"{pick.lat:.6f}",
                    "picked_lon": f"{pick.lon:.6f}",
                    "evidence": pick.reasoning[:240],
                }
            )
            writer.writerow(row)
            f.flush()
            n_ok += 1

            if err is not None:
                print(f"  -> ok | err={err:.2f} km | ({pick.lat:.5f}, {pick.lon:.5f})", flush=True)
            else:
                print(f"  -> ok (no GT centroids) | ({pick.lat:.5f}, {pick.lon:.5f})", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min. n_ok={n_ok}, n_err={n_err}.", flush=True)
    print(f"Wrote {out_csv}", flush=True)

    print_err_km_summary(out_csv)
    return 0


# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DATA_DIR),
        help=f"Eval data root. Default: {DATA_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help=f"Model alias or OpenRouter identifier. Default: {DEFAULT_VLM_MODEL}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default 0. Unlike segmentation we use temp 0 here, "
        "since for this task tehmp 0 did not cause API errors and reproducibality is preferrable.",
    )
    parser.add_argument(
        "--out",
        default=str(ABL_LOCATE_ONLY),
        help=f"Output root. A per-model subdir is created under it. "
        f"Default: {ABL_LOCATE_ONLY.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Space-separated case names; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke limit — evaluate only the first N cases.",
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
