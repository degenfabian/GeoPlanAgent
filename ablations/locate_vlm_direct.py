"""VLM-direct geocode ablation.

Sends the whole PDF binary to a single-shot VLM and parses a JSON
``{lat, lon, reasoning}`` response. No structured pdf_info, no tool
calls, no agent loop. Scored against the same nearest GT polygon-part
centroid metric as :mod:`locate_only_eval`.

Tests the paper claim "geographic localization needs tools" — directly
comparing a frontier VLM with full PDF access against our 6-tool locate
sub-agent. The VLM has every text and image signal in the PDF; what it
lacks is the ability to look up postcodes / grid refs / road geometry
against the actual OS datasets and verify against LA polygons.

Output sits next to the locate configs under the same root so the
aggregation step can pivot all configs into one table:

    ablations/locate_only_eval/
        full/                            # locate baseline
        no_postcode/, no_*/              # locate LOO variants
        vlm_direct_<model_alias>/        # this script
            locate_picks.csv             # same column schema
            trajectories/<case>.json
            run.log

Usage (from repo root):

    uv run python ablations/locate_vlm_direct.py --dump-prompt
    uv run python ablations/locate_vlm_direct.py --max-cases 3
    uv run python ablations/locate_vlm_direct.py
    uv run python ablations/locate_vlm_direct.py --vlm-model gemini-pro
    uv run python ablations/locate_vlm_direct.py --resume
"""
from __future__ import annotations

import argparse
import csv
import json
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

from ablations._shared import (  # noqa: E402
    CSV_FIELDNAMES, gt_part_centroids, nearest_part_err_km,
)
from geoplanagent.utils import resolve_model  # noqa: E402
from geoplanagent.run import extract_message_log_from_msgs  # noqa: E402
from geoplanagent.tools.pdf import resolve_case_pdf  # noqa: E402
from geoplanagent.metrics import load_geojson  # noqa: E402

load_dotenv()


DEFAULT_EVAL_DIR = REPO_ROOT / "evaluation_data"
DEFAULT_VLM_MODEL = "gemini-flash"
DEFAULT_OUT_ROOT = REPO_ROOT / "ablations" / "locate_only_eval"
DEFAULT_PROMPT_DUMP = REPO_ROOT / "ablations" / "prompts" / "vlm_direct_prompt.md"


# Prompt

VLM_DIRECT_PROMPT = """You are a UK planning permission geocoder. You will be given a UK
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


class VlmGeocodePick(BaseModel):
    """Single-shot VLM-direct geocode output. Mirrors the locate
    sub-agent's LocatePick on the fields that downstream scoring needs;
    omits sigma_m / confidence / verification (no tools available)."""
    lat: float = Field(
        description="WGS84 latitude of the application site. UK range "
                    "roughly 49.8 to 60.9.",
        ge=-90, le=90,
    )
    lon: float = Field(
        description="WGS84 longitude of the application site. UK range "
                    "roughly -8.2 to 1.9.",
        ge=-180, le=180,
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
    instructions=VLM_DIRECT_PROMPT,
)


# GT-centroid extraction + nearest-part scoring live in ablations._shared
# so the locate / VLM-direct / aggregation harnesses agree on the metric
# byte-for-byte. Imported above as gt_part_centroids and
# nearest_part_err_km.


def _model_label(model_name: str) -> str:
    """Filesystem-safe label derived from a model identifier."""
    return model_name.replace("/", "_").replace(":", "_")


# Prompt dump (no LLM calls)


def dump_prompt(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(VLM_DIRECT_PROMPT)
    print(f"Wrote prompt to {out_path.relative_to(REPO_ROOT)} "
          f"({len(VLM_DIRECT_PROMPT)} chars, "
          f"{VLM_DIRECT_PROMPT.count(chr(10)) + 1} lines)")


# Main eval


def evaluate(args: argparse.Namespace) -> int:
    config_label = f"vlm_direct_{_model_label(args.vlm_model)}"
    out_root = Path(args.out_root)
    out_dir = out_root / config_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "locate_picks.csv"
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:        {config_label}", flush=True)
    print(f"VLM model:     {args.vlm_model}", flush=True)
    print(f"Temperature:   {args.temperature}", flush=True)
    print(f"Output CSV:    {out_csv.relative_to(REPO_ROOT)}", flush=True)
    print(f"Trajectories:  {traj_dir.relative_to(REPO_ROOT)}/<case>.json",
          flush=True)

    eval_root = Path(args.eval_dir)
    all_cases = sorted(p.name for p in eval_root.iterdir() if p.is_dir())

    if args.only_cases:
        wanted = {c.strip() for c in args.only_cases.split(",") if c.strip()}
        cases = [c for c in all_cases if c in wanted]
        not_found = wanted - set(cases)
        if not_found:
            print(f"WARNING: --only-cases not in eval dir: "
                  f"{sorted(not_found)}", flush=True)
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
            print(f"--resume:      {len(already_done)} cases already in CSV",
                  flush=True)

    # Same column schema as locate_only_eval (imported from _shared) so
    # the aggregation step can union all CSVs cleanly. Fields VLM-direct
    # has no value for stay empty (sigma_m, confidence).
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
            gt_files = list(case_dir.glob("*.geojson"))
            gt_geojson = load_geojson(str(gt_files[0])) if gt_files else None
            centroids = gt_part_centroids(gt_geojson) if gt_geojson else []

            row = {fn: "" for fn in fieldnames}
            row["case"] = case
            row["n_gt_parts"] = len(centroids)
            row["picked_source"] = "vlm_direct"
            row["confidence"] = ""
            row["sigma_m"] = ""

            if pdf_path is None:
                row["error"] = "no PDF"
                writer.writerow(row); f.flush()
                n_err += 1
                print("  -> SKIP (no PDF)", flush=True)
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
                msgs = list(result.all_messages())
            except Exception as e:
                traceback.print_exc()
                row["error"] = f"vlm geocode raised: {e!s:.140}"
                writer.writerow(row); f.flush()
                n_err += 1
                print(f"  -> ERROR ({e!s:.80})", flush=True)
                continue

            err = nearest_part_err_km(pick.lat, pick.lon, centroids)
            row.update({
                "err_km": (f"{err:.3f}" if err is not None else ""),
                "picked_lat": f"{pick.lat:.6f}",
                "picked_lon": f"{pick.lon:.6f}",
                "evidence": pick.reasoning[:240],
            })
            writer.writerow(row); f.flush()
            n_ok += 1

            # Per-case trajectory JSON — same shape as locate harness so
            # aggregation tooling treats them uniformly. For VLM-direct
            # the trajectory is just the user prompt + the model's
            # response; total_tool_calls will typically be 1 (the
            # synthetic final_result tool pydantic-ai uses to emit
            # structured output).
            try:
                trajectory, traj_stats = extract_message_log_from_msgs(msgs)
                traj_payload = {
                    "case": case,
                    "config": {
                        "approach": "vlm_direct",
                        "vlm_model": args.vlm_model,
                        "temperature": args.temperature,
                    },
                    "pick": pick.model_dump(),
                    "err_km": err,
                    "gt_centroids": [
                        {"lat": lat, "lon": lon} for lat, lon in centroids
                    ],
                    "trajectory_stats": traj_stats,
                    "trajectory": trajectory,
                }
                fs_case = case.replace("/", "_").replace(":", "_")
                (traj_dir / f"{fs_case}.json").write_text(
                    json.dumps(traj_payload, indent=2, default=str)
                )
            except Exception as _e:
                print(f"  WARN: trajectory dump failed: {_e!s:.80}",
                      flush=True)

            if err is not None:
                print(f"  -> ok | err={err:.2f} km | ({pick.lat:.5f}, "
                      f"{pick.lon:.5f})", flush=True)
            else:
                print(f"  -> ok (no GT centroids) | ({pick.lat:.5f}, "
                      f"{pick.lon:.5f})", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. n_ok={n_ok}, n_err={n_err}.",
          flush=True)
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}", flush=True)

    if out_csv.exists():
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        errs = [float(r["err_km"]) for r in rows
                if r.get("err_km") and r["err_km"]]
        if errs:
            errs.sort()
            mean = sum(errs) / len(errs)
            median = errs[len(errs) // 2]
            print(f"err_km: n={len(errs)}  mean={mean:.2f} km  "
                  f"median={median:.2f} km  min={errs[0]:.2f}  "
                  f"max={errs[-1]:.2f}", flush=True)
    return 0


# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir", default=str(DEFAULT_EVAL_DIR),
        help=f"Eval data root. Default: "
             f"{DEFAULT_EVAL_DIR.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--vlm-model", default=DEFAULT_VLM_MODEL,
        help=f"Model alias or OpenRouter identifier. Default: "
             f"{DEFAULT_VLM_MODEL}",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. Default 0. Bump to 1 if the model "
             "loops on temp 0 (some Gemini-3 thinking models do).",
    )
    parser.add_argument(
        "--out-root", default=str(DEFAULT_OUT_ROOT),
        help=f"Output root. A per-model subdir is created under it. "
             f"Default: {DEFAULT_OUT_ROOT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--only-cases", default=None,
        help="Comma-separated case names; evaluate only these.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Smoke limit — evaluate only the first N cases.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cases already in the output CSV.",
    )
    parser.add_argument(
        "--dump-prompt", action="store_true",
        help=f"Write the VLM-direct prompt to "
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
