#!/usr/bin/env python3
"""
VLM Segmentation Failure Diagnostic
===================================

One-shot diagnostic. Runs a SINGLE case with Gemini Pro, captures the raw
text the model returned, and reports exactly why it tripped pydantic
validation (or, if it didn't, what the response looks like).

Use this on cases where the main ablation script reports
``UnexpectedModelBehavior: Exceeded maximum retries (0) for output
validation`` so we can finally see what Pro actually emits on those images
rather than inferring it.

Cost: ~$0.10-$0.30 per case (single API call, Gemini Pro).

Mechanism: pydantic-ai's NativeOutput path discards the raw model text
once a ``ValidationError`` is raised on it (only the structured error
survives via ``__cause__``). We monkey-patch
``pydantic_ai._output.ObjectOutputProcessor.process`` to stash the raw
text in a module-level list before validation runs, so we can dump it on
failure. This is a diagnostic-only hack and is local to this script.

Usage
-----
Diagnose one of the persistent failures (CPA4(1a) is the cheapest to
reason about — it has only 4 GT vertices, so token-budget cannot be the
cause; whatever shows up in the raw text is the true root cause):

    uv run python ablations/vlm_segmentation_diag.py \\
        --model gemini-pro \\
        --case 'CPA4(1a)' \\
        --jpeg-quality 90

Other useful targets:

    --case 'A4D14'   # mid-complexity failure
    --case '85'      # most-complex failure (671 GT vertices, 27 regions)
    --case '43'      # known coord-system-swap recovery at 32K

Quote case names that contain shell special chars (parens, colons).

The raw response and full diagnostic report are printed to stdout and
written to ``results/ablation_vlm_seg_diag/<sanitised-case>.json`` for
offline inspection.
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.usage import UsageLimits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.agent._model import resolve_model
from ablations.vlm_segmentation import VlmSegmentation, DEFAULT_PROMPT


# ── Monkey-patch pydantic-ai to capture raw response text ────────────────
# ObjectOutputProcessor.process(data, ...) is called with the raw model
# text right before validation. We append `data` to _raw_responses then
# delegate to the original implementation. On a failed run the exception
# bubbles up as normal, but we now have the text the validator saw.

import pydantic_ai._output as _po  # noqa: E402

_raw_responses: List[str] = []
_original_process = _po.ObjectOutputProcessor.process


async def _logging_process(self, data, **kwargs):
    if isinstance(data, str):
        _raw_responses.append(data)
    return await _original_process(self, data, **kwargs)


_po.ObjectOutputProcessor.process = _logging_process


# ── Permissive schema for after-the-fact testing ─────────────────────────
# If the strict (int) schema rejects the response, try float-relaxed —
# that tells us whether the issue is float emission vs structural.

class _PolygonFloat(BaseModel):
    vertices: List[List[float]]


class _VlmSegmentationFloat(BaseModel):
    polygons: List[_PolygonFloat]
    notes: str = ""


# ── Diagnostic checks against the captured raw text ──────────────────────

def _format_validation_errors(e: ValidationError, limit: int = 8) -> list:
    return [
        {"type": err["type"],
         "loc": [str(x) for x in err["loc"]],
         "msg": err["msg"][:160]}
        for err in e.errors(include_url=False)[:limit]
    ]


def diagnose_response(raw: str) -> dict:
    out = {
        "raw_text_length": len(raw),
        "raw_text_starts_with": raw[:200],
        "raw_text_ends_with": raw[-200:] if len(raw) > 200 else "",
        "starts_with_brace": raw.lstrip().startswith("{"),
        "ends_with_brace": raw.rstrip().endswith("}"),
        "contains_markdown_fence": "```" in raw,
        "contains_brace": "{" in raw,
    }

    # 1) Strict int validation — what the script's schema actually does.
    try:
        VlmSegmentation.model_validate_json(raw)
        out["strict_int"] = "PASSED"
    except ValidationError as e:
        out["strict_int"] = "FAILED"
        out["strict_int_errors"] = _format_validation_errors(e)

    # 2) Float-relaxed validation — diagnoses float-emission as failure.
    try:
        parsed_f = _VlmSegmentationFloat.model_validate_json(raw)
        out["float_relaxed"] = "PASSED"
        coords = [c for p in parsed_f.polygons for v in p.vertices for c in v]
        if coords:
            out["coord_min"] = min(coords)
            out["coord_max"] = max(coords)
            out["n_polygons"] = len(parsed_f.polygons)
            out["n_vertices_total"] = sum(len(p.vertices) for p in parsed_f.polygons)
            out["all_coords_integer_valued"] = all(c == int(c) for c in coords)
            out["coords_in_0_1000"] = (
                out["coord_min"] >= 0 and out["coord_max"] <= 1000
            )
    except ValidationError as e:
        out["float_relaxed"] = "FAILED"
        out["float_relaxed_errors"] = _format_validation_errors(e)

    # 3) Try extracting JSON from a prose preamble (find first { … last }).
    first = raw.find("{")
    last = raw.rfind("}")
    if first > 0:
        out["has_prose_preamble"] = True
        out["prose_preamble"] = raw[:first][:300]
    if 0 <= first < last:
        extracted = raw[first:last + 1]
        try:
            VlmSegmentation.model_validate_json(extracted)
            out["extracted_strict_int"] = "PASSED"
        except ValidationError:
            try:
                _VlmSegmentationFloat.model_validate_json(extracted)
                out["extracted_float_relaxed"] = "PASSED"
            except ValidationError:
                out["extracted_extract"] = "FAILED both modes"

    # 4) Markdown fence extraction (for the ``` ``` case).
    if "```" in raw:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                VlmSegmentation.model_validate_json(m.group(1))
                out["fence_strict_int"] = "PASSED"
            except ValidationError:
                pass

    return out


def _sanitise_case(case: str) -> str:
    return (case.replace(":", "_").replace("/", "_")
                .replace("(", "_").replace(")", "_"))


# ── Driver ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-pro",
                    help="OpenRouter alias (default: gemini-pro)")
    ap.add_argument("--case", required=True,
                    help="Single case identifier to diagnose")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="JPEG quality for input encoding (default: 90, "
                         "matches the production Pro run)")
    ap.add_argument("--max-image-dim", type=int, default=None,
                    help="Optional longest-side cap on input image")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="max_tokens cap (default: 32768 — generous, since "
                         "we want to see the FULL response, not a truncated one)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0 — matches the main "
                         "ablation script). Google's Gemini 3 docs warn that "
                         "thinking models like 3.1 Pro may loop at temperature "
                         "<1.0; pass --temperature 1.0 to test that hypothesis.")
    ap.add_argument("--out-dir", default="results/ablation_vlm_seg_diag")
    args = ap.parse_args()

    dataset_dir = REPO / "training" / "dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    entry = next((r for r in manifest if r["case"] == args.case), None)
    if entry is None:
        sys.exit(f"case {args.case!r} not in manifest")

    fname = entry["filename"]
    img_path = dataset_dir / "maps" / fname
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.width, img.height

    if args.max_image_dim and max(img.width, img.height) > args.max_image_dim:
        scale = args.max_image_dim / max(img.width, img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(args.jpeg_quality))
    png_bytes = buf.getvalue()
    media_type = "image/jpeg"

    agent = Agent(
        "test",
        output_type=NativeOutput(VlmSegmentation),
        retries=3,
        output_retries=0,
        model_settings={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        instructions=DEFAULT_PROMPT,
    )
    model = resolve_model(args.model)

    print(f"\n=== DIAGNOSTIC: {args.case} on {args.model} ===")
    print(f"  image: {img_path}  ({orig_w}x{orig_h})")
    if args.max_image_dim:
        print(f"  resized to: {img.width}x{img.height}")
    print(f"  payload: {len(png_bytes)/1024:.1f} KB JPEG{args.jpeg_quality}")
    print(f"  max_tokens: {args.max_tokens}  temperature: {args.temperature}")
    print()

    t0 = time.time()
    result = None
    err_str = None
    try:
        result = agent.run_sync(
            [
                BinaryContent(data=png_bytes, media_type=media_type),
                "Locate the drawn site boundary and output it per the schema.",
            ],
            model=model,
            usage_limits=UsageLimits(request_limit=4),
        )
        dt = time.time() - t0
        print(f"  case PASSED in {dt:.1f}s")
        print(f"    n_polygons: {len(result.output.polygons)}")
        print(f"    notes: {result.output.notes[:200]}")
    except Exception as e:
        dt = time.time() - t0
        err_str = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  case FAILED in {dt:.1f}s with {type(e).__name__}")
        print(f"  exception: {str(e)[:300]}")
        cause = e.__cause__
        while cause is not None:
            print(f"  caused by: {type(cause).__name__}: {str(cause)[:200]}")
            cause = cause.__cause__

    print(f"\n  raw responses captured: {len(_raw_responses)}")

    out_path = REPO / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    # Tag filename with temperature if it's non-default, so multiple runs
    # of the same case don't clobber each other.
    fname_stem = _sanitise_case(args.case)
    if args.temperature != 0.0:
        fname_stem = f"{fname_stem}_t{args.temperature}"
    out_file = out_path / f"{fname_stem}.json"

    diagnostics = []
    for i, raw in enumerate(_raw_responses):
        print(f"\n--- raw response #{i + 1} ({len(raw)} chars) ---")
        # Show first/last chunk so it's clear what the model emitted
        print(f"  FIRST 400 chars: {raw[:400]!r}")
        if len(raw) > 400:
            print(f"  LAST  200 chars: {raw[-200:]!r}")

        diag = diagnose_response(raw)
        diagnostics.append({"index": i, "raw_text": raw, "diagnosis": diag})

        print(f"\n  starts with '{{': {diag['starts_with_brace']}, "
              f"ends with '}}': {diag['ends_with_brace']}")
        print(f"  contains markdown fence (```): {diag['contains_markdown_fence']}")
        print(f"  strict int validation: {diag['strict_int']}")
        for err in diag.get("strict_int_errors", []):
            print(f"    - [{err['type']}] {err['loc']}: {err['msg']}")
        print(f"  float-relaxed validation: {diag['float_relaxed']}")
        if "coord_min" in diag:
            print(f"    polygons={diag['n_polygons']}, "
                  f"vertices={diag['n_vertices_total']}, "
                  f"coord range [{diag['coord_min']}, {diag['coord_max']}]")
            print(f"    coords integer-valued: {diag['all_coords_integer_valued']}")
            print(f"    coords in [0, 1000]:   {diag['coords_in_0_1000']}")
        for err in diag.get("float_relaxed_errors", []):
            print(f"    - [{err['type']}] {err['loc']}: {err['msg']}")
        if diag.get("has_prose_preamble"):
            print(f"  prose preamble before first '{{': {diag['prose_preamble'][:120]!r}")
        for key in ("extracted_strict_int", "extracted_float_relaxed", "fence_strict_int"):
            if key in diag:
                print(f"  {key}: {diag[key]}")

    out_file.write_text(json.dumps({
        "case": args.case,
        "model": args.model,
        "image_path": str(img_path),
        "image_size_orig": [orig_w, orig_h],
        "image_size_sent": [img.width, img.height],
        "jpeg_quality": args.jpeg_quality,
        "max_tokens": args.max_tokens,
        "elapsed_seconds": round(time.time() - t0, 2),
        "result_succeeded": result is not None,
        "exception": err_str,
        "n_raw_responses": len(_raw_responses),
        "diagnostics": diagnostics,
    }, indent=2, default=str))

    print(f"\nSaved: {out_file}")
    print(f"\nVerdict for {args.case}:")
    if not _raw_responses:
        print("  No raw text captured — the failure happened before validation "
              "(e.g., a network error or empty/thinking-only response). Check "
              "the exception above.")
    else:
        d = diagnostics[-1]["diagnosis"]
        if d["strict_int"] == "PASSED":
            print("  Strict-int passes — case would have been kept by the main "
                  "script. If you saw a failure, something else is going on.")
        elif d["float_relaxed"] == "PASSED":
            if not d.get("coords_in_0_1000", True):
                print(f"  ROOT CAUSE: Pro emitted coords in [{d['coord_min']:g}, "
                      f"{d['coord_max']:g}] — outside the prompted [0, 1000] "
                      f"range. Coord-convention violation.")
            elif not d.get("all_coords_integer_valued", True):
                print("  ROOT CAUSE: Pro emitted non-integer floats — fails the "
                      "strict int schema. Format-compliance violation.")
            else:
                print("  Float-relaxed passes with in-range integer-valued "
                      "coords — but strict-int still failed. Inspect "
                      "strict_int_errors above.")
        elif d.get("has_prose_preamble") and (
                d.get("extracted_strict_int") == "PASSED"
                or d.get("extracted_float_relaxed") == "PASSED"):
            print(f"  ROOT CAUSE: prose preamble before JSON "
                  f"(preamble = {d['prose_preamble'][:100]!r}). "
                  f"strip_markdown_fences misses this because the response "
                  f"doesn't start with '{{' and has no ``` fence.")
        elif not d["ends_with_brace"]:
            print("  ROOT CAUSE: response did NOT end with '}' — the model was "
                  "still emitting when the max_tokens cap hit. Truncated JSON.")
        else:
            print("  Failure mode UNCLEAR — inspect the raw text above for the "
                  "actual content (could be a refusal, an error message, or "
                  "something structurally malformed).")


if __name__ == "__main__":
    main()
