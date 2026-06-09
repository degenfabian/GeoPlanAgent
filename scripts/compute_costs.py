"""Audit per-case and aggregate $/doc costs from cached metrics.json files.

Reads every metrics.json under a results directory, decomposes tokens by
agent stage (reader / worker / locate), multiplies by per-MTok prices, and
emits a per-case CSV + a printed summary table comparing token-rate cost
vs the paper's reported $/doc.

If the locate-telemetry patches (locate_request_tokens / locate_response_tokens
/ locate_calls in agent_stats) are present, the cost includes Locate; on
older cached runs that pre-date the patch, Locate is 0 and the printed
"locate share" warns "TELEMETRY MISSING — re-run to populate".

When --query-openrouter is passed, also looks up exact billed cost per
captured generation_id via the OpenRouter /api/v1/generation endpoint and
prints a paired comparison.

Usage:
  uv run scripts/compute_costs.py results/cost_audit_v1
  uv run scripts/compute_costs.py results/cost_audit_v1 --model gemini-flash
  uv run scripts/compute_costs.py results/cost_audit_v1 --query-openrouter

The price table below is the paper's implicit assumption reverse-engineered
from cached token counts ÷ reported $/doc. Update if OpenRouter rates
change or you're auditing a different model.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# Per-MTok USD prices. Reverse-engineered from the paper's table:
#   cost_per_doc = mean_in × pin/1e6 + mean_out × pout/1e6
# These values yield numbers that match the paper rows exactly. If the
# OpenRouter rates change, update here and re-run the script.
PRICES: dict[str, tuple[float, float]] = {
    # OpenRouter model id (or short alias) → (input_$/MTok, output_$/MTok)
    "google/gemini-3-flash-preview":    (0.55,  2.20),
    "gemini-flash":                     (0.55,  2.20),
    "google/gemini-3.1-pro-preview":    (1.25, 12.50),
    "gemini-pro":                       (1.25, 12.50),
    "anthropic/claude-opus-4.7":        (5.00, 25.00),
    "claude-opus":                      (5.00, 25.00),
    "openai/gpt-5.5-pro":               (30.0, 180.0),
    "gpt-5.5-pro":                      (30.0, 180.0),
}

DEFAULT_MODEL = "gemini-flash"


# ── Token extraction ─────────────────────────────────────────────────────


def _pull_tokens(metrics: dict) -> dict:
    """Pull per-stage token counts from one case's metrics.json."""
    s = (metrics or {}).get("agent_stats", {}) or {}
    return {
        "reader_in":  int(s.get("reader_request_tokens",  0) or 0),
        "reader_out": int(s.get("reader_response_tokens", 0) or 0),
        "worker_in":  int(s.get("worker_request_tokens",  0) or 0),
        "worker_out": int(s.get("worker_response_tokens", 0) or 0),
        "locate_in":  int(s.get("locate_request_tokens",  0) or 0),
        "locate_out": int(s.get("locate_response_tokens", 0) or 0),
        "locate_n":   int(s.get("locate_n_calls",         0) or 0),
        "n_turns":    int(s.get("n_turns",                0) or 0),
        "validator_retries": int(s.get("validator_retries", 0) or 0),
        "locate_generation_ids": [
            c.get("generation_id") for c in (s.get("locate_calls") or [])
            if c.get("generation_id")
        ],
    }


# ── Cost arithmetic ──────────────────────────────────────────────────────


def _cost(tok_in: int, tok_out: int, pin: float, pout: float) -> float:
    return (tok_in * pin + tok_out * pout) / 1_000_000.0


def case_cost(tok: dict, pin: float, pout: float) -> dict:
    return {
        "reader_cost": _cost(tok["reader_in"], tok["reader_out"], pin, pout),
        "worker_cost": _cost(tok["worker_in"], tok["worker_out"], pin, pout),
        "locate_cost": _cost(tok["locate_in"], tok["locate_out"], pin, pout),
        "total_cost":  _cost(tok["reader_in"] + tok["worker_in"] + tok["locate_in"],
                              tok["reader_out"] + tok["worker_out"] + tok["locate_out"],
                              pin, pout),
    }


# ── OpenRouter /v1/generation lookup ─────────────────────────────────────


def query_openrouter_cost(gen_id: str, api_key: str) -> Optional[float]:
    """Return the exact billed `total_cost` for a generation id, or None."""
    url = f"https://openrouter.ai/api/v1/generation?id={gen_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        d = data.get("data") or {}
        return float(d.get("total_cost", 0.0))
    except urllib.error.HTTPError as e:
        print(f"  [openrouter] {gen_id[:24]}…: HTTP {e.code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [openrouter] {gen_id[:24]}…: {e!s:.80}", file=sys.stderr)
        return None


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path,
                    help="e.g. results/cost_audit_v1/gemini-flash/")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Pricing key (default: gemini-flash)")
    ap.add_argument("--query-openrouter", action="store_true",
                    help="Also hit /v1/generation?id= for each captured "
                         "generation_id and report exact billed cost. "
                         "Requires OPENROUTER_API_KEY.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Per-case CSV output (default: <results_dir>/cost_audit.csv)")
    args = ap.parse_args()

    if args.model not in PRICES:
        print(f"Unknown model '{args.model}'. Known: {sorted(PRICES)}",
              file=sys.stderr)
        return 1
    pin, pout = PRICES[args.model]

    # Recurse if the user passed the parent results/ rather than a model dir.
    if not any(args.results_dir.glob("*/metrics.json")):
        # try one level down (results/<output>/<model>/<case>/metrics.json)
        candidates = list(args.results_dir.glob("*/*/metrics.json"))
        if candidates:
            print(f"Note: passed parent; auditing {candidates[0].parent.parent}")
            args.results_dir = candidates[0].parent.parent

    case_dirs = sorted(
        c for c in args.results_dir.iterdir()
        if c.is_dir() and (c / "metrics.json").exists())
    if not case_dirs:
        print(f"No metrics.json under {args.results_dir}", file=sys.stderr)
        return 1
    print(f"Auditing {len(case_dirs)} cases under {args.results_dir}")
    print(f"Prices ({args.model}): in=${pin}/MTok, out=${pout}/MTok\n")

    rows = []
    for cd in case_dirs:
        metrics = json.loads((cd / "metrics.json").read_text())
        tok = _pull_tokens(metrics)
        cost = case_cost(tok, pin, pout)
        rows.append({
            "case": cd.name,
            **tok,
            **cost,
            "iou": metrics.get("iou"),
            "processing_time": metrics.get("processing_time"),
        })

    # ── Per-case CSV ─────────────────────────────────────────────────────
    out_csv = args.out_csv or (args.results_dir / "cost_audit.csv")
    fieldnames = [
        "case", "iou", "processing_time", "n_turns", "validator_retries",
        "reader_in", "reader_out", "worker_in", "worker_out",
        "locate_n", "locate_in", "locate_out",
        "reader_cost", "worker_cost", "locate_cost", "total_cost",
    ]
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"Wrote {out_csv}")

    # ── Aggregate summary ────────────────────────────────────────────────
    def mean(key):  return st.mean([r[key] for r in rows])
    def median(key): return st.median([r[key] for r in rows])

    has_locate = any(r["locate_n"] > 0 for r in rows)
    print("\n┌─ Aggregate cost summary ─────────────────────────────┐")
    print(f"│ Cases:                    {len(rows):>7d}                 │")
    print(f"│ Mean reader $/doc:        {mean('reader_cost'):>9.5f}             │")
    print(f"│ Mean worker $/doc:        {mean('worker_cost'):>9.5f}             │")
    if has_locate:
        print(f"│ Mean locate $/doc:        {mean('locate_cost'):>9.5f}  "
              f"({mean('locate_n'):.1f} calls/case avg) │")
    else:
        print(f"│ Mean locate $/doc:        {0.0:>9.5f}  TELEMETRY MISSING  │")
    print(f"│ Mean TOTAL $/doc:         {mean('total_cost'):>9.5f}             │")
    print(f"│ Median TOTAL $/doc:       {median('total_cost'):>9.5f}             │")
    print(f"│ Mean total tokens / doc:  "
          f"{int(mean('reader_in') + mean('worker_in') + mean('locate_in') + mean('reader_out') + mean('worker_out') + mean('locate_out')):>7d}                 │")
    print( "└──────────────────────────────────────────────────────┘")
    if not has_locate:
        print("\n⚠  No locate telemetry in any case. Either: (a) this run "
              "pre-dates the\n   locate-tracking patch in state.py / locate_agent.py / runtime.py, "
              "or\n   (b) every case was district_lookup (no propose_centers fired). "
              "Re-run\n   with the patches in place to get a locate-inclusive cost.")
        return 0

    # Compare paper's $0.019 vs our $0.04 gut
    paper = 0.019
    gut = 0.040
    me = mean("total_cost")
    print(f"\nComparison:")
    print(f"  Paper reported           ${paper:.4f}/doc  (reader + worker only)")
    print(f"  My gut guess             ${gut:.4f}/doc")
    print(f"  This audit (with Locate) ${me:.4f}/doc")
    if me > 0:
        print(f"  Locate fraction of cost:  "
              f"{mean('locate_cost')/me*100:.1f}%")

    # ── Optional: hit OpenRouter for exact billed cost ──────────────────
    if args.query_openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("\nOPENROUTER_API_KEY not set; skipping /v1/generation lookups")
            return 0
        print("\nQuerying OpenRouter /v1/generation for exact billed cost…")
        exact_total = 0.0
        n_ids = 0; n_ok = 0
        for r in rows:
            for gid in r.get("locate_generation_ids", []):
                n_ids += 1
                c = query_openrouter_cost(gid, api_key)
                if c is not None:
                    exact_total += c
                    n_ok += 1
        if n_ids:
            print(f"  {n_ok}/{n_ids} ids returned billed cost; "
                  f"summed locate cost (exact) = ${exact_total:.4f}")
            print(f"  Per case: ${exact_total / len(rows):.5f}/doc "
                  f"(token-rate estimate was ${mean('locate_cost'):.5f})")
        else:
            print("  No generation_ids captured in cached results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
