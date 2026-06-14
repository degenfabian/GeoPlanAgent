"""Per-case and aggregate $/doc costs from cached metrics.json files.

Decomposes token usage by stage (reader / worker / locate), multiplies by
per-MTok prices, and writes a per-case CSV next to a printed summary.
District-shortcut cases make no locate LLM call, so their locate column is
zero and the summary notes it.

Usage:
  uv run scripts/compute_costs.py results/cost_audit_v1
  uv run scripts/compute_costs.py results/cost_audit_v1 --model gemini-flash
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path


from _pricing import DEFAULT_MODEL, PRICES  # scripts/ on sys.path when run as a file


def _pull_tokens(metrics: dict) -> dict:
    s = (metrics or {}).get("agent_stats", {}) or {}
    return {
        "reader_in": int(s.get("reader_request_tokens", 0) or 0),
        "reader_out": int(s.get("reader_response_tokens", 0) or 0),
        "worker_in": int(s.get("worker_request_tokens", 0) or 0),
        "worker_out": int(s.get("worker_response_tokens", 0) or 0),
        "locate_in": int(s.get("locate_request_tokens", 0) or 0),
        "locate_out": int(s.get("locate_response_tokens", 0) or 0),
        "locate_n": int(s.get("locate_n_calls", 0) or 0),
        "n_turns": int(s.get("n_turns", 0) or 0),
        "validator_retries": int(s.get("validator_retries", 0) or 0),
    }


def _cost(tok_in: int, tok_out: int, pin: float, pout: float) -> float:
    return (tok_in * pin + tok_out * pout) / 1_000_000.0


def case_cost(tok: dict, pin: float, pout: float) -> dict:
    return {
        "reader_cost": _cost(tok["reader_in"], tok["reader_out"], pin, pout),
        "worker_cost": _cost(tok["worker_in"], tok["worker_out"], pin, pout),
        "locate_cost": _cost(tok["locate_in"], tok["locate_out"], pin, pout),
        "total_cost": _cost(
            tok["reader_in"] + tok["worker_in"] + tok["locate_in"],
            tok["reader_out"] + tok["worker_out"] + tok["locate_out"],
            pin,
            pout,
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="e.g. results/cost_audit_v1/gemini-flash/")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Pricing key (default: gemini-flash)")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Per-case CSV output (default: <results_dir>/cost_audit.csv)",
    )
    args = ap.parse_args()

    if args.model not in PRICES:
        print(f"Unknown model '{args.model}'. Known: {sorted(PRICES)}", file=sys.stderr)
        return 1
    pin, pout = PRICES[args.model]

    # accept the parent results/ dir as well as a model dir
    if not any(args.results_dir.glob("*/metrics.json")):
        candidates = list(args.results_dir.glob("*/*/metrics.json"))
        if candidates:
            print(f"Note: passed parent; auditing {candidates[0].parent.parent}")
            args.results_dir = candidates[0].parent.parent

    case_dirs = sorted(
        c for c in args.results_dir.iterdir() if c.is_dir() and (c / "metrics.json").exists()
    )
    if not case_dirs:
        print(f"No metrics.json under {args.results_dir}", file=sys.stderr)
        return 1
    print(f"Auditing {len(case_dirs)} cases under {args.results_dir}")
    print(f"Prices ({args.model}): in=${pin}/MTok, out=${pout}/MTok\n")

    rows = []
    for cd in case_dirs:
        metrics = json.loads((cd / "metrics.json").read_text())
        tok = _pull_tokens(metrics)
        rows.append(
            {
                "case": cd.name,
                **tok,
                **case_cost(tok, pin, pout),
                "iou": metrics.get("iou"),
                "processing_time": metrics.get("processing_time"),
            }
        )

    out_csv = args.out_csv or (args.results_dir / "cost_audit.csv")
    fieldnames = [
        "case",
        "iou",
        "processing_time",
        "n_turns",
        "validator_retries",
        "reader_in",
        "reader_out",
        "worker_in",
        "worker_out",
        "locate_n",
        "locate_in",
        "locate_out",
        "reader_cost",
        "worker_cost",
        "locate_cost",
        "total_cost",
    ]
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"Wrote {out_csv}")

    def mean(key):
        return st.mean([r[key] for r in rows])

    has_locate = any(r["locate_n"] > 0 for r in rows)
    total_tokens = int(
        mean("reader_in")
        + mean("worker_in")
        + mean("locate_in")
        + mean("reader_out")
        + mean("worker_out")
        + mean("locate_out")
    )
    print(f"\nCases:               {len(rows)}")
    print(f"Mean reader $/doc:   {mean('reader_cost'):.5f}")
    print(f"Mean worker $/doc:   {mean('worker_cost'):.5f}")
    if has_locate:
        print(
            f"Mean locate $/doc:   {mean('locate_cost'):.5f}  ({mean('locate_n'):.1f} calls/case)"
        )
    else:
        print("Mean locate $/doc:   0.00000  (no locate telemetry in this run)")
    print(f"Mean total $/doc:    {mean('total_cost'):.5f}")
    print(f"Median total $/doc:  {st.median([r['total_cost'] for r in rows]):.5f}")
    print(f"Mean tokens/doc:     {total_tokens}")

    if not has_locate:
        print(
            "\nEvery case took the district shortcut (no locate LLM call), "
            "so the total above excludes locate."
        )
        return 0

    me = mean("total_cost")
    if me > 0:
        print(f"\nLocate fraction of total cost: {mean('locate_cost') / me * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
