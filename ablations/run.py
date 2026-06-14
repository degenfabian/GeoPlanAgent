"""Single entry point for every paper ablation.

Each subcommand forwards to one self-contained harness and exposes that
harness's full CLI — run `<subcommand> -h` for its flags. All
subcommands except build-subset, audit-locate, reader-cache and
sam-prompts call OpenRouter and cost API credits; every harness
supports --max-cases N for a cheap smoke run first.

Usage:
    uv run ablations/run.py <subcommand> [harness flags...]

Subcommands and the paper rows they produce:
    vlm-e2e           Table 1 VLM end-to-end rows + per-model table
    vlm-seg           Segmentation comparison, VLM-direct rows
    sam-prompts       Vanilla-SAM3 prompt sweep (appendix)
    locate            Locate-stage table: place-only / 6-tool
    locate-vlm        Locate-stage table, VLM-direct row
    collapsed-reader  Table 1 Collapsed Reader row
                      (benchmark_runner.py with --no-reader injected)
    build-subset      Stratified 40-case subset (offline, seed 42)
    reader-cache      PDFInfo cache shared by the locate ablations
    audit-locate      Post-hoc audit of locate trajectories (offline)

The exact invocation behind every published row is listed in
ablations/README.md.
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COMMANDS = {
    "vlm-e2e": "ablations.vlm_e2e_pdf_to_geojson",
    "vlm-seg": "ablations.vlm_segmentation",
    "sam-prompts": "ablations.sam_base_prompt_search",
    "locate": "ablations.locate_only_eval",
    "locate-vlm": "ablations.locate_vlm_direct",
    "collapsed-reader": "benchmark_runner",
    "build-subset": "ablations.build_vlm_e2e_subset",
    "reader-cache": "ablations.extract_pdf_info_cache",
    "audit-locate": "ablations.audit_locate_results",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = sys.argv.pop(1)
    if cmd not in COMMANDS:
        print(f"Unknown subcommand '{cmd}'. Choose from: {' '.join(COMMANDS)}", file=sys.stderr)
        return 2
    if cmd == "collapsed-reader":
        sys.argv.insert(1, "--no-reader")
    sys.argv[0] = COMMANDS[cmd]
    runpy.run_module(COMMANDS[cmd], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
