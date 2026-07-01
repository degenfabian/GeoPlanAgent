# ablations

One self-contained script per ablation. Each writes its run artifacts under
`results/ablations/<name>/` (the exact directories are constants in
`geoplanagent/paths.py`). Run any script with `--help` for its flags. All LLM
ablations call OpenRouter and cost API credits.

| Script | Paper | What it does |
|---|---|---|
| `vlm_e2e_pdf_to_geojson.py` | Table 1 (baselines) | Single-call VLM baseline: the model reads the PDF and emits GeoJSON directly — no tools |
| `locate_only_eval.py` | Table 2 | Locate sub-agent in isolation, scored on centroid error; default is production (the single `place` tool), `--all-tools` enables all six geocoders |
| `locate_vlm_direct.py` | Table 2 | Single-shot VLM geocoding of the PDF (no gazetteer) |
| `vlm_segmentation.py` | Figure 3 | VLM traces the boundary polygon directly on the map image |
| `sam_base_prompt_search.py` | Figure 3, Table 12 | Vanilla SAM3 (no fine-tune) swept over five text prompts |
| `build_vlm_e2e_subset.py` | Table 4 | Builds the 40-case stratified subset (`subset_40.json`) used by the expensive baselines |

Shared inputs that live here:

- `subset_40.json` — the stratified 40-case subset definition (quality ×
  complexity strata).
- `cached_pdf_info_for_locate_ablations.json` — a frozen Reader output shared
  by the locate ablations, so every locate ablation sees identical information
  extracted from the PDF to isolate locate behavior
- `utils.py` — helper functions shared between the scripts.

The collapsed-reader ablation is not a script here — it's a flag on the main
benchmark (`benchmark_runner.py --no-reader`).
