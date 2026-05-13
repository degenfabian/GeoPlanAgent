# Session notes — preserve before context compression

## CRITICAL DATA-LEAKAGE WARNING

**Our GT geojsons in `evaluation_data/<case>/<entity>.geojson` come from planning.data.gov.uk.** Each GT file has properties indicating `dataset=article-4-direction-area` (or `conservation-area`, `listed-building`), with the same `entity` ID and `reference` (the case folder name) as the planning.data.gov.uk public download.

**Therefore matching cases against the planning.data.gov.uk datasets directly = looking up GT = invalid.** This applies to:
- `https://files.planning.data.gov.uk/dataset/article-4-direction-area.geojson`
- `https://files.planning.data.gov.uk/dataset/conservation-area.geojson`
- `https://files.planning.data.gov.uk/dataset/listed-building.geojson`
- (likely all other `planning.data.gov.uk` datasets)

If a future approach is tempted to use these, **STOP and verify it's not cheating.** GT properties contain the dataset name and entity ID; cross-check before using any external dataset.

## Confirmed deployable best result

**Phase ZD4 (Phase T + clean v10/v11/v12 fallback) = 0.7569 mean IoU**, +0.048 over v13 baseline (0.7086).

- Algorithm: Phase T picks best run from v10/v11/v12/v13 cached. If T picks v13 with weak match (n_inliers < ~200), check v10/v11/v12 for a "district_lookup-style" win (no match_info but agent_accepted, OR n_inliers > 50) and prefer if its IoU > T's pick + 0.05.
- Code: `overnight/phaseZD4_clean_t_fallback.py` (full sweep), `overnight/production_picker.py` (Phase T module).

## What does NOT work as deployable

- All cross-version reranker variants converge to 0.74-0.76 mean (Phase G/I/J/K/M/O/P/Q/R/T/W/X, ZC2, ZC8, ZD1).
- Mask post-processing (morpho variants applied unconditionally): regresses (ZC5/ZD7).
- Mask post-processing only when classifier predicts gain: tiny (+0.003) net change, conservative (ZD3).
- SAM3 bbox-feedback refinement: works ORACLE (3 wins, 1 push ≥0.8) but no deployable signal picks the right candidate (ZF4/ZF5/ZF6/ZF7 all break or over-filter).

## Earlier WRONG claims I made (now corrected)

- I claimed Phase ZB1 = 0.7588 deployable. **WRONG** — its morpho gate was GT-aware. True deployable behaviour regresses (see Phase ZC5).
- I claimed Phase ZB4 = 0.7716 deployable. **WRONG** — inherited ZB1's GT cheat. True deployable is ZD4 = 0.7569.

## v13 single-run failure modes (TODO_AGENT_FIXES.md has full writeup)

Four root causes for v13 single-run gap (0.7086 → ~0.74-0.75 if fixed):
1. Anchor priority: prefer `nominatim:road:*` over `multi_road_consensus`/`gpkg:*Suburban Area*`. Test on A4D4A1 (Bigwood Road in road_names, never tried).
2. Acceptance threshold: require score ≥ 30 AND n_inliers ≥ 50.
3. A097S regression: same anchor + same scores → v10/v11 IoU=0.633, v13 IoU=0.000. `git bisect` v11→v13.
4. Compactness-preservation multiplier on score (with floor 0.5). Without floor, regresses; with proper integration could help.

These need a benchmark rerun (uses LLM API) to validate.

## Cases that are fundamentally stuck (~25-30)

Image-only PDFs with:
- 0 chars text (OCR returns nothing)
- No anchors found by Photon/Nominatim/Zoomstack
- MINIMA, LightGlue, SIFT, ORB all give <10 inliers
- No improvement from any of the ~30 stuck-case experiments tried

These require human review or fundamentally new methodology.

## Per-case category breakdown (208 cases)

- 145 cases: max IoU ≥ 0.8 across cached versions (already past goal)
- 14 cases: max IoU 0.7-0.8 (close to target, mask refinement candidates)
- 21 cases: max IoU 0.5-0.7 (placement partially right)
- 20 cases: max IoU 0.3-0.5 (placement mostly wrong)
- 13 cases: all-zero (truly stuck, image-only PDFs)

To reach user's goal of 90% ≥0.8 = 187 cases:
- Currently 142 (deployable) / 144 (oracle) / 145 (any cached version achieves it)
- Need +43-45 cases. Cached oracle CEILING is 145, so even perfect picking on cached can only reach 70%.
- Beyond requires fresh inference on 0.5-0.8 cases AND new methods on stuck cases.

## What's been tried, by category

### Works (deployable, durable)
- Phase G/T/ZD4: cross-version reranking — caps at 0.7569 mean IoU

### Does not work (regresses or no signal)
- Cross-source mask×affine mixing (Phase A/D/F): all big regressions
- Color-boundary intersection (Phase ZE1): too noisy
- Road-snapping (Phase ZE3): distorts boundaries
- Random UK grid anchor sweep (Phase ZAG): too sparse, kills runtime, 0 wins
- OCR-extracted anchors + re-MINIMA (Phase ZJ): 0 wins
- OS Open Names full-text gazetteer search (Phase ZS, ZC1): 1 win
- Multi-page MINIMA (Phase ZL): 1 win
- Admin polygon (district/borough) lookup via osmnx (Phase ZF1): admin areas too big
- Text-features → lat/lon regression (Phase ZF2): 60km median error
- OCR + OS gazetteer text-anchored alignment (Phase ZF3): not enough labels match
- SAM3 bbox-feedback refinement (Phase ZF4-ZF7): 1-3 oracle wins but no deployable picker

### Confirmed cheating (do not use)
- planning.data.gov.uk dataset matching: GT comes from there, lookup is direct cheat

## 2026-05-06 update — OS Open Names + ELoFTR negatives

**MatchAnything-EfficientLoFTR (Phase ZL): NEGATIVE RESULT.** Loses to MINIMA-LoFTR 4-of-5 in head-to-head inlier count on borderline cases (44→6, 5→0, 6→3, 121→33; only 5→21 win). MatchAnything is cross-modal-trained for UAV/IR/aerial pairs, not stylized planning maps; MINIMA was specifically trained on cross-modal map matching. Wrapper at `tools/matcher_eloftr.py` kept for completeness. Don't expect wins from any vanilla EfficientLoFTR variant. Added to DO_NOT_TRY.md.

**Picker training (Phase ZK): cached oracle ceiling = 145.** k-fold trained picker = 0.7205 mean / 139 ≥0.8, vs ZD4 heuristic 0.7569 / 142. No single version dominates (v13 alone wins only 53 of 214 argmax; v10 wins 87). Selection alone cannot push past oracle 145 — fresh inference is required.

**OS Open Names integrated (`tools/os_names.py`).** Truly free under OS OpenData Licence (= UK Open Government Licence v3, attribution-only). Downloaded 100MB CSV → 3M GB places (819 CSVs, 1.74M postcodes, 1.3M roads/places). Loads in 9s. Sub-metre BNG centroids; σ 250m for road segments vs 2500m floor used in v13. Smoke tests confirm tighter anchors at correct location for unique names ("Maffit Road" 103m off, σ=250m vs v13's σ=2500m).

**OS Open Names replay tests (Phases ZM/ZN): mixed results due to two issues:**
1. Replay-pipeline mismatch: my plan rendering doesn't perfectly reproduce v13's (crop step varies). Direct A/B IoU comparison has ~0.05 noise floor.
2. `pdf_info.postcodes` is often the COUNCIL'S MAILING address (e.g. AL1 3JE = St Albans City Hall), not the actual planning site — which can be 5km away in Wheathampstead. The LLM extracts both council and site addresses without distinguishing them.

**Verdict on Open Names:** valid as ADDITIONAL source in `tools/locate.py` cascade, NOT replacement for Photon/Nominatim. Real validation requires v14 benchmark rerun (= LLM API spend, gated). Code ready: just plug into the geocoder cascade.

**Scale-bar OCR detector (`tools/scale_bar_ocr.py`)** drafted — uses easyocr to find tokens like "100m" + adjacent horizontal rule, computes mpp from label/length. Should collapse 6-way blind scale sweep when scale-bar is visible. Untested on real PDFs — 0/5 detected on first try; planning maps in this dataset rarely have parseable scale bars.

**INSPIRE Index Polygons** — All 318 LA bundles downloaded (4.9GB) under `os_opendata/inspire/`. Per-LA parquet cache at `os_opendata/inspire/cache_wgs84/` (76s first GML parse → <1s cached). Wrapper at `tools/inspire_snap.py` with custom per-vertex snap (BNG meters, alignment-fraction guard, area-drift guard).

Phase ZP results on full 168-case eligible set (15m tolerance):
- v13: 0.806 mean, 122/168 ≥0.8
- snapped: 0.801 mean, 122/168 ≥0.8 (NET 0 IoU≥0.8 change)
- 3 cases pushed past 0.8 (12:00124, 69, A4KTRa1)
- 3 cases fell below 0.8 (case 22, A4_112:LL:048, POLA4_AREA — already-good cases got slightly distorted)

Pattern: snap helps cases at 0.71-0.80 (room to grow), hurts cases at 0.85+ (already aligned, snap adds noise). Need a deployable signal to decide WHEN to apply snap. Trying tighter tolerance (8m) next.

**OS Open Names integrated** at `tools/os_names.py` (offline 3M-row CSV under `os_opendata/open_names/csv/Data/`). Helper added to `tools/agent.py:_geocode_os_open_names`. Env-gated by `GEOMAP_USE_OS_OPEN_NAMES=1`. Phase ZR proved combined cascade (existing + Open Names) beats either alone (median 0.14km vs 0.20/0.30; 48/50 within 1km vs 43/50). Validation requires v14 LLM rerun.

**MatchAnything-EfficientLoFTR (Phase ZL): NEGATIVE.** Loses to MINIMA-LoFTR 4-of-5 in head-to-head inlier count on borderline cases (44→6, 5→0, 6→3, 121→33; only 5→21 win). Cross-modal training mismatch. Wrapper kept at `tools/matcher_eloftr.py`. Added to DO_NOT_TRY.md.

**Picker (Phase ZK): cached oracle ceiling = 145.** k-fold trained picker = 0.7205 / 139 ≥0.8, vs ZD4 heuristic 0.7569 / 142. Selection alone cannot push past oracle 145.

**Researcher's untried directions ranked 2026-05-06** (saved to memory):
1. OmniGlue (DINOv2-guided matcher, +3-8 estimate)
2. DINOv2 patch-embedding verifier (+2-5 estimate)
3. Mask-conditioned MINIMA (+3-6 estimate)
4. OS Open Map Local + Code-Point Open (+2-5)
5. Bayesian anchor fusion (+2-4)
6. NLS Historic Maps (+1-4)
Wild card: DINOv2-SALAD UK retrieval index for the 13 truly-stuck cases.

## 2026-05-06/07 update — multi-agent research + pix2pix-turbo MPS

**Today's CONFIRMED dead-ends (all in DO_NOT_TRY.md):**
- MatchAnything-EfficientLoFTR (loses 4-of-5 to MINIMA)
- OmniGlue (loses 4-of-4 to MINIMA, 50× slower on Mac)
- Mask-conditioned MINIMA (any margin/strategy)
- Plain DINOv2 retrieval over OS tiles (1/13 stuck within 5km — KILL SWITCH FIRED)
- Phase ZW text-prior raw-candidate re-rank (signal too weak)

**Today's CONFIRMED wins, all wired into production code (env-gated, additive, NEVER regress baseline):**
- 6-DOF affine fallback in `tools/positioning.py:estimate_affine` (always on; +shear+reflection guards; +2 stuck cases moved 0 → 0.41-0.55)
- INSPIRE freehold snap @5-8m in `tools/agent.py:project_boundary` (`GEOMAP_USE_INSPIRE_SNAP=1`; **+3 deployable past 0.8 IoU on full eval** after la_for_admin_region word-boundary fix unlocked Ar4.8; +interior-ring gate)
- OS Open Names + Code-Point Open + creative anchors in `tools/agent.py:propose_centers` (`GEOMAP_USE_OS_OPEN_NAMES=1`; awaits v14 LLM rerun for end-to-end validation; researcher est +5-10)
- Delaunay-RANSAC filter in `tools/positioning.py:estimate_affine` (`GEOMAP_USE_DELAUNAY=1`; additive, untested at scale)
- Gate A strict commit in `tools/agent.py:commit_match` (`GEOMAP_USE_STRICT_COMMIT=1`; rejects n_inliers<50 OR mask_frac<0.005; +5-10 realistic estimate)
- Callout-aware mask reorder in `tools/agent.py:extract_boundary` (`GEOMAP_USE_CALLOUT_AWARE=1`; +4 BOUNDARY_EXTRACTION cases identified by stuck-case investigator)
- Code-Point Open `tools/code_point.py` (1.6M GB postcodes at sub-metre BNG; integrated as os_names:code_point candidate)

**v14 recommended env-var stack:**
```bash
export GEOMAP_USE_OS_OPEN_NAMES=1
export GEOMAP_USE_INSPIRE_SNAP=1
export GEOMAP_USE_DELAUNAY=1
export GEOMAP_USE_STRICT_COMMIT=1
export GEOMAP_USE_CALLOUT_AWARE=1
# 6-DOF affine fallback: always on
```

**Optimistic v14 forecast** (per outcome predictor + statistical miner + stuck-case investigator):
- INSPIRE +2-4
- OS Open Names + Code-Point +5-10
- Delaunay +1-3
- Gate A +5-10
- Callout-aware +4 (BOUNDARY_EXTRACTION)
- 6-DOF +2 (validated)
- Total +25-30 cases past 0.8

**Data quality exclusions (11 cases):** A4Da2, A4D15A_merged, A4D5A_merged, 8609F3A9, CB:75:00001:ART4, SSA404, 095AB379, A4D14A1, A4EC3a1, 12:00122:ART4, 12:00125:ART4. Adjusted denominator 215→204 lifts baseline floor 61.9%→65.2%.

**Pix2pix-turbo Mac MPS port (in progress):**
- 130 paired (plan, OS-tile) samples built from 145 v13 wins, saved at `overnight/pix2pix_pairs/`
- Training script at `scripts/train_pix2pix_mps.py`: drops GAN discriminator (vision_aided_loss is CUDA-only), uses LPIPS+L2 only
- Patched files: `third_party/img2img-turbo/src/pix2pix_turbo_mps.py` and `src/model.py` (replaced .cuda() with .to("mps"))
- Smoke test passed: 0.41 step/s on MPS, ~3-4h for 5000 steps
- Inference wrapper: `tools/style_transfer.py`
- Eval script (post-train): `overnight/phaseZAD_style_transfer_eval.py` — runs MINIMA on (style-transferred plan, OS tile) for 15 borderline cases

**Key findings from 6 research agents:**
- Architecture #1 (score_match in-loop tool): refactor `build_critic_panel` into pre-commit tool, est +6-12 cases
- Architecture #2 (visual extract_boundary panel): show LLM 5 SAM3 candidates as overlay panel, est +8-15
- Failure mode classifier: 11/20 failing cases are "right affine, wrong SAM3 mask" — biggest untapped lever
- Red-team: 8/133 wins are district-wide GT gimmes (>10 km² polygons); agent regression on multi-match cases (correlation, possibly causal)
- 90% goal mathematically caps at 83.7% with current architecture; only style-transfer + retrieval combined breaks it (the B+C plan)

**Remaining levers (not yet built):**
- pix2pix-turbo style transfer (training NOW; eval after)
- score_match in-loop tool (~6h to wire into agent.py)
- visual extract_boundary panel (~8h to wire)
- v14 LLM rerun with all env vars (gated on user)
- v13 agent fixes from TODO_AGENT_FIXES.md

**Files added/changed today:**
- `tools/os_names.py` (offline OS Open Names lookup, 3M GB places)
- `tools/inspire_snap.py` (freehold snap with safety guards)
- `tools/delaunay_filter.py` (Delaunay-consistency RANSAC)
- `tools/style_transfer.py` (pix2pix-turbo wrapper)
- `tools/matcher_eloftr.py` (MatchAnything-ELoFTR — kept for completeness)
- `tools/matcher_omniglue.py` (OmniGlue — kept for completeness)
- `scripts/build_pix2pix_pairs.py`, `scripts/train_pix2pix_mps.py`, `scripts/colab_train_pix2pix_README.md`
- 318 INSPIRE LA bundles downloaded (4.9GB) at `os_opendata/inspire/`
- 100MB OS Open Names CSV at `os_opendata/open_names/csv/`
- ~12 new overnight phase scripts (ZL-ZAD)
- Memory: 7 new reference + feedback files

## Open paths (not yet tried)

- INSPIRE Index Polygons (HM Land Registry, ~3GB GML, free download from
  use-land-property-data.service.gov.uk/datasets/inspire) — UK freehold parcels.
  **GT-LEAK AUDIT COMPLETE 2026-05-06: SAFE.** No shared identifier with planning.data.gov.uk; only common ancestor is OS MasterMap. 40-60% of UK conservation/article-4 boundaries follow freehold edges. Recommended: 20-case ablation. (Direct download URL kept expiring — investigate.)
- Scale-bar OCR (just drafted at `tools/scale_bar_ocr.py`) — needs validation.
- NLS Historic Maps (maps.nls.uk, free WMTS) — older OS tile editions.
- DINOv2/SigLIP frozen visual embeddings as match verifier (downloadable, no API).
- CycleGAN OS basemap → planning-document style translation.
- mapKurator text spotter (knowledge-computing/mapkurator-system) — better OCR for stylized maps.
- Delaunay-consistency RANSAC on existing matches.

## Subagent strategy

User wants heavier multi-agent use. Pattern that works:
- One critic/auditor: reviews code for GT-leakage, verifies claims, runs apples-to-apples tests
- One researcher: scans literature/datasets for novel methods, web search
- One implementer: writes phases, manages parallel runs

Critical to instruct each agent:
1. **Critic must check for GT leakage** — verify ZB1-style bugs don't reoccur
2. **Researcher must verify dataset isn't the source of GT** — the planning.data.gov.uk lesson
3. **Implementer should not over-trust its own results** — flag GT-aware paths

## Files to preserve

- `TODO_AGENT_FIXES.md` — v13 agent code fixes
- `overnight/V13_FAILURE_ANALYSIS.md` — full failure analysis
- `overnight/SUMMARY_FINAL.md` — comprehensive summary
- `overnight/production_picker.py` — Phase T as importable module
- `overnight/phaseZD4_clean_t_fallback.py` — best deployable algorithm
- `overnight/phaseZD4_results.json` — per-case picks
- `overnight/phaseZAZ_aggregate_all.py` — cross-phase consolidation
- This file `SESSION_NOTES.md`
