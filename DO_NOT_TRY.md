# Do Not Try (already tested, confirmed not the path)

If a future agent or session is tempted to try one of these — DON'T. The
user has explicitly rejected, or extensive testing has shown they don't help.

## Matchers / inference

- **RoMa v2 / RoMa v1** — User confirmed: "doesn't work". Skip both — no need to download weights or set up the inference path. Tested previously and the result was worse than MINIMA-LoFTR on this dataset.

- **SIFT / ORB sparse matchers** — Tested as ZAE/ZAF on stuck cases. Got 4-11 inliers max, far below acceptance threshold. CPU-only and slow, ~3 hours per phase with no wins.

- **kornia DISK + LightGlue** — Tested as ZV / ZAD with default and 4096 features. 0 wins on stuck cases.

- **MatchAnything-EfficientLoFTR** (`zju-community/matchanything_eloftr` via HF transformers) — Tested 2026-05-06 as Phase ZL head-to-head against MINIMA on 5 borderline (v13 IoU 0.5-0.7) cases at v13's matched window. MatchAnything lost 4/5 in inlier count: 44→6, 5→0, 6→3, 121→33; only 5→21 won. Mean inliers MINIMA 33 vs ELoFTR 22. MatchAnything's cross-modal training is on UAV/IR/aerial pairs, not stylized planning maps; MINIMA-LoFTR was specifically trained on cross-modal map matching for this domain. Wrapper at `tools/matcher_eloftr.py` is kept for completeness but don't expect wins.

- **OmniGlue** (Google Research, github.com/google-research/omniglue) — Tested 2026-05-06 as Phase ZT head-to-head against MINIMA on 4 borderline cases (one had window-OOB, skipped). OmniGlue lost 4/4 in inlier count. Mean inliers MINIMA 17 vs OmniGlue 3. Trained on natural photographs (MegaDepth) — claims domain generalization come from object-centric / aerial photo benchmarks, NOT stylized cartographic maps. Apple Silicon also painful (CPU only, ~8-12s/pair vs MINIMA's 0.2s). Wrapper at `tools/matcher_omniglue.py` kept for completeness. The MatchAnything verdict already implied this: if a CROSS-MODAL trained matcher loses to MINIMA, a photo-trained one will lose worse.

- **MINIMA-XoFTR** (2026-05-12) — Same MINIMA cross-modal training as the LoFTR variant we use, weights at `github.com/LSXI7/storage/releases/download/MINIMA/minima_xoftr.ckpt`. Three strikes:
  1. **MPS crash** in `MINIMA/third_party/XoFTR/src/xoftr/xoftr_module/fine_matching.py:87` — empty `conf_matrix_fine` triggers `index 14 out of bounds: 0, range 0 to 1` on MPS when the coarse stage produces no matches above threshold. CUDA likely handles silently; MPS doesn't. Fixable with a `numel() == 0` guard but the patch needs upstream-vs-fork triage.
  2. **8× slower on MPS** — measured 2.33 s/pair vs LoFTR 0.29 s/pair on a 512×635 panel. Paper estimated +14% on CUDA; reality on Apple Silicon is 8× regardless of why. A 3.5h LoFTR benchmark would take ~16h on XoFTR. Hard ship blocker on Mac.
  3. **No MINIMA fine-tuning recipe** — `MINIMA/train_orders/` ships `minima_loftr.sh`, `minima_lightglue.sh`, `minima_roma.sh` but NO `minima_xoftr.sh` and no `xoftr_index_preparation.py`. Upstream `OnderT/XoFTR` has a `train.py` but it's MegaDepth-only with a different (focal + sub-pixel regression) loss head than LoFTR's dual-softmax. Adapting it to MINIMA's cross-modal pipeline would be 1-2 days of porting work with uncertain payoff. So even if we patched the MPS crash and ate the 8× slowdown, we couldn't fine-tune to close the planning-map domain gap.
  Combined verdict: revisit only if (a) we move to Linux+CUDA AND (b) inference-only XoFTR wins ≥5 cases on a calibrated A/B AND (c) someone is willing to port the training recipe. Not before. Setup state preserved: submodule `MINIMA/third_party/XoFTR/` restored, weights at `MINIMA/weights/weights_xoftr_640.ckpt`, `load_xoftr` in `MINIMA/load_model.py`, A/B harness at `tests/offline/xoftr_vs_loftr_ab.py`.

- **Mask-conditioned MINIMA** (any margin/exclusion strategy tested 2026-05-06 in Phase ZS) — Three variants tried:
  - 50px fixed margin: net -23 cases (catastrophic; too few inliers for RANSAC)
  - 200px fixed margin: aborted before completion
  - title-block-aware exclusion (no fixed margin): net -2 cases (3 pushed past 0.8, 5 fell below)
  Hypothesis was that title-block / scale-bar features create false-positive matches that outscore boundary-area matches. Either the hypothesis is wrong, OR my replay pipeline (different rendering path from cached v13) introduces enough noise to mask any real signal. Either way: don't pursue mask-conditioning of MINIMA's keypoints as a deployable lever. Wrapper at `overnight/phaseZS_mask_cond_minima.py` kept for reference.

- **Plain DINOv2 visual retrieval over OS Zoomstack tiles** (Phase ZAC, 2026-05-06) — DINOv2 ViT-L/14 CLS features on 2000 cached z18 tiles, FAISS-Flat IP index, queried with planning-map crops for the 13 stuck cases. Result: 1/13 within 5km of GT (KILL SWITCH at ≥3 fired). The cross-modal gap between stylized planning maps and OS Zoomstack vector tiles is too large for DINOv2 (trained on natural photos) to bridge. Caveats: this was plain CLS, NOT AnyLoc-VLAD aggregation; index was sparse (2000 tiles biased toward case locations), not full UK. AnyLoc-VLAD on full UK index might do slightly better but unlikely to clear the cross-modal gap without style transfer. The viable architecture for stuck cases is style-transfer FIRST (pix2pix-turbo on 145 paired wins) THEN retrieval on translated plans. Don't try plain DINOv2 retrieval again.

- **Pix2pix-turbo style-transfer with LPIPS+L2-only loss (no GAN)** (Phase ZAD, 2026-05-07) — Trained on 130 paired (plan, OS-tile) crops for 5000 steps on Apple Silicon MPS (the GAN discriminator vision_aided_loss is CUDA-only, so we dropped it). Eval on 15 borderline cases: 0/12 wins, mean fake_inliers=0 vs raw_inliers=63. The style-transferred output is too blurry/low-frequency for MINIMA to find any correspondence at all. Possible alternatives: (a) full GAN training on Colab T4 with vision_aided_loss enabled, (b) different image-to-image method (CycleGAN, ControlNet on canny edges, IP-Adapter), but all are heavy ML lifts. Don't try LPIPS-only finetune on Mac MPS again.

- **Pure geometric parcel-snap to OS Open Map Local buildings** (Phase ZAF, 2026-05-07) — Replaces predicted polygon with UNION of OS Open Map Local building footprints whose centroids fall inside it. Tested on 50 cases (mix of borderline 0.5-0.8 + already-good 0.8-0.95) with strict guards (predicted <0.05 km² + 50%-inside threshold + area band [0.7, 1.3]). Result: snap fires on almost no cases; 0 wins, 0 pushes past 0.8, 3 minor losses. Without strict guards: catastrophic regressions on large polygons (0.95→0.30). The geometric "buildings inside polygon" approach doesn't replicate the human "select TOID parcel" workflow because humans use the ADDRESS (UPRN → specific building) AND visual matching, not just containment. Don't try pure geometric parcel-snap again. To make this work: integrate OS Open UPRN + Code-Point Open + house_number_road_pairs to anchor to a SPECIFIC building, then grow with curtilage. Significant additional engineering work.

- **Random UK grid anchor sweep** — Tested as ZAG. 1.5° grid spacing with sigma=50km. Too sparse: best n_inliers per case was ~28, below the score floor. ~5 hours runtime for full set.

- **Postcode/address-anchored polygon translate or building-snap on cached predictions** (Phase ZAH-diag, 2026-05-07) — Diagnostic on 40 cases with both `house_number_road_pairs` AND a full postcode (Code-Point Open). Findings that kill the approach:
  1. **Postcode→pred distance is INVERTED vs IoU.** Failing cases (IoU<0.8) median distance to postcode = 148m. Passing cases (IoU≥0.8) median = 3.9km. PASSING cases routinely 5-10km from their postcode (e.g. A4D03 IoU 0.998 is 3.9km away; 983981FC IoU 0.968 is 10.9km away). The postcode in PDFs is NOT a boundary anchor — it's metadata referring to council offices, building inside a much-larger Article-4 zone, or a reference address.
  2. **"Replace pred with nearest OS Open Map Local building to postcode" fails.** On A002S (single-cottage boundary): closest building to postcode is the wrong building (3m away, IoU=0); the right one is 2nd-nearest at 20m. We can't reliably pick the correct building without paid AddressBase Premium (which has UPRN→building linkage). OS Open data only has anonymous building polygons.
  3. **Even with oracle building selection, IoU ceiling is ~0.63.** GT polygons are not aligned to OS Open Map Local building edges — they're traced on planning maps. The data-source mismatch caps the achievable IoU regardless of algorithmic perfection.
  Translate-anchor would damage the 11 already-near-postcode failing cases (their position is fine; only shape/scale is off). Building-level snap can't pick the right building.
  Don't pursue address-anchor or address-snap as a deployable lever on cached predictions. The real value of postcodes is at MATCHING TIME (Code-Point Open BNG point as Gaussian prior for sliding-window MINIMA). That requires a v14 rerun.

## Datasets / GT-leak risk

- **planning.data.gov.uk** (any dataset on `files.planning.data.gov.uk`) — **THIS IS THE GROUND TRUTH SOURCE**. Each evaluation case's GT GeoJSON has properties indicating `dataset=article-4-direction-area` (or `conservation-area`, etc.) with the same `entity` ID and `reference` as the matching planning.data.gov.uk download. Matching against any dataset on this domain = direct GT lookup = invalid. Verified on case 05D21091-B835-402E-B64F-C5DEB8D59D46 (entity 7010002644 matches identically between our GT and the public download).

## Selection / reranking

- **Cross-source mask × affine mixing** (Phase A/D/F): -11 to -15pp deploy. Mixing one version's mask through another version's affine breaks the per-run optimisation. Each version's (mask, affine) was co-tuned; cross-mixing them is consistently bad.

- **Color-boundary intersection** (Phase ZE1): plans have noise in red (text, titles, borders). Intersection of color mask with SAM3 mask collapses to 0.30 mean from 0.59. Color-only alone is 0.25 mean. Not the path.

- **Road-snapping projected boundary to OS roads** (Phase ZE3): all snap distances tested (10m, 30m, 50m) regress (0.71 → 0.50). Snap distorts the polygon shape too much.

- **Admin polygon (district/borough) lookup as direct prediction** (Phase ZF1): admin areas are too big — site boundaries are typically a single building or small zone, not a whole borough. ~0 correct picks.

- **Text-features → lat/lon regression** (Phase ZF2): 60km median error. Text features alone aren't precise enough for matching.

- **OCR + OS gazetteer text-anchored alignment** (Phase ZF3): 18/64 cases have <4 OCR labels at all; 46/64 have OCR labels but <4 match OS gazetteer. The OS Open Names gazetteer covers major places but not the street labels / POIs typical of plan documents.

- **Mask post-processing applied unconditionally** (Phase ZC5/ZD7): morpho variants over-fire. Conservative rules (compactness < 0.2 only) give +0.0001 mean. Aggressive rules regress to 0.64.

- **SAM3 bbox-feedback mask refinement** (Phase ZF4-ZF7): yields 1-3 oracle wins per phase, but no deployable signal reliably picks the right candidate. Strict gates over-filter (0/0 wins/losses); loose gates have a 1:1 win/loss ratio.

## Specific stuck cases

These ~13 cases have ALL versions returning IoU=0 and no extractable text:
- 118, 12:00115:ART4, 12:00125:ART4, 23:53159:ART4, 5B10B5A8-...,
  A4_102:LL:077, CB:75:00001:ART4, ED3ECD0D-..., SSA404, SSA405, SSA410, SSA416

Image-only PDFs with no text. Phase ZJ (OCR), Phase ZAG (UK grid), Phase ZF1 (admin lookup), Phase ZS (Zoomstack gazetteer) all 0 wins. These need a fundamentally different approach (likely a vision-language verifier, or human review).

## Lessons about validation

- **Threshold sweeps that pick "best mean IoU" on the full set leak GT.** Evidence: my ZD4 was claimed at 0.7569 with thr=200; k-fold-CV across cases also picks thr=200 unanimously and the held-out mean is also 0.7569 — so this particular case is robust, but if I'd published a different threshold without CV-checking it would have been GT-tuned.

- **Variant-selection gates that compare `if new_iou > old_iou: swap` use GT.** The `iou` value comes from `metrics.json` which compares to GT. Phase ZB1 / ZB4 had this bug; gave inflated 0.7588 / 0.7716 numbers. True deployable was 0.7569.

## Truly clean deployable best (as of writing)

**Phase ZD4 = 0.7569 mean IoU** (k-fold-CV-verified threshold), 142/208 ≥0.8 IoU. Cached oracle ceiling 144/208 = 69.2%.
