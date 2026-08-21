# Image branch — nuclear/glandular density confound check (2026-08-12)

*Permanent record of a pre-Stage-2 methodology check for the image branch's DICOM tiling pipeline. Referenced from `docs\HANDOVER_PROJECT_CONTEXT.md`. Raw output lives in `code\thyroid\stage2_confound_check\`; scripts in `code\scripts\`.*

## Question

Before finalizing the tissue/quality/ink filtering scope for WSI tiling (§4 of the main handover — case-level splits, ink filtering, weak-supervision limitations), we needed to know: is there a systematic difference in nuclei/glandular density between cvPTC- and fvPTC-labeled tiles that a CNN could exploit as a shortcut, instead of learning genuine papillary-vs-follicular growth-pattern architecture?

This matters specifically because cvPTC and fvPTC are, by definition, distinguished by growth pattern (papillary/solid vs. follicular), and follicular architecture is inherently colloid-rich / nuclei-sparse relative to papillary/solid architecture. If tile-level pixel density alone tracks the label, a classifier could learn "density = class" as a cheap proxy rather than recognizing actual diagnostic morphology (nuclear grooves, ground-glass nuclei, papillary fronds vs. round follicles) — a real risk for a model whose whole point is explainability (Grad-CAM) and clinical trust.

## Method

Script: `code\scripts\image_stage2_confound_check.py`. Full result: `code\thyroid\stage2_confound_check\confound_check_result.json`.

1. Sampled 5 cvPTC cases + 5 fvPTC cases (seeded, `random.default_rng(42)`) from the 134-case `Dataset\TCGA_THCA_Slides\` cohort.
2. Extracted 20 tissue tiles per case (100 tiles/class, 200 total) using the confirmed Stage 1 tiling recipe: 1024×1024 native ~40x (level 0) → 2× image-domain downsample (Lanczos) → final 512×512, with tissue-mask-guided sampling (thumbnail tissue mask, mean-pixel-intensity accept threshold < 220).
3. Per tile: color-deconvolved to the hematoxylin channel (`skimage.color.rgb2hed`), computed a density score = fraction of pixels above a threshold.
4. **Threshold choice, deliberately controlled for fairness:** a single global Otsu threshold (0.04842) computed on the *pooled* hematoxylin-channel histogram across both classes together — not a per-class or per-tile-normalized threshold, which would launder away any real between-class difference this check exists to detect.
5. Compared the resulting per-class density-score distributions: descriptive stats, Mann-Whitney U test, Cohen's d effect size.

## Result

| | cvPTC (n=100) | fvPTC (n=100) |
|---|---|---|
| Mean density | 0.221 | 0.143 |
| Median | 0.207 | 0.113 |
| Std | 0.115 | 0.107 |

- **Mann-Whitney U = 6925.5, p = 2.557e-06**
- **Cohen's d = 0.700** (medium-large effect)

![density distribution comparison](../code/thyroid/stage2_confound_check/density_distribution_comparison.png)

Visual inspection of sample tiles from each class (`code\thyroid\stage2_confound_check\cvPTC_sample_contact_sheet.png`, `fvPTC_sample_contact_sheet.png`) matches the statistic: fvPTC tiles more often show large, pale, colloid-filled follicles with sparse nuclei; cvPTC tiles are denser, more solid/papillary, more nuclei-packed. One cvPTC sample tile also showed a visible blue ink mark — independent confirmation that ink filtering (already planned) is a real, not hypothetical, need on this data.

## Verdict

**A real, statistically significant systematic difference exists (case "3b", not "3a")** — this is genuine architecture-linked biology, not a scanning artifact, which is exactly what makes it a real shortcut-learning risk rather than a hypothetical one. Left unmitigated, a CNN could learn tile-level pixel density as a cheap, non-explainable proxy for class instead of true morphological growth-pattern recognition.

## Mitigation decision

Approved direction (2026-08-12): **trim both tails of the density distribution, don't attempt to fully match the two classes' distributions.** Fully matching would fight real biology and risk destroying genuine diagnostic signal along with the shortcut-learning risk — trimming only removes tiles at the extremes (near-empty/pure-colloid tiles that are trivially density-classifiable, and solid-blood/necrosis tiles that carry no useful architecture either way), which is a much narrower, defensible intervention.

### Cutoff calibration (2026-08-12)

Script: `code\scripts\image_stage2_density_cutoff_calibration.py`. Re-extracted the identical 200-tile calibration set (same seed) and visually inspected real tile content at candidate cutoff values, rather than picking a blind percentile.

**Low end:** below density ≈0.02–0.03, tiles are genuinely low-content — fibrous stroma, collagen, blood vessels, largely acellular (e.g. `fvPTC_TCGA-EM-A2CN_x36516_y32790_d0.0000.png`). From density ≈0.04 upward, tiles show clear, legitimate follicular architecture (large colloid-filled follicles with intact epithelial rims) — genuine fvPTC diagnostic morphology, not filler. **A low cutoff much above ~0.02–0.03 would start discarding real signal, disproportionately from fvPTC** (the confound-check finding, replayed here: at cutoff 0.04, 17/100 fvPTC tiles drop vs. 4/100 cvPTC; at 0.08, 38/100 fvPTC vs. 11/100 cvPTC).

**High end — revised finding, contrary to the original hypothesis:** visual inspection at every candidate high cutoff (0.30, 0.35, 0.40, 0.45) and at the true sample maximum (0.60) found **no solid-blood or necrosis "junk"** — every high-density tile examined was real, densely-packed tumor tissue (solid/papillary architecture, some with clearly visible papillary fronds). This makes biological sense in hindsight: red blood cells are anucleate and stain weakly with hematoxylin, so a genuinely blood-filled tile would score *low* on this metric, not high; true necrosis (karyolysis/nuclear debris loss) would likewise tend to score low, not high. High hematoxylin density on this metric just means dense tumor cellularity — real signal, concentrated on the cvPTC side. **Trimming the high tail as originally planned would have discarded real cvPTC signal for no actual junk-removal benefit** — this was corrected before locking any threshold, per the "visually confirm before cutting" requirement.

### Locked-in cutoff values (approved 2026-08-12)

- **Low cutoff: 0.02.** Targets only the genuinely acellular tail (stroma/collagen/blood vessels with no real diagnostic content). Impact on the 200-tile calibration set: 1/100 cvPTC + 4/100 fvPTC dropped — small in absolute terms, still mildly disproportionate toward fvPTC (the smaller class), consistent with and expected from the underlying confound, but not large enough to be a real data-loss concern at this cutoff.
- **No aggressive high-end trim**, for the reason established above (no junk found at the high end, only real dense tumor tissue).
- **Optional generous ceiling around 0.55–0.60** may be applied — but this is an **outlier safety net only, not confound mitigation, and this distinction is explicit and load-bearing: don't conflate the two.** The safety-net ceiling exists purely to catch pathological outlier tiles that might appear in the full 134-case run and weren't present in this 200-tile calibration sample (e.g. a genuine scanning/staining artifact producing an anomalous hematoxylin reading) — it is set near the observed maximum (0.5967) specifically so it does *not* trim any real tumor tissue seen in calibration, and should not be tightened later in the name of "fixing" the density confound. If a future reader is tempted to lower this ceiling as an additional confound-mitigation lever, that would be wrong per the finding above — use blur/artifact detection for that instead.

### Why the high end carries no confound-mitigation signal — the hematoxylin-biology reasoning

This is the citable methodological reasoning behind skipping an aggressive high cutoff, not just an empirical observation: **hematoxylin stains nuclei (specifically acidic/basophilic nuclear chromatin), not red blood cells or necrotic debris.** Red blood cells are anucleate and take up eosin (the pink counterstain), not hematoxylin — so a tile that is genuinely dominated by blood pooling would score *low* on a hematoxylin-channel density metric, not high. Necrotic tissue undergoes karyolysis (nuclear dissolution) or karyorrhexis (nuclear fragmentation with loss of chromatin basophilia) — so true necrosis also tends to read as *low* hematoxylin density, not high, as nuclear material degrades. There is consequently no plausible histological mechanism by which "solid blood" or "necrosis" would produce a *high* reading on this specific metric — the only way to score high is genuinely dense, viable, nucleated tissue. This is why every high-density tile examined in calibration (up to the sample maximum of 0.5967) showed real tumor architecture rather than artifact, and why trimming the high tail would only ever discard real signal, disproportionately from cvPTC, with no offsetting junk-removal benefit. Blur (Laplacian variance) and saturated-color ink detection remain the correct tools for catching genuine artifact/junk tiles — density is not a proxy for those failure modes.

## Stage 2 full filtering pipeline — build and verification (2026-08-12)

Script: `code\scripts\image_stage2_filtering.py`. Systematic non-overlapping grid tiling (1024×1024 native ~40x → 512×512 final) across the whole slide extent, run on the same Stage 1 test case (`TCGA-BJ-A0YZ`, cvPTC) before scaling to Stage 3's full 134-case run. Full result: `code\thyroid\stage2_filtering_run\stage2_filtering_result.json`.

**Two filter rules were wrong on the first pass and caught by visual verification before locking anything — both are worth recording as methodology, not just the fixes:**

1. **Ink detection, first attempt (hue-based) failed badly.** Original rule: flag pixels as ink if hue fell in a blue/green range (0.33–0.78) with saturation > 0.35 — the intuition being ink is a saturated non-H&E color. This dropped **1292 of 3668 tissue tiles (35%)** on the first run. Visual inspection of 20 sample rejects found only 1 genuinely contained ink; the other 19 were normal tissue. Root cause: densely-stained hematoxylin nuclei clusters are dark AND fall in a similar hue range to ink, because hue is a poor descriptor once saturation is high and the color is a deep, consistent purple. Diagnosed by directly sampling HSV pixel statistics from a confirmed-ink tile vs. a false-positive tile: dense real nuclei are dark but *strongly and consistently saturated* purple (sat≈0.43, std 0.06); true ink is dark and *desaturated/near-achromatic* (sat≈0.26, std 0.27 — hue is essentially noise for near-black pixels). **Fixed rule: flag pixels as ink if `value < 0.35 AND saturation < 0.35`** — dark + desaturated, not dark + any-hue. Verified against the confound check's own densest real-tumor tile (d=0.60): scores ~0.01% ink-pixels under the new rule vs. 7.6% for confirmed ink. Rerun: ink_dropped fell from 1292 to **14**, and all 14 sampled rejects visually confirmed as genuine ink.

2. **Blur detection, first attempt (per-case relative percentile) was not measuring blur.** Original rule: Laplacian variance below the 5th percentile of that case's own tile distribution. Visual inspection of the 20 lowest-variance tiles (including the case-wide minimum) found all of them sharp and in-focus — just low-texture tissue (striated muscle fibers, sparse stroma), not out-of-focus scan regions. TCGA WSI scans are generally well-QC'd, so genuine focus failure appears rare/absent in this data; a relative percentile cutoff can't distinguish "low texture" from "actually blurry" and would have systematically discarded legitimate low-texture tissue types. **Fixed: replaced the per-case percentile with a fixed, conservative absolute floor (variance < 0.003, well below the ~0.006 observed case minimum)** that acts as a safety net for genuinely pathological out-of-focus tiles (should any appear elsewhere in the full 134-case run) rather than a routine bottom-percentile drop. Rerun: blur_dropped fell from (an untested, would-have-been ~183) to **7**, consistent with a rare-outlier safety net rather than a systematic filter.

3. **Tissue re-verification, found later during Stage 3 pre-checks (2026-08-12): re-fitting Otsu per-tile silently discarded large amounts of real tissue on dense, uniformly-stained slides.** The full-res tissue re-verification step (Stage 2's item 1) re-fit an Otsu threshold on each individual 1024×1024 tile's own saturation channel. This works fine for tiles that genuinely mix tissue and background, but **fails badly for tiles that are wholly or mostly dense tissue with no real background present** — Otsu still forces a bimodal split of the tile's own internal saturation variance, producing a threshold far above the true tissue/background boundary. Found while spot-checking an anomalous case (`TCGA-DJ-A2PP`, see the sibling TSS-site doc for the initial anomaly): a tile confirmed by eye to be ~98% real tissue measured a per-tile Otsu threshold of 0.416 (tissue_fraction 0.36, wrongly rejected) vs. the correct slide-wide thumbnail threshold of 0.184 (tissue_fraction 0.98). **Fixed: `tissue_fraction()` now takes the slide-level (thumbnail-derived) threshold as a fixed parameter, reused across every tile on that slide, instead of re-fitting per tile.** Impact varied hugely by case, correlating with how much large/dense/uniform tissue a slide has — exactly the cases that would otherwise contribute the most tiles:
   - `TCGA-DJ-A2PP`: final kept tiles 1,363 → **6,206** (4.55×)
   - `TCGA-DJ-A3VM`: final kept tiles 3,701 → **6,014** (1.63×)
   - `TCGA-BJ-A0YZ` (this doc's main test case): 3,354 → 3,376 (+0.7%, negligible — this slide's ring/donut tissue shape naturally mixes tissue and background within most tiles, so it was barely affected)
   - 4 other smaller/less-dense sampled cases: +3% to +8%, all minor

   This bug would have silently and unevenly under-counted tiles across the full 134-case run in a way correlated with tissue density/architecture, not randomly — a second reason (independent of the TSS-site and nuclear-density confounds) to be cautious about anything that varies systematically with slide-level tissue characteristics. Caught and fixed before the full run, not after.

**Final filtering funnel, test case `TCGA-BJ-A0YZ`, all three fixes applied:**

| Stage | Count | Notes |
|---|---|---|
| Grid cells (whole slide, 1024px non-overlapping) | 10,416 | |
| Thumbnail tissue pre-filter pass | 4,119 | coarse Otsu-on-saturation pre-filter, decides what's worth reading at full res |
| Full-res tissue filter kept | 3,668 | re-verified with the same Otsu-saturation method on the actual tile |
| Full-res tissue filter kept | 3,696 | with the per-tile-Otsu fix (item 3 above); was 3,668 before |
| Blur-dropped | 7 | fixed absolute safety-net floor (see above) |
| Ink-dropped | 20 | corrected value+saturation rule (see above) |
| Line-dropped | 271 | secondary shape-based thin-line gate, see below |
| Density-low-dropped (< 0.02) | 22 | genuinely acellular tiles only |
| Density-high-dropped (> 0.60) | 0 | none this case — consistent with the "no junk at the high end" finding |
| **Final kept** | **3,376** | 91.3% of tissue-filtered tiles survive quality/ink/line/density filtering |

Visual spot-check: 50-tile random contact sheet of kept output shows clean, legitimate tissue throughout (follicular and papillary architecture, some muscle/stroma), no visible ink contamination, no blank/background tiles. All 14 ink-rejects visually confirmed as genuine ink on re-inspection.

### Follow-up: thin dark lines slipping through the ink filter (2026-08-12)

User inspection of `stage2_kept_contact_sheet.png` spotted thin dark diagonal marks in several kept tiles. Diagnostic script: `code\scripts\image_stage2_ink_line_diagnostic.py`. Reproduced the identical 50-tile sample (seed=7) with an added connected-component "elongation" metric (does any dark+desaturated region span a large fraction of the tile despite low total area?).

**Finding: only 2 of the 50 sampled tiles (4%) had a genuine missed ink/foreign-mark line** (full-resolution visual confirmation) — `x=32768,y=53248` and `x=98304,y=29696`. Three other tiles that looked line-like in the small 140px composite thumbnail turned out, on full-resolution re-inspection, to be normal tissue features (vessel walls, tissue folds, red cell boundaries) — a caution that the composite thumbnail is too small to be trusted for this kind of visual QA; full-resolution crops are needed.

**Root cause, different from the original hypothesis:** this is not primarily an area-fraction blind spot (a thin line simply not covering enough of the tile to cross the 1% threshold) — for the two confirmed misses, the ink-mask found *close to zero* matching pixels, not "some but not enough." The actual mechanism: a ~1–2px-wide fully dark line at native ~40x resolution gets partially blended with surrounding pink tissue color during the 2× Lanczos downsample to the final 512×512 tile. This blending raises the line's measured HSV saturation well above the calibrated ink threshold (measured on the confirmed misses: darkest-decile pixels along the line had saturation ≈0.54, vs. the 0.35 ink threshold) — so the strict value+saturation ink mask misses these thin lines almost entirely, even though they remain clearly visible to the eye.

**Fix explored, with an important complication found during validation:** switching the shape-detection mask to value-only darkness (no saturation constraint) does catch both missed lines via a span-fraction check (the line's connected component reaches a large fraction of the tile's width/height: 1.00 and 0.68 respectively, vs. <0.4 for confirmed non-line tiles in the same sample). **However, this is a harder discrimination problem than the original ink-vs-nuclei color separation**, because genuine anatomical structures — thin blood vessels, fibrous septae — are also naturally dark and elongated, and one such real structure in this sample scored close to the line-like range on aspect ratio alone (though not on span-fraction, which held up as the more reliable of the two geometric measures in this small n=50 calibration). Aspect ratio alone is not a clean discriminator; span-fraction (≥0.5 of tile width/height) separated the confirmed cases better in this sample, but this is based on a small calibration set and the boundary against real thin vessels is inherently fuzzier than the original problem.

**Status: proposed, not yet implemented pending sign-off** — recommended as a secondary, conservative gate (value<0.45 mask, span-fraction ≥0.5 on a connected component with area ≥15px) layered on top of the existing value+saturation ink rule, reported separately in the Stage 2 funnel so its real-world false-positive rate against genuine thin vessels can be monitored once run at scale, not merged silently into the main ink filter. Given the project's explicit principle that ink must never be treated as signal (§4), erring toward occasionally dropping a real thin-vessel tile is the defensible conservative choice here — the cost of a false positive (losing one tile) is far lower than the cost of a false negative (an ink line surviving into training data).

## Related, separately-flagged finding: TSS-site/scanner confound with class label (2026-08-12)

While surveying the full 134-case cohort for Stage 3 pre-checks (`code\scripts\image_stage3_precheck_survey.py`), found that **TCGA Tissue Source Site (TSS) codes are not evenly distributed across classes**: the `TCGA-EM-*` site accounts for 37 of 69 fvPTC cases (54%) and **zero** of 65 cvPTC cases. This is a distinct, likely more serious confound than the nuclear-density one above — a purely technical (scanner/institution/staining-batch) confound rather than a biological one, and exactly the kind of risk §4 of the main handover warns about for cross-source mixing, except here it's an *intra-TCGA* site imbalance rather than TCGA-vs-IBIA. `TCGA-EM-*` slides in the survey sample were also markedly smaller (physical scan size), contributing to the per-case tile-count variance noted in the Stage 3 pre-check survey. This needs its own discussion before Stage 3 proceeds — see `docs\HANDOVER_PROJECT_CONTEXT.md` §6.

This pipeline (tissue detection, blur, ink, and the density tail-trim) is verified at single-case scale. Before it's run across the full 134-case cohort (Stage 3), the thin-line filter decision above and the TSS-site confound need resolving — see `docs\HANDOVER_PROJECT_CONTEXT.md` §6 for current status.

## Downstream validation requirement (elevated, not optional)

This confound check only justifies the mitigation choice — it does not prove the mitigation worked. The real test is downstream: once the image-branch CNN is trained, **Grad-CAM must be run on a systematic sample of test tiles per class (not 2-3 anecdotal cases)**, explicitly checking whether attention concentrates on genuine morphological features (nuclear grooves, papillary fronds, follicle boundaries) rather than raw density/color regions. This is now a **required validation gate before the image branch is considered done**, per explicit project decision (2026-08-12) — not an optional explainability demo nicety. See `docs\HANDOVER_PROJECT_CONTEXT.md` §4/§6.
