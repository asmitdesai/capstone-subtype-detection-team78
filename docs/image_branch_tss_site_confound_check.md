# Image branch — TCGA Tissue Source Site (TSS) confound check (2026-08-12)

*Permanent record of a second, distinct confound found during pre-Stage-3 checks for the image branch's DICOM tiling pipeline. Referenced from `docs\HANDOVER_PROJECT_CONTEXT.md` §6. Related to, but a different mechanism from, the nuclear-density confound in `docs\image_branch_nuclear_density_confound_check.md`.*

## Finding

While surveying the full 134-case cohort for Stage 3 pre-checks (`code\scripts\image_stage3_precheck_survey.py`), TCGA case IDs were grouped by their Tissue Source Site (TSS) code — the second hyphen-delimited segment of a TCGA barcode (e.g. `TCGA-EM-A2CN` → site `EM`), which identifies the submitting institution/hospital, not the patient. Result, counted directly from `Dataset\TCGA_THCA_Slides\` folder names:

| Site | cvPTC cases | fvPTC cases |
|---|---|---|
| BJ | 34 | 5 |
| CE | 9 | 0 |
| DE | 9 | 1 |
| DJ | 13 | 21 |
| DO | 0 | 1 |
| E3 | 0 | 2 |
| EL | 0 | 2 |
| **EM** | **0** | **37** |
| **Total** | **65** | **69** |

**`TCGA-EM-*` alone accounts for 37 of 69 fvPTC cases (53.6%) and zero of 65 cvPTC cases.** No other site is anywhere near this exclusive — BJ, DE, and DJ all have cases in both classes (BJ: 34cv/5fv; DE: 9cv/1fv; DJ: 13cv/21fv), so only EM is a fully class-exclusive site at this scale.

This was also visible in the Stage 3 tile-count-variance survey: `TCGA-EM-*` slides in the 24-case sample ran markedly smaller physically than average, meaning this site also disproportionately drives the ~32× per-case tile-count variance already documented in the handover — a class-exclusive site is also a size-atypical site here, compounding the risk.

**Full breakdown confirmed (2026-08-12), using all 24 cases from the thumbnail pre-check survey (`code\thyroid\stage2_filtering_run\stage3_precheck_survey.json`), split by site:**

| Group | n cases | Mean est. tiles | Std | Min | Max |
|---|---|---|---|---|---|
| **EM** | 7 | 668 | 285 | 285 | 1,056 |
| **Non-EM** | 17 | 4,401 | 1,805 | 1,388 | 9,151 |

**Complete separation — zero overlap.** Every EM case in the sample falls below every non-EM case (EM max 1,056 < non-EM min 1,388). This confirms EM cases are systematically, severely under-tiled relative to the rest of the cohort — not a marginal effect. Combined with EM being 100% fvPTC, this means the final tile pool will contain proportionally far fewer tiles per EM-sourced fvPTC case than per any other case, on top of the case-count imbalance already noted above. (Note: this estimate used the pre-tissue-fraction-bug-fix thumbnail proxy, which is only mildly affected by that bug since it's thumbnail-only, not per-tile — the separation is real and would not disappear with the fix.)

### Severity update (2026-08-12): EM's actual share of the fvPTC TILE pool is smaller than its case-count share suggests

Full 134-case thumbnail survey (`code\scripts\image_stage3_build_split.py`, tile estimates scaled by the real measured 70% post-bug-fix survival rate, verified by direct recount from the split CSV): **EM cases total 37/134 (27.6% of all cases, 53.6% of fvPTC cases by count) but contribute only 33,076 of 286,852 total estimated tiles (11.5% of the whole tile pool).** Put differently: EM is over half of the fvPTC case *count* but roughly a ninth of the whole cohort's *tile* pool, because each EM case individually contributes so few tiles. This means EM's practical influence on model training (and on the confidence/attention patterns Grad-CAM will assess) is smaller than a naive case-count-based read of the confound would suggest — the confound is real, but its tile-level footprint is more modest than its case-level footprint. This doesn't reduce the underlying risk (the deterministic-proxy problem — any EM tile is 100% certainly fvPTC — doesn't care how many tiles there are), but it's a relevant severity-calibration data point for interpreting the mandatory EM-vs-non-EM subgroup evaluation results once available: a null result there wouldn't be surprising given the smaller tile footprint, and a positive (site-shortcut) result would carry weight despite the smaller sample. See `docs\HANDOVER_PROJECT_CONTEXT.md` §6 for the case-level split's resulting test-set tile balance.

## Why this matters — and why it's more serious than the density confound

The density confound (see the sibling doc) is a *biological* difference — real growth-pattern architecture that happens to be exploitable as a shortcut, but is at least tied to genuine diagnostic signal along a continuous spectrum. **The TSS-site confound is a purely technical one**: different institutions use different scanners, staining protocols, tissue-processing batches, and digitization pipelines. A CNN has no biological reason to prefer this kind of shortcut over real morphology — color-calibration quirks, batch-specific staining tone, or scanner-specific noise characteristics are exactly the sort of cheap, non-explainable signal a CNN is prone to exploit when available, and here it is available: **any tile from an EM-sourced slide is deterministically fvPTC, with 100% certainty, regardless of what's on it.** This is precisely the failure mode `docs\HANDOVER_PROJECT_CONTEXT.md` §4 already warns about for cross-dataset mixing (TCGA vs. IBIA) — except here it's *within* TCGA, at the institution level, and was not something the original single-sourcing decision anticipated or could have prevented (both classes are still exclusively TCGA-THCA; the imbalance is in which TSS sites happened to contribute cases to each class within that single source).

## Why it cannot be "fixed" by the tiling pipeline or the split alone

This imbalance is baked into which 134 cases were downloaded, not introduced by tiling. **No case-level train/test split strategy can balance the confound out of the data itself**, because EM contributes zero cvPTC cases to balance against — there is no EM-sourced cvPTC case to move into either split. Stratifying the split by site (see next section) is still worth doing, but it only prevents an *additional, avoidable* problem (a train/test site mismatch on top of the underlying imbalance) — it does not, and cannot, remove the underlying risk that the model learns site-as-proxy-for-class from the training data itself.

## Decisions (2026-08-12)

1. **Stratified case-level split by TSS site, as a baseline** — the Stage 3 case-level train/test split will explicitly balance site representation (in addition to class) across train and test, so evaluation isn't further distorted by a site mismatch on top of the pre-existing imbalance. This is close to free and strictly better than an unstratified split, but see above: it does not resolve the shortcut-learning risk, only prevents making it worse.

2. **Required subgroup evaluation (elevated to mandatory, not optional) — once the image CNN is trained:** explicitly compare model confidence/accuracy on **EM-sourced fvPTC test cases vs. non-EM-sourced fvPTC test cases**. This is now a **named, required step in the evaluation plan**, alongside the Grad-CAM systematic-sample requirement already established (`docs\image_branch_nuclear_density_confound_check.md`, `docs\HANDOVER_PROJECT_CONTEXT.md` §6):
   - If EM-sourced fvPTC cases show **meaningfully higher** confidence/accuracy than non-EM fvPTC cases, that is direct, reportable evidence the model is partly keying off site/scanner cues rather than morphology — to be reported as a finding, not filed away as a caveat.
   - If EM and non-EM fvPTC performance are comparable, that's supporting (not conclusive — n is small either way) evidence the model generalized past the site cue.
   - This check should be read together with Grad-CAM: a model that both (a) shows an EM/non-EM performance gap and (b) shows Grad-CAM attention on non-morphological regions would be strong combined evidence of shortcut learning; either signal alone is suggestive but not sufficient on its own.

## Follow-up: does thin-ink-line prevalence (the Stage 2 filter fix) correlate with site?

Prompted by finding one technical confound (site), checked whether another already-identified issue — thin ink lines slipping past the original Stage 2 ink filter (see the sibling doc) — is itself site-correlated, as supporting evidence for how much site effects matter here generally.

Script: `code\scripts\image_stage2_tss_site_line_check.py`. Random tissue-tile sampling (100 tiles/case, not full grid, for speed) across a site-diverse case selection: BJ (both classes), DJ (both classes), CE (cvPTC-only), EM (fvPTC-only) — chosen specifically so site and class can be told apart (BJ/DJ have both classes, so any pattern isn't just "fvPTC has more lines"). Applied the full filter chain including the new shape-based line detector (`line_span_fraction`, span-fraction ≥0.5).

**Result: yes, a large, clean correlation — and in a direction that adds a second technical shortcut on top of the site confound itself.** Full data: `code\thyroid\stage2_filtering_run\tss_site_line_check_result.json`.

| Site group | Tiles reaching line check | Line-flagged | Rate |
|---|---|---|---|
| **EM** (4 cases, all fvPTC) | 358 | 1 | **0.28%** |
| **Non-EM** (10 cases, mixed BJ/DJ/CE, mixed class) | 717 | 65 | **9.07%** |

Per-case rates ranged from 0% to 38.6% among non-EM cases (highest: `TCGA-BJ-A45F`, cvPTC, 27/70 = 38.6%), while all 4 EM cases sampled showed 0–1.1% — essentially none. This is a **~32× difference**, and it held consistently across all 4 EM cases sampled, not just in aggregate.

**Interpretation:** this is not itself a color/architecture confound like the density one — it looks like an institutional specimen-handling signature (some source labs' slides routinely carry thin ink marks or processing-crack artifacts; EM's evidently don't, or use methods that don't leave them). Combined with the base site/class imbalance, this means **"near-zero probability of any thin dark line in a tile" is itself a second, independently-learnable, near-perfect proxy for fvPTC** in this cohort (since EM = 100% fvPTC and EM tiles are almost always line-free, while a large fraction of cvPTC-heavy non-EM tiles carry these marks). This strengthens rather than resolves the concern above: it's supporting evidence that site-level technical signatures are a real, multi-pronged risk here, not a one-off finding limited to raw pixel color. It also means the Stage 2 line filter itself, ironically, differentially affects data volume by site (removing a meaningfully larger fraction of non-EM tiles than EM tiles) — worth being aware of when interpreting per-class/per-site final tile counts in the Stage 3 run, so an observed site-skewed tile count isn't mistaken for a tiling bug when it's actually this real, upstream artifact-rate difference.

This makes the required EM-vs-non-EM subgroup evaluation (above) even more important: if the trained model ends up weighting thin-line presence/absence as a feature (directly or indirectly, e.g. via whatever the line-filtering leaves behind), that's an additional plausible shortcut mechanism to check for, not just raw color/density.
