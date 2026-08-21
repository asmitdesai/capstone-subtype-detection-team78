# Project Context Handover — Multimodal PTC Subtype Classification

*Paste this at the start of every new Claude Code session on this project. It covers the science, the data, and the findings — not the download/infrastructure history (batching, network issues, folder cleanup). That operational history lives in `docs\HANDOVER_2026-07-28.md` and `docs\label_corrections_log.txt` if ever needed, but is not required to continue the actual work.*

---

## 1. Project Overview

**Goal:** distinguish classical PTC (cvPTC) from follicular-variant PTC (fvPTC) — the two major papillary thyroid carcinoma subtypes — using two independent, explainable branches fused at decision level.

**Clinical motivation:** cvPTC vs fvPTC distinction affects prognosis and management; fvPTC in particular is a source of inter-pathologist diagnostic disagreement. A fused, explainable second-opinion tool built from molecular (methylation) and morphological (histopathology image) evidence is intended to reduce that ambiguity, especially on borderline cases.

**Architecture — two independently-explainable branches fused at decision level:**
- **Methylation branch:** 1D CNN, fine-tuned from a pan-cancer-pretrained checkpoint (`Dataset\Models\pan-cancer-leaky-relu\`) on cvPTC/fvPTC beta-value data. Explainability via **SHAP** (CpG-site feature importance).
- **Histopathology image branch:** 2D CNN over WSI tiles from TCGA-THCA DICOM slides. **Not yet built** — see §6. Explainability via **Grad-CAM** (tile/region localization).
- **Fusion:** confidence-weighted decision-level combination of both branches' predictions (not built yet — depends on the image branch existing first).

> ⚠️ **Not independently verified**: I could not find any saved file (README, doc, notebook markdown cell) stating this clinical-motivation / two-branch-fusion framing anywhere in the current project tree, `docs\`, or `archive\`. This section is written from context supplied directly by the user (Team78) rather than a file I could cross-check. If a fuller project proposal/report exists outside this folder, it's worth re-syncing this section against it.

### Current methylation branch results — read carefully, two different numbers exist

Source: `code\thyroid\FinalModel_modified.ipynb` (current, most-developed notebook). The notebook itself flags a critical distinction that **must not be collapsed into one number**:

**(a) In-sample evaluation** (model evaluated on the same 461 samples it was trained on — optimistic, not a generalization estimate):
- MCC: **0.5745** (notebook's own annotation: "`>0.7 = strong`" — this falls short of that bar)
- AUC-ROC: **0.9040**
- fvPTC Recall: **0.6296**
- MC Dropout (50 forward passes, training=True, on a stratified 50-sample subset of the same training data): raw MCC **0.7669**; on the 26/50 samples flagged "confident" (uncertainty below the mean threshold): MCC **0.8909**, with 24/50 flagged for expert review.

**(b) Stratified 5-fold cross-validation** (base model frozen, only the ~130k-param classification head retrained per fold — every prediction made on genuinely held-out data): printed in the notebook under the explicit header **"CROSS-VALIDATION SUMMARY — REPORT THESE NUMBERS"**, immediately preceded by a code comment in the notebook itself: *"Cross-validation in next cell is the reported result."*
- Aggregate MCC (threshold=0.5): **-0.0018** (chance level)
- Aggregate AUC-ROC: **0.5169** (chance level)
- Aggregate fvPTC Recall: **0.3951**
- Best MCC found via threshold sweep: **0.1067** at threshold=0.40 — still far below the 0.7 "strong" bar

**Bottom line:** the notebook's own author considers (b), not (a), the real result — the in-sample numbers in (a) are inflated by evaluating on training data, and (a)'s MC Dropout "confidence" was computed on a subset of that same training data, not held-out data. The cross-validation result indicates the current methylation model is **not yet performing above chance on genuinely unseen data**. This is the actual state of the methylation branch — do not report (a) alone as "current performance" in any future summary without this context, and don't let this quietly get papered over in future sessions.

### Update (2026-08-11): root cause found and a working fix confirmed

**Root cause:** it is NOT a data problem. `FinalModel_modified.ipynb`'s CV cell trains a 64-unit Dense head directly on the frozen pan-cancer base model's flattened embedding (1,942,272-dim) with only ~234 training samples per fold and `Adam(lr=1e-3)`. That samples-to-effective-parameters ratio causes every single fold to collapse to predicting one constant class (exact 0.0 MCC in all 5 folds, confirmed via per-fold confusion matrices — folds 1-3 always predict cvPTC, folds 4-5 always predict fvPTC). This is a classifier-head/training-regime bug, not evidence the methylation data lacks signal, and not a data-leakage bug (the CV setup itself — per-fold model rebuild, per-fold class-weight recompute, stratified split done once upstream — is methodologically sound).

**Diagnostic sequence** (scripts in `code\scripts\`, all runnable standalone via `PYTHONPATH=code python code\scripts\<name>.py`):
1. `step1_classical_baseline.py` — QDA/SVC on the raw 485,577-probe beta values (the CNN's exact input), same `StratifiedKFold(5, seed=42)`. QDA collapses to chance (covariance matrix singular at this dimensionality with ~294 samples/fold); SVC shows unstable but positive signal (agg. MCC 0.106, AUC 0.686). This ruled out "raw features have no signal at this dimensionality" as the sole explanation and pointed at the classifier/head, not the data.
2. `step2_embedding_diagnostic.py` — extracts the frozen pan-cancer base model's embedding (`base_model.layers[-2].output`) for all 368 training samples, reduces to 50 components via **per-fold-fit PCA** (no leakage), then runs QDA/SVC. **This is the key result: SVC on PCA-reduced frozen embeddings reaches CV MCC 0.536 / AUC 0.844, every fold positive, no collapse.** This proves the frozen pan-cancer conv features do carry a strong, transferable cvPTC/fvPTC signal — the bottleneck was specifically the missing dimensionality reduction before the classifier head, not the pretrained features or the raw data.
3. `step3_fixed_methylation_model.py` — reruns the full pipeline end-to-end (raw beta values → frozen embedding → per-fold PCA(50) → SVC) with the same CV protocol and per-fold collapse-diagnostic printout as the original notebook, to confirm the fix holds outside the precomputed-embedding shortcut. Result:

| | Original CNN head (chance) | Fixed: frozen embedding + PCA(50) + SVC |
|---|---|---|
| Aggregate MCC | -0.0018 | **0.536** |
| Aggregate AUC | 0.5169 | **0.844** |
| Aggregate fvPTC recall | 0.3951 | **0.769** |
| Any fold collapsed to constant prediction | yes (all 5 folds) | **no (0 of 5 folds)** |

**Also applied as part of this fix:** the previously-pending `TCGA-FY-A4B0` relabel (see §3) is now applied in `code\preprocessing\ThyroidDataset.ipynb`'s "Final Model" cell (and standalone in `code\scripts\rebuild_final_data.py`, which was used to rebuild `FinalData.npy` with the correction). `code\.env` was also fixed — it held stale Linux lab-machine paths that meant `FinalModel_modified.ipynb` could not resolve data/model paths correctly via `config.py` on this Windows machine; paths now point at `D:\Capstone_Team78\...`.

### Update (2026-08-11, continued): fine-tuning pass — final locked-in model and true held-out test result

Following the fix above, a staged fine-tuning pass was run to push MCC/accuracy/recall as high as possible while **keeping the branch a 1D CNN** (frozen pan-cancer Conv1D backbone as feature extractor, neural classifier head — not classical ML) per explicit project requirement. Full detail, all scripts, and a seeded reproducibility harness are in `code\scripts\step5_seed_stability_harness.py` through `step11_final_lockin_and_test_eval.py`.

**Two real bugs found and fixed before any tuning:**
1. **The held-out 93-sample test set had never received its 3 pending label corrections** (`docs\label_corrections_log.txt` — only the 1 train-split correction had been applied). Fixed in `code\scripts\rebuild_final_data.py`: corrections are now applied *after* the split, directly to each patient's recorded `meth_split` partition (not before, which would risk reshuffling which split a patient lands in). Test set is now 92 samples, 69 cvPTC / 23 fvPTC — exactly matching the independently-reported figure in §3.
2. **No random seed was fixed anywhere** in the CV pipeline (only `StratifiedKFold`/`PCA` were seeded) — Keras weight init/dropout were not, so re-running the same sweep twice picked different "best" configs. Fixed via `step5_seed_stability_harness.py`: every config below is evaluated over 5 seeds × 5 folds = 25 runs, reporting mean ± std, not a single run.

**What was tried and what worked (all under the 25-run seeded harness unless noted):**

| Experiment | Result | Verdict |
|---|---|---|
| Baseline: raw probes, PCA(50)+Dense(16) head | MCC 0.5039±0.0864, AUC 0.8275±0.0483, recall 0.6801±0.0993 | Reference point |
| LDA (standalone 1D, and PCA+LDA hybrid) | MCC 0.05–0.08 | Failed badly, dropped |
| 3-model soft-vote ensemble (Dense NN+SVC+QDA) | MCC 0.405–0.417 | Worse than best single model (QDA drags it down) |
| SVC alone on PCA(50) | MCC 0.5300±0.0599, AUC 0.8438±0.0438, recall 0.5891±0.0727 | Best MCC found, but classical ML (not a CNN classifier) and lower recall — not chosen |
| Filtered probe set (394,337 probes) + Dense NN head | MCC 0.5289±0.1117, AUC 0.8471±0.0434, recall 0.7299±0.1198 | **Chosen** — MCC tied with SVC, best recall, stays a full neural pipeline |
| Differential probe set (10,899 probes) + Dense NN head | MCC 0.5133±0.1169, AUC 0.8419±0.0653 | Not better than filtered |
| SMOTE oversampling | — | Skipped by explicit decision — synthetic (non-real-patient) data cuts against the project's transparency goal; `class_weight='balanced'` kept instead |
| Nested, leakage-free per-fold threshold tuning | MCC 0.443±0.103 (vs 0.466±0.089 default threshold, same reduced-data setup) | No improvement, dropped |
| Partial fine-tuning of the backbone's last Conv1D block (frozen BatchNorm, LR 1e-5, `GlobalAveragePooling1D` replacing `Flatten` to avoid re-triggering the original bug) | Smoke test: `val_accuracy` frozen at exactly 0.2162 for 13+ consecutive epochs | **Collapsed**, same failure mode as the original bug, despite precautions — matches the documented full-unfreeze collapse precedent in `FinalModel_modified.ipynb`. Aborted, not pursued further. |

**Locked-in final model:** frozen pan-cancer 1D-CNN backbone → filtered-probe-set input (394,337 real probes, zero-padded to 485,577) → PCA(50, fit once on all 368 training samples) → `Dense(16, L2=0.01) → Dropout(0.3) → Dense(2, softmax)`, `Adam(lr=1e-3)`, `class_weight='balanced'`.

**Final result — the one and only held-out test evaluation** (92 samples, evaluated exactly once, no further tuning after seeing it):

| Metric | 25-run seeded CV estimate | **Final test set (92 samples)** |
|---|---|---|
| MCC | 0.5289 ± 0.1117 | **0.5831** |
| AUC | 0.8471 ± 0.0434 | **0.8097** |
| Accuracy | — | **0.8478** |
| fvPTC Recall | 0.7299 ± 0.1198 | **0.6522** |
| fvPTC Precision | — | **0.7143** |

Confusion matrix (rows=true, cols=pred, [cvPTC, fvPTC]): `[[63, 6], [8, 15]]` — 63/69 cvPTC correct, 15/23 fvPTC correct, no collapse. The test MCC (0.583) landing *above* the CV mean and well within its std band is a good sign of genuine generalization, not CV overfitting.

### Update (2026-08-11, continued): alternate pan-cancer checkpoint tried — no real difference

Two other pan-cancer pretrained checkpoints exist on disk (`Dataset\Models\pan-cancer-standard-relu`, `pan-cancer-solid-only`) that had never been used — every experiment above used `pan-cancer-leaky-relu` only. Both are architecturally identical drop-ins (58 layers, same input/embedding shapes), so all three were compared under the same 25-run seeded harness (filtered probe set, locked PCA(50)+Dense(16) head, `code\scripts\step12_alt_checkpoints.py`):

| Checkpoint | CV MCC | CV AUC | CV Recall |
|---|---|---|---|
| **pan-cancer-standard-relu** | **0.5588 ± 0.0712** | **0.8634 ± 0.0396** | **0.7362 ± 0.0877** |
| pan-cancer-leaky-relu (used above) | 0.5289 ± 0.1117 | 0.8471 ± 0.0434 | 0.7299 ± 0.1198 |
| pan-cancer-solid-only | 0.4912 ± 0.0771 | 0.8411 ± 0.0365 | 0.6990 ± 0.0824 |

`standard-relu` looked like a clean win in CV — better mean *and* lower variance on every metric, not a tradeoff. So the one-and-only test evaluation was re-run once more with `standard-relu` substituted (`code\scripts\step13_final_lockin_v2_standard_relu.py`), since the checkpoint choice was made purely from CV evidence, before ever looking at test performance for this checkpoint:

| Metric | leaky-relu (production, test) | standard-relu (test) |
|---|---|---|
| MCC | **0.5831** | 0.5591 |
| AUC | 0.8097 | **0.8488** |
| Accuracy | **0.8478** | 0.8370 |
| fvPTC Recall | 0.6522 | 0.6522 (tied) |
| fvPTC Precision | **0.7143** | 0.6818 |

**The CV win did not hold up on the true 92-sample test set** — the two checkpoints are essentially tied there (`leaky-relu` ahead on MCC/accuracy/precision, `standard-relu` ahead on AUC, recall identical). With n=92, this gap is well within noise either direction. **Conclusion: checkpoint choice doesn't meaningfully matter here — `pan-cancer-leaky-relu` stays the production checkpoint** (already validated, no clear reason to add the complexity of switching). This negative-but-useful result is itself worth keeping on record so nobody re-runs this same comparison later expecting a different answer.

### Update (2026-08-12): MC Dropout calibration CHECKED on the TRUE held-out test set — confidence signal holds up

The original MC Dropout "confident subset" result (§1a: MCC 0.8909 on 26/50 samples, 24 flagged) was computed **in-sample**, on a stratified subset of the 368-sample training data — explicitly flagged as not a valid generalization estimate. This has now been rerun on the genuine 92-sample held-out test set, against the actual locked-in production model (frozen `pan-cancer-leaky-relu` → filtered probe set → PCA(50) → `Dense(16, L2=0.01) → Dropout(0.3) → Dense(2, softmax)`, reproduced exactly per `step11_final_lockin_and_test_eval.py`'s recipe), using the identical MC Dropout technique and confident/flagged rule as the original (`code\thyroid\FinalModel_modified.ipynb`): 50 forward passes with dropout active (`training=True`), per-sample uncertainty = std across passes of the predicted class's probability, `uncertainty_threshold = round(mean(uncertainty), 2)`, confident = below threshold. Script: `code\scripts\step14_mc_dropout_true_holdout.py`. Full per-sample result: `code\thyroid\mc_dropout_true_holdout_result.json`.

**Sanity check passed:** standard (non-MC, dropout-off) inference on the reproduced model gave MCC 0.5831 / AUC 0.8097 / Accuracy 0.8478 — an exact match to the original one-shot test result, confirming the same model/pipeline/test samples were used.

| | Full test set (n=92) | Confident subset | Flagged subset |
|---|---|---|---|
| n | 92 | 50 (54%) | 42 (46%) |
| MCC | 0.5591 | **0.7379** | 0.3977 |
| AUC | 0.8595 | 0.9356 | 0.6941 |
| Accuracy | 0.8370 | 0.9400 | 0.7143 |
| fvPTC Recall | 0.6522 | 0.8333 | 0.5882 |
| fvPTC Precision | 0.6818 | 0.7143 | 0.6667 |

Note the full-test-set MC Dropout numbers (MCC 0.5591) differ slightly from the deterministic single-pass number (MCC 0.5831) — expected, since MC Dropout averages 50 stochastic forward passes rather than one fixed-weights pass; the sanity check above confirms the underlying model is identical.

**Verdict: the confidence signal genuinely transfers to held-out data.** The confident subset (54% of the test set) scores MCC 0.74 vs. 0.56 on the full set and 0.40 on the flagged subset — a real, non-trivial separation, not just an artifact of in-sample evaluation. It is a smaller effect than the in-sample number suggested (0.89 in-sample vs. 0.74 here) and a larger fraction of cases get flagged for review here (46% vs. 48% in-sample — comparable, actually), but the core claim — that MC Dropout uncertainty identifies a subset of higher-accuracy predictions — holds up on genuinely unseen data. This closes the "not yet done" calibration-check item from §4/§6.

### Update (2026-08-12): PCA component count sweep — 50 confirmed as a reasonable choice, no clear improvement found

PCA was locked at 50 components after fixing the original collapse bug (§1, "root cause found"), but 50 itself had never been swept as a hyperparameter under the 25-run seeded harness. Swept `n_components ∈ {20, 35, 50, 75, 100, 150}` using the exact same harness, filtered-probe-set embeddings, and Dense(16) head config that produced the chosen result (`step5_seed_stability_harness.seeded_cv`, 5 seeds × 5 folds = 25 runs per component count). Script: `code\scripts\step15_pca_component_sweep.py`. Full result: `code\thyroid\pca_component_sweep_result.json`.

| PCA components | MCC (mean±std) | AUC (mean±std) | fvPTC Recall (mean±std) |
|---|---|---|---|
| 20 | 0.5181±0.0787 | 0.8340±0.0462 | 0.7568±0.0722 |
| 35 | 0.5292±0.0757 | 0.8451±0.0530 | 0.7466±0.0965 |
| **50 (current)** | **0.5289±0.1117** | **0.8471±0.0434** | **0.7299±0.1198** |
| 75 | 0.5210±0.0908 | 0.8467±0.0462 | 0.7026±0.1140 |
| 100 | 0.5379±0.1066 | 0.8472±0.0380 | 0.7006±0.0978 |
| 150 | 0.5335±0.0986 | 0.8444±0.0413 | 0.6906±0.1077 |

The 50-component row exactly reproduces the previously-reported CV number (MCC 0.5289±0.1117), confirming the sweep harness is consistent with prior runs. **No component count beats 50 by more than roughly one baseline std** (largest delta: pca_dim=100 at +0.0089 MCC, well inside the ±0.1117 noise band) — all six values are statistically indistinguishable given the run-to-run variance. Recall trends slightly downward as component count increases past 50 (0.76 at 20 → 0.69 at 150), a mild but not dramatic pattern. **Conclusion: 50 components stays the production choice — this was a legitimate check, not a wasted one, since it confirms 50 wasn't an arbitrary or lucky pick, but no re-run of the one-shot held-out test evaluation is warranted** (no candidate cleared the "beat baseline by >1 std" bar set in advance for burning that one-shot evaluation).

**Status / what's still open:**
- This is now the methylation branch's production model, still architecturally a 1D CNN throughout (feature extraction and classification both neural) as required.
- MC Dropout confidence calibration is now validated on the true held-out test set (above) — the confidence signal is real, though weaker than the in-sample estimate suggested. Fusion-branch design (once the image branch exists) can treat MC Dropout uncertainty as a meaningful per-case confidence signal, with the caveat that ~46% of cases get flagged at this threshold, not a small fraction.
- `code\thyroid\Explainability.ipynb` (SHAP) still has no completed/trustworthy result and now also needs updating to target this locked-in model/pipeline (its `DeepExplainer` setup assumed the original, broken architecture) — separate, not yet addressed.
- Not tried: fine-tuning deeper (non-last-block) CNN layers, or the full backbone with much heavier regularization than attempted here — the partial-fine-tuning attempt collapsed with only the last block unfrozen, so further unfreezing seems unlikely to help without a fundamentally different regularization/data-scale approach; deprioritized given the collapse evidence.

---

## 2. The Core Data Problem That Was Found and Fixed

This is the most important section to understand before touching the image branch.

**What was wrong:** the original `Classical_PTC_PNG` image folder (root-level, now archived) was built by filtering the IBIA/AIIMS thyroid histology dataset (HISTOS_1000000014) for filenames containing "Papillae"/"Pappilae" and calling the result "Classical PTC." "Papillae" is a **morphological/architecture** bucket in IBIA's own folder taxonomy, not a confirmed diagnosis label. The same mistake was made building `FVPTC_PNG_Images` (filtered on "FVPTC" in filenames).

**What was discovered:** the IBIA/AIIMS dataset has **no usable classic-PTC (CPTC) label anywhere in its released data** — this was checked exhaustively across all 20 of its folder paths. The only real diagnosis labels present are:
- `NIFTP`, `IEFVPTC`, `IFSPTC` — all fvPTC-family / follicular-pattern diagnoses, not cvPTC
- `FND` — genuinely benign (follicular nodular disease)
- Everything else (`Papillae`, `Follicles`, `Discard`, "PTC like features" / "Non-PTC like features") is architecture/presentation description, not a diagnosis (see `archive\superseded_data\label_mapping_results.txt` for the full crosstab that established this).

**The fix:** TCGA-THCA carries real ICD-O-coded diagnoses — cvPTC = morphology code `8260/3`, fvPTC = `8340/3`. Both malignant classes (cvPTC and fvPTC) are now sourced **exclusively from TCGA-THCA**, never mixed with IBIA. This is a deliberate, load-bearing decision — see §4 for why single-sourcing per class matters.

**IBIA's role now:** benign class (FND) and NIFTP/nuclear-feature explainability material only. It is **not** used for cvPTC/fvPTC subtype classification anywhere in the current pipeline. The current benign download (`Dataset\Datasets\ibia_metadata.xlsx` → `Dataset\Benign_PNG_Images\`, built by `code\scripts\download_benign.py`) deliberately restricts to `STAGE1_BTID` folders only, excludes `MT_Discard`, and pulls `MT_FND`/`MT_Follicles`/`MT_Papillae`/`ET_FND`/`ET_Follicles`/`ET_Pappilae` — i.e. benign/architecture categories, correctly scoped.

---

## 3. The Paired Multimodal Cohort

Verified directly against the raw data files in this session (not just against the log — the raw `.txt` headers, the merged CSV, and the demo-set CSV were all cross-checked and are internally consistent):

- **461 real methylation samples exist** (359 cvPTC + 102 fvPTC, counted directly from `Dataset\RawData\cvPTC_beta_values_unprocessed.txt` / `fvPTC_beta_values_unprocessed.txt` headers). The "368" figure that appears in some older comments/logs is **not** a different/smaller cohort — it's the 80% train-split of these same 461 (`train_test_split(test_size=0.2, random_state=2569)`, reproduced in `code\scripts\intersect_methylation_image.py`), i.e. 368 train / 93 test.
- **457 of 461 (99.1%) have a matching TCGA-THCA image case** — verified via `Dataset\Datasets\paired_multimodal_cases_full.csv` (457 data rows). True patient-paired multimodal fusion is possible for this cohort, not just unpaired decision-level fusion.
- **All 93 of the held-out (test-split) samples are among the 457 paired ones — zero loss on the test side.** The 4 unpaired samples are all on the train side (3 cvPTC + 1 fvPTC). This means the test-split is a clean, fully-paired, genuinely-unseen demo/evaluation set.
- **4 label disagreements** were found between the methylation cohort's baked-in cvPTC/fvPTC label and TCGA's clinical ICD-O record (all 4 were methylation-says-cvPTC / TCGA-says-fvPTC, morphology 8340/3). Resolved via TCGA's own `consistent_pathology_review` field:
  - **Relabeled to fvPTC** (TCGA central pathology re-review agreed): `TCGA-BJ-A45K` (test), `TCGA-J8-A3O0` (test), `TCGA-FY-A4B0` (train)
  - **Excluded** (TCGA re-review itself flagged disagreement — genuinely unresolved, dropped rather than guessed): `TCGA-FY-A3R9` (test)
  - Full reasoning: `docs\label_corrections_log.txt`

**Where these corrections currently stand (checked against the actual files, not assumed):**
- `Dataset\Datasets\PairedDemoSet\methylation\labels.csv` (the 92-patient demo set extracted from the test-split) **has the corrections applied**: `TCGA-BJ-A45K,fvPTC` and `TCGA-J8-A3O0,fvPTC` are both present with the corrected label; `TCGA-FY-A3R9` is correctly absent (excluded). 92 = 93 test-split − 1 excluded. Class balance is 69 cvPTC / 23 fvPTC, which matches `docs\HANDOVER_2026-07-28.md`'s figure exactly once you trace the arithmetic (72 cv/21 fv → −1 cv for the exclusion → −2 cv/+2 fv for the two relabels → 69 cv/23 fv).
- `Dataset\Datasets\paired_multimodal_cases_full.csv` (the full 457-case cohort table) **does NOT have the corrections applied** — all 4 disputed cases still show their original `meth_label` (`cvPTC_classical`) with `label_match=False`. This is expected and matches what `docs\label_corrections_log.txt` says: the corrections were applied only to the demo-set extraction step, not back to this upstream table.
- **`TCGA-FY-A4B0` specifically (the train-split case): its relabel is confirmed still NOT applied anywhere**, consistent with the "PENDING — NOT YET APPLIED" note in `docs\label_corrections_log.txt`. It won't matter until a training-set build happens — but if/when the methylation model is retrained, **apply this relabel first** (flip `TCGA-FY-A4B0` from cvPTC_classical to fvPTC_follicular_variant before building the training tensors).

---

## 4. Key Methodological Decisions

**Why single-source per class (never IBIA-for-one-class, TCGA-for-another):** mixing image sources across classes risks the model learning scanner/institution/staining artifacts instead of actual tumor morphology. This is why §2's fix routes both cvPTC and fvPTC through TCGA-THCA exclusively, and IBIA is confined to a different role (benign class, explainability material) rather than ever supplying a malignant subtype class.

**Why case-level (not patch/tile-level) train/test splitting is mandatory for the image branch:** tiles from the same slide/patient are highly correlated; splitting at the tile level would leak patient identity across train/test and produce inflated, meaningless performance numbers. Splits must be done at the case (patient) level before any tiling happens.

**Weak/slide-level supervision limitation:** tiles inherit their whole slide's label. A slide labeled fvPTC may still contain patches of normal tissue, necrosis, or even other tissue architecture — this is expected noise inherent to weak supervision, not a pipeline bug. Multiple Instance Learning (MIL) is the principled future direction if/when patch-level noise becomes a limiting factor, but is not in scope for the current pipeline.

**Ink artifact handling:** surgical margin ink marks specimen **orientation**, not tumor location. It must be filtered out as background/artifact during tissue/quality/ink filtering at tiling time — never treated as a signal correlated with tumor presence.

**Confidence validation plan:** MC Dropout-derived confidence for the methylation branch has now been validated via a true held-out calibration check (2026-08-12, §1) — confident-subset predictions do score meaningfully higher (MCC 0.74 vs 0.56 full-set) on genuinely unseen data, not just in-sample. Grad-CAM as an independent cross-check is still pending for the image branch (see §6) once that branch exists. Raw softmax/uncertainty output should not be reported as "confidence" without this kind of validation.

---

## 5. Current Folder Structure

```
D:\Capstone_Team78\
├── Dataset\
│   ├── TCGA_THCA_Slides\          verified 134-case / 7-batch DICOM WSI download
│   │                               (cvPTC_classical\, fvPTC_follicular_variant\ subfolders)
│   ├── RawData\                   raw methylation source files (cvPTC/fvPTC/norm, beta+m values,
│   │                               processed + unprocessed, plus differential-CpG tables)
│   ├── Benign_PNG_Images\         current, correctly-scoped IBIA benign set (MT/ET × FND/Follicles/Pappilae)
│   ├── Models\                    pan-cancer-* checkpoints (base, pretrained) +
│   │                               thyroid-cancer / thyroid-subtype(.h5/.keras, weighted, optimized)
│   ├── Results\, Tensorboard\, conda\
│   ├── tcga_thca_download_subset.csv   the 134-case download plan (hardcoded path — do not move)
│   └── Datasets\
│       ├── PairedDemoSet\         92-patient paired demo/eval set (methylation + matched image case ids)
│       │   ├── image_case_ids.csv
│       │   └── methylation\beta_values_demo_test.tsv, labels.csv
│       ├── download_batches\      the 7 batch CSVs used for the TCGA_THCA_Slides download
│       ├── *.npy (6 files)        pan-cancer + thyroid-subtype train tensors (FinalData.npy etc.)
│       ├── GDC_samples.csv        provenance/citation reference for the pan-cancer pretraining data
│       │                          (the 121GB raw pan-cancer download itself was deleted — checkpoint is kept instead)
│       ├── paired_multimodal_cases_full.csv   the 457-case merged cohort table (§3)
│       └── ibia_metadata.xlsx, ibia_samples.xlsx, ibia_experiments.xlsx,
│           tcga_thca_cvptc_fvptc_case_list.csv, methylation_case_barcodes.csv
├── code\
│   ├── config.py, .env, .git\     (.env currently holds paths for a different, Linux lab machine —
│   │                                not yet configured for local Windows runs, pre-existing/unrelated to any recent work)
│   ├── thyroid\                   FinalModel.ipynb, FinalModel_modified.ipynb (current, see §1),
│   │                               Explainability.ipynb (SHAP), MachineLearningModels.ipynb
│   ├── preprocessing\             ThyroidDataset.ipynb
│   └── scripts\                   download_tcga_thca_slides.py, run_all_batches.ps1,
│                                   download_benign.py, intersect_methylation_image.py
├── docs\
│   ├── HANDOVER_PROJECT_CONTEXT.md   this file
│   ├── HANDOVER_2026-07-28.md, label_corrections_log.txt, follow_up_items.txt
│   └── LICENSE, Readme.txt
└── archive\                       superseded data + one-off scripts, kept (not deleted) for the
                                    "what we found and fixed" narrative — see archive\superseded_data\,
                                    archive\debug_scripts\, archive\pancancer_training\
```

---

## 6. What's Done / What's Pending

**Done:**
- cvPTC/fvPTC case list built and paired against the methylation cohort (§3)
- Paired multimodal cohort verified (457/461, 99.1%; test-split fully paired with zero loss)
- 134-case TCGA-THCA DICOM data downloaded and integrity-verified
- Benign IBIA data downloaded, correctly scoped (§2)
- Label disagreements resolved and reasoning documented; corrections applied to the demo set (§3)
- Methylation branch trained (1D CNN, fine-tuned from pan-cancer checkpoint) — but see §1 for the honest performance picture
- Folder structure cleaned and reorganized

**Pending — image branch, in order** (in progress as of 2026-08-12, staged build with verification at each stage — see below):
1. DICOM tiling pipeline — **use `wsidicom`, not OpenSlide** (confirmed necessary: `Dataset\TCGA_THCA_Slides\` cases are multi-instance DICOM WSIs, one `.dcm` file per pyramid level; OpenSlide is built for SVS/TIFF and is the common wrong assumption here).
   - **Stage 1 (done):** single-slide proof of concept (`code\scripts\image_stage1_poc.py`). Pyramid has only 4 levels per case — `[0, 2, 4, 5]` (~40x/~10x/~2.5x/~1.2x); levels 1 and 3 (~20x/~5x) do not exist in this download batch (confirmed via direct pydicom inspection on 2 cases, same pattern both times). No native ~20x level, so tiles are extracted at 1024×1024 native ~40x (level 0) then image-domain 2× downsampled (Lanczos) to the final 512×512 (~20x-equivalent field of view, ~512μm/side) — avoids pyramid-interpolation blur, standard practice when the desired level doesn't exist natively. Real tissue confirmed via contact sheet (first attempt used the thumbnail tissue-mask *centroid* as the sample point and got all-background tiles because this tissue is ring/donut-shaped with a hollow center — fixed by sampling actual tissue-mask pixel coordinates directly, not their centroid).
   - **Pre-Stage-2 empirical check (done):** tested whether cvPTC/fvPTC tiles differ systematically in nuclei/glandular density (shortcut-learning risk, since growth pattern *is* the diagnostic distinction) — see `docs\image_branch_nuclear_density_confound_check.md` for the full permanent record. **Result: yes, a real, statistically significant difference** (p=2.56e-6, Cohen's d=0.70, cvPTC denser than fvPTC) — confirmed both by hematoxylin-channel color deconvolution and direct visual inspection. Approved mitigation: trim both tails of the density distribution (not full distribution-matching, which would fight real biology) — cutoff values being set by visual calibration against real tile content, not blind percentiles (in progress; the high-density tail turned out on inspection to be real dense tumor tissue, not blood/necrosis junk as originally hypothesized — see the doc for why, and for the still-open exact cutoff decision).
2. Tissue / quality / ink / density filtering on tiles (ink = artifact to filter out, never a signal — §4; density tail-trim per the confound check above) — **core pipeline done and verified on the Stage 1 test case** (`code\scripts\image_stage2_filtering.py`), **two open items before Stage 3 proceeds** (below). Two filter rules were wrong on the first pass and fixed after visual verification caught them (full story in `docs\image_branch_nuclear_density_confound_check.md`): a hue-based ink detector false-positived on 35% of tiles (dense hematoxylin nuclei look similar to ink in hue space) — fixed to a value+saturation rule (dark AND desaturated, not dark AND any-hue), which dropped the false-positive rate to genuine ink only (14/3668, all visually confirmed); a per-case relative-percentile blur detector was found to be measuring "low texture" not "blur" (TCGA scans are generally well-focused) — replaced with a fixed conservative absolute safety-net floor. Final funnel on the test case: 10,416 grid cells → 3,668 tissue-kept → 3,625 final (blur -7, ink -14, density-low -22, density-high 0) — 98.8% survival past tissue filtering, visually spot-checked clean.

   **Open item A — thin ink lines, RESOLVED (2026-08-12): shape-based secondary filter applied.** 2 of 50 sampled kept tiles had a genuine thin dark line missed by the value+saturation ink rule — root cause is 2× downsample blending raising the line's measured saturation above threshold, not an area-fraction blind spot. Added a secondary gate (value-only darkness `<0.45` + connected-component span-fraction `≥0.5`, min area 15px), reported as a separate `line_dropped` funnel count, not merged into `ink_dropped`. Applied and reverified on the test case: **271/3647 tiles (7.4%) flagged and dropped** (higher than the 4% seen in the small 50-tile sample, but consistent — all 20 sampled rejects visually confirmed as genuine thin dark artifacts, likely a mix of ink and tissue-processing cracks). Final test-case funnel: 10,416 grid → 3,668 tissue-kept → blur -7, ink -14, **line -271**, density-low -22 → **3,354 final kept** (91.4% of tissue-filtered tiles). Accepted, known trade-off: this shape-based rule can in principle also catch genuine thin anatomical structures (vessels, septae) — no clean way to fully separate that from real artifact at this scale, but erring toward dropping a rare real thin-vessel tile is the defensible choice per the project's "ink is never signal" principle (§4).

   **Open item B — TSS-site/scanner confound with class label, found during Stage 3 pre-check survey. Decisions made (2026-08-12), full record in `docs\image_branch_tss_site_confound_check.md`:** `TCGA-EM-*` (a single Tissue Source Site) accounts for 37/69 fvPTC cases (54%) and 0/65 cvPTC cases — a purely technical (scanner/institution/staining-batch) confound, not a biological one, and more serious than the nuclear-density confound already mitigated: any tile from an EM-sourced slide is deterministically fvPTC regardless of content. **This cannot be fixed by the split alone** — there is no EM-sourced cvPTC case to balance against, so no split strategy removes the underlying shortcut-learning risk, only avoids compounding it. Decisions:
   - **Baseline (do both, not either/or):** Stage 3's case-level split will be explicitly stratified by TSS site (in addition to class) — close to free, strictly better than an unstratified split, but explicitly does NOT resolve the confound itself.
   - **Elevated to a REQUIRED evaluation step (not optional):** once the image CNN is trained, compare model confidence/accuracy on EM-sourced vs. non-EM-sourced fvPTC test cases. A meaningful gap is a reportable finding of site-based shortcut learning, to be read together with the Grad-CAM systematic check already required.
   - **Follow-up finding, supporting the severity of this confound:** checked whether thin-ink-line prevalence (item A above) itself correlates with site (`code\scripts\image_stage2_tss_site_line_check.py`, 14-case site-diverse sample). **Result: EM-site tiles are flagged at 0.28% (1/358) vs. 9.07% (65/717) for non-EM sites — a ~32× difference**, consistent across all 4 EM cases sampled. This means thin-line presence/absence is itself a second, near-perfect, independently-learnable proxy for class in this cohort (on top of raw color/density) — strengthens the case that site-level technical signatures are a real, multi-pronged risk here, not a one-off. Also means Stage 3's per-site tile counts will differ partly *because* of this real artifact-rate gap, not a tiling bug — worth remembering when interpreting those numbers.

   **Pre-Stage-3 scaling checks — resolved with real (not extrapolated) data (2026-08-12):**

   **A real, second pipeline bug found and fixed during this check** (not just an estimate refinement): the full-res tissue re-verification step re-fit an Otsu threshold per-tile instead of reusing the slide-level threshold, which silently discarded large amounts of real tissue on slides with big, dense, uniformly-stained regions — confirmed on `TCGA-DJ-A2PP` (a tile visually ~98% real tissue measured tissue_fraction 0.36 under the old per-tile rule vs. 0.98 with the fix). Fixed in `tissue_fraction()` (now takes a fixed slide-level threshold). Impact was uneven and correlated with tissue density, not random: `TCGA-DJ-A2PP` final tiles 1,363→**6,206** (4.55×), `TCGA-DJ-A3VM` 3,701→**6,014** (1.63×), the original test case and 4 smaller/less-dense sampled cases only +0.7% to +8%. Full writeup: `docs\image_branch_nuclear_density_confound_check.md`.

   **Real per-case tile-count variance, measured (not thumbnail-estimated) on 6 size-diverse cases, fix applied:** min=183 (`TCGA-EM-A2CQ`), max=6,206 (`TCGA-DJ-A2PP`), median=2,943, mean=3,091±2,563 — a real ~34× range. `TCGA-EM-*` cases confirmed systematically and severely smaller: full 24-case thumbnail survey shows **EM mean 668 tiles (range 285–1,056) vs. non-EM mean 4,401 (range 1,388–9,151) — zero overlap between the two groups.** Full breakdown: `docs\image_branch_tss_site_confound_check.md`.

   **Revised full-134-case estimate**, using the real measured aggregate survival rate (70.0%, up from the bug-affected 42.2%) applied to the 24-case thumbnail-based tile-count proxy: **≈314,000 total tiles, ≈147GB disk, ≈34.4 hours single-threaded processing time** (time estimate essentially unchanged by the tissue-fraction fix, since it doesn't change how many tiles get read/processed, only how many survive).

   **Resumability — built and verified working.** `code\scripts\image_stage3_full_run.py` writes a per-case JSON manifest only after that case's tiles are fully saved; on restart, any case with an existing manifest is skipped. Demo: processed 2 cases (4,042 tiles), rerun completed in 1.1s with both correctly skipped. A crash hit mid-build (unrelated bug, fixed) didn't corrupt state, since the manifest write happens before the process could fail on that code path — real evidence the design is robust to interruption, not just in theory.

   **Parallelization — real measured data, worker count decided.** Machine has 32 logical cores and 128GB RAM (107.7GB free). Measured peak memory per worker during a 6-way parallel run: 280–374MB — memory is a complete non-issue at any plausible worker count (even 32 workers ≈ 12GB). The naive 6-cases-on-6-workers measured speedup (2.61×) is an artifact of task count equaling worker count (wall time bounded by the single slowest case, others sit idle) and is not representative of the real 134-case run, where a work queue keeps every worker fed. Using the real measured per-tile rate (0.2757s/tile) and total estimated read-tile count across 134 cases: serial ≈34.4 hours; with N workers, wall-time ≈ serial/N + a tail buffer for the largest single case still finishing (~0.78hr).

   **Decision: 20 workers.** Estimated wall time ≈2.5 hours. Not maxing all 32 logical cores — leaves 12 (37.5%) headroom for the OS, the orchestrating process, and disk-I/O contention across many concurrent WSI reads on a single drive (untested at higher concurrency); memory is not the constraint (128GB total makes even 32 workers trivial, ~12GB). 20 was chosen over a higher count because the marginal wall-time gain going from 20→24 or 24→32 is small (diminishing returns once workers ≫ number of large/slow cases in flight at once) while headroom loss is not — a deliberate, moderate choice, not a memory-driven one.

   **Per-case tile cap: not needed for time or disk reasons** (both comfortably resolved above) — if a cap is still wanted, it would now be a training-time class/case-balance decision (should one 6,000-tile case dominate a batch relative to a 200-tile case?), not a tiling-pipeline necessity. Not yet decided; can be revisited at the CNN-training stage independent of the tiling run.

   **Case-level split — built (2026-08-12), before any tiling.** `code\scripts\image_stage3_build_split.py`, saved to `Dataset\Datasets\image_branch_case_split.csv` (134 rows: case_id, label, tss_site, split, forced_demo_overlap, est_final_tiles). Method: the 27 cases overlapping `PairedDemoSet` were force-assigned to test (confirmed exactly 27, matching the known follow-up item); this alone landed at 20.1% of 134 cases — almost exactly the target 20% test fraction (consistent with the methylation branch's split ratio) — so zero additional cases needed stratified selection on top. Practical consequence: **this split's test set is entirely determined by which patients happened to have both a methylation and an image record, not by an independent stratified draw** — there was no slack left to correct for any resulting site/class skew.

   **Case-count balance:** train 107 (53 cvPTC/54 fvPTC), test 27 (12 cvPTC/15 fvPTC) — reasonably proportional. Per-site case counts also reasonable (e.g. EM 31 train/6 test ≈ 16% test rate, somewhat below the 20% target but not alarming at this sample size; full table in the split CSV).

   **Tile-count balance — this is where the EM effect actually shows up, as anticipated.** Total estimated tiles: train 223,633, test 63,219 (22.0% of total — close to the case-based 20% target, so the *overall* split isn't tile-skewed). But **within the fvPTC test cases specifically, EM contributes only 22.4% of tiles (8,121 of 36,276) despite being 40% of fvPTC test *cases* (6 of 15)** — confirming the concern raised before building this: a case-count-balanced split still under-represents EM at the tile level, because each EM case is individually so much smaller. Full cohort context: EM is 37/134 cases (53.6% of all fvPTC cases by count) but only 33,076/286,852 estimated tiles cohort-wide (11.5% of the total tile pool) — see `docs\image_branch_tss_site_confound_check.md` for the full severity write-up.

   **What this means for the mandatory EM-vs-non-EM subgroup evaluation:** the absolute EM test-tile count (8,121) is still large enough to support a meaningful tile-pooled comparison — not a "too few tiles to trust" situation. But the EM test **case** count (6) is thin for a case-level aggregated comparison (e.g. majority-vote-per-case accuracy) — a gap found at n=6 cases could plausibly be noise. Recommendation for the eventual evaluation: **report both tile-pooled and case-level EM-vs-non-EM metrics**, and treat a finding as more trustworthy if it's consistent across both (thousands of tiles + 6 cases) rather than relying on either alone. This refines, not replaces, the mandatory-evaluation decision already made.
3. Macenko stain normalization
4. 2D CNN training with case-level (not patch-level) train/test split — §4
5. **Confidence calibration check + Grad-CAM validation for the image branch — §4. Grad-CAM is now an ELEVATED, REQUIRED validation gate (2026-08-12 decision), not an optional explainability demo:** once the image CNN is trained, Grad-CAM must run on a systematic sample of test tiles per class (not 2-3 anecdotal cases), explicitly checking whether attention concentrates on genuine morphological features (nuclear grooves, papillary fronds, follicle boundaries) rather than raw density/color regions. This is the real test of whether the density-confound mitigation above actually worked, and the image branch is not considered done without it.
6. Fusion wiring (confidence-weighted decision-level combination of both branches)
7. Demo build on the paired held-out cases (`PairedDemoSet\`)

**Pending — methylation branch:** the original chance-level CV result (§1b) was superseded by the root-cause fix and fine-tuning pass (§1, 2026-08-11/12) — production model now at test MCC 0.5831, with MC Dropout calibration validated on the true held-out set and a PCA component count sweep confirming 50 as a reasonable choice. Remaining open items are SHAP (below) and the still-pending `TCGA-FY-A4B0` relabel (§3), which must be applied before any retraining/rebuild of the training tensors.

**Still open from the original methylation work (lower priority):** SHAP analysis completion, CpG probe annotation, a results table (referred to elsewhere as "Table 4.1"), Docker/ONNX packaging.

**Known open follow-up (not blocking):** only 27 of the 92 `PairedDemoSet` patients currently have a downloaded TCGA-THCA image (the 134-case image subset wasn't originally scoped for full demo coverage — see `docs\follow_up_items.txt`). May be worth expanding later so held-out performance can be reported on the full 92-patient paired cohort rather than a 27-case subset, but this doesn't block getting the image pipeline built and working end-to-end first.

---

## 7. Working Conventions

- **Verify against real files/output before reporting anything done.** This project has repeatedly found that trusting summaries or exit codes without checking actual file counts or content leads to silent errors (e.g. a folder move landing one level too deep, discovered only by a file-count mismatch — not by the move command's exit code).
- **Flag ambiguity rather than guess or silently adapt.** If something about scope, naming, or a hardcoded path is unclear, ask before acting.
- **Don't silently pick a side on a genuine data/label ambiguity** — present the evidence and ask, the way the FY-A3R9 exclusion was handled (neither source trusted over the other, dropped rather than guessed).
- `"duck"` = end-of-response signal used in this project's sessions.
