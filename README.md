# Multimodal PTC Subtype Classification — Team 78

Distinguishing **classical papillary thyroid carcinoma (cvPTC)** from **follicular-variant PTC (fvPTC)** using two independently-explainable branches — DNA methylation and histopathology — fused at decision level.

Built on **TCGA-THCA**. Capstone project, PES University (UE23CS441A).

---

## Why

The cvPTC/fvPTC distinction affects prognosis and management, and fvPTC in particular is a well-known source of inter-pathologist disagreement. The goal here is a *second-opinion* tool that produces two independent, cross-checkable lines of evidence — molecular and morphological — rather than a single black-box score, so a borderline case can be reasoned about instead of just labelled.

Two design commitments follow from that, and they constrain most of the decisions in this repo:

- **Explainability is a gate, not a demo.** SHAP for the methylation branch, Grad-CAM for the image branch. The image branch is not considered done until Grad-CAM has been run systematically (not on 2–3 cherry-picked tiles) and shown to attend to real morphology.
- **Confidence must be validated before it is reported.** Raw softmax is not confidence. The methylation branch's MC Dropout uncertainty was only accepted as a confidence signal after it was re-checked on the true held-out test set (see below).

---

## Status at a glance

| Branch | State | Headline |
|---|---|---|
| **Methylation (1D CNN)** | ✅ Production model locked in | Test MCC **0.583**, AUC **0.810**, accuracy **0.848** on 92 held-out samples |
| **MC Dropout confidence** | ✅ Validated on held-out data | Confident 54% of cases: MCC **0.738** vs 0.559 full-set |
| **SHAP explainability** | ⚠️ Not complete | `Explainability.ipynb` still targets the *old, broken* architecture |
| **Image (2D CNN)** | 🔨 In progress — tiling | Stage 3 full-run at **36 / 134** cases |
| **Fusion** | ⛔ Not started | Blocked on the image branch |

Full narrative, including every negative result: **[`docs/HANDOVER_PROJECT_CONTEXT.md`](docs/HANDOVER_PROJECT_CONTEXT.md)** — read this before touching anything.

---

## Data

Both malignant classes come **exclusively from TCGA-THCA**, using real ICD-O morphology codes: cvPTC = `8260/3`, fvPTC = `8340/3`.

This is load-bearing. An earlier version of this project built its cvPTC class by filtering the IBIA/AIIMS histology dataset for filenames containing "Papillae" — but "Papillae" is an *architecture* bucket in IBIA's taxonomy, not a diagnosis, and IBIA turns out to carry **no usable classic-PTC label anywhere** in its released data. IBIA is now confined to the benign class (FND) and NIFTP explainability material, and never supplies a malignant subtype. Mixing sources across classes would also invite the model to learn scanner and staining artifacts instead of tumor morphology.

**Cohort:**

- 461 methylation samples (359 cvPTC / 102 fvPTC) → 368 train / 93 test (`random_state=2569`)
- 457 of 461 (99.1%) have a matching TCGA-THCA image case — real patient-level pairing is possible
- All 93 test-split samples are paired; the held-out set is fully multimodal
- 4 label disagreements against TCGA's clinical record were resolved via `consistent_pathology_review`: 3 relabelled to fvPTC, 1 (`TCGA-FY-A3R9`) **excluded** rather than guessed. Reasoning in [`docs/label_corrections_log.txt`](docs/label_corrections_log.txt).
- Final held-out test set: **92 samples, 69 cvPTC / 23 fvPTC**

> ⚠️ **`TCGA-FY-A4B0`'s relabel is logged but not yet applied** to the training tensors. Apply it (via `code/scripts/rebuild_final_data.py`) before any retrain.

Raw TCGA data is **not** in this repo — download it yourself via the GDC/IDC portals under their data use policies. `.gitignore` excludes all slides, tiles, `.npy` tensors and model checkpoints.

---

## Methylation branch

### Architecture

```
raw beta values (485,577 probes)
      │  filtered probe set — 394,337 real probes, zero-padded back to 485,577
      ▼
frozen pan-cancer 1D-CNN backbone       ← pan-cancer-leaky-relu checkpoint, not retrained
      │  penultimate layer embedding (1,942,272-dim)
      ▼
PCA(50)                                 ← fit once on all 368 training samples
      ▼
Dense(16, L2=0.01) → Dropout(0.3) → Dense(2, softmax)
                                          Adam(lr=1e-3), class_weight='balanced'
```

### The bug that defined this branch

The original model scored **MCC −0.0018 / AUC 0.517** under 5-fold CV — chance. Every one of the 5 folds collapsed to predicting a single constant class.

It was not a data problem and not a leakage problem. The CV cell trained a 64-unit Dense head directly on the frozen backbone's **1.94-million-dimensional flattened embedding** with ~234 training samples per fold. The samples-to-parameters ratio guaranteed collapse.

Three diagnostics established this rather than assuming it:

| Step | Script | Finding |
|---|---|---|
| 1 | `step1_classical_baseline.py` | QDA/SVC on raw probes — QDA singular, SVC weak but positive (MCC 0.106). Rules out "no signal in the data". |
| 2 | `step2_embedding_diagnostic.py` | SVC on **per-fold-fit** PCA(50) of frozen embeddings → MCC **0.536** / AUC 0.844, zero folds collapsed. The features were fine; the missing dimensionality reduction was the bug. |
| 3 | `step3_fixed_methylation_model.py` | Same result end-to-end from raw beta values, not the precomputed shortcut. |

### What was tried afterwards

Every config below was evaluated over **5 seeds × 5 folds = 25 runs** (`step5_seed_stability_harness.py`), because the original pipeline seeded only `StratifiedKFold` and `PCA` — Keras weight init was unseeded, so reruns picked different "best" configs.

| Experiment | CV MCC | Verdict |
|---|---|---|
| Baseline: raw probes + PCA(50) + Dense(16) | 0.504 ± 0.086 | Reference |
| **Filtered probe set (394,337) + Dense(16)** | **0.529 ± 0.112** | ✅ **Chosen** — best recall (0.730), stays fully neural |
| SVC on PCA(50) | 0.530 ± 0.060 | Tied on MCC but classical ML and lower recall |
| LDA (standalone and PCA-hybrid) | 0.05–0.08 | Failed badly |
| 3-model soft-vote ensemble | 0.405–0.417 | Worse than best single model |
| Differential probe set (10,899) | 0.513 ± 0.117 | No gain |
| Nested per-fold threshold tuning | 0.443 ± 0.103 | No gain, dropped |
| Partial backbone fine-tune (last Conv1D block) | — | **Collapsed** — `val_accuracy` frozen at 0.2162 for 13+ epochs |
| Alternate checkpoints (`standard-relu`, `solid-only`) | 0.559 / 0.491 | CV win did **not** hold on test — checkpoint choice doesn't matter here |
| PCA sweep {20, 35, 50, 75, 100, 150} | all within ±1 std | 50 confirmed as reasonable, not lucky |

SMOTE was **skipped by decision** — synthetic non-patient data cuts against the transparency goal. `class_weight='balanced'` is used instead.

### Final result — one held-out evaluation, run exactly once

| Metric | 25-run seeded CV | **Held-out test (n=92)** |
|---|---|---|
| MCC | 0.5289 ± 0.1117 | **0.5831** |
| AUC | 0.8471 ± 0.0434 | **0.8097** |
| Accuracy | — | **0.8478** |
| fvPTC recall | 0.7299 ± 0.1198 | **0.6522** |
| fvPTC precision | — | **0.7143** |

Confusion matrix `[[63, 6], [8, 15]]` — no collapse. Test MCC landing *above* the CV mean and inside its std band suggests genuine generalization rather than CV overfitting.

### MC Dropout confidence — validated, and weaker than it first looked

The original "confident subset MCC 0.89" figure was computed **in-sample** on training data. Re-run on the true held-out set (`step14_mc_dropout_true_holdout.py`, 50 stochastic passes, threshold = rounded mean per-sample uncertainty):

| | Full test (n=92) | Confident (n=50) | Flagged (n=42) |
|---|---|---|---|
| MCC | 0.5591 | **0.7379** | 0.3977 |
| Accuracy | 0.8370 | **0.9400** | 0.7143 |
| fvPTC recall | 0.6522 | **0.8333** | 0.5882 |

The signal is real and transfers — but it is 0.74, not 0.89, and **46% of cases get flagged for review**, not a small fraction. Fusion design should treat that as a meaningful per-case confidence signal with that caveat attached.

---

## Image branch (in progress)

TCGA-THCA slides here are **multi-instance DICOM WSIs** — one `.dcm` per pyramid level. Use **`wsidicom`, not OpenSlide**; OpenSlide targets SVS/TIFF and is the common wrong assumption.

**Pipeline stages:**

1. **Stage 1 — POC** (`image_stage1_poc.py`) ✅ Pyramid has only 4 levels `[0, 2, 4, 5]`; there is no native ~20× level, so tiles are read at 1024×1024 native ~40× then Lanczos-downsampled 2× to 512×512.
2. **Stage 2 — filtering** (`image_stage2_filtering.py`) ✅ Tissue → blur → ink → thin-line → density. Test-case funnel: 10,416 grid cells → 3,668 tissue-kept → **3,354 final** (91.4% survival).
3. **Stage 3 — full 134-case run** (`image_stage3_full_run.py`) 🔨 **36/134 done.** ~314k tiles, ~147GB, ~2.5h wall time at 20 workers.
4. Macenko stain normalization — pending
5. 2D CNN training, **case-level split** — pending
6. Grad-CAM validation gate + EM-site subgroup eval — pending
7. Fusion wiring — pending
8. Demo on `PairedDemoSet` — pending

### Three confounds found before training anything

Each is documented permanently rather than silently mitigated.

**Nuclear density** ([doc](docs/image_branch_nuclear_density_confound_check.md)) — cvPTC tiles are genuinely denser than fvPTC (p=2.6e-6, Cohen's d=0.70). Since growth pattern *is* the diagnostic distinction, full distribution-matching would fight real biology; both tails are trimmed instead, with cutoffs set by visual calibration rather than blind percentiles.

**TSS site / scanner** ([doc](docs/image_branch_tss_site_confound_check.md)) — the single site `TCGA-EM-*` accounts for **37/69 fvPTC cases and 0/65 cvPTC cases**. Any tile from an EM slide is deterministically fvPTC regardless of content. There is no EM-sourced cvPTC to balance against, so **no split strategy fixes this** — the split is stratified by site anyway (free, strictly better), and an EM-vs-non-EM subgroup evaluation is now a **required** post-training step, reported both tile-pooled and case-level.

**Thin-line artifacts as a second site proxy** — EM tiles are flagged for thin dark lines at 0.28% vs 9.07% elsewhere, a **~32× gap**. Line presence is itself an independently-learnable proxy for class. Two filter rules were also caught wrong by visual inspection: a hue-based ink detector false-positived on 35% of tiles (dense hematoxylin resembles ink in hue space) and a "blur" detector was actually measuring low texture.

A **fourth bug** was found during scaling checks: full-res tissue re-verification re-fit an Otsu threshold *per tile* instead of reusing the slide-level one, silently discarding real tissue on densely-stained slides. One case went from 1,363 → **6,206** tiles after the fix (4.55×).

### Split

`image_stage3_build_split.py` → `Dataset/Datasets/image_branch_case_split.csv`. Split is **case-level, before tiling** — tile-level splitting would leak patient identity across train/test.

Train 107 (53 cv / 54 fv), test 27 (12 cv / 15 fv). The 27 test cases are exactly the `PairedDemoSet` overlap, which landed at 20.1% of cases on its own — so **the test set is determined by which patients happened to have both records**, with no slack left to correct site/class skew.

---

## Repository layout

Only code, docs and small metadata are tracked. Bulk data, tiles, tensors and checkpoints are gitignored.

```
├── code/
│   ├── config.py                 loads paths from .env (not committed)
│   ├── preprocessing/            ThyroidDataset.ipynb — builds training tensors
│   ├── thyroid/
│   │   ├── FinalModel_modified.ipynb    current methylation notebook
│   │   ├── Explainability.ipynb         SHAP — ⚠️ targets the OLD architecture
│   │   ├── MethylationRemediation.ipynb / MethylationResultsVisualization.ipynb
│   │   └── *_result.json                MC Dropout + PCA sweep results
│   └── scripts/
│       ├── step1–step4      diagnosis of the collapse bug, and the fix
│       ├── step5–step15     seeded harness, fine-tuning, lock-in, calibration, sweeps
│       ├── image_stage1–3   POC → filtering → full tiling run
│       ├── download_*.py    TCGA-THCA slides (IDC), IBIA benign
│       └── rebuild_final_data.py  rebuilds tensors with label corrections applied
├── Dataset/
│   ├── Datasets/            cohort tables, case lists, split CSV, IBIA metadata
│   ├── TCGA_Tiles/_manifests/   per-case tiling manifests (resumability state)
│   ├── conda/spec-file.txt  conda environment spec (linux-64)
│   └── tcga_thca_download_subset.csv   the 134-case download plan
└── docs/                    handovers, confound checks, label log, follow-ups
```

---

## Setup

```bash
conda create --name capstone78 --file Dataset/conda/spec-file.txt
conda activate capstone78
```

Then create `code/.env` — `config.py` reads these and nothing resolves without them:

```dotenv
CODE_PATH=/abs/path/to/code
GDC_PATH=/abs/path/to/TCGA/gdc-downloads
RAW_THYROID_PATH=/abs/path/to/rawdata
THYROID_PATH=/abs/path/to/Dataset/Datasets
MODEL_PATH=/abs/path/to/Dataset/Models
RESULTS_PATH=/abs/path/to/Dataset/Results
TENSORBOARD_PATH=/abs/path/to/Dataset/Tensorboard
```

The image-branch scripts additionally use hardcoded `SLIDES_ROOT` / `OUT_ROOT` constants at the top of each file — set them for your machine.

### Running

Methylation scripts run standalone from the repo root:

```bash
PYTHONPATH=code python code/scripts/step3_fixed_methylation_model.py   # the fix, end to end
PYTHONPATH=code python code/scripts/step11_final_lockin_and_test_eval.py
PYTHONPATH=code python code/scripts/step14_mc_dropout_true_holdout.py
```

Image branch, in order — smoke-test on a subset before the full run:

```bash
python code/scripts/image_stage3_smoketest.py       # small isolated subset
python code/scripts/image_stage3_build_split.py     # split BEFORE tiling
python code/scripts/image_stage3_full_run.py        # 20 workers, resumable
```

The full run is **resumable and fault-isolated**: a per-case manifest is written to `Dataset/TCGA_Tiles/_manifests/` only after that case's tiles are fully saved, so an interrupted case is reprocessed cleanly rather than left half-written. Per-case exceptions are caught and logged instead of propagating — an earlier crash took down all 20 workers from a single `wsidicom` read error and sat idle for ~13–22 hours before anyone noticed.

---

## Working conventions

These come from repeated experience on this project, not from style preference:

- **Verify against real files and output before reporting anything done.** Trusting exit codes and summaries here has repeatedly hidden silent errors — a folder move landing one level too deep was caught only by a file-count mismatch, not by a non-zero exit.
- **Flag ambiguity instead of guessing or silently adapting.** Unclear scope, naming or hardcoded paths get asked about.
- **Never silently pick a side on a genuine label ambiguity.** `TCGA-FY-A3R9` was dropped, not resolved by preferring one source.
- **Negative results stay on record**, so nobody re-runs the same comparison later expecting a different answer.

---

## Known gaps

- SHAP has no trustworthy result and needs retargeting at the locked-in pipeline
- `TCGA-FY-A4B0` relabel still unapplied to training tensors
- Only 27 of 92 `PairedDemoSet` patients have a downloaded slide ([`docs/follow_up_items.txt`](docs/follow_up_items.txt))
- Not attempted: deeper backbone fine-tuning — deprioritized given the last-block collapse
- CpG probe annotation, results table, Docker/ONNX packaging

---

## Attribution & license

The pan-cancer methylation backbone and the original preprocessing approach derive from prior published work — see [`docs/Readme.txt`](docs/Readme.txt) for the upstream reproduction instructions and `Dataset/Datasets/GDC_samples.csv` for pan-cancer pretraining data provenance. The 121GB raw pan-cancer download itself was not retained; the trained checkpoints are.

Licensed under **GPL-3.0** — see [`docs/LICENSE`](docs/LICENSE).
