# Subtypes Detection of Papillary Thyroid Cancer (PTC) using Multimodal Deep Learning

A multimodal deep learning pipeline for classifying subtypes of Papillary Thyroid Cancer (PTC) by combining **DNA methylation data** (via a 1D-CNN) with **histopathology images** (via EfficientNetB0), trained and evaluated on the **TCGA-THCA** dataset. Model predictions are made interpretable using **SHAP** (for methylation features) and **Grad-CAM** (for image features).

## Overview

Papillary Thyroid Cancer has multiple histological subtypes that can be difficult to distinguish using a single data modality. This project explores whether fusing molecular (epigenetic) and visual (histopathological) information improves subtype classification accuracy over unimodal approaches, while keeping the model's decisions explainable to support clinical trust.

**Key components:**
- **1D-CNN branch** — learns patterns from DNA methylation beta-value profiles (TCGA-THCA)
- **EfficientNetB0 branch** — extracts features from whole-slide/histopathology image patches
- **Fusion layer** — combines both modalities for final subtype prediction
- **Explainability** — SHAP values for methylation feature importance, Grad-CAM heatmaps for image regions driving predictions

## Motivation

Traditional PTC subtyping relies heavily on pathologist interpretation of tissue morphology, which can be subjective and subtype-ambiguous in borderline cases. DNA methylation patterns carry complementary epigenetic signal that doesn't depend on visual morphology. Combining both modalities aims to:
- Improve classification robustness on ambiguous cases
- Provide two independent, cross-checkable lines of evidence
- Offer interpretable outputs rather than a black-box prediction

## Dataset

- **Source:** [The Cancer Genome Atlas — Thyroid Carcinoma (TCGA-THCA)](https://portal.gdc.cancer.gov/projects/TCGA-THCA)
- **Modalities used:**
  - Illumina DNA methylation beta values (450K/EPIC array)
  - Diagnostic histopathology whole-slide images (patched/tiled for input)
- **Labels:** PTC subtypes (e.g., classical, follicular-variant, tall-cell, etc., depending on annotation availability)

> Note: TCGA data must be downloaded separately via the GDC Data Portal in accordance with its data use policies. Raw data is not included in this repository.

## Architecture

```
                ┌─────────────────────┐
 Methylation ──▶│      1D-CNN          │──┐
   (beta        │  (Conv1D blocks +    │  │
   values)      │   pooling + dense)   │  │
                └─────────────────────┘  │
                                          ├──▶ Fusion (concat/dense) ──▶ Softmax ──▶ Subtype
                ┌─────────────────────┐  │
 Histopathology│    EfficientNetB0     │──┘
   image tiles ▶│  (ImageNet pretrained,│
                │   fine-tuned)         │
                └─────────────────────┘
```

- **1D-CNN branch:** stacked 1D convolutional layers over methylation feature vectors, with batch normalization, ReLU activations, and max-pooling, followed by dense layers.
- **Image branch:** EfficientNetB0 backbone (pretrained on ImageNet) fine-tuned on histopathology patches, with a global pooling head.
- **Fusion:** learned embeddings from both branches are concatenated and passed through fully connected layers to the final classification output.

## Explainability

- **SHAP** is used on the methylation branch to identify which CpG sites / methylation features contribute most to a given prediction.
- **Grad-CAM** is used on the EfficientNetB0 branch to visualize which regions of a histopathology image the model attended to when making its prediction.

This dual explainability approach lets predictions be validated against known biological/morphological markers rather than trusted blindly.

## Repository Structure

```
Subtypes-Detection-of-PTC-1-D-CNN/
├── data/               # Data loading/preprocessing scripts (raw data not included)
├── models/             # 1D-CNN, EfficientNetB0, and fusion model definitions
├── notebooks/          # Exploratory analysis, training, and evaluation notebooks
├── explainability/      # SHAP and Grad-CAM analysis scripts
├── utils/               # Helper functions (metrics, preprocessing, visualization)
├── requirements.txt      # Python dependencies
└── README.md
```



## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow / Keras (or PyTorch, depending on implementation)
- Access to TCGA-THCA methylation and histopathology data via [GDC Data Portal](https://portal.gdc.cancer.gov/)

### Installation

```bash
git clone https://github.com/asmitdesai/Subtypes-Detection-of-PTC-1-D-CNN.git
cd Subtypes-Detection-of-PTC-1-D-CNN
pip install -r requirements.txt
```

### Usage

```bash
# Preprocess raw TCGA data into model-ready format
python data/preprocess.py

# Train the multimodal model
python models/train.py

# Generate SHAP / Grad-CAM explanations for a trained model
python explainability/explain.py
```

> Update these commands to match your actual script names and entry points.

## Results

| Metric | Value |
|---|---|
| Accuracy | TBD |
| AUC | TBD |
| F1-score | TBD |



## Future Work

- Expand to additional PTC subtypes with more balanced class representation
- Explore attention-based fusion instead of simple concatenation
- Validate on an independent thyroid cancer cohort outside TCGA



## License
MIT
