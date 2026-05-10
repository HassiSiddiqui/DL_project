# Semantic-DTFD-MIL: Foundation Model Extension for Multiple Instance Learning

> **Course:** Deep Learning — FAST-NUCES  
> **Team:** Abdul Haseeb Siddiqui (23I-2654) · Issa Sultan (23I-2596) · Ahmed Sajid (23I-2598)  
> **Instructors:** Dr. Qurat Ul Ain · Dr. Zohair Ahmed · Mr. Ubaid Ur Rehman

## Overview

This project extends the **DTFD-MIL** (CVPR 2022) framework for Whole-Slide Image (WSI) classification by replacing the original ResNet-50 backbone and random pseudo-bag splitting with:

1. **PLIP** (Pathology Language-Image Pre-training) — 512-dim medical ViT embeddings
2. **SAM** (Segment Anything Model) — structural feature extraction for spatial-aware K-Means pseudo-bag clustering
3. **Double-Tier Feature Distillation** — the original AFS attention distillation pipeline

### Key Results

| Model | AUC | Accuracy | F1-Score |
|-------|-----|----------|----------|
| Baseline (ResNet-50 + Random Split) | 0.552 | 0.450 | 0.420 |
| Ablation (PLIP + Random Split) | 0.641 | 0.612 | 0.540 |
| **Proposed (PLIP + SAM K-Means)** | **0.800** | **0.800** | **0.800** |

> Improvement over baseline: **+0.248 AUC** on the PCam subset (40 training bags, 10 test bags).

---

## Repository Structure

```
project-root/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml                # All hyperparameters and paths
├── train.py                   # Training script (reads config.yaml)
├── inference.py               # Inference script (loads checkpoint)
│
├── data/
│   └── sample_data.csv        # 8 sample bags with feature statistics
│
├── notebooks/
│   └── 01_inference_demo.ipynb  # Full inference demo with visualizations
│
├── src/
│   ├── model.py               # SemanticDTFDMIL model wrapper
│   ├── dataset.py             # PCamBagDataset loader
│   └── utils.py               # Evaluation, checkpointing, logging
│
├── results/
│   ├── baseline_metrics.json  # Baseline model performance
│   ├── improved_metrics.json  # Proposed model performance
│   └── training_log.csv       # Epoch-by-epoch training log
│
├── checkpoints/
│   └── best_model.pth         # Trained model weights (263K params)
│
├── Model/                     # Core DTFD-MIL architecture
│   ├── Attention.py           # Attention_Gated, Attention_with_Classifier
│   └── network.py             # DimReduction, Classifier_1fc
│
└── assignment 3/              # Full experimental pipeline
    ├── Main_DTFD_MIL.py       # Original training loop
    ├── create_pcam_bags.py    # PLIP + SAM feature extraction
    ├── run_inference.py       # Standalone inference script
    ├── Dataset/               # Pre-extracted PKL bag files
    └── debug_log/             # TensorBoard logs & checkpoints
```

---

## Quick Start

### 1. Setup
```bash
git clone https://github.com/HassiSiddiqui/DL_project.git
cd DL_project
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Run Inference
```bash
python inference.py
```
Output:
```
Test bags   : 10  |  Feature dim: 512
AUC         : 0.8000
Accuracy    : 0.8000  (8/10 bags)
F1-Score    : 0.8000
```

### 3. Train from Scratch
```bash
python train.py --epochs 50 --lr 1e-4
```
Training reads from `config.yaml` and saves:
- Best checkpoint → `checkpoints/best_model.pth`
- Training log → `results/training_log.csv`
- Final metrics → `results/improved_metrics.json`

### 4. Run the Notebook
Open `notebooks/01_inference_demo.ipynb` in VS Code or Jupyter and **Run All Cells** to see:
- Per-bag predictions table
- ROC curve & confusion matrix
- Training loss/AUC curves
- Baseline vs Proposed comparison bar chart

---

## Architecture

```
Raw Patch (96×96×3)
       │
       ▼ PLIP ViT (vinid/plip, frozen)
   512-dim medical embeddings
       │
       ▼ SAM ViT (facebook/sam-vit-base, frozen)
   256-dim structural features → K-Means (K=5) → semantic pseudo-bags
       │
       ▼ DimReduction (Linear: 512 → 256)
   256-dim hidden state
       │
       ▼ Tier-1: Attention_Gated (per pseudo-bag)
   Weighted aggregation → 1 distilled representation per pseudo-bag
       │
       ▼ Tier-2: Attention_with_Classifier (across K pseudo-bags)
   Global attention → bag-level classification
       │
       ▼ Output: [P(Normal), P(Tumor)]
```

**Trainable parameters:** 263,942 (only the attention/classification heads)

---

## Model Components

| Component | Input → Output | Parameters | Role |
|-----------|---------------|------------|------|
| `DimReduction` | (N, 512) → (N, 256) | 131,072 | Feature projection |
| `Attention_Gated` | (N, 256) → (1, N) | 65,921 | Tier-1 local attention |
| `Attention_with_Classifier` | (K, 256) → (1, 2) | 66,435 | Tier-2 global classifier |
| `Classifier_1fc` | (1, 256) → (1, 2) | 514 | Tier-1 auxiliary head |

---

## Dataset

- **Source:** PatchCamelyon (PCam) — binary classification of breast cancer metastasis
- **Patches:** 1,000 images (96×96 pixels), organized into 50 bags of 20 patches each
- **Split:** 40 training bags / 10 test bags
- **Features:** Pre-extracted PLIP 512-dim embeddings stored as `.pkl` files

---

## Hardware

All experiments ran on local hardware:
- **CPU:** AMD Ryzen 5 7535HS
- **GPU:** AMD Radeon RX 6550M (8GB VRAM)
- **RAM:** 16 GB

SAM inference required sequential patch processing (1-at-a-time) due to ViT memory scaling.

---

## References

1. Zhang et al., "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning," CVPR 2022
2. Huang et al., "PLIP: A Visual-Language Foundation Model for Pathology," Nature Medicine 2023
3. Kirillov et al., "Segment Anything," ICCV 2023
