# Plant Disease Classification — Cross-Domain Benchmark and Ensemble Pipeline

This repository contains the complete consolidated research code for cross-domain plant disease classification, organized according to a 16-stage research pipeline. The objective is to close the cross-domain generalization gap between a laboratory-condition dataset (PlantVillage) and a field-condition dataset (PlantDoc).

**Final result:** 80.08% cross-domain accuracy on the PlantDoc held-out test set — an improvement of 47.45 percentage points over the VGG16 baseline (32.63%) — with an F1 score of 79.04%.

## Environment

The experiments were developed and executed on Kaggle Notebooks (free tier, GPU T4), with approximately 117 GPU hours of total training compute. The notebook is portable and can also be run on Google Colab or a local Jupyter environment. The PlantVillage and PlantDoc datasets are publicly available and are used here independently of the Kaggle platform.

- **Dataset paths.** The code internally expects a `/kaggle/input/...` directory layout. The portable setup cell recreates this layout via symlinks from a single `DATA_ROOT` variable, so no other path needs to be changed regardless of platform.
- **GPU.** A GPU is required only for retraining (~117 GPU hours for full training). With `SKIP_TRAINING = True` (default), all results are reproduced by loading cached checkpoints, and no GPU is required.
- **Checkpoint caching.** Later stages load model checkpoints (`.pth`) produced by earlier stages. The skip logic detects these cached checkpoints and bypasses re-training.

## How to Reproduce

1. Set the `DATA_ROOT` variable in the portable setup cell to the folder containing the PlantVillage dataset, the PlantDoc dataset, and the pre-computed checkpoints. Run that cell first.
2. Keep `SKIP_TRAINING = True` to reproduce all results from cached checkpoints (no GPU needed). Set it to `False` only if you intend to retrain.
3. Run the stages in order. Stage 13 produces the final fifteen-model weighted ensemble result.

For local Jupyter only (not needed on Colab or Kaggle), run once in a terminal so the symlink layout can be created:
```sudo mkdir -p /kaggle && sudo chown -R $USER /kaggle```

**Dependencies:** `torch`, `torchvision` (matching your CUDA or CPU build), `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pillow`.

## Pipeline Overview

The pipeline spans 16 stages across 6 categories:

| Category | Stages |
|---|---|
| Data Foundation | 1–3 |
| Initial Experiments | 4–6 |
| Mixed-Domain Training | 7–9 |
| Multi-Architecture Push | 10–12 |
| Final Ensemble | 13 |
| Analysis and Submission | 14–16 |

## Datasets

- **PlantVillage** (laboratory conditions) — Hughes & Salathé (2015)
- **PlantDoc** (field conditions) — Singh et al. (2020)

Both datasets are publicly available through their original publications.
