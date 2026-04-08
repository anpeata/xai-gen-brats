# Research Summary

## Project
Explainable & Generative AI for Brain Tumor MRI Analysis (BraTS 2023)

## Motivation
This project addresses a practical clinical-AI gap: achieving strong segmentation performance while keeping model behavior transparent and uncertainty-aware. In oncology settings, trust and interpretability are essential for decision support.

## What Was Implemented
1. Multi-modal MRI segmentation using MONAI + PyTorch (T1, T1ce, T2, FLAIR).
2. Baseline 3-class tumor segmentation model (ET, TC, WT-focused analysis).
3. Explainability modules:
   - Grad-CAM for spatial attention visualization.
   - SHAP-inspired modality attribution for MRI channel relevance.
4. Uncertainty estimation:
   - Monte Carlo Dropout to produce voxel-wise predictive variance maps.
5. Generative extension:
   - Lightweight VAE for synthetic MRI slice generation and data augmentation prototyping.

## Evaluation Protocol
- Dice score (segmentation overlap quality)
- HD95 (boundary accuracy)
- ECE (probability calibration)

## Research Value
- Directly aligned with medical image analysis and oncology-focused PhD tracks.
- Demonstrates practical understanding of explainability and uncertainty in high-stakes AI.
- Includes generative modeling extension for scarce/imbalanced tumor subtypes.

## Next Experiments
1. Compare U-Net vs transformer-based lightweight encoder.
2. Quantify performance gain from synthetic augmentation.
3. Add cross-institution validation split to test generalization.
4. Expand uncertainty analysis to class-wise calibration curves.

## Personal Learning Outcome
This project strengthened end-to-end research engineering skills in biomedical AI: data handling, model development, explainability, uncertainty estimation, and scientific reporting.
