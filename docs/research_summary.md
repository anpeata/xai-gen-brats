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

## Current Status
- Current phase: Phase 4 (Comparative validation and optimization).
- Repository and pipeline scaffolding are complete.
- Final claims should be updated only after full experiments are executed and logged.
- Use the templates in `docs/experiment_log_template.md` and `docs/insights_report_template.md` for evidence-backed reporting.

## Required Evidence Before Finalizing Results
1. Baseline metrics from `scripts/evaluate.py` saved to `results/metrics/`.
2. Qualitative figures from XAI and uncertainty scripts saved to `results/figures/`.
3. Ablation summary table comparing baseline vs augmentation settings.
4. Consolidated insights report based on quantitative and qualitative findings.

## Research Value
- Directly aligned with medical image analysis and oncology-focused research tracks.
- Demonstrates practical understanding of explainability and uncertainty in high-stakes AI.
- Includes generative modeling extension for scarce/imbalanced tumor subtypes.

## Next Experiments
1. Compare U-Net vs transformer-based lightweight encoder.
2. Quantify performance gain from synthetic augmentation.
3. Add cross-institution validation split to test generalization.
4. Expand uncertainty analysis to class-wise calibration curves.

## Future GPU Scale-Up (Estimates)

These estimates are practical planning ranges for this codebase and preprocessing choices.

- CPU baseline used here:
   - Segmentation medium run (2 epochs, bounded batches): about 2.5-3.5 minutes total.
   - VAE run (2 epochs over full case list): about 22-24 minutes total.
- Single modern GPU (for example RTX 4090 / A5000 / A100) expected speed-up:
   - Segmentation training throughput: typically ~6x to ~20x faster than CPU-only runs.
   - VAE training throughput: typically ~8x to ~25x faster than CPU-only runs.
- Expected runtime examples with same settings:
   - Segmentation medium run: roughly 20-40 seconds.
   - VAE 2-epoch run: roughly 1-3 minutes.
- What this enables for better insights:
   - Longer segmentation runs (20-100 epochs) and stronger convergence trends.
   - Multi-seed experiments for variance/confidence intervals.
   - More cases for XAI/uncertainty and richer failure-pattern analysis.
   - Hyperparameter sweeps (crop size, model choice, loss variants).

## Suggested 3-Week Timeline

- Week 1: Data pipeline + baseline segmentation.
- Week 2: Explainability and uncertainty analysis.
- Week 3: Generative extension + final reporting.

## Future Work

- Replace VAE with diffusion-based synthesis for higher realism.
- Add cross-site external validation.
- Incorporate calibration plots and decision-threshold analysis.
- Extend to transformer segmentation backbones.

## Personal Learning Outcome
This project strengthened end-to-end research engineering skills in biomedical AI: data handling, model development, explainability, uncertainty estimation, and scientific reporting.
