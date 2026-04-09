# Insights Report

## Experimental Setup

- Data split: Random case-level split (`val_ratio=0.2`) on `data/processed/BraTS2023`.
- Preprocessing: Load four modalities, channel-first, intensity normalization (`nonzero=True`, `channel_wise=True`), training random crop (`96^3`, smoke run with `num_samples=1`).
- Model: MONAI 3D UNet (`in_channels=4`, `out_channels=4`) trained for CPU smoke validation.
- Optimizer and schedule: Adam (`lr=1e-3`), no scheduler in smoke run.
- Evaluation metrics: Dice mean, HD95 mean, ECE.

## Quantitative Results

| Setting | Dice Mean | Dice ET | Dice TC | Dice WT | HD95 Mean | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline segmentation | 0.0891 | - | - | - | 100.7852 | 0.0688 |
| Baseline segmentation (CPU medium) | 0.0525 | - | - | - | 94.1585 | 0.5455 |
| Baseline + uncertainty analysis | 0.0891 | - | - | - | 100.7852 | 0.0688 |
| Baseline + synthetic augmentation |  |  |  |  |  |  |

## Qualitative Findings

- Case 1 (`BraTS-GLI-00000-000`): Grad-CAM and uncertainty maps were generated successfully from the smoke-v2 checkpoint.
- Case 2 (`BraTS-GLI-00001-000`): Grad-CAM, SHAP, and uncertainty artifacts generated with CPU-bounded settings.
- Case 3 (`BraTS-GLI-00001-001`): Grad-CAM, SHAP, and uncertainty artifacts generated with CPU-bounded settings.

## Explainability Findings

- Grad-CAM: generated for 3 cases under `results/xai/smoke_v2/` with stable execution on padded full volumes.
- SHAP modality attribution: generated for 3 cases with CPU-bounded `nsamples=20`.

## Uncertainty Findings

- Boundary uncertainty: generated for 3 cases under `results/uncertainty/smoke_v2/` using MC Dropout (`passes=10`), suitable for visual boundary-risk inspection.
- Calibration behavior: global ECE in smoke setting is 0.0688; larger runs are needed for stable calibration claims.

## Synthetic Data Findings

- Sample realism: 8 synthetic panels were generated from the 2-epoch CPU VAE checkpoint (`results/generated_smoke_v2/`), with coherent multi-modal structure and expected blur for early-stage training.
- Performance impact: not yet measured; synthetic-to-real augmentation experiment is pending.

## Conclusions

- Best current setting: CPU smoke-v2 baseline provides validated execution path for training, evaluation, XAI, and uncertainty artifacts.
- Main failure modes: dependency/import gaps, class-index mismatch in early metric code, and full-volume shape mismatch before divisible padding.
- Next three experiments: (1) run 20+ epoch GPU segmentation baseline with full-validation metrics; (2) run multi-seed XAI/uncertainty analysis over larger case subsets; (3) quantify synthetic augmentation impact against a no-augmentation control.

## GPU Scale-Up Notes

- Estimated acceleration for this pipeline with a modern single GPU is approximately 6x-20x for segmentation and 8x-25x for VAE training versus current CPU runs.
- This enables longer and more statistically stable studies in practical wall-clock time: deeper training schedules, ablations, and case-wise qualitative reviews.
- Recommended first GPU campaign: full-train baseline, followed by uncertainty calibration and synthetic augmentation A/B comparison under matched splits.
