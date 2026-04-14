# Insights Report

## Experimental Setup

- Data split: Deterministic case-level split (`val_ratio=0.2`) on `data/processed/BraTS2023` using sorted case order (`split_seed=null` / no shuffle).
- Preprocessing: Load four modalities, channel-first, intensity normalization (`nonzero=True`, `channel_wise=True`), training random crop (`96^3`, smoke run with `num_samples=1`).
- Model: MONAI 3D UNet (`in_channels=4`, `out_channels=4`) trained for CPU smoke validation.
- Optimizer and schedule: Adam (`lr=1e-3`), no scheduler in smoke run.
- Evaluation metrics: Dice mean, HD95 mean, ECE.

## Quantitative Results

| Setting | Dice Mean | Dice ET | Dice TC | Dice WT | HD95 Mean | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline segmentation | 0.0891 | 0.0287 | 0.0155 | 0.0817 | 100.7852 | 0.0566 |
| Baseline segmentation (CPU medium) | 0.0525 | 0.0158 | 0.1165 | 0.0250 | 94.1585 | 0.5455 |
| SegResNet (CPU medium quick-eval) | 0.0009 | 0.0000 | 0.0000 | 0.0028 | 175.0522 | 0.3150 |
| Baseline segmentation (CPU long v1) | 0.1240 | 0.0032 | 0.3188 | 0.0498 | 91.6189 | 0.3570 |
| Baseline + uncertainty analysis | 0.0891 | 0.0287 | 0.0155 | 0.0817 | 100.7852 | 0.0566 |
| Baseline + synthetic augmentation |  |  |  |  |  |  |

## Qualitative Findings

- Case 1 (`BraTS-GLI-00000-000`): segmentation overlay, Grad-CAM, and uncertainty maps were generated successfully from CPU checkpoints.
- Case 2 (`BraTS-GLI-00001-000`): segmentation overlay, Grad-CAM, SHAP, and uncertainty artifacts generated with CPU-bounded settings.
- Case 3 (`BraTS-GLI-00001-001`): segmentation overlay, Grad-CAM, SHAP, and uncertainty artifacts generated with CPU-bounded settings.

## Explainability Findings

- Grad-CAM: generated for 3 cases under `results/xai/smoke_v2/` with stable execution on padded full volumes.
- SHAP modality attribution: generated for 3 cases with CPU-bounded `nsamples=20`.

## Uncertainty Findings

- Boundary uncertainty: generated for 3 cases under `results/uncertainty/smoke_v2/` using MC Dropout (`passes=10`), suitable for visual boundary-risk inspection.
- Calibration behavior: global ECE in smoke setting is 0.0688; larger runs are needed for stable calibration claims.

## Synthetic Data Findings

- Sample realism: 8 synthetic panels were generated from the 2-epoch CPU VAE checkpoint (`results/generated_smoke_v2/`), with coherent multi-modal structure and expected blur for early-stage training.
- Performance impact: not yet measured; synthetic-to-real augmentation experiment is pending because current generated outputs are unlabeled 2D panels.

## Conclusions

- Best current setting: CPU long v1 baseline (`epochs=5`, `case-limit=64`) currently provides the strongest segmentation metrics in this repository.
- Main failure modes: dependency/import gaps, class-index mismatch in early metric code, and full-volume shape mismatch before divisible padding.
- Next three experiments: (1) run longer CPU baselines (10+ epochs) with fixed random seeds; (2) run multi-seed XAI/uncertainty analysis over larger case subsets; (3) quantify synthetic augmentation impact against a no-augmentation control.

## GPU Scale-Up Notes

- Estimated acceleration for this pipeline with a modern single GPU is approximately 6x-20x for segmentation and 8x-25x for VAE training versus current CPU runs.
- This enables longer and more statistically stable studies in practical wall-clock time: deeper training schedules, ablations, and case-wise qualitative reviews.
- Recommended first GPU campaign: full-train baseline, followed by uncertainty calibration and synthetic augmentation A/B comparison under matched splits.

No GPU runs were executed in this project environment so far; all reported metrics and artifacts were generated on CPU only.
