# XAI-BraTS: Explainable & Generative AI for Brain Tumor MRI Analysis

Research-grade project for brain tumor MRI segmentation and synthesis in medical imaging, oncology AI, explainability, and generative modeling.

## Why This Project Matters

This repository focuses on a high-impact clinical AI task: multi-modal brain tumor analysis on BraTS. It combines:

- Strong segmentation performance targets.
- Transparent decision support through explainability.
- Uncertainty-aware outputs for safer interpretation.
- Generative modeling to address rare tumor patterns and data scarcity.

## Key Features

- 🧠 Spatio-Temporal Segmentation: Multi-sequence MRI analysis with T1, T1ce, T2, FLAIR.
- 🎨 Generative Data Augmentation: VAE-based synthetic MRI generation scaffold for rare pattern support.
- 🔍 Explainable AI (XAI): Grad-CAM spatial explanations and modality-level SHAP attribution.
- 🎲 Uncertainty Quantification: Monte Carlo Dropout-based uncertainty maps.
- 📊 Evaluation: Dice, HD95 (Hausdorff 95), and ECE (Expected Calibration Error).

## Dataset

BraTS 2023 (public benchmark) is used as the core dataset.

- Official BraTS 2023 challenge distribution is on Synapse; Kaggle/Hugging Face mirrors and manually prepared case folders are also accepted if they match required file naming.
- Place case folders under `data/processed/BraTS2023/`.
- Each case folder should include:
	- `*_t1.nii.gz`
	- `*_t1ce.nii.gz`
	- `*_t2.nii.gz`
	- `*_flair.nii.gz`
	- `*_seg.nii.gz`

See `data/README.md` for details.

## Repository Structure

```text
xai-gen-brats/
├── checkpoints/
├── data/
│   └── README.md
├── docs/
│   ├── experiment_log.md
│   ├── insights_report.md
│   ├── research_summary.md
│   ├── research_summary.pdf
│   └── runbook.md
├── generation/
│   ├── __init__.py
│   └── vae.py
├── models/
│   ├── __init__.py
│   └── segmentation.py
├── notebooks/
│   └── project_walkthrough.ipynb
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── tables/
│   ├── uncertainty/
│   └── xai/
├── scripts/
│   ├── dataset.py
│   ├── download_brats.py
│   ├── evaluate.py
│   ├── generate_samples.py
│   ├── predict_overlay.py
│   ├── run_xai.py
│   ├── train_segmentation.py
│   ├── train_vae.py
│   └── uncertainty.py
├── assets/
│   ├── example_prediction.svg
│   ├── gradcam_overlay.svg
│   └── uncertainty_map.svg
├── xai/
│   ├── __init__.py
│   ├── gradcam.py
│   └── modality_shap.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Repository Policy: Maximum File Size

This repository enforces a strict maximum file size policy:

- No file larger than 5 MB is allowed in git history.

Enforcement mechanisms:

- Local pre-push hook at `.githooks/pre-push`.
- GitHub Actions workflow at `.github/workflows/file-size-policy.yml`.

Enable local hook policy after cloning:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_hooks.ps1
```

## Runbook and Reporting Workflow

For end-to-end execution and documentation workflow, use:

- `docs/runbook.md`
- `docs/experiment_log_template.md`
- `docs/insights_report_template.md`
- `results/README.md`

## Phase 1: Baseline Segmentation (Core)

Train baseline model:

```bash
python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --model unet --epochs 50 --quiet-warnings
```

Evaluate metrics (Dice, HD95, ECE):

```bash
python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model.pt --quiet-warnings
```

## Phase 2: Explainability + Uncertainty

Run Grad-CAM and modality-level SHAP for a case:

```bash
python scripts/run_xai.py --checkpoint checkpoints/best_model.pt --case-dir data/processed/BraTS2023/<CASE_ID>
```

CPU smoke shortcut (Grad-CAM only):

```bash
python scripts/run_xai.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir data/processed/BraTS2023/<CASE_ID> --device cpu --out-dir results/xai/smoke_v2 --skip-shap
```

CPU smoke SHAP (reduced cost):

```bash
python scripts/run_xai.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir data/processed/BraTS2023/<CASE_ID> --device cpu --out-dir results/xai/smoke_v2 --shap-nsamples 20
```

Run Monte Carlo Dropout uncertainty mapping:

```bash
python scripts/uncertainty.py --checkpoint checkpoints/best_model.pt --case-dir data/processed/BraTS2023/<CASE_ID> --passes 20
```

## Phase 3: Generative Extension

Train lightweight VAE on slice-wise multi-modal inputs:

```bash
python scripts/train_vae.py --data-dir data/processed/BraTS2023 --epochs 30
```

Generate synthetic MRI panels:

```bash
python scripts/generate_samples.py --checkpoint checkpoints/vae.pt --n 8
```

## Visual Evidence

- Segmentation example: `assets/example_prediction.svg`
- Grad-CAM example: `assets/gradcam_overlay.svg`
- Uncertainty map example: `assets/uncertainty_map.svg`
- Synthetic panels (CPU VAE run): `results/generated_smoke_v2/`

## Results (Fill With Real Runs)

Add final values from generated files under `results/metrics/`.

| Experiment Setting | Dice Mean | Dice ET | Dice TC | Dice WT | HD95 Mean | ECE | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline UNet (CPU smoke) | 0.0891 | 0.0287 | 0.0155 | 0.0817 | 100.7852 | 0.0566 | case-limit=16, max-train-batches=5, max-val-batches=2 |
| Baseline UNet (CPU medium) | 0.0525 | 0.0158 | 0.1165 | 0.0250 | 94.1585 | 0.5455 | case-limit=32, epochs=2, max-train-batches=20, max-val-batches=6 |
| SegResNet (CPU medium quick-eval) | 0.0009 | 0.0000 | 0.0000 | 0.0028 | 175.0522 | 0.3150 | checkpoint from bounded run, eval max-val-batches=2; quick screening only |
| Baseline UNet (CPU long v1) | 0.1240 | 0.0032 | 0.3188 | 0.0498 | 91.6189 | 0.3570 | case-limit=64, epochs=5, max-train-batches=40, max-val-batches=10 |
| Baseline + Uncertainty Analysis | 0.0891 | 0.0287 | 0.0155 | 0.0817 | 100.7852 | 0.0566 | uncertainty maps generated for 3 cases under `results/uncertainty/smoke_v2/` |
| Baseline + Synthetic Augmentation (label-preserving) | 0.0335 | 0.0147 | 0.0707 | 0.0150 | 121.3415 | 0.4941 | train-extra-dir=`data/processed/BraTS2023_SYN`, 16 synthetic labeled cases, same medium bounded run config |

### Qualitative Evidence Checklist

- [x] Segmentation overlay for at least 3 representative cases.
- [x] Grad-CAM figure for at least 3 representative cases.
- [x] Uncertainty map for at least 3 representative cases.
- [x] Synthetic sample panel generated from VAE.

### Key Insights (Replace With Project Findings)

1. Across current CPU runs, TC has the highest Dice and ET the lowest (for long v1: ET=0.0032, TC=0.3188, WT=0.0498), indicating core-region learning is stronger than enhancing-tumor delineation under bounded training.
2. Grad-CAM overlays were successfully generated for three representative cases and are qualitatively centered around lesion regions in those examples.
3. Uncertainty maps for the same three cases show elevated variance near tumor boundaries, consistent with expected boundary ambiguity.
4. Label-preserving synthetic augmentation (41 train cases total with 16 synthetic additions) under the bounded medium setup reduced Dice versus baseline medium (0.0335 vs 0.0525), suggesting the current synthetic generation recipe needs refinement.
5. Stronger 3-seed medium A/B (`seed=42,43,44`, 2 epochs, 20 train batches, 6 val batches) showed a mean Dice decrease (`delta_dice=-0.001367`) and HD95 worsening (`delta_hd95=+2.750`) for synthetic augmentation, while mean ECE slightly improved (`delta_ece=-0.005809`); see `results/tables/seed_ablation_medium_summary.csv`.

Current qualitative artifact paths (smoke-v2):

- `results/predictions/smoke_v2/overlay_BraTS-GLI-00000-000.png`
- `results/predictions/smoke_v2/overlay_BraTS-GLI-00001-000.png`
- `results/predictions/smoke_v2/overlay_BraTS-GLI-00001-001.png`
- `results/xai/smoke_v2/gradcam_overlay.png`
- `results/xai/smoke_v2/BraTS-GLI-00001-000/gradcam_overlay.png`
- `results/xai/smoke_v2/BraTS-GLI-00001-001/gradcam_overlay.png`
- `results/uncertainty/smoke_v2/uncertainty_map_BraTS-GLI-00000-000.png`
- `results/uncertainty/smoke_v2/uncertainty_map_BraTS-GLI-00001-000.png`
- `results/uncertainty/smoke_v2/uncertainty_map_BraTS-GLI-00001-001.png`

Synthetic sample paths (CPU VAE):

- `results/generated_smoke_v2/synthetic_case_00.png`
- `results/generated_smoke_v2/synthetic_case_01.png`
- `results/generated_smoke_v2/synthetic_case_02.png`
- `results/generated_smoke_v2/synthetic_case_03.png`

## Future GPU Scale-Up (Estimates)

These estimates are practical planning ranges for this codebase and current preprocessing choices.

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

## Reproducibility Checklist

- [x] Dataset source and version are documented.
- [x] Train/validation split details are documented.
- [x] Environment versions are logged.
- [x] Random seeds are fixed and recorded.
- [x] Commands for each run are logged.
- [x] Metrics are saved as machine-readable files.
- [x] Figures in README point to generated files in `results/`.
- [x] Commit hash used for final results is recorded.

## Research Comparison Angle

In reports or interviews, compare:

- Black-box segmentation baseline only.
- Interpretable segmentation with XAI maps.
- Segmentation with synthetic augmentation support.

Focus questions:

- Do explanations align with known tumor regions?
- Where is model confidence low, and why?
- Does synthetic data improve minority sub-region performance?

## Suggested 3-Week Timeline

- Week 1: Data pipeline + baseline segmentation.
- Week 2: Explainability and uncertainty analysis.
- Week 3: Generative extension + final reporting.

## Research Strengths

This project demonstrates:

- Hands-on biomedical imaging workflow (BraTS benchmark).
- Clinical trust-focused AI (explainability + uncertainty).
- Oncology relevance (brain tumor analysis).
- Generative modeling capability (VAE augmentation).
- Research communication readiness (figures, notebook, summary PDF).

## Bonus: Research Summary

One-page summary is available in:

- `docs/research_summary.md`
- `docs/research_summary.pdf`

Use these companion files for complete reporting:

- `docs/experiment_log_template.md`
- `docs/insights_report_template.md`

## Future Work

- Replace VAE with diffusion-based synthesis for higher realism.
- Add cross-site external validation.
- Incorporate calibration plots and decision-threshold analysis.
- Extend to transformer segmentation backbones.