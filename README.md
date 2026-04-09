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

- Kaggle and Hugging Face distributions are both acceptable.
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
XAI-Gen-BraTS/
├── data/
│   └── README.md
├── models/
│   ├── __init__.py
│   └── segmentation.py
├── xai/
│   ├── __init__.py
│   ├── gradcam.py
│   └── modality_shap.py
├── generation/
│   ├── __init__.py
│   └── vae.py
├── scripts/
│   ├── dataset.py
│   ├── train_segmentation.py
│   ├── evaluate.py
│   ├── run_xai.py
│   ├── uncertainty.py
│   ├── train_vae.py
│   └── generate_samples.py
├── notebooks/
│   └── project_walkthrough.ipynb
├── docs/
│   ├── research_summary.md
│   └── research_summary.pdf
├── assets/
│   ├── example_prediction.svg
│   ├── gradcam_overlay.svg
│   └── uncertainty_map.svg
├── results/
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
python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --model unet --epochs 50
```

Evaluate metrics (Dice, HD95, ECE):

```bash
python scripts/evaluate.py --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model.pt
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
| Baseline UNet (CPU smoke) | 0.0891 | - | - | - | 100.7852 | 0.0688 | case-limit=16, max-train-batches=5, max-val-batches=2 |
| Baseline UNet (CPU medium) | 0.0525 | - | - | - | 94.1585 | 0.5455 | case-limit=32, epochs=2, max-train-batches=20, max-val-batches=6 |
| Baseline UNet (CPU long v1) | 0.1240 | - | - | - | 91.6189 | 0.3570 | case-limit=64, epochs=5, max-train-batches=40, max-val-batches=10 |
| Baseline + Uncertainty Analysis | 0.0891 | - | - | - | 100.7852 | 0.0688 | uncertainty maps generated for 3 cases under `results/uncertainty/smoke_v2/` |
| Baseline + Synthetic Augmentation |  |  |  |  |  |  | VAE trained on CPU (2 epochs), synthetic panels generated in `results/generated_smoke_v2/` |

### Qualitative Evidence Checklist

- [ ] Segmentation overlay for at least 3 representative cases.
- [x] Grad-CAM figure for at least 3 representative cases.
- [x] Uncertainty map for at least 3 representative cases.
- [x] Synthetic sample panel generated from VAE.

### Key Insights (Replace With Project Findings)

1. Which tumor sub-region achieves highest and lowest Dice, and why.
2. Whether Grad-CAM aligns with lesion regions or reveals spurious focus.
3. Whether uncertainty peaks correlate with boundary errors.
4. Whether synthetic augmentation improves minority-region performance.

Current qualitative artifact paths (smoke-v2):

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

- [ ] Dataset source and version are documented.
- [ ] Train/validation split details are documented.
- [ ] Environment versions are logged.
- [ ] Random seeds are fixed and recorded.
- [ ] Commands for each run are logged.
- [ ] Metrics are saved as machine-readable files.
- [ ] Figures in README point to generated files in `results/`.
- [ ] Commit hash used for final results is recorded.

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