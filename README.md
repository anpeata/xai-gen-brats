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

- Official BraTS 2023 challenge distribution is on Synapse.
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

For deeper project narrative and roadmap (timeline, future work, scaling outlook), see:

- `docs/research_summary.md`