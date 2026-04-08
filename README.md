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

## Research Comparison Angle

In your reports or interviews, compare:

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

## Future Work

- Replace VAE with diffusion-based synthesis for higher realism.
- Add cross-site external validation.
- Incorporate calibration plots and decision-threshold analysis.
- Extend to transformer segmentation backbones.