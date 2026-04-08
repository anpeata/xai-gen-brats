# Execution Runbook

This runbook helps generate real evidence quickly and consistently.

## Step 1: Set up environment

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Confirm GPU and package versions.

## Step 2: Prepare data

1. Download BraTS 2023 using one of the supported options below.
2. Use the dataset preparation script to normalize file layout.
3. Verify each prepared case has T1, T1ce, T2, FLAIR, and segmentation mask.

Option A: Kaggle API
- Configure Kaggle credentials (`~/.kaggle/kaggle.json` or environment variables).
- Download and prepare:
	- `python scripts/download_brats.py --source kaggle --kaggle-dataset <owner/dataset-slug> --extract-zips`
	- or `python scripts/download_brats.py --source kaggle --kaggle-competition <competition-slug> --extract-zips`

Option B: Hugging Face Hub
- Authenticate if required (`huggingface-cli login`).
- Download and prepare:
	- `python scripts/download_brats.py --source huggingface --hf-repo-id <dataset-repo-id> --extract-zips`

If you already downloaded data manually:
- Put raw files under `data/raw/BraTS2023/` and run:
	- `python scripts/download_brats.py --source none --extract-zips`

Recommended for Synapse zip downloads in this repository layout:
- Place archives under `data/raw/BraTS2023/archives/GLI`.
- Prepare GLI cases for segmentation with:
	- `python scripts/download_brats.py --source none --raw-dir data/raw/BraTS2023/archives --processed-dir data/processed/BraTS2023 --extract-zips --link-mode hardlink --include-case-prefix BraTS-GLI`

Optional challenge-specific preparation commands:
- MEN:
	- `python scripts/download_brats.py --source none --raw-dir data/raw/BraTS2023/archives --processed-dir data/processed/BraTS2023-MEN --extract-zips --link-mode hardlink --include-case-prefix BraTS-MEN`
- PED:
	- `python scripts/download_brats.py --source none --raw-dir data/raw/BraTS2023/archives --processed-dir data/processed/BraTS2023-PED --extract-zips --link-mode hardlink --include-case-prefix BraTS-PED`

Quick verification command:
- `python scripts/download_brats.py --source none --dry-run`

## Step 3: Baseline segmentation

Run:
- `python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --model unet --epochs 50`

Then evaluate:
- `python scripts/evaluate.py --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model.pt --out results/metrics/baseline_metrics.json`

## Step 4: XAI and uncertainty

Run XAI:
- `python scripts/run_xai.py --checkpoint checkpoints/best_model.pt --case-dir data/processed/BraTS2023/<CASE_ID> --out-dir results/figures`

Run uncertainty:
- `python scripts/uncertainty.py --checkpoint checkpoints/best_model.pt --case-dir data/processed/BraTS2023/<CASE_ID> --out results/figures/uncertainty_case_<CASE_ID>.png`

## Step 5: Generative extension

Train VAE:
- `python scripts/train_vae.py --data-dir data/processed/BraTS2023 --epochs 30 --out checkpoints/vae.pt`

Generate samples:
- `python scripts/generate_samples.py --checkpoint checkpoints/vae.pt --n 8 --out-dir results/figures`

## Step 6: Fill logs and reports

1. Copy `docs/experiment_log_template.md` to `docs/experiment_log.md` and fill all fields.
2. Copy `docs/insights_report_template.md` to `docs/insights_report.md` and add findings.
3. Update `README.md` Results section with final numbers and links to figures.

## Step 7: Final quality check

- Verify every metric cited in README has a corresponding file in `results/`.
- Verify every figure in README points to a generated output.
- Keep one clear ablation table and one concise conclusion section.
