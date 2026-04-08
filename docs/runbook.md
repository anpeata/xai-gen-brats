# Execution Runbook

This runbook helps generate real evidence quickly and consistently.

## Step 1: Set up environment

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Confirm GPU and package versions.

## Step 2: Prepare data

1. Download BraTS 2023.
2. Place cases under `data/processed/BraTS2023/`.
3. Verify each case has T1, T1ce, T2, FLAIR, and segmentation mask.

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
