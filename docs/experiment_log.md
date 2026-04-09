# Experiment Log

Use this file as the primary source of truth for all executed runs.

## Environment

- Date: 2026-04-09
- Machine/GPU: CPU-only smoke run
- Python version: 3.12.8
- Torch version: 2.9.0
- MONAI version: 1.5.2
- Commit hash: 0cf87bd
- Dataset source/version: BraTS 2023 GLI, prepared under `data/processed/BraTS2023`

## Run Table

| Run ID | Objective | Command | Train Cases | Val Cases | Key Hyperparameters | Best Epoch | Dice Mean | HD95 Mean | ECE | Runtime | Output Paths |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| run-001 | CPU smoke baseline | `python scripts/train_segmentation.py ... --case-limit 16 --max-train-batches 5 --max-val-batches 2` | 12 | 4 | UNet, spatial_size=96, num_samples=1, epochs=1 | 1 | 0.0891 | 100.7852 | 0.0688 | ~1-2 min | `checkpoints/best_model_cpu_smoke_v2.pt`, `results/metrics/baseline_metrics_smoke_v2.json` |
| run-002 | XAI (Grad-CAM, single case) | `python scripts/run_xai.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir data/processed/BraTS2023/BraTS-GLI-00000-000 --device cpu --out-dir results/xai/smoke_v2 --skip-shap` | - | - | CPU, single case, divisible padding enabled | - | - | - | - | ~30-60 s | `results/xai/smoke_v2/gradcam_overlay.png` |
| run-003 | XAI (SHAP modality attribution, single case) | `python scripts/run_xai.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir data/processed/BraTS2023/BraTS-GLI-00000-000 --device cpu --out-dir results/xai/smoke_v2 --shap-nsamples 20` | - | - | CPU, SHAP `nsamples=20` | - | - | - | - | ~20-40 s | `results/xai/smoke_v2/shap_modalities.png` |
| run-004 | MC Dropout uncertainty (single case) | `python scripts/uncertainty.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir data/processed/BraTS2023/BraTS-GLI-00000-000 --device cpu --passes 10 --out results/uncertainty/uncertainty_map_smoke_v2.png` | - | - | CPU, passes=10 | - | - | - | - | ~30-90 s | `results/uncertainty/uncertainty_map_smoke_v2.png` |
| run-005 | XAI (Grad-CAM + SHAP, two extra cases) | `python scripts/run_xai.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir <CASE_DIR> --device cpu --out-dir results/xai/smoke_v2/<CASE_ID> --shap-nsamples 20` | - | - | CPU, SHAP `nsamples=20`, two cases | - | - | - | - | ~40-60 s total | `results/xai/smoke_v2/BraTS-GLI-00001-000/*`, `results/xai/smoke_v2/BraTS-GLI-00001-001/*` |
| run-006 | MC Dropout uncertainty (two extra cases) | `python scripts/uncertainty.py --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --case-dir <CASE_DIR> --device cpu --passes 10 --out results/uncertainty/smoke_v2/uncertainty_map_<CASE_ID>.png` | - | - | CPU, passes=10, two cases | - | - | - | - | ~60-180 s total | `results/uncertainty/smoke_v2/uncertainty_map_BraTS-GLI-00001-000.png`, `results/uncertainty/smoke_v2/uncertainty_map_BraTS-GLI-00001-001.png` |

## Run Notes

### run-001
- Intended change: Validate full training/evaluation loop on CPU with bounded batches.
- Result summary: Training and evaluation completed successfully with smoke metrics saved.
- Failure modes: CPU runtime remains high for full-scale training; smoke metrics are low by design due to constrained run size.
- Next action: Increase case and batch limits gradually or move to GPU for full baseline reporting.

### run-002
- Intended change: Generate Grad-CAM overlay from the smoke-v2 checkpoint.
- Result summary: Grad-CAM output generated after input padding fix.
- Failure modes: Initial run failed because `xai/modality_shap.py` had an indentation error, and UNet inference failed on non-divisible volume size (`RuntimeError: Sizes of tensors must match ...`).
- Fixes applied: corrected indentation in `xai/modality_shap.py`; added `DivisiblePadd(k=16)` in `scripts/run_xai.py` input transform.

### run-003
- Intended change: Produce SHAP modality attribution with CPU-feasible settings.
- Result summary: SHAP figure generated successfully with reduced sampling.
- Failure modes: default SHAP sampling can be slow on CPU.
- Fixes applied: added CLI control `--shap-nsamples`; set `--shap-nsamples 20` for smoke execution.

### run-004
- Intended change: Produce MC Dropout uncertainty map on the same case/checkpoint.
- Result summary: Uncertainty map saved successfully.
- Failure modes: same potential tensor shape mismatch risk as XAI path on full-volume inference.
- Fixes applied: added `DivisiblePadd(k=16)` in `scripts/uncertainty.py` input transform.

### run-005
- Intended change: Extend XAI qualitative evidence to three total cases.
- Result summary: Grad-CAM and SHAP outputs generated for `BraTS-GLI-00001-000` and `BraTS-GLI-00001-001`.
- Failure modes: none observed in this run after prior XAI fixes.
- Fixes applied: not required.

### run-006
- Intended change: Extend uncertainty qualitative evidence to three total cases.
- Result summary: additional uncertainty maps saved for `BraTS-GLI-00001-000` and `BraTS-GLI-00001-001`.
- Failure modes: none observed in this run after prior uncertainty fixes.
- Fixes applied: not required.
