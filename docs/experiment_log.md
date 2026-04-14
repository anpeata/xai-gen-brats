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
| run-007 | CPU medium baseline segmentation | `python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --device cpu --epochs 2 --num-workers 0 --case-limit 32 --spatial-size 96 --num-samples 1 --max-train-batches 20 --max-val-batches 6 --out checkpoints/best_model_cpu_mid.pt` | 25 | 7 | UNet, spatial_size=96, num_samples=1, epochs=2 | 2 | 0.0525 | 94.1585 | 0.5455 | ~2.5-3.5 min | `checkpoints/best_model_cpu_mid.pt`, `results/metrics/baseline_metrics_cpu_mid.json` |
| run-008 | VAE training (CPU medium) | `python scripts/train_vae.py --data-dir data/processed/BraTS2023 --device cpu --epochs 2 --batch-size 8 --out checkpoints/vae_cpu_mid.pt` | - | - | latent_dim=128, beta=1.0, batch=8 | 2 | - | - | - | ~22-24 min | `checkpoints/vae_cpu_mid.pt` |
| run-009 | Synthetic generation from VAE | `python scripts/generate_samples.py --checkpoint checkpoints/vae_cpu_mid.pt --n 8 --out-dir results/generated_smoke_v2 --device cpu` | - | - | n=8 | - | - | - | - | <1 min | `results/generated_smoke_v2/synthetic_case_*.png` |
| run-010 | CPU long baseline segmentation | `python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --device cpu --epochs 5 --num-workers 0 --case-limit 64 --spatial-size 96 --num-samples 1 --max-train-batches 40 --max-val-batches 10 --out checkpoints/best_model_cpu_long_v1.pt` | 51 | 13 | UNet, spatial_size=96, num_samples=1, epochs=5 | 5 | 0.1240 | 91.6189 | 0.3570 | ~12-15 min | `checkpoints/best_model_cpu_long_v1.pt`, `results/metrics/baseline_metrics_cpu_long_v1.json` |
| run-011 | Segmentation overlays (3 cases) | `python scripts/predict_overlay.py --checkpoint checkpoints/best_model_cpu_long_v1.pt --case-dir <CASE_DIR> --device cpu --out results/predictions/smoke_v2/overlay_<CASE_ID>.png` | - | - | CPU, 3 representative cases | - | - | - | - | ~30-90 s total | `results/predictions/smoke_v2/overlay_BraTS-GLI-00000-000.png`, `results/predictions/smoke_v2/overlay_BraTS-GLI-00001-000.png`, `results/predictions/smoke_v2/overlay_BraTS-GLI-00001-001.png` |
| run-012 | Re-eval smoke with class-wise metrics | `python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model_cpu_smoke_v2.pt --device cpu --out results/metrics/baseline_metrics_smoke_v2.json --num-workers 0 --case-limit 16 --max-val-batches 2 --spatial-size 96 --seed 42 --split-seed -1 --val-ratio 0.2` | - | 4 | deterministic split order, seed=42, split_seed=null | - | 0.0891 | 100.7852 | 0.0566 | ~2-4 min | `results/metrics/baseline_metrics_smoke_v2.json` |
| run-013 | Re-eval medium with class-wise metrics | `python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model_cpu_mid.pt --device cpu --out results/metrics/baseline_metrics_cpu_mid.json --num-workers 0 --case-limit 32 --max-val-batches 6 --spatial-size 96 --seed 42 --split-seed -1 --val-ratio 0.2` | - | 7 | deterministic split order, seed=42, split_seed=null | - | 0.0525 | 94.1585 | 0.5455 | ~5-8 min | `results/metrics/baseline_metrics_cpu_mid.json` |
| run-014 | Re-eval long v1 with class-wise metrics | `python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model_cpu_long_v1.pt --device cpu --out results/metrics/baseline_metrics_cpu_long_v1.json --num-workers 0 --case-limit 64 --max-val-batches 10 --spatial-size 96 --seed 42 --split-seed -1 --val-ratio 0.2` | - | 13 | deterministic split order, seed=42, split_seed=null | - | 0.1240 | 91.6189 | 0.3570 | ~10-15 min | `results/metrics/baseline_metrics_cpu_long_v1.json` |
| run-015 | SegResNet bounded training (CPU medium) | `python -m scripts.train_segmentation --data-dir data/processed/BraTS2023 --model segresnet --device cpu --epochs 2 --num-workers 0 --case-limit 32 --spatial-size 96 --num-samples 1 --max-train-batches 20 --max-val-batches 6 --seed 42 --split-seed -1 --out checkpoints/best_model_cpu_mid_segresnet.pt --quiet-warnings` | 25 | 7 | SegResNet, bounded train/val batches | 1* | - | - | - | ~8-12 min | `checkpoints/best_model_cpu_mid_segresnet.pt` |
| run-016 | SegResNet quick evaluation (CPU medium subset) | `python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model_cpu_mid_segresnet.pt --device cpu --out results/metrics/baseline_metrics_cpu_mid_segresnet_quick.json --num-workers 0 --case-limit 32 --max-val-batches 2 --spatial-size 96 --seed 42 --split-seed -1 --val-ratio 0.2 --quiet-warnings` | - | 7 | quick screen, eval max-val-batches=2 | - | 0.0009 | 175.0522 | 0.3150 | ~2-4 min | `results/metrics/baseline_metrics_cpu_mid_segresnet_quick.json` |
| run-017 | Generate label-preserving synthetic cases | `python -m scripts.generate_label_preserving_synthetic --source-dir data/processed/BraTS2023 --out-dir data/processed/BraTS2023_SYN --num-synthetic 16 --base-case-limit 32 --seed 42` | - | - | paired flips + intensity transforms, synthetic=16 | - | - | - | - | <1 min | `data/processed/BraTS2023_SYN/*` |
| run-018 | Baseline + synthetic augmentation training/eval (CPU medium) | `python -m scripts.train_segmentation --data-dir data/processed/BraTS2023 --train-extra-dir data/processed/BraTS2023_SYN --device cpu --epochs 2 --num-workers 0 --case-limit 32 --spatial-size 96 --num-samples 1 --max-train-batches 20 --max-val-batches 6 --seed 42 --split-seed -1 --out checkpoints/best_model_cpu_mid_plus_synth.pt --quiet-warnings` then `python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model_cpu_mid_plus_synth.pt --device cpu --out results/metrics/baseline_metrics_cpu_mid_plus_synth.json --num-workers 0 --case-limit 32 --max-val-batches 6 --spatial-size 96 --seed 42 --split-seed -1 --val-ratio 0.2 --quiet-warnings` | 41 | 7 | +16 synthetic train-only cases from `BraTS2023_SYN` | 2 | 0.0335 | 121.3415 | 0.4941 | ~4-6 min | `checkpoints/best_model_cpu_mid_plus_synth.pt`, `results/metrics/baseline_metrics_cpu_mid_plus_synth.json` |
| run-019 | Multi-seed quick A/B (baseline vs +synthetic) | loop over `seed=42,43,44` with `python -m scripts.train_segmentation` (epochs=1, case-limit=32, max-train-batches=10, max-val-batches=3) and paired `python -m scripts.evaluate` for baseline and `--train-extra-dir data/processed/BraTS2023_SYN` | 25/41 | 7 | quick-screen configuration for variance check | 1 | 0.0116 (baseline mean) | 142.2647 (baseline mean) | 0.1074 (baseline mean) | ~20-30 min | `results/metrics/seed_*_baseline_quick.json`, `results/metrics/seed_*_plus_synth_quick.json`, `results/tables/seed_ablation_quick_summary.csv` |

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

### run-007
- Intended change: run a stronger CPU baseline than smoke for better reporting.
- Result summary: run completed and produced `baseline_metrics_cpu_mid.json` with improved HD95 but lower Dice/higher ECE than smoke-v2.
- Failure modes: initial attempt failed with label out-of-bounds in DiceCE one-hot conversion.
- Fixes applied: sanitized training labels before loss (`long`, shape-safe, clamp to class range) and re-ran successfully.

### run-008
- Intended change: train VAE on CPU for synthetic panel generation.
- Result summary: training completed; loss improved from 0.0737 to 0.0174 in 2 epochs.
- Failure modes: none observed.
- Fixes applied: not required.

### run-009
- Intended change: generate synthetic modality panels for reporting evidence.
- Result summary: eight synthetic panels generated under `results/generated_smoke_v2`.
- Failure modes: none observed.
- Fixes applied: not required.

### run-010
- Intended change: obtain stronger CPU-only segmentation evidence with a longer bounded run.
- Result summary: completed successfully with improved Dice and HD95 versus earlier CPU runs.
- Failure modes: none observed after prior label-safety fixes.
- Fixes applied: not required.

### run-011
- Intended change: complete qualitative checklist with prediction-vs-label overlays for representative cases.
- Result summary: overlays generated for three GLI cases using the CPU long checkpoint.
- Failure modes: none observed.
- Fixes applied: not required.

### run-012
- Intended change: backfill class-wise Dice/HD95 and seed metadata for smoke baseline evaluation output.
- Result summary: metric file updated with `dice_classwise`, `hd95_classwise`, and reproducibility fields (`seed`, `split_seed`, `val_ratio`).
- Failure modes: no blocking failures; runtime still includes expected MONAI/TensorFlow warning noise.
- Fixes applied: used module invocation (`python -m scripts.evaluate`) and explicit `--split-seed -1` to preserve historical split behavior.

### run-013
- Intended change: backfill class-wise Dice/HD95 and seed metadata for medium baseline evaluation output.
- Result summary: metric file updated and values aligned with previously reported aggregate metrics.
- Failure modes: no blocking failures.
- Fixes applied: same reproducibility settings as run-012 for split compatibility.

### run-014
- Intended change: complete class-wise Dice/HD95 backfill for long baseline output.
- Result summary: long metric file now includes class-wise fields and reproducibility metadata.
- Failure modes: runtime was substantially longer than smoke/medium and required asynchronous terminal execution.
- Fixes applied: none required in model code; completion achieved with the same module-invocation and split-compatible settings.

### run-015
- Intended change: run a second architecture baseline (SegResNet) under medium bounded CPU settings.
- Result summary: checkpoint was produced; runtime was significantly slower than UNet with this configuration.
- Failure modes: initial command form (`python scripts/...`) failed with module import path errors; corrected to module invocation (`python -m ...`).
- Fixes applied: reran with module invocation and recorded checkpoint artifact.

### run-016
- Intended change: obtain a quick SegResNet comparison metric without waiting for full bounded validation completion.
- Result summary: quick-screen metrics generated; SegResNet underperformed UNet in this bounded setup.
- Failure modes: full `max-val-batches=6` evaluation was too slow for interactive completion.
- Fixes applied: completed a transparent quick-screen evaluation with `max-val-batches=2` and clearly labeled it as non-final.

### run-017
- Intended change: create trainable synthetic data with labels preserved for a real augmentation A/B.
- Result summary: 16 synthetic cases generated from first 32 base cases under `data/processed/BraTS2023_SYN`.
- Failure modes: none observed.
- Fixes applied: not required.

### run-018
- Intended change: run a true baseline+synthetic augmentation experiment with synthetic data added to training only.
- Result summary: experiment completed; compared with baseline medium, Dice decreased (0.0335 vs 0.0525) and HD95 worsened.
- Failure modes: none blocking after adding `--train-extra-dir` support.
- Fixes applied: implemented train-only extra directory loading in `scripts/dataset.py` and `scripts/train_segmentation.py`.

### run-019
- Intended change: reduce single-seed bias by running a quick 3-seed A/B stress check under bounded settings.
- Result summary: near-parity aggregate outcome (`mean delta_dice=-0.000115`, `mean delta_hd95=-0.361`, `mean delta_ece=+0.0019`), indicating no robust improvement from current synthetic settings.
- Failure modes: none blocking.
- Fixes applied: exported a machine-readable summary table at `results/tables/seed_ablation_quick_summary.csv`.
