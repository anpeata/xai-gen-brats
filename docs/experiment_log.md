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

## Run Notes

### run-001
- Intended change: Validate full training/evaluation loop on CPU with bounded batches.
- Result summary: Training and evaluation completed successfully with smoke metrics saved.
- Failure modes: CPU runtime remains high for full-scale training; smoke metrics are low by design due to constrained run size.
- Next action: Increase case and batch limits gradually or move to GPU for full baseline reporting.
