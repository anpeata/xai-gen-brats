# Experiment Log Template

Track every run in one place for reproducibility.

## Environment

- Date:
- Machine/GPU:
- Python version:
- Torch version:
- MONAI version:
- Commit hash:
- Dataset source/version:

## Run Table

| Run ID | Objective | Command | Train Cases | Val Cases | Key Hyperparameters | Best Epoch | Dice Mean | HD95 Mean | ECE | Runtime | Output Paths |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| run-001 | Baseline UNet | python scripts/train_segmentation.py ... |  |  |  |  |  |  |  |  |  |
| run-002 | + XAI analysis | python scripts/run_xai.py ... |  |  |  | - | - | - | - |  |  |
| run-003 | + MC Dropout | python scripts/uncertainty.py ... |  |  |  | - | - | - | - |  |  |
| run-004 | VAE synthesis | python scripts/train_vae.py ... |  |  |  |  |  |  |  |  |  |

## Notes Per Run

### run-001
- Intended change:
- Result summary:
- Failure modes:
- Next action:

### run-002
- Intended change:
- Result summary:
- Failure modes:
- Next action:

### run-003
- Intended change:
- Result summary:
- Failure modes:
- Next action:

### run-004
- Intended change:
- Result summary:
- Failure modes:
- Next action:
