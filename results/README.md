# Results Directory Guide

Use this folder for real outputs produced by experiments.

## Recommended Layout

- `results/metrics/`
  - `baseline_metrics.json`
  - `xai_metrics.json`
  - `augmentation_metrics.json`
- `results/figures/`
  - `segmentation_overlay_case_<id>.png`
  - `gradcam_case_<id>.png`
  - `uncertainty_case_<id>.png`
  - `synthetic_samples_panel.png`
- `results/tables/`
  - `ablation_summary.csv`
  - `classwise_metrics.csv`
- `results/logs/`
  - training logs and runtime notes

## Minimum Evidence Checklist

- [x] Baseline metrics available.
- [x] Class-wise Dice reported.
- [x] HD95 and ECE reported.
- [ ] At least 3 representative figures included.
- [x] One ablation table included.
- [ ] Insights summarized in `docs/insights_report.md`.

Current machine-readable artifacts:

- `results/tables/ablation_summary.csv`
- `results/tables/classwise_metrics.csv`
- `results/tables/seed_ablation_quick_summary.csv`
- `results/tables/seed_ablation_medium_summary.csv`
- `results/tables/seed_ablation_medium_tuned_summary.csv`
- `results/tables/seed_ablation_medium_tuned64_summary.csv`
