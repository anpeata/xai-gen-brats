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

- [ ] Baseline metrics available.
- [ ] Class-wise Dice reported.
- [ ] HD95 and ECE reported.
- [ ] At least 3 representative figures included.
- [ ] One ablation table included.
- [ ] Insights summarized in `docs/insights_report.md`.
