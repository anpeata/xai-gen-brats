# Insights Report Template

## 1. Experimental Setup

- Data split:
- Preprocessing:
- Model:
- Optimizer and schedule:
- Evaluation metrics:

## 2. Quantitative Results

| Setting | Dice Mean | Dice ET | Dice TC | Dice WT | HD95 Mean | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline segmentation |  |  |  |  |  |  |
| Baseline + uncertainty analysis |  |  |  |  |  |  |
| Baseline + synthetic augmentation |  |  |  |  |  |  |

## 3. Qualitative Analysis

- Case A:
  - What worked:
  - What failed:
  - Clinical relevance:
- Case B:
  - What worked:
  - What failed:
  - Clinical relevance:

## 4. Explainability Findings

- Grad-CAM consistency with lesion location:
- Modality attribution ranking from SHAP:
- Mismatch cases and hypotheses:

## 5. Uncertainty Findings

- High-uncertainty regions:
- Correlation with boundary errors:
- Calibration interpretation:

## 6. Synthetic Data Findings

- Visual quality observations:
- Performance impact after augmentation:
- Risks (artifact amplification, bias):

## 7. Final Conclusions

- Most reliable configuration:
- Current limitations:
- Top 3 next experiments:
