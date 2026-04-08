# Data Layout

This project expects BraTS files in a structure similar to:

- `data/raw/BraTS2023/` for downloaded source files
- `data/processed/BraTS2023/` for preprocessed cases

Each case directory should include the MRI modalities and segmentation mask:

- `*_t1.nii.gz`
- `*_t1ce.nii.gz`
- `*_t2.nii.gz`
- `*_flair.nii.gz`
- `*_seg.nii.gz`

You can download BraTS from Kaggle or Hugging Face and then place files under `data/raw/`.
