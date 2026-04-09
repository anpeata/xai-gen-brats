# Data Layout

This repository uses the following data organization:

- `data/raw/BraTS2023/archives/` stores downloaded challenge archives.
	- `GLI/`, `MEN/`, `PED/`, and `misc/` group archives by challenge family.
- `data/raw/BraTS2023/extracted/` stores extracted support packages.
- `data/raw/BraTS2023/incomplete/` stores unfinished browser downloads.
- `data/processed/BraTS2023/` stores training-ready case folders.

## Processed Case Format

Each processed case directory must contain:

- `*_t1.nii.gz`
- `*_t1ce.nii.gz`
- `*_t2.nii.gz`
- `*_flair.nii.gz`
- `*_seg.nii.gz`

## Source Naming Notes

Synapse BraTS archives typically use modality suffixes below:

- `-t1n` (mapped to `t1`)
- `-t1c` (mapped to `t1ce`)
- `-t2w` (mapped to `t2`)
- `-t2f` (mapped to `flair`)
- `-seg` (segmentation mask)

The preparation script converts these files into the processed case format expected by the training pipeline.

## Storage and Source Policy

- The full BraTS 2023 dataset is too large to upload to git (well above 100 GB in full archive plus extracted form).
- Data is stored locally only and is excluded from version control.
- The dataset used in this repository was downloaded from the BraTS 2023 Challenge distribution on Synapse | Sage Bionetworks.
