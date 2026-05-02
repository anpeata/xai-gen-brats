# Progress Log

## 2026-05-01
- Fixed `scripts.evaluate` to report BraTS ET/TC/WT region metrics under `dice_mean`/`hd95_mean`.
- Added auxiliary raw-label (NCR/ED/ET) metrics as `label_*` fields for debugging.
- Note: run scripts as modules (e.g. `python -m scripts.evaluate`) so imports like `models.*` resolve.
- Smoke tests: `python -m compileall scripts` OK; `python -m scripts.evaluate --help` OK.

## 2026-05-02
- Fixed `scripts.train_segmentation`: `--out` now preserves the best checkpoint (no longer overwritten by later epochs).
- Added `--out-latest` to optionally persist the latest epoch checkpoint separately.
- Hardened `scripts.summarize_variant_tradeoffs` to accept either `synth_*` or `variant_*` CSV columns.
- Sped up `scripts.dataset.get_dataloaders` when `case_limit>0` by stopping case validation early.
- Smoke tests: `python -m compileall scripts`, `python -m scripts.train_segmentation --help`, `python -m scripts.summarize_variant_tradeoffs --help`.
