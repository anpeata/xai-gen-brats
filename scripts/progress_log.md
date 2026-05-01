# Progress Log

## 2026-05-01
- Fixed `scripts.evaluate` to report BraTS ET/TC/WT region metrics under `dice_mean`/`hd95_mean`.
- Added auxiliary raw-label (NCR/ED/ET) metrics as `label_*` fields for debugging.
- Note: run scripts as modules (e.g. `python -m scripts.evaluate`) so imports like `models.*` resolve.
- Smoke tests: `python -m compileall scripts` OK; `python -m scripts.evaluate --help` OK.
