from __future__ import annotations

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate label-preserving synthetic BraTS cases using paired spatial/intensity transforms."
    )
    p.add_argument("--source-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--out-dir", type=str, default="data/processed/BraTS2023_SYN")
    p.add_argument("--num-synthetic", type=int, default=16)
    p.add_argument("--base-case-limit", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_cases(source_dir: Path) -> list[Path]:
    return [p for p in sorted(source_dir.iterdir()) if p.is_dir()]


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def apply_spatial(arr: np.ndarray, flips: tuple[bool, bool, bool]) -> np.ndarray:
    out = arr
    if flips[0]:
        out = np.flip(out, axis=0)
    if flips[1]:
        out = np.flip(out, axis=1)
    if flips[2]:
        out = np.flip(out, axis=2)
    return out.copy()


def apply_intensity(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    scale = rng.uniform(0.9, 1.1)
    shift = rng.uniform(-0.1, 0.1)
    gamma = rng.uniform(0.85, 1.15)
    out = normalize(arr)
    out = np.clip(out * scale + shift, 0.0, 1.0)
    out = np.power(out, gamma, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cases = find_cases(source_dir)
    if args.base_case_limit > 0:
        base_cases = base_cases[: args.base_case_limit]

    if not base_cases:
        raise RuntimeError("No base cases found for synthetic generation.")

    generated = 0
    for idx in range(args.num_synthetic):
        base = rng.choice(base_cases)
        stem = base.name

        seg_path = base / f"{stem}_seg.nii.gz"
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata().astype(np.int16)

        flips = (rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5)
        seg_syn = apply_spatial(seg, flips)

        syn_id = f"{stem}_SYN{idx:03d}"
        syn_case_dir = out_dir / syn_id
        syn_case_dir.mkdir(parents=True, exist_ok=True)

        for mod in MODALITIES:
            mod_path = base / f"{stem}_{mod}.nii.gz"
            img = nib.load(str(mod_path))
            arr = img.get_fdata().astype(np.float32)
            arr_syn = apply_spatial(arr, flips)
            arr_syn = apply_intensity(arr_syn, rng)

            out_img = nib.Nifti1Image(arr_syn, affine=img.affine, header=img.header)
            nib.save(out_img, str(syn_case_dir / f"{syn_id}_{mod}.nii.gz"))

        out_seg = nib.Nifti1Image(seg_syn.astype(np.int16), affine=seg_img.affine, header=seg_img.header)
        nib.save(out_seg, str(syn_case_dir / f"{syn_id}_seg.nii.gz"))
        generated += 1

    print(f"Base cases considered: {len(base_cases)}")
    print(f"Synthetic cases generated: {generated}")
    print(f"Synthetic output directory: {out_dir}")


if __name__ == "__main__":
    main()
