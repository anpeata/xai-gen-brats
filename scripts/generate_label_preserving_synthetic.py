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
    p.add_argument("--min-tumor-voxels", type=int, default=500)
    p.add_argument("--flip-prob", type=float, default=0.35)
    p.add_argument("--scale-min", type=float, default=0.95)
    p.add_argument("--scale-max", type=float, default=1.05)
    p.add_argument("--shift-min", type=float, default=-0.03)
    p.add_argument("--shift-max", type=float, default=0.03)
    p.add_argument("--gamma-min", type=float, default=0.95)
    p.add_argument("--gamma-max", type=float, default=1.05)
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


def apply_intensity(arr: np.ndarray, rng: random.Random, args: argparse.Namespace) -> np.ndarray:
    scale = rng.uniform(args.scale_min, args.scale_max)
    shift = rng.uniform(args.shift_min, args.shift_max)
    gamma = rng.uniform(args.gamma_min, args.gamma_max)
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

    filtered_cases: list[Path] = []
    for case in base_cases:
        stem = case.name
        seg_path = case / f"{stem}_seg.nii.gz"
        if not seg_path.exists():
            continue
        seg = nib.load(str(seg_path)).get_fdata()
        if int((seg > 0).sum()) >= args.min_tumor_voxels:
            filtered_cases.append(case)

    base_cases = filtered_cases

    if not base_cases:
        raise RuntimeError("No base cases found for synthetic generation.")

    generated = 0
    for idx in range(args.num_synthetic):
        base = rng.choice(base_cases)
        stem = base.name

        seg_path = base / f"{stem}_seg.nii.gz"
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata().astype(np.int16)

        flips = (
            rng.random() < args.flip_prob,
            rng.random() < args.flip_prob,
            rng.random() < args.flip_prob,
        )
        seg_syn = apply_spatial(seg, flips)

        syn_id = f"{stem}_SYN{idx:03d}"
        syn_case_dir = out_dir / syn_id
        syn_case_dir.mkdir(parents=True, exist_ok=True)

        for mod in MODALITIES:
            mod_path = base / f"{stem}_{mod}.nii.gz"
            img = nib.load(str(mod_path))
            arr = img.get_fdata().astype(np.float32)
            arr_syn = apply_spatial(arr, flips)
            arr_syn = apply_intensity(arr_syn, rng, args)

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
