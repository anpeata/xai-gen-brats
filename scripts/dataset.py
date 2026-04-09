from __future__ import annotations

from pathlib import Path
from typing import Iterable

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)

MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]


def _find_cases(data_dir: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for seg_path in sorted(data_dir.rglob("*_seg.nii.gz")):
        stem = seg_path.name.replace("_seg.nii.gz", "")
        image_paths = []
        missing = False
        for mod in MODALITY_SUFFIXES:
            p = seg_path.with_name(f"{stem}_{mod}.nii.gz")
            if not p.exists():
                missing = True
                break
            image_paths.append(str(p))
        if missing:
            continue
        cases.append({"image": image_paths, "label": str(seg_path)})
    return cases


def split_cases(items: list[dict[str, str]], val_ratio: float = 0.2):
    split_idx = int(len(items) * (1 - val_ratio))
    return items[:split_idx], items[split_idx:]


def build_transforms(
    pixdim: Iterable[float] = (1.0, 1.0, 1.0),
    spatial_size: tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 2,
):
    train = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train, val


def get_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    cache_rate: float = 0.1,
    val_ratio: float = 0.2,
    case_limit: int = 0,
    spatial_size: tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 2,
):
    root = Path(data_dir)
    cases = _find_cases(root)

    if case_limit > 0:
        cases = cases[:case_limit]

    if len(cases) < 2:
        raise ValueError("Expected at least 2 cases. Verify BraTS files are under data_dir.")

    train_cases, val_cases = split_cases(cases, val_ratio=val_ratio)
    train_tf, val_tf = build_transforms(spatial_size=spatial_size, num_samples=num_samples)

    train_ds = CacheDataset(data=train_cases, transform=train_tf, cache_rate=cache_rate)
    val_ds = Dataset(data=val_cases, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(train_cases), len(val_cases)


def post_pred_transform():
    return AsDiscrete(argmax=True, to_onehot=3)


def post_label_transform():
    return AsDiscrete(to_onehot=3)
