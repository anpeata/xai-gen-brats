from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="Protobuf gencode version", category=UserWarning)

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from models.segmentation import create_segmentation_model
from scripts.dataset import get_dataloaders


BRATS_REGION_NAMES = ["ET", "TC", "WT"]
LABEL_CLASS_NAMES = ["NCR", "ED", "ET"]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_onehot_tensor(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().clamp(0, num_classes - 1)
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return onehot.permute(0, 4, 1, 2, 3).float()


def to_brats_region_masks(labels: torch.Tensor) -> torch.Tensor:
    """Convert BraTS label ids {0,1,2,3} into ET/TC/WT multi-label masks.

    BraTS conventions:
    - ET: label == 3
    - TC: label in {1, 3}
    - WT: label in {1, 2, 3}

    Returns a float tensor of shape [B, 3, H, W, D] with channels [ET, TC, WT].
    """

    labels = labels.long()
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4:
        raise ValueError(f"Expected labels [B,H,W,D] or [B,1,H,W,D], got shape {tuple(labels.shape)}")

    et = labels == 3
    tc = (labels == 1) | (labels == 3)
    wt = labels > 0
    return torch.stack([et, tc, wt], dim=1).float()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation checkpoint on BraTS validation set.")
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="results/metrics.json")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--spatial-size", type=int, default=128)
    p.add_argument("--quiet-warnings", action="store_true")
    return p.parse_args()


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def main():
    args = parse_args()
    if args.quiet_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    set_global_seed(args.seed)
    split_seed = None if args.split_seed < 0 else args.split_seed
    device = torch.device(args.device)

    _, val_loader, _, n_val = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        split_seed=split_seed,
        case_limit=args.case_limit,
        spatial_size=(args.spatial_size, args.spatial_size, args.spatial_size),
        num_samples=1,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "unet")
    model = create_segmentation_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Primary metrics: BraTS ET/TC/WT regions (multi-label masks).
    region_dice_metric = DiceMetric(include_background=True, reduction="mean")
    region_hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    region_dice_metric_classwise = DiceMetric(include_background=True, reduction="mean_batch")
    region_hd95_metric_classwise = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    # Auxiliary diagnostics: raw label ids 1/2/3 (NCR/ED/ET), excluding background (0).
    label_dice_metric = DiceMetric(include_background=False, reduction="mean")
    label_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    label_dice_metric_classwise = DiceMetric(include_background=False, reduction="mean_batch")
    label_hd95_metric_classwise = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")

    confidence_values = []
    correctness_values = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if labels.ndim == 4:
                labels = labels.unsqueeze(1)
            labels = labels.long().clamp(0, 3)
            logits = sliding_window_inference(
                images,
                roi_size=(args.spatial_size, args.spatial_size, args.spatial_size),
                sw_batch_size=1,
                predictor=model,
            )
            probs = torch.softmax(logits, dim=1)

            pred_cls = torch.argmax(logits, dim=1)

            # BraTS region metrics (ET/TC/WT).
            region_pred = to_brats_region_masks(pred_cls)
            region_label = to_brats_region_masks(labels)
            region_dice_metric(y_pred=region_pred, y=region_label)
            region_hd95_metric(y_pred=region_pred, y=region_label)
            region_dice_metric_classwise(y_pred=region_pred, y=region_label)
            region_hd95_metric_classwise(y_pred=region_pred, y=region_label)

            # Raw label metrics (1/2/3).
            pred_onehot = to_onehot_tensor(pred_cls, num_classes=logits.shape[1])
            label_onehot = to_onehot_tensor(labels, num_classes=logits.shape[1])
            label_dice_metric(y_pred=pred_onehot, y=label_onehot)
            label_hd95_metric(y_pred=pred_onehot, y=label_onehot)
            label_dice_metric_classwise(y_pred=pred_onehot, y=label_onehot)
            label_hd95_metric_classwise(y_pred=pred_onehot, y=label_onehot)

            conf, voxel_pred_cls = torch.max(probs, dim=1)
            true_cls = labels[:, 0, ...].clamp(0, logits.shape[1] - 1)
            correct = (voxel_pred_cls == true_cls).float()
            confidence_values.extend(conf.cpu().numpy().ravel().tolist())
            correctness_values.extend(correct.cpu().numpy().ravel().tolist())

            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break

    dice_classwise_values = region_dice_metric_classwise.aggregate().detach().cpu().numpy().tolist()
    hd95_classwise_values = region_hd95_metric_classwise.aggregate().detach().cpu().numpy().tolist()

    label_dice_classwise_values = label_dice_metric_classwise.aggregate().detach().cpu().numpy().tolist()
    label_hd95_classwise_values = label_hd95_metric_classwise.aggregate().detach().cpu().numpy().tolist()

    metrics = {
        "n_validation_cases": n_val,
        "dice_mean": float(region_dice_metric.aggregate().item()),
        "hd95_mean": float(region_hd95_metric.aggregate().item()),
        "dice_classwise": {
            cls: float(val) for cls, val in zip(BRATS_REGION_NAMES, dice_classwise_values)
        },
        "hd95_classwise": {
            cls: float(val) for cls, val in zip(BRATS_REGION_NAMES, hd95_classwise_values)
        },
        "label_dice_mean": float(label_dice_metric.aggregate().item()),
        "label_hd95_mean": float(label_hd95_metric.aggregate().item()),
        "label_dice_classwise": {
            cls: float(val) for cls, val in zip(LABEL_CLASS_NAMES, label_dice_classwise_values)
        },
        "label_hd95_classwise": {
            cls: float(val) for cls, val in zip(LABEL_CLASS_NAMES, label_hd95_classwise_values)
        },
        "ece": expected_calibration_error(
            np.asarray(confidence_values, dtype=np.float32),
            np.asarray(correctness_values, dtype=np.float32),
            n_bins=15,
        ),
        "seed": args.seed,
        "split_seed": split_seed,
        "val_ratio": args.val_ratio,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
