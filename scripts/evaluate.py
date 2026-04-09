from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from models.segmentation import create_segmentation_model
from scripts.dataset import get_dataloaders


def to_onehot_tensor(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().clamp(0, num_classes - 1)
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return onehot.permute(0, 4, 1, 2, 3).float()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation checkpoint on BraTS validation set.")
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="results/metrics.json")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--spatial-size", type=int, default=128)
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
    device = torch.device(args.device)

    _, val_loader, _, n_val = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=args.num_workers,
        case_limit=args.case_limit,
        spatial_size=(args.spatial_size, args.spatial_size, args.spatial_size),
        num_samples=1,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "unet")
    model = create_segmentation_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    confidence_values = []
    correctness_values = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(
                images,
                roi_size=(args.spatial_size, args.spatial_size, args.spatial_size),
                sw_batch_size=1,
                predictor=model,
            )
            probs = torch.softmax(logits, dim=1)

            pred_cls = torch.argmax(logits, dim=1)
            pred_onehot = to_onehot_tensor(pred_cls, num_classes=logits.shape[1])
            label_onehot = to_onehot_tensor(labels, num_classes=logits.shape[1])
            dice_metric(y_pred=pred_onehot, y=label_onehot)
            hd95_metric(y_pred=pred_onehot, y=label_onehot)

            conf, pred_cls = torch.max(probs, dim=1)
            true_cls = labels[:, 0, ...]
            correct = (pred_cls == true_cls).float()
            confidence_values.extend(conf.cpu().numpy().ravel().tolist())
            correctness_values.extend(correct.cpu().numpy().ravel().tolist())

            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break

    metrics = {
        "n_validation_cases": n_val,
        "dice_mean": float(dice_metric.aggregate().item()),
        "hd95_mean": float(hd95_metric.aggregate().item()),
        "ece": expected_calibration_error(
            np.asarray(confidence_values, dtype=np.float32),
            np.asarray(correctness_values, dtype=np.float32),
            n_bins=15,
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
