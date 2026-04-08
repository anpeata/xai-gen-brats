from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from models.segmentation import create_segmentation_model
from scripts.dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation checkpoint on BraTS validation set.")
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="results/metrics.json")
    p.add_argument("--num-workers", type=int, default=4)
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
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "unet")
    model = create_segmentation_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])
    post_label = AsDiscrete(to_onehot=3)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    confidence_values = []
    correctness_values = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(images, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model)
            probs = torch.softmax(logits, dim=1)

            pred_onehot = [post_pred(i) for i in logits]
            label_onehot = [post_label(i) for i in labels]
            dice_metric(y_pred=pred_onehot, y=label_onehot)
            hd95_metric(y_pred=pred_onehot, y=label_onehot)

            conf, pred_cls = torch.max(probs, dim=1)
            true_cls = labels[:, 0, ...]
            correct = (pred_cls == true_cls).float()
            confidence_values.extend(conf.cpu().numpy().ravel().tolist())
            correctness_values.extend(correct.cpu().numpy().ravel().tolist())

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
