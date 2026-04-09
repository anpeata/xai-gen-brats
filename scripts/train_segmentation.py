from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm

from models.segmentation import create_segmentation_model
from scripts.dataset import get_dataloaders


def to_onehot_tensor(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().clamp(0, num_classes - 1)
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return onehot.permute(0, 4, 1, 2, 3).float()


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline BraTS segmentation model.")
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--model", type=str, default="unet", choices=["unet", "segresnet"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-rate", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--spatial-size", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--quick-cpu", action="store_true")
    p.add_argument("--out", type=str, default="checkpoints/best_model.pt")
    return p.parse_args()


def main():
    args = parse_args()

    # Fast smoke-test profile for CPU-only environments.
    if args.quick_cpu:
        if args.epochs == 50:
            args.epochs = 1
        if args.num_workers == 4:
            args.num_workers = 0
        if args.case_limit == 0:
            args.case_limit = 16
        if args.spatial_size == 128:
            args.spatial_size = 96
        if args.num_samples == 2:
            args.num_samples = 1
        if args.max_train_batches == 0:
            args.max_train_batches = 10
        if args.max_val_batches == 0:
            args.max_val_batches = 4

    device = torch.device(args.device)
    spatial_size = (args.spatial_size, args.spatial_size, args.spatial_size)

    train_loader, val_loader, n_train, n_val = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        val_ratio=args.val_ratio,
        case_limit=args.case_limit,
        spatial_size=spatial_size,
        num_samples=args.num_samples,
    )
    print(f"Loaded dataset: train={n_train}, val={n_val}, device={device}")
    print(
        "Run config: "
        f"spatial_size={spatial_size}, num_samples={args.num_samples}, "
        f"max_train_batches={args.max_train_batches}, max_val_batches={args.max_val_batches}"
    )

    model = create_segmentation_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(progress, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

        avg_loss = epoch_loss / max(1, len(train_loader))

        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = sliding_window_inference(images, roi_size=spatial_size, sw_batch_size=1, predictor=model)
                pred_cls = torch.argmax(logits, dim=1)
                pred_onehot = to_onehot_tensor(pred_cls, num_classes=logits.shape[1])
                label_onehot = to_onehot_tensor(labels, num_classes=logits.shape[1])
                dice_metric(y_pred=pred_onehot, y=label_onehot)

                if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                    break

        val_dice = float(dice_metric.aggregate().item())
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "model_name": args.model,
                },
                out_path,
            )
            print(f"Saved new best checkpoint to {out_path} (val_dice={best_dice:.4f})")

        # Always save the latest epoch checkpoint to guarantee a usable artifact.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "model_name": args.model,
            },
            out_path,
        )

    print(f"Training complete. Best validation Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
