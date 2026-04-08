from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, Compose
from tqdm import tqdm

from models.segmentation import create_segmentation_model
from scripts.dataset import get_dataloaders, post_label_transform, post_pred_transform


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline BraTS segmentation model.")
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--model", type=str, default="unet", choices=["unet", "segresnet"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="checkpoints/best_model.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, n_train, n_val = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Loaded dataset: train={n_train}, val={n_val}")

    model = create_segmentation_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), post_pred_transform()])
    post_label = post_label_transform()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, len(train_loader))

        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = sliding_window_inference(images, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model)
                preds = [post_pred(i) for i in logits]
                gts = [post_label(i) for i in labels]
                dice_metric(y_pred=preds, y=gts)

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

    print(f"Training complete. Best validation Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
