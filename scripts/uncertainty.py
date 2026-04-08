from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd

from models.segmentation import create_segmentation_model


def enable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo Dropout uncertainty for BraTS segmentation.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--case-dir", type=str, required=True)
    p.add_argument("--passes", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="results/uncertainty/uncertainty_map.png")
    return p.parse_args()


def load_case(case_dir: Path):
    stem = case_dir.name
    sample = {
        "image": [
            str(case_dir / f"{stem}_t1.nii.gz"),
            str(case_dir / f"{stem}_t1ce.nii.gz"),
            str(case_dir / f"{stem}_t2.nii.gz"),
            str(case_dir / f"{stem}_flair.nii.gz"),
        ]
    }
    tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
    out = tf(sample)
    return out["image"].unsqueeze(0)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = create_segmentation_model(ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    enable_dropout(model)

    x = load_case(Path(args.case_dir)).to(device)
    probs = []
    with torch.no_grad():
        for _ in range(args.passes):
            p = torch.softmax(model(x), dim=1)
            probs.append(p)

    stack = torch.stack(probs, dim=0)
    uncertainty = stack.var(dim=0).mean(dim=1).squeeze().cpu().numpy()
    flair = x[0, 3].detach().cpu().numpy()
    z = flair.shape[0] // 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(flair[z], cmap="gray")
    plt.title("FLAIR slice")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(uncertainty[z], cmap="inferno")
    plt.title("MC Dropout Uncertainty")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved uncertainty map to {args.out}")


if __name__ == "__main__":
    main()
