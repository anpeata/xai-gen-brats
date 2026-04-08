from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generation.vae import BrainMRIVAE


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic MRI slices with trained VAE.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/vae.pt")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--out-dir", type=str, default="results/generated")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = BrainMRIVAE(latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(args.n, args.latent_dim, device=device)
        samples = model.decode(z).cpu()

    for i in range(args.n):
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for c, title in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            axes[c].imshow(samples[i, c], cmap="gray")
            axes[c].set_title(title)
            axes[c].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"synthetic_case_{i:02d}.png", dpi=160)
        plt.close(fig)

    print(f"Saved {args.n} synthetic sample panels to {out_dir}")


if __name__ == "__main__":
    main()
