from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from generation.vae import BrainMRIVAE, vae_loss


class BratsSliceDataset(Dataset):
    def __init__(self, data_dir: str, target_size: int = 128):
        self.case_dirs = [p for p in sorted(Path(data_dir).iterdir()) if p.is_dir()]
        self.target_size = target_size

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case = self.case_dirs[idx]
        stem = case.name
        vols = []
        for mod in ["t1", "t1ce", "t2", "flair"]:
            path = case / f"{stem}_{mod}.nii.gz"
            arr = nib.load(str(path)).get_fdata().astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            vols.append(arr)

        mid = vols[0].shape[2] // 2
        slices = [v[:, :, mid] for v in vols]
        x = np.stack(slices, axis=0)
        x = torch.from_numpy(x).unsqueeze(0)
        x = torch.nn.functional.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return x.squeeze(0)


def parse_args():
    p = argparse.ArgumentParser(description="Train a lightweight VAE for MRI slice synthesis.")
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="checkpoints/vae.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ds = BratsSliceDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = BrainMRIVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"VAE epoch {epoch}/{args.epochs}"):
            x = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, recon_loss, kld_loss = vae_loss(recon, x, mu, logvar, beta=args.beta)
            loss.backward()
            opt.step()
            running += loss.item()

        avg = running / max(1, len(loader))
        print(f"Epoch {epoch}: loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model_state_dict": model.state_dict(), "loss": avg}, out_path)

    print(f"Saved best VAE checkpoint to {out_path} (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
