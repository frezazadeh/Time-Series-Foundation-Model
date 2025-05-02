#!/usr/bin/env python3
"""
Fine‑tune a pre‑trained masked‑patch backbone on **one KPI channel**.

Example
-------
python scripts/finetune.py \
        --ckpt checkpoints/model_epoch_50.pt \
        --x /content/x.npy --y /content/y.npy \
        --kpi_idx 6 --epochs 100
"""
from __future__ import annotations
import argparse, itertools, os
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tsfound.models import TSBackboneFeature, SupervisedTSModel
from tsfound.utils  import get_device, seed_everything


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True, help="Backbone .pt checkpoint")
    p.add_argument("--x",      required=True, help="(N,L,C) numpy inputs")
    p.add_argument("--y",      required=True, help="(N,C)  numpy targets")
    p.add_argument("--kpi_idx",type=int, default=6,   help="Which KPI column to train on")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--out",    default="supervised_model.pt")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────
class SeriesDS(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):  return len(self.x)
    def __getitem__(self, i):  return self.x[i], self.y[i]


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    seed_everything()
    device, _ = get_device()

    # 1) Load data
    x = np.load(args.x)                   # (N,L,C)
    y_all = np.load(args.y)               # (N,C)
    n_ch = x.shape[2]

    # 2) choose one KPI channel
    y = y_all[:, args.kpi_idx:args.kpi_idx + 1]    # (N,1)

    # 3) scale X per‑channel, y separately
    x_scalers = [StandardScaler().fit(x[:, :, c].reshape(-1, 1)) for c in range(n_ch)]
    for c, sc in enumerate(x_scalers):
        x[:, :, c] = sc.transform(x[:, :, c].reshape(-1, 1)).reshape(x[:, :, c].shape)
    y_scaler = StandardScaler().fit(y)
    y = y_scaler.transform(y)

    ds = SeriesDS(x, y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=min(2, os.cpu_count()))

    # 4) model
    backbone = TSBackboneFeature(n_channels=n_ch).to(device)
    backbone.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)

    model = SupervisedTSModel(backbone).to(device)
    for p in backbone.parameters(): p.requires_grad = False
    for p in itertools.chain(backbone.blocks[-1].parameters(),
                             backbone.ln_final.parameters()):
        p.requires_grad = True    # unfreeze last block

    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.MSELoss()

    # 5) training
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, yb in tqdm(dl, desc=f"Ep{ep}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep:03d} | train MSE {np.mean(losses):.6f}")

    # 6) save
    torch.save({
        "model_state_dict": model.state_dict(),
        "y_scaler": y_scaler,
        "x_scalers": x_scalers,
    }, args.out)
    print("✅ Fine‑tuned model saved →", Path(args.out).resolve())


if __name__ == "__main__":
    main()
