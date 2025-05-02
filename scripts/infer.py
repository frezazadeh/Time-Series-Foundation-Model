#!/usr/bin/env python3
"""
Generate predictions for *one* KPI channel and (optionally) compute metrics.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

from tsfound.models import TSBackboneFeature, SupervisedTSModel
from tsfound.utils  import get_device, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="supervised_model.pt")
    p.add_argument("--x",    required=True, help="(N,L,C) inputs")
    p.add_argument("--y",                help="(optional) (N,C) ground truth")
    p.add_argument("--kpi_idx", type=int, default=6, help="column predicted")
    p.add_argument("--out",  default="y_pred.npy")
    p.add_argument("--batch",type=int, default=256)
    return p.parse_args()


class XDS(Dataset):
    def __init__(self, x): self.x = torch.from_numpy(x).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def main():
    args = parse_args()
    seed_everything()
    device, _ = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    x = np.load(args.x)
    n_ch = x.shape[2]

    # scale inputs
    for c, sc in enumerate(ckpt["x_scalers"]):
        x[:, :, c] = sc.transform(x[:, :, c].reshape(-1, 1)).reshape(x[:, :, c].shape)

    dl = DataLoader(XDS(x), batch_size=args.batch, shuffle=False,
                    num_workers=min(2, os.cpu_count()))

    backbone = TSBackboneFeature(n_channels=n_ch).to(device)
    model = SupervisedTSModel(backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    preds = []
    with torch.no_grad():
        for xb in dl:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.vstack(preds)                       # (N,1)
    np.save(args.out, preds)
    print("✅ predictions →", Path(args.out).resolve())

    # optional metrics + plot
    if args.y:
        y_full = np.load(args.y)                   # (N,C)
        y_true = y_full[:, args.kpi_idx:args.kpi_idx + 1]
        y_pred = ckpt["y_scaler"].inverse_transform(preds)
        mse  = mean_squared_error(y_true, y_pred)
        print(f"MSE={mse:.6f}, RMSE={mse**0.5:.6f}")


if __name__ == "__main__":
    main()
