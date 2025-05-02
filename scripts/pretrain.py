#!/usr/bin/env python3
"""
Pre‑train the masked‑patch Transformer on TimeSeries‑PILE (or a folder of CSVs).
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List

import pandas as pd
import torch, torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm

from tsfound.datasets import MaskedPatchDataset
from tsfound.models   import TSFoundationTransformer
from tsfound.utils    import get_device, seed_everything


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token", required=True)
    p.add_argument("--repo", default="AutonLab/Timeseries-PILE")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--patch_len", type=int, default=8)
    p.add_argument("--mask", type=float, default=0.30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chk_dir", default="checkpoints")
    return p.parse_args()

# ── helpers ───────────────────────────────────────────────────────────────
def load_csv_series(csv_paths: List[Path]) -> List[torch.Tensor]:
    series = []
    for pth in tqdm(csv_paths, desc="CSV"):
        df  = pd.read_csv(pth).select_dtypes(include=["number"])
        arr = torch.tensor(df.values, dtype=torch.float32)
        if arr.dim() == 1:
            arr = arr.unsqueeze(-1)
        series.append(arr)
    return series

def pad_channels(series: List[torch.Tensor]) -> List[torch.Tensor]:
    n_channels = max(s.size(1) for s in series)
    out = []
    for arr in series:
        if arr.size(1) < n_channels:
            pad = torch.zeros((arr.size(0), n_channels - arr.size(1)))
            arr = torch.cat([arr, pad], dim=1)
        out.append(arr)
    return out

# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device, dtype = get_device()          # dtype = bfloat16 on CUDA, else float32
    seed_everything()

    # 1) fetch dataset
    print("Downloading dataset from HuggingFace …")
    repo_path = snapshot_download(args.repo, token=args.hf_token, repo_type="dataset")
    csv_paths = [Path(r) / f
                 for r, _, fs in os.walk(repo_path)
                 for f in fs if f.endswith(".csv")]

    # 2) load & pad
    series     = pad_channels(load_csv_series(csv_paths))
    n_channels = series[0].size(1)

    # 3) dataset / loader
    ds = MaskedPatchDataset(series,
                            seq_len=args.seq_len,
                            patch_len=args.patch_len,
                            mask_ratio=args.mask)
    dl = DataLoader(ds,
                    batch_size=args.batch,
                    shuffle=True,
                    num_workers=min(2, os.cpu_count()))

    # 4) model
    model = TSFoundationTransformer(n_channels=n_channels,
                                    seq_len=args.seq_len,
                                    patch_len=args.patch_len).to(device=device,
                                                                 dtype=dtype)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    os.makedirs(args.chk_dir, exist_ok=True)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 5) training loop
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_loss = 0.0

        for x, tgt, m in tqdm(dl, desc=f"E{epoch}/{args.epochs}"):
            x   = x.to(device=device, dtype=dtype)   # <── CAST HERE
            tgt = tgt.to(device=device, dtype=dtype)
            m   = m.to(device)                       # bool mask

            pred = model(x, m)
            loss = F.mse_loss(pred[m], tgt[m])

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch:02d} | train MSE {epoch_loss/len(dl):.6f}")
        torch.save(model.state_dict(),
                   Path(args.chk_dir) / f"model_epoch_{epoch}.pt")

    print("✅ Pre‑training complete!")

if __name__ == "__main__":
    main()
