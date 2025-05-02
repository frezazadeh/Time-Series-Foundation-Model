#!/usr/bin/env python3
"""
Plot ground‑truth vs. predicted values for one KPI.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--y",     required=True, help="(N,C) ground truth .npy")
    p.add_argument("--pred",  required=True, help="(N,1) predicted .npy")
    p.add_argument("--ckpt",  required=True, help="supervised_model.pt (scaler)")
    p.add_argument("--kpi_idx", type=int, default=6)
    p.add_argument("--png",   default="kpi_true_vs_pred.png")
    return p.parse_args()


def main():
    a = parse_args()

    y_all   = np.load(a.y)                       # (N,C)
    y_true  = y_all[:, a.kpi_idx]                # (N,)
    y_pred_scaled = np.load(a.pred)              # (N,1)

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    y_pred = ckpt["y_scaler"].inverse_transform(y_pred_scaled).ravel()

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="Ground truth", lw=2)
    plt.plot(y_pred, label="Predicted", ls="--")
    plt.xlabel("Sample index"); plt.ylabel(f"KPI {a.kpi_idx}")
    plt.title(f"KPI‑{a.kpi_idx} • True vs. Predicted")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(a.png, dpi=150)
    plt.show()
    print("✅ plot saved →", Path(a.png).resolve())


if __name__ == "__main__":
    main()
