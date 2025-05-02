#!/usr/bin/env python3
"""Fine‑tune the backbone on a KPI prediction target."""
import itertools 
import argparse, os, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tsfound.models import TSBackboneFeature, SupervisedTSModel
from tsfound.utils  import get_device

class SeriesDS(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',   type=str, required=True,
                   help='Pre‑trained backbone checkpoint')
    p.add_argument('--x',      type=str, required=True)
    p.add_argument('--y',      type=str, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch',  type=int, default=256)
    p.add_argument('--lr',     type=float, default=1e-4)
    args = p.parse_args()

    x = np.load(args.x); y = np.load(args.y)
    # assume y is (N, 1)
    n_ch = x.shape[2]
    device, _ = get_device()

    # scaling
    scalers = [StandardScaler().fit(x[:,:,c].reshape(-1,1)) for c in range(n_ch)]
    for c, sc in enumerate(scalers):
        x[:,:,c] = sc.transform(x[:,:,c].reshape(-1,1)).reshape(x[:,:,c].shape)
    y_scaler = StandardScaler().fit(y); y = y_scaler.transform(y)

    ds = SeriesDS(x, y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4)

    backbone = TSBackboneFeature(n_channels=n_ch).to(device)
    state = torch.load(args.ckpt, map_location='cpu')
    backbone.load_state_dict(state, strict=False)

    model = SupervisedTSModel(backbone).to(device)
    for p in backbone.parameters(): p.requires_grad = False
    for p in itertools.chain(backbone.blocks[-1].parameters(),
                             backbone.ln_final.parameters()):
        p.requires_grad = True   # unfreeze last block

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.MSELoss()

    for ep in range(1, args.epochs+1):
        model.train(); losses=[]
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f'Ep{ep} | train MSE {sum(losses)/len(losses):.4f}')

    torch.save({'model_state_dict': model.state_dict(),
                'y_scaler': y_scaler,
                'x_scalers': scalers},
               'supervised_model.pt')

if __name__ == '__main__':
    main()
