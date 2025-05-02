#!/usr/bin/env python3
"""Run inference with the fine‑tuned model."""
import argparse, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from tsfound.models import TSBackboneFeature, SupervisedTSModel
from tsfound.utils  import get_device
from sklearn.metrics import mean_squared_error

class XDS(Dataset):
    def __init__(self, x): self.x = torch.from_numpy(x).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--x',    type=str, required=True)
    p.add_argument('--y',    type=str)
    p.add_argument('--out',  type=str, default='y_pred.npy')
    p.add_argument('--batch',type=int, default=256)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    x    = np.load(args.x)
    y    = np.load(args.y) if args.y else None

    n_ch = x.shape[2]
    for c, sc in enumerate(ckpt['x_scalers']):
        x[:,:,c] = sc.transform(x[:,:,c].reshape(-1,1)).reshape(x[:,:,c].shape)

    ds = XDS(x)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)
    device, _ = get_device()

    backbone = TSBackboneFeature(n_channels=n_ch).to(device)
    model = SupervisedTSModel(backbone).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.vstack(preds)
    np.save(args.out, preds)
    print('Saved', args.out)

    if y is not None:
        from sklearn.preprocessing import StandardScaler
        y_pred = ckpt['y_scaler'].inverse_transform(preds)
        mse  = mean_squared_error(y, y_pred)
        rmse = mse ** 0.5
        print(f'Test MSE={mse:.4f}  RMSE={rmse:.4f}')

if __name__ == '__main__':
    main()
