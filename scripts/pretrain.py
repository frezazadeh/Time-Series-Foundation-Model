#!/usr/bin/env python3
"""Train the masked‑patch Transformer on TimeSeries‑PILE (or any CSV archive)."""
import argparse, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsfound.datasets import MaskedPatchDataset
from tsfound.models   import TSFoundationTransformer
from tsfound.utils    import get_device, seed_everything
from huggingface_hub  import snapshot_download

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hf_token', type=str,    required=True)
    p.add_argument('--repo',     type=str,    default='AutonLab/Timeseries-PILE')
    p.add_argument('--seq_len',  type=int,    default=512)
    p.add_argument('--patch_len',type=int,    default=8)
    p.add_argument('--epochs',   type=int,    default=50)
    p.add_argument('--batch',    type=int,    default=256)
    p.add_argument('--lr',       type=float,  default=1e-4)
    p.add_argument('--mask',     type=float,  default=0.3)
    p.add_argument('--chk_dir',  type=str,    default='checkpoints')
    args = p.parse_args()

    device, dtype = get_device()
    seed_everything()

    print('Downloading dataset...')
    repo_path = snapshot_download(args.repo, token=args.hf_token, repo_type='dataset')
    # Gather CSVs
    csv_paths = []
    for root, _, files in os.walk(repo_path):
        csv_paths += [os.path.join(root, f) for f in files if f.endswith('.csv')]

    # Load series
    series = []
    for pth in tqdm(csv_paths, desc='CSV'):
        import pandas as pd
        df = pd.read_csv(pth).select_dtypes(include=['number'])
        arr = torch.tensor(df.values, dtype=torch.float32)
        if arr.dim() == 1: arr = arr.unsqueeze(-1)
        series.append(arr)
    n_ch = max(s.size(1) for s in series)

    ds = MaskedPatchDataset(series, seq_len=args.seq_len,
                            patch_len=args.patch_len,
                            mask_ratio=args.mask)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4)

    model = TSFoundationTransformer(n_channels=n_ch,
                                    seq_len=args.seq_len,
                                    patch_len=args.patch_len).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    os.makedirs(args.chk_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train(); epoch_loss = 0
        for x, tgt, m in tqdm(dl, desc=f'E{epoch}/{args.epochs}'):
            x, tgt, m = x.to(device), tgt.to(device), m.to(device)
            pred = model(x, m)
            loss = F.mse_loss(pred[m], tgt[m])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch} loss: {epoch_loss/len(dl):.4f}')
        torch.save(model.state_dict(),
                   os.path.join(args.chk_dir, f'model_epoch_{epoch}.pt'))

if __name__ == '__main__':
    main()
