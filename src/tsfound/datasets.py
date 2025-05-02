"""Dataset and data‑loading utilities."""
import torch
from torch.utils.data import Dataset

class MaskedPatchDataset(Dataset):
    """Generate sliding windows and apply BERT‑style random masking."""
    def __init__(self, series_list, seq_len=512, patch_len=8,
                 mask_ratio=0.3, stride=1, scale=True):
        self.windows = []
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        for arr in series_list:
            # per‑series z‑score
            if scale:
                mean = arr.mean(dim=0, keepdim=True)
                std  = arr.std(dim=0, keepdim=True) + 1e-6
                arr  = (arr - mean) / std
            for s in range(0, arr.size(0) - seq_len + 1, stride):
                self.windows.append(arr[s:s+seq_len])
        self.num_patches = seq_len // patch_len

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        seq   = self.windows[idx]
        # patchify
        patches = seq.unfold(0, self.patch_len, self.patch_len)
        patches = patches.contiguous().view(self.num_patches, -1)
        # random mask
        mask = torch.rand(self.num_patches) < self.mask_ratio
        x = patches.clone()
        x[mask] = 0
        return x, patches, mask
