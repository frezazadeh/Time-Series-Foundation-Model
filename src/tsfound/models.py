"""Transformer backbones and heads."""
import torch, torch.nn as nn, torch.nn.functional as F

class TSFoundationTransformer(nn.Module):
    """Masked‑patch encoder‑decoder (pre‑training)."""
    def __init__(self, n_channels, seq_len=512, patch_len=8,
                 embed=384, layers=7, heads=6, dropout=0.05):
        super().__init__()
        self.patch_dim = patch_len * n_channels
        self.embed     = nn.Linear(self.patch_dim, embed)
        self.mask_tok  = nn.Parameter(torch.zeros(1, embed))
        self.num_p     = seq_len // patch_len
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_p, embed))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed, heads,
                                       dim_feedforward=6*embed,
                                       dropout=dropout,
                                       batch_first=True,
                                       activation='gelu')
            for _ in range(layers)
        ])
        self.ln_final = nn.LayerNorm(embed)
        self.decoder  = nn.Linear(embed, self.patch_dim)

    def forward(self, x, mask):
        x = self.embed(x)
        B, M, E = x.shape
        mask = mask.unsqueeze(-1)
        x = torch.where(mask, self.mask_tok, x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        return self.decoder(x)

class TSBackboneFeature(nn.Module):
    """Feature extractor for fine‑tuning (no decoder)."""
    def __init__(self, n_channels, seq_len=32, patch_len=8,
                 embed=384, layers=7, heads=6, dropout=0.05):
        super().__init__()
        self.patch_dim = patch_len * n_channels
        self.embed     = nn.Linear(self.patch_dim, embed)
        self.num_p     = seq_len // patch_len
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_p, embed))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed, heads,
                                       dim_feedforward=6*embed,
                                       dropout=dropout,
                                       batch_first=True,
                                       activation='gelu')
            for _ in range(layers)
        ])
        self.ln_final = nn.LayerNorm(embed)

    def forward(self, x):
        patches = x.unfold(1, self.num_p, self.num_p)
        patches = patches.contiguous().view(x.size(0), -1, self.patch_dim)
        x = self.embed(patches) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.ln_final(x)

class SupervisedTSModel(nn.Module):
    def __init__(self, backbone: TSBackboneFeature, out_dim=1):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(backbone.ln_final.normalized_shape[0], out_dim)

    def forward(self, x):
        feats = self.backbone(x).mean(1)
        return self.head(feats)
