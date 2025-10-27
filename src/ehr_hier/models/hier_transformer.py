import torch, torch.nn as nn
from .layers import SimpleTransformerLayer
from .attention.masks import patient_block_mask
from .attention.local_window import local_window_mask

class EventEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, window=128):
        super().__init__()
        self.layers = nn.ModuleList([SimpleTransformerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.window = window
    def forward(self, x, pid, is_static=None):
        mask = patient_block_mask(pid, is_static)
        mask = local_window_mask(mask, self.window)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class VisitAggregator(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
    def forward(self, x, visit_id):
        B,T,D = x.shape
        xg = x * self.gate(x)
        V = max(T // 32, 1)
        visits = xg[:, ::max(T//V,1), :][:, :V, :]
        return visits  # [B,V,D]

class VisitTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([SimpleTransformerLayer(d_model, n_heads) for _ in range(n_layers)])
    def forward(self, v):
        for layer in self.layers:
            v = layer(v)
        return v

class HierModel(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers_l0=4, n_layers_l1=2, window=128, use_alibi=True):
        super().__init__()
        self.event_enc = EventEncoder(d_model, n_heads, n_layers_l0, window)
        self.agg = VisitAggregator(d_model)
        self.visit_tf = VisitTransformer(d_model, n_heads, n_layers_l1)
        self.head = nn.Linear(d_model, 1)  # example: mortality logit
    def forward(self, batch):
        B,T = batch["pid"].shape
        x = torch.randn(B, T, 256, device=batch["pid"].device)  # placeholder embeddings
        x = self.event_enc(x, batch["pid"])
        v = self.agg(x, batch["visit_id"])
        v = self.visit_tf(v)
        return self.head(v[:, -1, :]).squeeze(-1)
