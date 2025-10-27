import torch
import torch.nn as nn
from .layers import SimpleTransformerLayer
from .attention.masks import patient_block_mask
from .attention.local_window import local_window_mask


class EventEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, window=128):
        super().__init__()
        self.layers = nn.ModuleList([SimpleTransformerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.window = window

    def forward(self, x, pid, is_static=None):
        # Build keep-mask (B,T,T) even if SimpleTransformerLayer ignores it for now
        _mask = local_window_mask(patient_block_mask(pid, is_static), self.window)
        for layer in self.layers:
            x = layer(x, _mask)
        return x


class VisitAggregator(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, visit_id):
        # TODO: replace with true per-visit grouping; this is a stride placeholder
        B, T, D = x.shape
        xg = x * self.gate(x)
        V = max(T // 32, 1)
        visits = xg[:, ::max(T // V, 1), :][:, :V, :]
        return visits  # [B, V, D]


class VisitTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([SimpleTransformerLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, v):
        for layer in self.layers:
            v = layer(v)
        return v


class HierModel(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_layers_l0=4,
        n_layers_l1=2,
        window=128,
        use_alibi=True,  # for flashattention later; not used in this stub
    ):
        super().__init__()
        self.event_enc = EventEncoder(d_model, n_heads, n_layers_l0, window)
        self.agg = VisitAggregator(d_model)
        self.visit_tf = VisitTransformer(d_model, n_heads, n_layers_l1)
        self.head = nn.Linear(d_model, 1)  # example: mortality logit

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use forward_with_embeds(x, batch)")

    def forward_with_embeds(self, x: torch.Tensor, batch: dict) -> torch.Tensor:
        """
        x:     [B, T, D] embeddings from tokenizer encoder (Triplet/EIC).
        batch: dict with keys:
               - 'pid'           [B, T]  patient ids per token
               - 'visit_id'      [B, T]  visit/session ids per token
               - 'attention_mask'[B, T]  (optional) True for real tokens
               - 'is_static'     [B, T]  (optional) mark static tokens
        """
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            x = x.masked_fill(~attn_mask.unsqueeze(-1), 0)

        # L0: event-scale encoder (masked/self-attn over events)
        x = self.event_enc(x, pid=batch["pid"], is_static=batch.get("is_static", None))

        # Event → Visit aggregation
        v = self.agg(x, visit_id=batch["visit_id"])   # [B, V, D]

        # L1: visit-scale transformer
        v = self.visit_tf(v)                          # [B, V, D]

        # Prediction head (take last visit token by default)
        logits = self.head(v[:, -1, :]).squeeze(-1)   # [B]
        return logits
