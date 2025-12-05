import torch
import torch.nn as nn
from .encoder import AETCausalAttention  # Re-use our custom attention!


class AETGlobalAggregator(nn.Module):
    """
    The "Spine" of the architecture.
    Models the trajectory of Window Summaries.
    """

    def __init__(self, config, rope_module):
        super().__init__()
        self.config = config

        # We reuse the same EncoderLayer structure but apply it globally
        # (Standard GPT-style causal modeling)
        self.layers = nn.ModuleList([
            AETGlobalLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                rope_module=rope_module,
                dropout=config.dropout
            ) for _ in range(config.num_global_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, window_summaries, window_times, padding_mask, prev_context_state=None):
        """
        Args:
            window_summaries: (Batch, Num_Windows, Dim) - From Local Encoder
            window_times: (Batch, Num_Windows) - Float time of each window start
            padding_mask: (Batch, Num_Windows) - 1 for Real, 0 for Pad
            prev_context_state: (Batch, Dim) - Optional G_final from previous sequence
        """
        x = window_summaries

        # --- 1. History Injection ---
        if prev_context_state is not None:
            # seed window 0 only where padding_mask is valid to avoid in-place pad pollution
            valid = (padding_mask[:, 0] > 0).unsqueeze(-1)
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + prev_context_state * valid

        # --- 2. Transformer Layers ---
        # Note: We use the same cRoPE module.
        # window_times represents the "Macro Clock" (When did this window happen?)

        for layer in self.layers:
            x = layer(x, window_times, padding_mask)

        x = self.norm(x)

        return x


class AETGlobalLayer(nn.Module):
    """
    Identical to Local Layer, just naming separation for clarity.
    """

    def __init__(self, d_model, num_heads, d_ff, rope_module, dropout=0.1):
        super().__init__()
        self.attn = AETCausalAttention(d_model, num_heads, rope_module, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, times, mask=None):
        x = x + self.attn(self.norm1(x), times, mask)
        x = x + self.ffn(self.norm2(x))
        return x
