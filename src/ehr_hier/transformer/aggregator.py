import torch
import torch.nn as nn
from .encoder import AETCausalAttention  # Re-use our custom attention!


class AETIntraWindowAggregator(nn.Module):
    """
    Models chunk-to-chunk dynamics within a semantic window.

    Input/Output contract:
      - input chunk summaries: (B, W, C, D)
      - chunk times: (B, W, C) relative to semantic-window start
      - chunk mask: (B, W, C)
      - output chunk states: (B, W, C, D)
      - output semantic summaries: (B, W, D), taken from the last real chunk state
    """

    def __init__(self, config, rope_module):
        super().__init__()
        num_layers = int(getattr(config, "num_chunk_layers", 1))
        self.layers = nn.ModuleList([
            AETGlobalLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                rope_module=rope_module,
                dropout=config.dropout,
                enable_alibi_hours_bias=bool(getattr(config, "enable_alibi_hours_bias", False)),
                alibi_hours_max=float(getattr(config, "alibi_hours_max", 28.0 * 24.0)),
                alibi_hours_slope_scale=float(getattr(config, "alibi_hours_slope_scale", 1.0)),
            ) for _ in range(max(0, num_layers))
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, chunk_summaries, chunk_times, chunk_mask):
        if chunk_summaries.ndim != 4:
            raise ValueError(
                f"chunk_summaries must be (B,W,C,D), got shape {tuple(chunk_summaries.shape)}"
            )
        B, W, C, D = chunk_summaries.shape
        if C == 0:
            return chunk_summaries, chunk_summaries.new_zeros((B, W, D))

        x = chunk_summaries.view(B * W, C, D)
        times = chunk_times.view(B * W, C)
        mask = chunk_mask.view(B * W, C)

        for layer in self.layers:
            x = layer(x, times, mask)

        x = self.norm(x)
        x = x * mask.unsqueeze(-1)

        chunk_states = x.view(B, W, C, D)
        n_real = chunk_mask.to(dtype=torch.long).sum(dim=2).clamp(min=1)
        last_idx = (n_real - 1).clamp(min=0)
        batch_idx = torch.arange(B, device=chunk_summaries.device)[:, None]
        win_idx = torch.arange(W, device=chunk_summaries.device)[None, :]
        semantic_summaries = chunk_states[batch_idx, win_idx, last_idx, :]
        semantic_summaries = semantic_summaries * (chunk_mask.any(dim=2).unsqueeze(-1))
        return chunk_states, semantic_summaries


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
                dropout=config.dropout,
                enable_alibi_hours_bias=bool(getattr(config, "enable_alibi_hours_bias", False)),
                alibi_hours_max=float(getattr(config, "alibi_hours_max", 28.0 * 24.0)),
                alibi_hours_slope_scale=float(getattr(config, "alibi_hours_slope_scale", 1.0)),
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
        # window_times represents the "Macro Clock"

        for layer in self.layers:
            x = layer(x, window_times, padding_mask)

        x = self.norm(x)
        x = x * padding_mask.unsqueeze(-1)

        return x


class AETGlobalLayer(nn.Module):
    """
    Identical to Local Layer, just naming separation for clarity.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        rope_module,
        dropout=0.1,
        *,
        enable_alibi_hours_bias: bool = False,
        alibi_hours_max: float = 28.0 * 24.0,
        alibi_hours_slope_scale: float = 1.0,
    ):
        super().__init__()
        self.attn = AETCausalAttention(
            d_model,
            num_heads,
            rope_module,
            dropout,
            enable_alibi_hours_bias=enable_alibi_hours_bias,
            alibi_hours_max=alibi_hours_max,
            alibi_hours_slope_scale=alibi_hours_slope_scale,
        )
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
