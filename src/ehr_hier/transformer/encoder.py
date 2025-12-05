import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AETCausalAttention(nn.Module):
    """
    Custom Multi-Head Attention that supports Continuous RoPE (cRoPE).
    Enforces Causal Masking (Auto-regressive) + Padding Masking.
    """

    def __init__(self, d_model, num_heads, rope_module, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.rope = rope_module  # Instance of ContinuousRotaryPositionalEmbedding

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, times, attention_mask=None):
        """
        Args:
            x: (Batch, Seq_Len, d_model)
            times: (Batch, Seq_Len) - Cumulative float times for RoPE
            attention_mask: (Batch, Seq_Len) - 1 for Real, 0 for Pad
        """
        B, S, D = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, S, d_head)
        k = self.k_proj(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        # 2. Apply cRoPE (Rotation)

        q_rot = self.rope(q.transpose(1, 2).reshape(B, S, -1), times)  # (B, S, D)
        k_rot = self.rope(k.transpose(1, 2).reshape(B, S, -1), times)

        # Reshape back to heads
        q = q_rot.view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, S, d_h)
        k = k_rot.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # (B, H, S, S)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 4. Masking
        # A. Causal Mask (Upper Triangular = -inf)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(causal_mask, float('-inf'))

        # B. Padding Mask (if provided)
        if attention_mask is not None:
            # attention_mask is (B, S). We need (B, 1, 1, S) -> (B, 1, 1, S)
            # We mask columns (keys) that are padding.
            # Mask logic: 0 is Pad. fill -inf where mask is 0.
            extended_mask = attention_mask[:, None, None, :]  # (B, 1, 1, S)
            attn_weights = attn_weights.masked_fill(extended_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Output
        out = torch.matmul(attn_weights, v)  # (B, H, S, d_h)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class AETEncoderLayer(nn.Module):
    """
    Standard Transformer Block:
    Input -> LayerNorm -> CausalAttn -> Add -> LayerNorm -> FFN -> Add
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
        # Pre-Norm Architecture (Better stability)
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, times, mask)
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x


class AETLocalEncoder(nn.Module):
    """
    The "Ribs" of the architecture.
    Processes (Batch, Windows, Tokens) by flattening Batch & Windows.
    """

    def __init__(self, config, rope_module):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            AETEncoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                rope_module=rope_module,
                dropout=config.dropout
            ) for _ in range(config.num_local_layers)
        ])

        # Final Norm before passing to Global Aggregator
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, times, attention_mask):
        """
        Args:
            x: (Batch, Num_Windows, Window_Len, Dim)
            times: (Batch, Num_Windows, Window_Len)
            attention_mask: (Batch, Num_Windows, Window_Len)
        Returns:
            x: Contextualized tokens (Same Shape)
            window_summaries: (Batch, Num_Windows, Dim) - The [CLS] tokens
        """
        B, W, L, D = x.shape

        # 1. Flatten Batch and Windows
        # We treat every window as an independent sequence
        x_flat = x.view(B * W, L, D)
        times_flat = times.view(B * W, L)
        mask_flat = attention_mask.view(B * W, L)

        # 2. Pass through Transformer Layers
        for layer in self.layers:
            x_flat = layer(x_flat, times_flat, mask_flat)

        x_flat = self.norm(x_flat)

        # 3. Unflatten
        x_out = x_flat.view(B, W, L, D)

        # 4. Extract Summaries
        # Assuming the Structural/CLS Token is ALWAYS at index 0 of the window
        # (This is enforced by our Collator/Builder)
        window_summaries = x_out[:, :, 0, :]  # (Batch, Windows, Dim)

        return x_out, window_summaries