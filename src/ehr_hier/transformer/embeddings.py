import torch
import torch.nn as nn
import math


class ContinuousRotaryPositionalEmbedding(nn.Module):
    """
    Continuous Rotary Positional Embeddings (cRoPE).
    Unlike standard RoPE which uses integer positions (0, 1, 2...),
    this uses continuous time values (0.0, 0.5, 12.2...) to drive the rotation.
    """

    def __init__(self, d_model, max_period=10000.0):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // 2  # RoPE usually applied to half the dim or pairs

        # Precompute frequencies: 1 / (10000 ^ (2i / d))
        # We only compute half because sin/cos pairs share frequency
        dim_t = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        inv_freq = 1.0 / (max_period ** (dim_t / self.d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, times):
        """
        Args:
            x: Query or Key tensor of shape (Batch, Seq_Len, Dim)
            times: Float tensor of shape (Batch, Seq_Len) representing cumulative time.
        """
        # 1. Compute angles: time * freq
        # times: (B, S), inv_freq: (D/2) -> (B, S, D/2)
        sinusoid_inp = torch.einsum("bs,d->bsd", times, self.inv_freq)

        # 2. Create sin/cos
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()

        # 3. Repeat to match full dimension (D) instead of (D/2)
        # We interleave: [sin1, sin1, sin2, sin2...] to match the rotation logic
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        cos = torch.repeat_interleave(cos, 2, dim=-1)

        # 4. Apply rotation
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)


class AETEmbeddings(nn.Module):
    """
    The Input Adapter.
    Fuses the discrete Token Identity with the continuous Numeric Side-Channel.
    """

    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Side-Channel Encoder
        # Projects scalar "value" (e.g., log1p dosage) to vector space
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh()  # Tanh helps scale values to match embedding distribution
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, numeric_values):
        """
        Args:
            input_ids: (Batch, Seq) LongTensor
            numeric_values: (Batch, Seq, 1) FloatTensor.
                            Note: Must be 0.0 for tokens without values!
        """
        # 1. Embed Identity
        x = self.token_embedding(input_ids)

        # 2. Embed Value (Side Channel)
        # numeric_values is (B, S, 1)
        val_emb = self.value_encoder(numeric_values)

        # 3. Fuse
        # x = Identity + Value
        # For tokens where value=0, val_emb should be close to 0 vector (due to Linear bias init)
        # Ideally, Linear bias should be 0 init, or use Masking if strict 0 is needed.
        x = x + val_emb

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x