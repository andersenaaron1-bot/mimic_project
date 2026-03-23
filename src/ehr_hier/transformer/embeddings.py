import torch
import torch.nn as nn
import math


class ContinuousRotaryPositionalEmbedding(nn.Module):
    """
    Continuous Rotary Positional Embeddings (cRoPE).
    Unlike standard RoPE which uses integer positions (0, 1, 2...),
    this uses continuous time values (0.0, 0.5, 12.2...) to drive the rotation.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim

        dim_t = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (max_period ** (dim_t / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Query or Key tensor of shape (Batch*, Seq, Dim)
            times: Float tensor of shape (Batch*, Seq) representing cumulative time.
        """
        sinusoid_inp = torch.einsum("bs,d->bsd", times, self.inv_freq)  # (B*, S, D/2)
        sin = torch.repeat_interleave(sinusoid_inp.sin(), 2, dim=-1)
        cos = torch.repeat_interleave(sinusoid_inp.cos(), 2, dim=-1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)


class TimeEmbedding(nn.Module):
    """
    Additive time embedding for irregular timestamps.

    Intended use:
      - Compute a normalized scalar time feature from hours:
            t' = log1p(clamp(t_hours, 0, max_hours)) / log1p(max_hours)
      - Project t' -> d_model via a small MLP and add to token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        *,
        max_hours: float = 28.0 * 24.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_hours = float(max_hours)
        self._denom = float(math.log1p(self.max_hours)) if self.max_hours > 0 else 1.0

        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, t_hours: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hours: float tensor of shape (...,) in hours.
        Returns:
            emb: float tensor of shape (..., d_model)
        """
        if t_hours.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            t_hours = t_hours.to(dtype=torch.float32)

        t = t_hours.clamp(min=0.0, max=self.max_hours)
        t = torch.log1p(t) / self._denom
        emb = self.mlp(t.unsqueeze(-1))
        emb = self.dropout(emb)
        return emb


class AETEmbeddings(nn.Module):
    """
    The Input Adapter.
    Fuses the discrete Token Identity with the continuous Numeric Side-Channel.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.1,
        *,
        num_window_types: int = 0,
        special_type_id: int = 0,
        exclude_special_from_window_type: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.special_type_id = int(special_type_id)
        self.exclude_special_from_window_type = bool(exclude_special_from_window_type)
        self.window_type_embedding = (
            nn.Embedding(int(num_window_types), d_model) if int(num_window_types) > 0 else None
        )

        # Side-Channel Encoder
        # Projects scalar "value" (e.g., log1p dosage) to vector space
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh()  # Tanh helps scale values to match embedding distribution
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids,
        numeric_values,
        *,
        numeric_mask=None,
        window_type_ids=None,
        token_type_ids=None,
    ):
        """
        Args:
            input_ids: (Batch, Windows, Len) or (Batch, Windows, Chunks, Len) LongTensor
            numeric_values: matching float tensor with trailing singleton channel.
                            Note: Must be 0.0 for tokens without values!
            numeric_mask: Optional mask matching input_ids without trailing channel.
                          When provided, value projections are applied only where mask==1.
            window_type_ids: Optional (Batch, Num_Windows) LongTensor of per-window type ids.
            token_type_ids: Optional (Batch, Num_Windows, Seq) LongTensor of TokenCategory ids.
        """
        # 1. Embed Identity
        x = self.token_embedding(input_ids)

        # 2. Embed Value (Side Channel)
        # numeric_values is (..., 1)
        val_emb = self.value_encoder(numeric_values)
        if numeric_mask is None:
            numeric_mask = numeric_values.ne(0).any(dim=-1)
        else:
            numeric_mask = numeric_mask.to(dtype=torch.bool)
        val_emb = val_emb * numeric_mask.unsqueeze(-1).to(dtype=val_emb.dtype)

        # 3. Fuse
        # x = Identity + masked Value
        x = x + val_emb

        # 4. Window type segment embedding (optional)
        if self.window_type_embedding is not None and window_type_ids is not None:
            win_emb = self.window_type_embedding(window_type_ids)
            if input_ids.ndim == 3:
                win_emb = win_emb.unsqueeze(2)  # (B, W, 1, D)
            elif input_ids.ndim == 4:
                win_emb = win_emb.unsqueeze(2).unsqueeze(3)  # (B, W, 1, 1, D)
            else:
                raise ValueError(f"input_ids must be 3D or 4D, got shape {tuple(input_ids.shape)}")
            if self.exclude_special_from_window_type and token_type_ids is not None:
                mask = (token_type_ids != self.special_type_id).unsqueeze(-1)
                x = x + win_emb * mask
            else:
                x = x + win_emb

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
