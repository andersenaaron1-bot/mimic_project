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


class MultiScaleTimeEmbedding(nn.Module):
    """
    Additive time embedding over multiple clinical clocks.

    Intended use:
      - local hours within a chunk
      - semantic hours within a window
      - global hours within the full subject timeline

    This keeps the hierarchy explicit: dense intra-stay timing and sparse lifetime
    timing are both available to MLPs/heads, while cRoPE still controls attention.
    """

    def __init__(
        self,
        d_model: int,
        *,
        local_max_hours: float = 72.0,
        semantic_max_hours: float = 31.0 * 24.0,
        global_max_hours: float = 365.25 * 24.0 * 10.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.local_max_hours = float(local_max_hours)
        self.semantic_max_hours = float(semantic_max_hours)
        self.global_max_hours = float(global_max_hours)
        self._local_denom = float(math.log1p(self.local_max_hours)) if self.local_max_hours > 0 else 1.0
        self._semantic_denom = float(math.log1p(self.semantic_max_hours)) if self.semantic_max_hours > 0 else 1.0
        self._global_denom = float(math.log1p(self.global_max_hours)) if self.global_max_hours > 0 else 1.0

        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _coerce_float(t_hours: torch.Tensor) -> torch.Tensor:
        if t_hours.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return t_hours.to(dtype=torch.float32)
        return t_hours

    @staticmethod
    def _normalize_hours(t_hours: torch.Tensor, *, max_hours: float, denom: float) -> torch.Tensor:
        t = MultiScaleTimeEmbedding._coerce_float(t_hours).clamp(min=0.0, max=float(max_hours))
        return torch.log1p(t) / float(max(1e-6, denom))

    def forward(
        self,
        *,
        local_time_hours: torch.Tensor,
        semantic_time_hours: torch.Tensor,
        global_time_hours: torch.Tensor,
    ) -> torch.Tensor:
        local = self._normalize_hours(
            local_time_hours,
            max_hours=self.local_max_hours,
            denom=self._local_denom,
        )
        semantic = self._normalize_hours(
            semantic_time_hours,
            max_hours=self.semantic_max_hours,
            denom=self._semantic_denom,
        )
        global_t = self._normalize_hours(
            global_time_hours,
            max_hours=self.global_max_hours,
            denom=self._global_denom,
        )
        features = torch.stack([local, semantic, global_t], dim=-1)
        emb = self.mlp(features)
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
        num_token_types: int = 8,
        special_type_id: int = 0,
        exclude_special_from_window_type: bool = True,
        condition_numeric_on_token_type: bool = True,
        numeric_value_transform: str = "signed_log1p",
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.special_type_id = int(special_type_id)
        self.exclude_special_from_window_type = bool(exclude_special_from_window_type)
        self.window_type_embedding = (
            nn.Embedding(int(num_window_types), d_model) if int(num_window_types) > 0 else None
        )
        self.numeric_value_transform = str(numeric_value_transform).strip().lower() or "identity"
        if self.numeric_value_transform not in {"identity", "signed_log1p"}:
            raise ValueError(
                f"Unsupported numeric_value_transform={self.numeric_value_transform!r}; expected identity|signed_log1p"
            )

        # Side-Channel Encoder
        # Projects scalar "value" (e.g., log1p dosage) to vector space
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh()  # Tanh helps scale values to match embedding distribution
        )
        self.condition_numeric_on_token_type = bool(condition_numeric_on_token_type)
        self.num_token_types = max(0, int(num_token_types))
        if self.condition_numeric_on_token_type and self.num_token_types > 0:
            self.numeric_type_scale = nn.Embedding(self.num_token_types, d_model)
            self.numeric_type_bias = nn.Embedding(self.num_token_types, d_model)
            with torch.no_grad():
                self.numeric_type_scale.weight.fill_(1.0)
                self.numeric_type_bias.weight.zero_()
        else:
            self.numeric_type_scale = None
            self.numeric_type_bias = None

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _transform_numeric_values(self, numeric_values: torch.Tensor) -> torch.Tensor:
        if self.numeric_value_transform == "identity":
            return numeric_values
        values = numeric_values
        if values.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            values = values.to(dtype=torch.float32)
        return values.sign() * torch.log1p(values.abs())

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
        transformed_numeric_values = self._transform_numeric_values(numeric_values)
        val_emb = self.value_encoder(transformed_numeric_values)
        if numeric_mask is None:
            numeric_mask = numeric_values.ne(0).any(dim=-1)
        else:
            numeric_mask = numeric_mask.to(dtype=torch.bool)
        if (
            self.numeric_type_scale is not None
            and self.numeric_type_bias is not None
            and token_type_ids is not None
        ):
            safe_type_ids = token_type_ids.clamp(min=0, max=max(0, self.num_token_types - 1))
            val_emb = (
                val_emb * self.numeric_type_scale(safe_type_ids)
            ) + self.numeric_type_bias(safe_type_ids)
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
