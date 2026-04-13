from __future__ import annotations

import torch
import torch.nn as nn

from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_ORDER


class AETEventComposer(nn.Module):
    """
    Compose one hidden state per EventFrame from its token bundle plus event metadata.

    The local encoder can then model event trajectories, while token-level heads remain
    available by scattering event states back onto the bundle-token lattice.
    """

    def __init__(
        self,
        *,
        d_model: int,
        dropout: float = 0.1,
        num_payload_types: int | None = None,
        num_token_types: int = 8,
        max_bundle_slots: int = 32,
        numeric_value_transform: str = "signed_log1p",
    ) -> None:
        super().__init__()
        self.max_bundle_slots = max(1, int(max_bundle_slots))
        self.num_token_types = max(0, int(num_token_types))
        self.numeric_value_transform = str(numeric_value_transform).strip().lower() or "identity"
        if self.numeric_value_transform not in {"identity", "signed_log1p"}:
            raise ValueError(
                f"Unsupported numeric_value_transform={self.numeric_value_transform!r}; expected identity|signed_log1p"
            )

        payload_count = int(num_payload_types or len(EVENT_PAYLOAD_KIND_ORDER))
        self.slot_embedding = nn.Embedding(self.max_bundle_slots, d_model)
        self.payload_embedding = nn.Embedding(payload_count, d_model)
        self.event_type_embedding = (
            nn.Embedding(self.num_token_types, d_model) if self.num_token_types > 0 else None
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))

    def _transform_numeric_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.numeric_value_transform == "identity":
            return values
        if values.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            values = values.to(dtype=torch.float32)
        return values.sign() * torch.log1p(values.abs())

    def forward(
        self,
        token_embeddings: torch.Tensor,
        *,
        token_event_index: torch.Tensor,
        token_event_slot_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_type_ids: torch.Tensor | None = None,
        event_payload_ids: torch.Tensor | None = None,
        event_numeric_values: torch.Tensor | None = None,
        event_numeric_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_embeddings.ndim != 5:
            raise ValueError(
                f"token_embeddings must be (B,W,C,L,D), got shape {tuple(token_embeddings.shape)}"
            )
        B, W, C, L, D = token_embeddings.shape
        if token_event_index.shape != (B, W, C, L):
            raise ValueError(
                "token_event_index must match token_embeddings on (B,W,C,L); "
                f"got {tuple(token_event_index.shape)} vs {(B, W, C, L)}"
            )
        if token_event_slot_ids.shape != (B, W, C, L):
            raise ValueError(
                "token_event_slot_ids must match token_embeddings on (B,W,C,L); "
                f"got {tuple(token_event_slot_ids.shape)} vs {(B, W, C, L)}"
            )
        if attention_mask.shape != (B, W, C, L):
            raise ValueError(
                f"attention_mask must be (B,W,C,L), got shape {tuple(attention_mask.shape)}"
            )
        if event_attention_mask.ndim != 4 or event_attention_mask.shape[:3] != (B, W, C):
            raise ValueError(
                "event_attention_mask must be (B,W,C,E); "
                f"got {tuple(event_attention_mask.shape)}"
            )

        E = int(event_attention_mask.shape[-1])
        if E == 0:
            return token_embeddings.new_zeros((B, W, C, 0, D))

        flat_tokens = token_embeddings.reshape(B * W * C, L, D)
        flat_event_index = token_event_index.reshape(B * W * C, L)
        flat_slot_ids = token_event_slot_ids.reshape(B * W * C, L).clamp(min=0, max=self.max_bundle_slots - 1)
        flat_token_mask = (
            attention_mask.reshape(B * W * C, L).to(dtype=torch.bool)
            & flat_event_index.ge(0)
        )

        token_repr = flat_tokens + self.slot_embedding(flat_slot_ids)
        token_repr = token_repr * flat_token_mask.unsqueeze(-1).to(dtype=token_repr.dtype)

        safe_event_index = flat_event_index.clamp(min=0, max=max(0, E - 1))
        scatter_index = safe_event_index.unsqueeze(-1).expand(-1, -1, D)
        event_sums = token_repr.new_zeros((B * W * C, E, D))
        event_sums.scatter_add_(dim=1, index=scatter_index, src=token_repr)

        counts = token_repr.new_zeros((B * W * C, E, 1))
        counts.scatter_add_(
            dim=1,
            index=safe_event_index.unsqueeze(-1),
            src=flat_token_mask.unsqueeze(-1).to(dtype=token_repr.dtype),
        )
        event_repr = event_sums / counts.clamp(min=1.0)
        event_repr = event_repr.reshape(B, W, C, E, D)

        if event_type_ids is not None and self.event_type_embedding is not None:
            safe_event_type_ids = event_type_ids.clamp(min=0, max=max(0, self.num_token_types - 1))
            event_repr = event_repr + self.event_type_embedding(safe_event_type_ids)

        if event_payload_ids is not None:
            safe_payload_ids = event_payload_ids.clamp(
                min=0,
                max=max(0, int(self.payload_embedding.num_embeddings) - 1),
            )
            event_repr = event_repr + self.payload_embedding(safe_payload_ids)

        if event_numeric_values is not None:
            numeric_values = self._transform_numeric_values(event_numeric_values)
            numeric_repr = self.numeric_encoder(numeric_values)
            numeric_valid = (
                event_numeric_mask.to(dtype=torch.bool)
                if event_numeric_mask is not None
                else event_numeric_values.squeeze(-1).ne(0)
            )
            event_repr = event_repr + numeric_repr * numeric_valid.unsqueeze(-1).to(dtype=event_repr.dtype)

        event_repr = self.layer_norm(event_repr)
        event_repr = self.dropout(event_repr)
        event_repr = event_repr * event_attention_mask.unsqueeze(-1).to(dtype=event_repr.dtype)
        return event_repr

    @staticmethod
    def scatter_to_tokens(
        event_hidden: torch.Tensor,
        *,
        token_event_index: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if event_hidden.ndim != 5:
            raise ValueError(
                f"event_hidden must be (B,W,C,E,D), got shape {tuple(event_hidden.shape)}"
            )
        B, W, C, E, D = event_hidden.shape
        if token_event_index.ndim != 4 or token_event_index.shape[:3] != (B, W, C):
            raise ValueError(
                "token_event_index must be (B,W,C,L); "
                f"got {tuple(token_event_index.shape)}"
            )
        if attention_mask.shape != token_event_index.shape:
            raise ValueError(
                "attention_mask must match token_event_index; "
                f"got {tuple(attention_mask.shape)} vs {tuple(token_event_index.shape)}"
            )

        L = int(token_event_index.shape[-1])
        if E == 0:
            return event_hidden.new_zeros((B, W, C, L, D))

        flat_hidden = event_hidden.reshape(B * W * C, E, D)
        flat_event_index = token_event_index.reshape(B * W * C, L)
        safe_event_index = flat_event_index.clamp(min=0, max=max(0, E - 1))
        gather_index = safe_event_index.unsqueeze(-1).expand(-1, -1, D)
        token_hidden = torch.gather(flat_hidden, dim=1, index=gather_index)
        token_hidden = token_hidden.reshape(B, W, C, L, D)
        valid = (
            attention_mask.to(dtype=torch.bool)
            & token_event_index.ge(0)
        ).unsqueeze(-1)
        return token_hidden * valid.to(dtype=token_hidden.dtype)
