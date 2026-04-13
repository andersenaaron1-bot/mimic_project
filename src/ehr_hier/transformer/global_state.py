import torch
import torch.nn as nn
import torch.nn.functional as F


class AETLatentHealthState(nn.Module):
    """
    Persistent latent state over semantic windows.

    This is the state-only WP5 step: a gap-aware recurrent state update that
    consumes one semantic-window summary at a time and returns one latent health
    state per window. It is designed as a clean precursor to a future Mamba-like
    state path, while already enforcing the intended persistent-state contract.
    """

    def __init__(self, config) -> None:
        super().__init__()
        d_model = int(config.d_model)
        dropout = float(getattr(config, "dropout", 0.0))
        self.num_window_types = max(0, int(getattr(config, "num_window_types", 0)))
        self.use_window_type = bool(getattr(config, "latent_state_use_window_type", True))
        self.use_absolute_time = bool(getattr(config, "latent_state_use_absolute_time", True))

        self.window_type_embedding = (
            nn.Embedding(self.num_window_types, d_model)
            if self.use_window_type and self.num_window_types > 0
            else None
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.decay_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update_head = nn.Linear(2 * d_model, d_model)
        self.reset_head = nn.Linear(2 * d_model, d_model)
        self.candidate_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _derive_gap_prev_hours(
        *,
        window_start_times: torch.Tensor,
        semantic_duration_hours: torch.Tensor | None,
    ) -> torch.Tensor:
        if semantic_duration_hours is None:
            semantic_duration_hours = torch.zeros_like(window_start_times)
        prev_window_end = window_start_times + semantic_duration_hours.to(dtype=window_start_times.dtype)
        prev_window_end_shift = torch.zeros_like(prev_window_end)
        if window_start_times.shape[1] > 1:
            prev_window_end_shift[:, 1:] = prev_window_end[:, :-1]
        return (window_start_times - prev_window_end_shift).clamp(min=0.0)

    def forward(
        self,
        *,
        window_summaries: torch.Tensor,
        window_start_times: torch.Tensor,
        padding_mask: torch.Tensor,
        semantic_duration_hours: torch.Tensor | None = None,
        window_type_ids: torch.Tensor | None = None,
        prev_context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if window_summaries.ndim != 3:
            raise ValueError(
                f"window_summaries must be (B,W,D), got shape {tuple(window_summaries.shape)}"
            )
        if window_start_times.ndim != 2 or window_start_times.shape[:2] != window_summaries.shape[:2]:
            raise ValueError(
                "window_start_times must be (B,W) aligned with window_summaries; "
                f"got {tuple(window_start_times.shape)} vs {tuple(window_summaries.shape[:2])}"
            )
        if padding_mask.ndim != 2 or padding_mask.shape != window_start_times.shape:
            raise ValueError(
                f"padding_mask must be (B,W) aligned with window_start_times, got {tuple(padding_mask.shape)}"
            )

        B, W, D = window_summaries.shape
        if W == 0:
            return window_summaries.new_zeros((B, W, D))

        if semantic_duration_hours is None:
            semantic_duration_hours = torch.zeros_like(window_start_times)
        elif semantic_duration_hours.shape != window_start_times.shape:
            raise ValueError(
                "semantic_duration_hours must match window_start_times; "
                f"got {tuple(semantic_duration_hours.shape)} vs {tuple(window_start_times.shape)}"
            )

        gap_prev_h = self._derive_gap_prev_hours(
            window_start_times=window_start_times,
            semantic_duration_hours=semantic_duration_hours,
        )
        if not self.use_absolute_time:
            abs_start_h = torch.zeros_like(window_start_times)
        else:
            abs_start_h = window_start_times.clamp(min=0.0)

        prev_h = (
            prev_context_state
            if prev_context_state is not None
            else window_summaries.new_zeros((B, D))
        )
        if prev_h.shape != (B, D):
            raise ValueError(
                f"prev_context_state must be (B,D) aligned with window_summaries, got {tuple(prev_h.shape)}"
            )

        outputs: list[torch.Tensor] = []
        for t in range(W):
            valid = padding_mask[:, t].to(dtype=torch.bool).unsqueeze(-1)
            summary_t = window_summaries[:, t, :]
            meta_t = torch.stack(
                [
                    torch.log1p(gap_prev_h[:, t].clamp(min=0.0)),
                    torch.log1p(semantic_duration_hours[:, t].clamp(min=0.0)),
                    torch.log1p(abs_start_h[:, t].clamp(min=0.0)),
                ],
                dim=-1,
            )
            meta_emb = self.meta_proj(meta_t)
            if self.window_type_embedding is not None and window_type_ids is not None:
                safe_type_ids = window_type_ids[:, t].clamp(min=0, max=max(0, self.num_window_types - 1))
                type_emb = self.window_type_embedding(safe_type_ids)
            else:
                type_emb = torch.zeros_like(summary_t)

            input_emb = self.input_proj(torch.cat([summary_t, meta_emb, type_emb], dim=-1))

            decay_in = torch.cat([prev_h, input_emb], dim=-1)
            decay_rate = F.softplus(self.decay_head(decay_in))
            gap_scale = torch.log1p(gap_prev_h[:, t].clamp(min=0.0)).unsqueeze(-1).to(dtype=prev_h.dtype)
            prev_h_decayed = prev_h * torch.exp(-decay_rate * gap_scale)

            gate_in = torch.cat([prev_h_decayed, input_emb], dim=-1)
            update = torch.sigmoid(self.update_head(gate_in))
            reset = torch.sigmoid(self.reset_head(gate_in))
            candidate = torch.tanh(
                self.candidate_head(torch.cat([reset * prev_h_decayed, input_emb], dim=-1))
            )
            new_h = self.state_norm((1.0 - update) * prev_h_decayed + update * candidate)
            new_h = self.dropout(new_h)

            out_t = self.output_norm(new_h + self.output_proj(torch.cat([new_h, input_emb], dim=-1)))
            out_t = self.dropout(out_t)

            prev_h = torch.where(valid, new_h, prev_h)
            outputs.append(torch.where(valid, out_t, torch.zeros_like(out_t)))

        return torch.stack(outputs, dim=1)
