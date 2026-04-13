import torch
import torch.nn as nn
from .encoder import AETCausalAttention  # Re-use our custom attention!
from .world_model_contract import STATE_PACKET_SLOT_ORDER, WindowStatePacket


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
        self.semantic_summary_mode = str(getattr(config, "semantic_summary_mode", "gated")).lower()
        if self.semantic_summary_mode not in {"last", "mean", "gated"}:
            raise ValueError(
                f"Unsupported semantic_summary_mode={self.semantic_summary_mode!r}; expected one of last|mean|gated"
            )
        self.semantic_summary_gate = (
            nn.Sequential(
                nn.Linear((2 * config.d_model) + 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if self.semantic_summary_mode == "gated"
            else None
        )
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
        semantic_last = chunk_states[batch_idx, win_idx, last_idx, :]
        chunk_mask_f = chunk_mask.to(dtype=chunk_states.dtype).unsqueeze(-1)
        semantic_mean = (chunk_states * chunk_mask_f).sum(dim=2) / chunk_mask_f.sum(dim=2).clamp(min=1.0)

        if self.semantic_summary_mode == "last":
            semantic_summaries = semantic_last
        elif self.semantic_summary_mode == "mean":
            semantic_summaries = semantic_mean
        else:
            assert self.semantic_summary_gate is not None
            valid = chunk_mask.to(dtype=torch.bool)
            pos_inf = torch.tensor(float("inf"), device=chunk_times.device, dtype=chunk_times.dtype)
            neg_inf = torch.tensor(float("-inf"), device=chunk_times.device, dtype=chunk_times.dtype)
            t_first = torch.where(valid, chunk_times, pos_inf).amin(dim=2)
            t_last = torch.where(valid, chunk_times, neg_inf).amax(dim=2)
            t_first = torch.where(torch.isfinite(t_first), t_first, torch.zeros_like(t_first))
            t_last = torch.where(torch.isfinite(t_last), t_last, torch.zeros_like(t_last))
            duration_h = (t_last - t_first).clamp(min=0.0)
            n_chunks = chunk_mask.to(dtype=chunk_summaries.dtype).sum(dim=2)
            meta = torch.stack([torch.log1p(n_chunks), torch.log1p(duration_h)], dim=-1)
            gate_in = torch.cat([semantic_last, semantic_mean, meta], dim=-1)
            alpha = torch.sigmoid(self.semantic_summary_gate(gate_in))
            semantic_summaries = alpha * semantic_last + (1.0 - alpha) * semantic_mean

        semantic_summaries = semantic_summaries * (chunk_mask.any(dim=2).unsqueeze(-1))
        return chunk_states, semantic_summaries


class AETWindowStatePacketBuilder(nn.Module):
    """
    Build the canonical boundary-state packet from chunk-level window states.

    This is the migration bridge from the legacy single-vector semantic summary
    to the world-model interface. The packet carries multiple slots with fixed
    functional roles, plus a temporary query summary that can be projected into
    the current global latent path.
    """

    def __init__(self, config):
        super().__init__()
        d_model = int(config.d_model)
        self.num_slots = int(len(STATE_PACKET_SLOT_ORDER))
        self.max_write_tokens = max(0, int(getattr(config, "window_packet_write_tokens", 2)))
        self.meta_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.volatility_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.intervention_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.physiology_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_proj = nn.Sequential(
            nn.Linear((self.num_slots + 1) * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.write_score_head = (
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            if self.max_write_tokens > 0
            else None
        )

    @staticmethod
    def _first_last_indices(chunk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = chunk_mask.to(dtype=torch.bool)
        counts = valid.to(dtype=torch.long).sum(dim=2).clamp(min=1)
        last_idx = (counts - 1).clamp(min=0)
        chunk_ids = torch.arange(chunk_mask.shape[2], device=chunk_mask.device).view(1, 1, -1)
        first_idx = torch.where(
            valid,
            chunk_ids,
            torch.full_like(chunk_ids, fill_value=chunk_mask.shape[2]),
        ).amin(dim=2)
        first_idx = first_idx.clamp(min=0, max=max(0, int(chunk_mask.shape[2]) - 1))
        return first_idx, last_idx

    @staticmethod
    def _gather_chunk_states(chunk_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            1,
            chunk_states.shape[-1],
        )
        return torch.gather(chunk_states, dim=2, index=gather_index).squeeze(2)

    def forward(
        self,
        *,
        base_summary: torch.Tensor,
        chunk_states: torch.Tensor,
        chunk_mask: torch.Tensor,
        chunk_start_offsets: torch.Tensor,
        window_start_times: torch.Tensor,
        semantic_duration_hours: torch.Tensor,
        chunk_token_counts: torch.Tensor,
        chunk_duration_hours: torch.Tensor,
        window_mask: torch.Tensor | None = None,
        window_type_ids: torch.Tensor | None = None,
    ) -> WindowStatePacket:
        if chunk_states.ndim != 4:
            raise ValueError(
                f"chunk_states must be (B,W,C,D), got shape {tuple(chunk_states.shape)}"
            )
        if base_summary.shape != chunk_states.shape[:2] + (chunk_states.shape[-1],):
            raise ValueError(
                "base_summary must match chunk_states on (B,W,D); "
                f"got {tuple(base_summary.shape)} vs {(chunk_states.shape[0], chunk_states.shape[1], chunk_states.shape[-1])}"
            )
        B, W, C, D = chunk_states.shape
        device = chunk_states.device
        valid_window = (
            window_mask.to(dtype=torch.bool)
            if window_mask is not None
            else chunk_mask.any(dim=2).to(dtype=torch.bool)
        )
        valid_chunks = chunk_mask.to(dtype=torch.bool)
        chunk_mask_f = valid_chunks.to(dtype=chunk_states.dtype).unsqueeze(-1)
        denom = chunk_mask_f.sum(dim=2).clamp(min=1.0)

        first_idx, last_idx = self._first_last_indices(chunk_mask)
        first_chunk = self._gather_chunk_states(chunk_states, first_idx)
        end_chunk = self._gather_chunk_states(chunk_states, last_idx)
        mean_chunk = (chunk_states * chunk_mask_f).sum(dim=2) / denom
        delta_raw = end_chunk - first_chunk
        deviation = (chunk_states - mean_chunk.unsqueeze(2)).abs() * chunk_mask_f
        volatility_raw = deviation.sum(dim=2) / denom

        prev_window_end = window_start_times + semantic_duration_hours.to(dtype=window_start_times.dtype)
        prev_window_end_shift = torch.zeros_like(prev_window_end)
        if W > 1:
            prev_window_end_shift[:, 1:] = prev_window_end[:, :-1]
        gap_prev_h = (window_start_times - prev_window_end_shift).clamp(min=0.0)

        n_chunks = valid_chunks.to(dtype=chunk_states.dtype).sum(dim=2)
        mean_chunk_tokens = chunk_token_counts.to(dtype=chunk_states.dtype).sum(dim=2) / n_chunks.clamp(min=1.0)
        mean_chunk_duration = chunk_duration_hours.to(dtype=chunk_states.dtype).sum(dim=2) / n_chunks.clamp(min=1.0)
        meta = torch.stack(
            [
                torch.log1p(semantic_duration_hours.to(dtype=chunk_states.dtype).clamp(min=0.0)),
                torch.log1p(gap_prev_h.to(dtype=chunk_states.dtype).clamp(min=0.0)),
                torch.log1p(mean_chunk_tokens.clamp(min=0.0)),
                torch.log1p(mean_chunk_duration.clamp(min=0.0)),
            ],
            dim=-1,
        )
        meta_emb = self.meta_proj(meta)

        end_token = end_chunk
        burden_token = base_summary
        delta_token = self.delta_proj(torch.cat([delta_raw, meta_emb], dim=-1))
        volatility_token = self.volatility_proj(torch.cat([volatility_raw, meta_emb], dim=-1))
        intervention_token = self.intervention_proj(
            torch.cat([end_token, burden_token, meta_emb], dim=-1)
        )
        physiology_token = self.physiology_proj(
            torch.cat([mean_chunk, delta_token, volatility_token, meta_emb], dim=-1)
        )

        slot_tokens = torch.stack(
            [
                end_token,
                burden_token,
                delta_token,
                volatility_token,
                intervention_token,
                physiology_token,
            ],
            dim=2,
        )
        slot_mask = valid_window.unsqueeze(-1).expand(B, W, self.num_slots)
        slot_tokens = slot_tokens * slot_mask.unsqueeze(-1).to(dtype=slot_tokens.dtype)

        query_input = torch.cat(
            [
                slot_tokens.reshape(B, W, self.num_slots * D),
                meta_emb,
            ],
            dim=-1,
        )
        query_token = self.query_proj(query_input)
        query_token = query_token * valid_window.unsqueeze(-1).to(dtype=query_token.dtype)

        write_tokens = None
        write_mask = None
        if self.write_score_head is not None and self.max_write_tokens > 0 and C > 0:
            write_meta = meta_emb.unsqueeze(2).expand(B, W, C, D)
            write_scores = self.write_score_head(
                torch.cat([chunk_states, write_meta], dim=-1)
            ).squeeze(-1)
            write_scores = write_scores.masked_fill(~valid_chunks, float("-inf"))
            top_k = min(int(self.max_write_tokens), int(C))
            top_scores, top_idx = torch.topk(write_scores, k=top_k, dim=2)
            safe_top_idx = top_idx.clamp(min=0, max=max(0, C - 1))
            write_tokens = torch.gather(
                chunk_states,
                dim=2,
                index=safe_top_idx.unsqueeze(-1).expand(-1, -1, -1, D),
            )
            write_mask = torch.isfinite(top_scores) & valid_window.unsqueeze(-1)
            write_tokens = write_tokens * write_mask.unsqueeze(-1).to(dtype=write_tokens.dtype)

        return WindowStatePacket(
            slot_tokens=slot_tokens,
            slot_mask=slot_mask,
            query_token=query_token,
            write_tokens=write_tokens,
            write_mask=write_mask,
            window_type_ids=window_type_ids,
            start_hours=window_start_times,
            duration_hours=semantic_duration_hours,
            gap_prev_hours=gap_prev_h,
        )


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
