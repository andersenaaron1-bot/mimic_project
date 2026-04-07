import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import AETGlobalAggregator, AETIntraWindowAggregator
from .embeddings import (
    AETEmbeddings,
    ContinuousRotaryPositionalEmbedding,
    MultiScaleTimeEmbedding,
)
from .encoder import AETLocalEncoder
from .heads import AETOutputHeads


class AdaptiveEpisodicTransformer(nn.Module):
    def __init__(self, config, vocab_config):
        super().__init__()
        self.config = config
        self.vocab_config = dict(vocab_config)

        window_markers_cfg = dict(self.vocab_config.get("window_markers", {}) or {})
        self.window_marker_type_offset = int(window_markers_cfg.get("type_token_offset", 0))
        self.window_marker_num_types = int(window_markers_cfg.get("num_types", 0))
        self.window_marker_end_token_id = window_markers_cfg.get("end_token_id", None)
        self.window_marker_end_mode = str(window_markers_cfg.get("end_mode", "end_token"))
        self.window_marker_continue_token_id = window_markers_cfg.get("continue_token_id", None)

        offsets_cfg = dict(self.vocab_config.get("offsets", {}) or {})
        self.special_token_offset = int(offsets_cfg.get("SPECIAL", 0))
        self.size_special = int(self.vocab_config.get("size_special", 0))

        num_window_types = int(
            getattr(
                config,
                "num_window_types",
                vocab_config.get("window_markers", {}).get("num_types", 16),
            )
        )
        self.num_window_types = max(0, num_window_types)
        if self.window_marker_num_types <= 0:
            self.window_marker_num_types = self.num_window_types

        self.rope = ContinuousRotaryPositionalEmbedding(
            dim=config.d_model // config.num_heads,
            max_period=config.rope_max_period,
        )

        self.embeddings = AETEmbeddings(
            vocab_size=vocab_config["total_size"],
            d_model=config.d_model,
            dropout=config.dropout,
            num_window_types=self.num_window_types,
            num_token_types=int(getattr(config, "num_token_types", 8)),
            special_type_id=int(getattr(config, "special_type_id", 0)),
            exclude_special_from_window_type=True,
            condition_numeric_on_token_type=bool(
                getattr(config, "condition_numeric_on_token_type", True)
            ),
            numeric_value_transform=str(
                getattr(config, "numeric_value_transform", "signed_log1p")
            ),
        )

        self.time_embedding: MultiScaleTimeEmbedding | None = None
        self.time_embedding_scale: nn.Parameter | None = None
        if bool(getattr(config, "enable_time_embedding", False)):
            semantic_max_hours = float(getattr(config, "time_embedding_max_hours", 31.0 * 24.0))
            local_max_hours = float(
                getattr(config, "local_time_embedding_max_hours", 72.0)
            )
            global_max_hours = float(
                getattr(config, "global_time_embedding_max_hours", 365.25 * 24.0 * 10.0)
            )
            dropout = float(getattr(config, "time_embedding_dropout", 0.0))
            scale_init = float(getattr(config, "time_embedding_scale_init", 1.0))
            self.time_embedding = MultiScaleTimeEmbedding(
                config.d_model,
                local_max_hours=local_max_hours,
                semantic_max_hours=semantic_max_hours,
                global_max_hours=global_max_hours,
                dropout=dropout,
            )
            self.time_embedding_scale = nn.Parameter(torch.tensor(scale_init))

        self.chunk_meta_proj = (
            nn.Sequential(
                nn.Linear(3, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            if bool(getattr(config, "enable_chunk_meta_sidechannel", False))
            else None
        )
        self.chunk_meta_scale = (
            nn.Parameter(torch.tensor(1.0))
            if self.chunk_meta_proj is not None
            else None
        )
        self.window_sequence_meta_proj = (
            nn.Sequential(
                nn.Linear(4, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            if bool(getattr(config, "enable_window_sequence_meta", False))
            else None
        )
        self.window_sequence_meta_scale = (
            nn.Parameter(torch.tensor(1.0))
            if self.window_sequence_meta_proj is not None
            else None
        )

        self.local_encoder = AETLocalEncoder(config, self.rope)
        self.chunk_aggregator = AETIntraWindowAggregator(config, self.rope)
        self.global_aggregator = AETGlobalAggregator(config, self.rope)
        self.use_unified_token_head = bool(getattr(config, "use_unified_token_head", True))
        self.emit_switched_heads = bool(getattr(config, "emit_switched_heads", True))
        self.heads = AETOutputHeads(
            config.d_model,
            vocab_config,
            use_unified_token_head=self.use_unified_token_head,
            emit_switched_heads=self.emit_switched_heads,
        )

        self.next_window_type_head = (
            nn.Linear(config.d_model, self.num_window_types) if self.num_window_types > 0 else None
        )
        self.enable_transition_control_heads = bool(
            getattr(config, "enable_transition_control_heads", True)
        )
        self.transition_boundary_head = (
            nn.Linear(config.d_model, 2) if self.enable_transition_control_heads else None
        )
        self.boundary_next_window_type_head = (
            nn.Linear(config.d_model, self.num_window_types)
            if self.enable_transition_control_heads and self.num_window_types > 0
            else None
        )
        self.window_len_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_window_len_head", False))
            else None
        )
        self.window_time_nll_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_window_time_nll_head", False))
            else None
        )
        self.window_time_nll_min_sigma = float(getattr(config, "window_time_nll_min_sigma", 0.1))
        self.event_time_nll_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_event_time_nll_head", False))
            else None
        )
        self.event_time_nll_min_sigma = float(getattr(config, "event_time_nll_min_sigma", 0.1))
        self.window_type_control_embedding = (
            nn.Embedding(self.num_window_types, config.d_model) if self.num_window_types > 0 else None
        )
        self.chunk_len_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_chunk_len_head", False))
            else None
        )

        self.enable_transition_bias = bool(getattr(config, "enable_transition_bias", False))
        self.transition_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.transition_hazard_scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_transition_hazard_scale = nn.Parameter(torch.tensor(1.0))

        self.context_adapter = nn.Linear(config.d_model, config.d_model)
        self.chunk_context_adapter = nn.Linear(config.d_model, config.d_model)
        self.global_fusion_mode = str(getattr(config, "global_fusion_mode", "add")).lower()
        if self.global_fusion_mode not in {"add", "film"}:
            raise ValueError(f"Unsupported global_fusion_mode={self.global_fusion_mode!r}; expected 'add' or 'film'.")
        self.exclude_special_from_global_fusion = bool(getattr(config, "exclude_special_from_global_fusion", False))
        self.context_film = nn.Linear(config.d_model, 2 * config.d_model) if self.global_fusion_mode == "film" else None

    def _window_end_token_local_index(self) -> int:
        end_token_id = (
            int(self.window_marker_end_token_id)
            if self.window_marker_end_token_id is not None
            else int(self.window_marker_type_offset) + int(self.window_marker_num_types)
        )
        return int(end_token_id) - int(self.special_token_offset)

    def _window_continue_token_local_index(self) -> int:
        if self.window_marker_continue_token_id is not None:
            continue_token_id = int(self.window_marker_continue_token_id)
        else:
            end_token_id = (
                int(self.window_marker_end_token_id)
                if self.window_marker_end_token_id is not None
                else int(self.window_marker_type_offset) + int(self.window_marker_num_types)
            )
            continue_token_id = int(end_token_id) + 1
        return int(continue_token_id) - int(self.special_token_offset)

    def _marker_masks_from_input_ids(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_start = int(self.window_marker_type_offset)
        num_types = int(self.window_marker_num_types)
        if num_types > 0:
            type_end = type_start + num_types
            type_mask = (input_ids >= type_start) & (input_ids < type_end)
        else:
            type_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        end_token_id = (
            int(self.window_marker_end_token_id)
            if self.window_marker_end_token_id is not None
            else int(self.window_marker_type_offset) + int(self.window_marker_num_types)
        )
        continue_token_id = (
            int(self.window_marker_continue_token_id)
            if self.window_marker_continue_token_id is not None
            else int(end_token_id) + 1
        )
        end_mask = input_ids == int(end_token_id)
        continue_mask = input_ids == int(continue_token_id)
        return type_mask, end_mask, continue_mask

    def _predictive_transition_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.shape[-1] < 2:
            empty = torch.zeros_like(input_ids, dtype=torch.bool)
            return empty, empty, empty

        next_ids = input_ids[..., 1:]
        next_type_mask, next_end_mask, next_continue_mask = self._marker_masks_from_input_ids(next_ids)
        source_valid = attention_mask[..., :-1].to(dtype=torch.bool) & attention_mask[..., 1:].to(dtype=torch.bool)
        if token_type_ids is not None:
            source_non_special = token_type_ids[..., :-1] != int(getattr(self.config, "special_type_id", 0))
        else:
            source_non_special = torch.ones_like(source_valid)

        predict_continue = source_valid & source_non_special & next_continue_mask
        predict_end = source_valid & source_non_special & (next_end_mask | next_type_mask)
        predict_type = source_valid & source_non_special & next_type_mask

        pad = torch.zeros_like(input_ids[..., :1], dtype=torch.bool)
        return (
            torch.cat([predict_continue, pad], dim=-1),
            torch.cat([predict_end, pad], dim=-1),
            torch.cat([predict_type, pad], dim=-1),
        )

    @staticmethod
    def _chunk_end_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Build a boolean mask with exactly one True at the last valid token of each chunk.
        Shapes:
            attention_mask: (B,W,C,L) or (B,W,L)
            returns: same shape, bool
        """
        if attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(2)
            squeeze = True
        elif attention_mask.ndim == 4:
            squeeze = False
        else:
            raise ValueError(
                f"attention_mask must be 3D or 4D, got shape {tuple(attention_mask.shape)}"
            )

        am = attention_mask.to(dtype=torch.bool)
        B, W, C, L = am.shape
        out = torch.zeros_like(am, dtype=torch.bool)
        if L > 0:
            lengths = am.to(dtype=torch.long).sum(dim=-1)  # (B,W,C)
            has_any = lengths > 0
            last_idx = (lengths - 1).clamp(min=0, max=max(0, L - 1))
            b_idx, w_idx, c_idx = torch.where(has_any)
            if b_idx.numel() > 0:
                out[b_idx, w_idx, c_idx, last_idx[b_idx, w_idx, c_idx]] = True
        if squeeze:
            out = out.squeeze(2)
        return out

    def forward(
        self,
        input_ids,
        time_ids,
        numeric_values,
        token_type_ids,
        attention_mask,
        numeric_mask=None,
        prev_global_state=None,
        window_start_times=None,
        window_mask=None,
        window_type_ids=None,
        chunk_mask=None,
        chunk_start_offsets=None,
        chunk_is_last=None,
        semantic_token_counts=None,
        semantic_duration_hours=None,
        chunk_token_counts=None,
        chunk_duration_hours=None,
    ):
        squeeze_chunk_axis = False
        if input_ids.ndim == 3:
            squeeze_chunk_axis = True
            input_ids = input_ids.unsqueeze(2)
            time_ids = time_ids.unsqueeze(2)
            numeric_values = numeric_values.unsqueeze(2)
            if numeric_mask is not None:
                numeric_mask = numeric_mask.unsqueeze(2)
            token_type_ids = token_type_ids.unsqueeze(2)
            attention_mask = attention_mask.unsqueeze(2)
        elif input_ids.ndim != 4:
            raise ValueError(f"input_ids must be 3D or 4D, got shape {tuple(input_ids.shape)}")

        B, W, C, L = input_ids.shape
        D = self.config.d_model

        if window_mask is None:
            window_mask = attention_mask.any(dim=-1).any(dim=-1).to(dtype=torch.long)
        if chunk_mask is None:
            chunk_mask = attention_mask.any(dim=-1).to(dtype=torch.long)
        if chunk_start_offsets is None:
            chunk_start_offsets = torch.zeros((B, W, C), device=time_ids.device, dtype=time_ids.dtype)
        if chunk_is_last is None:
            chunk_is_last = torch.zeros((B, W, C), device=input_ids.device, dtype=torch.long)
            if C > 0:
                n_real_chunks = chunk_mask.to(dtype=torch.long).sum(dim=2).clamp(min=1)
                last_idx = (n_real_chunks - 1).clamp(min=0)
                batch_idx = torch.arange(B, device=input_ids.device)[:, None]
                win_idx = torch.arange(W, device=input_ids.device)[None, :]
                chunk_is_last[batch_idx, win_idx, last_idx] = 1
                chunk_is_last = chunk_is_last * chunk_mask.to(dtype=torch.long)
        if window_start_times is None:
            window_start_times = torch.zeros((B, W), device=time_ids.device, dtype=time_ids.dtype)

        semantic_time_ids = time_ids.clamp(min=0.0) + chunk_start_offsets.unsqueeze(-1)
        global_time_ids = semantic_time_ids + window_start_times.unsqueeze(-1).unsqueeze(-1)

        x = self.embeddings(
            input_ids,
            numeric_values,
            numeric_mask=numeric_mask,
            window_type_ids=window_type_ids,
            token_type_ids=token_type_ids,
        )
        if self.time_embedding is not None and self.time_embedding_scale is not None:
            x = x + (
                self.time_embedding_scale
                * self.time_embedding(
                    local_time_hours=time_ids,
                    semantic_time_hours=semantic_time_ids,
                    global_time_hours=global_time_ids,
                )
            )

        local_hidden, chunk_summaries = self.local_encoder(x, time_ids, attention_mask, token_type_ids=token_type_ids)
        content_mask = attention_mask.to(dtype=torch.bool)
        if token_type_ids is not None:
            content_mask = content_mask & (token_type_ids != int(getattr(self.config, "special_type_id", 0)))

        if chunk_token_counts is None:
            chunk_token_counts = content_mask.to(dtype=torch.float32).sum(dim=-1)
        if chunk_duration_hours is None:
            neg_inf = torch.tensor(float("-inf"), device=time_ids.device, dtype=time_ids.dtype)
            chunk_time_masked = torch.where(content_mask, time_ids, neg_inf)
            chunk_duration_hours = chunk_time_masked.max(dim=-1).values
            chunk_duration_hours = torch.where(
                torch.isfinite(chunk_duration_hours),
                chunk_duration_hours,
                torch.zeros_like(chunk_duration_hours),
            )
            chunk_duration_hours = chunk_duration_hours.clamp(min=0.0)
        if semantic_token_counts is None:
            semantic_token_counts = chunk_token_counts.sum(dim=2)
        if semantic_duration_hours is None:
            neg_inf = torch.tensor(float("-inf"), device=semantic_time_ids.device, dtype=semantic_time_ids.dtype)
            sem_time_masked = torch.where(content_mask, semantic_time_ids, neg_inf)
            semantic_duration_hours = sem_time_masked.amax(dim=-1).amax(dim=-1)
            semantic_duration_hours = torch.where(
                torch.isfinite(semantic_duration_hours),
                semantic_duration_hours,
                torch.zeros_like(semantic_duration_hours),
            )
            semantic_duration_hours = semantic_duration_hours.clamp(min=0.0)

        if self.chunk_meta_proj is not None and self.chunk_meta_scale is not None:
            chunk_meta = torch.stack(
                [
                    torch.log1p(chunk_token_counts.to(dtype=x.dtype).clamp(min=0.0)),
                    torch.log1p(chunk_duration_hours.to(dtype=x.dtype).clamp(min=0.0)),
                    torch.log1p(chunk_start_offsets.to(dtype=x.dtype).clamp(min=0.0)),
                ],
                dim=-1,
            )
            chunk_meta_emb = self.chunk_meta_proj(chunk_meta) * self.chunk_meta_scale
            chunk_summaries = chunk_summaries + (
                chunk_meta_emb * chunk_mask.to(dtype=chunk_meta_emb.dtype).unsqueeze(-1)
            )

        chunk_states, semantic_summaries = self.chunk_aggregator(chunk_summaries, chunk_start_offsets, chunk_mask)

        if self.window_sequence_meta_proj is not None and self.window_sequence_meta_scale is not None:
            prev_window_end = window_start_times + semantic_duration_hours.to(dtype=window_start_times.dtype)
            prev_window_end_shift = torch.zeros_like(prev_window_end)
            if W > 1:
                prev_window_end_shift[:, 1:] = prev_window_end[:, :-1]
            gap_prev_h = (window_start_times - prev_window_end_shift).clamp(min=0.0)
            window_meta = torch.stack(
                [
                    torch.log1p(semantic_token_counts.to(dtype=x.dtype).clamp(min=0.0)),
                    torch.log1p(semantic_duration_hours.to(dtype=x.dtype).clamp(min=0.0)),
                    torch.log1p(window_start_times.to(dtype=x.dtype).clamp(min=0.0)),
                    torch.log1p(gap_prev_h.to(dtype=x.dtype).clamp(min=0.0)),
                ],
                dim=-1,
            )
            window_meta_emb = self.window_sequence_meta_proj(window_meta) * self.window_sequence_meta_scale
            semantic_summaries = semantic_summaries + (
                window_meta_emb * window_mask.to(dtype=window_meta_emb.dtype).unsqueeze(-1)
            )

        global_states = self.global_aggregator(
            semantic_summaries,
            window_start_times,
            window_mask,
            prev_context_state=prev_global_state,
        )

        global_context = self.context_adapter(global_states)
        shifted_context = torch.zeros_like(global_context)
        if prev_global_state is not None:
            shifted_context[:, 0, :] = self.context_adapter(prev_global_state)
        if W > 1:
            shifted_context[:, 1:, :] = global_context[:, :-1, :]

        chunk_context = self.chunk_context_adapter(chunk_states)
        shifted_chunk_context = torch.zeros_like(chunk_context)
        if C > 1:
            shifted_chunk_context[:, :, 1:, :] = chunk_context[:, :, :-1, :]

        if self.global_fusion_mode == "film" and self.context_film is not None:
            gamma_beta = self.context_film(shifted_context)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma)
            fused_representation = local_hidden * (1.0 + gamma.unsqueeze(2).unsqueeze(3)) + beta.unsqueeze(2).unsqueeze(3)
            fused_representation = fused_representation + shifted_chunk_context.unsqueeze(3)
        else:
            fused_representation = (
                local_hidden
                + shifted_context.unsqueeze(2).unsqueeze(3)
                + shifted_chunk_context.unsqueeze(3)
            )

        if self.exclude_special_from_global_fusion and token_type_ids is not None:
            mask = (token_type_ids != int(getattr(self.config, "special_type_id", 0))).unsqueeze(-1)
            fused_representation = torch.where(mask, fused_representation, local_hidden)

        fused_representation = fused_representation * attention_mask.unsqueeze(-1)
        fused_representation = torch.nan_to_num(
            fused_representation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        logits_dict = self.heads(fused_representation)
        if self.transition_boundary_head is not None:
            logits_dict["logits_transition_boundary"] = self.transition_boundary_head(
                fused_representation
            )
        if self.boundary_next_window_type_head is not None:
            logits_dict["logits_boundary_next_window_type"] = (
                self.boundary_next_window_type_head(fused_representation)
            )
        if self.next_window_type_head is not None:
            logits_dict["logits_next_window_type"] = self.next_window_type_head(global_states)

        if self.event_time_nll_head is not None:
            raw = self.event_time_nll_head(fused_representation)
            logits_dict["pred_dt_next_mu"] = raw[..., 0]
            logits_dict["pred_dt_next_sigma"] = F.softplus(raw[..., 1]) + float(self.event_time_nll_min_sigma)

        control_ctx = None
        if self.window_len_head is not None or self.window_time_nll_head is not None:
            control_ctx = shifted_context
            if window_type_ids is not None and self.window_type_control_embedding is not None:
                safe_ids = window_type_ids.clamp(min=0, max=max(0, self.num_window_types - 1))
                control_ctx = control_ctx + self.window_type_control_embedding(safe_ids)

        if self.window_len_head is not None and control_ctx is not None:
            raw = self.window_len_head(control_ctx)
            logits_dict["pred_window_len_tokens"] = F.softplus(raw[..., 0]) + 1.0
            logits_dict["pred_window_len_hours"] = F.softplus(raw[..., 1]) + 0.25

        if self.window_time_nll_head is not None and control_ctx is not None:
            raw = self.window_time_nll_head(control_ctx)
            logits_dict["pred_window_dur_mu"] = raw[..., 0]
            logits_dict["pred_window_dur_sigma"] = F.softplus(raw[..., 1]) + float(self.window_time_nll_min_sigma)

        if self.chunk_len_head is not None:
            chunk_control_ctx = shifted_chunk_context
            if window_type_ids is not None and self.window_type_control_embedding is not None:
                safe_ids = window_type_ids.clamp(min=0, max=max(0, self.num_window_types - 1))
                chunk_control_ctx = chunk_control_ctx + self.window_type_control_embedding(safe_ids).unsqueeze(2)
            raw = self.chunk_len_head(chunk_control_ctx)
            logits_dict["pred_chunk_len_tokens"] = F.softplus(raw[..., 0]) + 1.0
            logits_dict["pred_chunk_len_hours"] = F.softplus(raw[..., 1]) + 0.25

        if self.enable_transition_bias and self.size_special > 0:
            content_mask = attention_mask.to(dtype=torch.bool)
            if token_type_ids is not None:
                content_mask = content_mask & (token_type_ids != int(getattr(self.config, "special_type_id", 0)))

            predict_continue_mask, predict_end_mask, _ = self._predictive_transition_masks(
                input_ids,
                attention_mask,
                token_type_ids,
            )

            pred_len_tokens = logits_dict.get("pred_window_len_tokens", None)
            pred_len_hours = logits_dict.get("pred_window_len_hours", None)
            if pred_len_hours is None:
                mu = logits_dict.get("pred_window_dur_mu", None)
                sigma = logits_dict.get("pred_window_dur_sigma", None)
                if mu is not None and sigma is not None:
                    pred_len_hours = torch.exp(mu + 0.5 * sigma.square()) - 1.0
                    pred_len_hours = pred_len_hours.clamp(min=0.25)

            if pred_len_hours is not None:
                flat_content = content_mask.reshape(B, W, C * L).to(dtype=torch.float32)
                pos_flat = torch.cumsum(flat_content, dim=2).reshape(B, W, C, L)
                t_sem = time_ids.clamp(min=0.0) + chunk_start_offsets.unsqueeze(-1)
                eps = 1e-6
                prog_time = t_sem / (pred_len_hours.unsqueeze(-1).unsqueeze(-1) + eps)
                if pred_len_tokens is not None:
                    prog_tokens = pos_flat / (pred_len_tokens.unsqueeze(-1).unsqueeze(-1) + eps)
                    prog = 0.5 * (prog_tokens + prog_time)
                else:
                    prog = prog_time
                hazard_logit = self.transition_hazard_scale * (prog - 1.0)
                hazard_logit = hazard_logit * chunk_is_last.to(dtype=hazard_logit.dtype).unsqueeze(-1)
            else:
                hazard_logit = time_ids.new_zeros((B, W, C, L))

            pred_chunk_len_tokens = logits_dict.get("pred_chunk_len_tokens", None)
            pred_chunk_len_hours = logits_dict.get("pred_chunk_len_hours", None)
            if pred_chunk_len_hours is not None:
                chunk_pos = torch.cumsum(content_mask.to(dtype=torch.float32), dim=-1)
                eps = 1e-6
                chunk_prog_time = time_ids.clamp(min=0.0) / (pred_chunk_len_hours.unsqueeze(-1) + eps)
                if pred_chunk_len_tokens is not None:
                    chunk_prog_tokens = chunk_pos / (pred_chunk_len_tokens.unsqueeze(-1) + eps)
                    chunk_prog = 0.5 * (chunk_prog_tokens + chunk_prog_time)
                else:
                    chunk_prog = chunk_prog_time
                chunk_hazard_logit = self.chunk_transition_hazard_scale * (chunk_prog - 1.0)
                nonfinal = (chunk_mask.to(dtype=torch.bool) & ~chunk_is_last.to(dtype=torch.bool)).to(dtype=chunk_prog.dtype)
                chunk_hazard_logit = chunk_hazard_logit * nonfinal.unsqueeze(-1)
            else:
                chunk_hazard_logit = time_ids.new_zeros((B, W, C, L))

            # Localized transition biasing: only at predictive chunk-end content positions.
            if "logits_transition_boundary" in logits_dict and logits_dict["logits_transition_boundary"] is not None:
                logits_tb = logits_dict["logits_transition_boundary"]
                end_apply = predict_end_mask.to(dtype=logits_tb.dtype)
                cont_apply = predict_continue_mask.to(dtype=logits_tb.dtype)

                logits_tb[..., 1] = logits_tb[..., 1] + (hazard_logit * end_apply)
                logits_tb[..., 0] = logits_tb[..., 0] + (chunk_hazard_logit * cont_apply)
                logits_dict["logits_transition_boundary"] = logits_tb

            if (
                self.window_marker_num_types > 0
                and self.next_window_type_head is not None
                and "logits_boundary_next_window_type" in logits_dict
                and logits_dict["logits_boundary_next_window_type"] is not None
            ):
                logits_bnt = logits_dict["logits_boundary_next_window_type"]  # (B,W,C,L,K)
                K = int(self.window_marker_num_types)
                prior = logits_dict.get("logits_next_window_type", None)
                if prior is not None and prior.shape[-1] == K:
                    prior_logp = torch.log_softmax(prior, dim=-1)
                else:
                    prior_logp = logits_bnt.new_zeros((B, W, K))

                next_window_exists = torch.zeros((B, W), device=input_ids.device, dtype=torch.bool)
                if W >= 2:
                    next_window_exists[:, :-1] = (
                        window_mask[:, :-1].to(dtype=torch.bool) & window_mask[:, 1:].to(dtype=torch.bool)
                    )
                type_boundary = (
                    predict_end_mask
                    & next_window_exists.unsqueeze(-1).unsqueeze(-1)
                ).to(dtype=logits_bnt.dtype)  # (B,W,C,L)

                if type_boundary.any():
                    logits_bnt = logits_bnt + (
                        type_boundary.unsqueeze(-1)
                        * (
                            hazard_logit.unsqueeze(-1)
                            + (self.transition_prior_scale * prior_logp).unsqueeze(2).unsqueeze(2)
                        )
                    )
                    logits_dict["logits_boundary_next_window_type"] = logits_bnt

        for key, value in list(logits_dict.items()):
            if torch.is_tensor(value):
                logits_dict[key] = torch.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        if W == 0:
            final_state = torch.zeros((B, D), device=global_states.device, dtype=global_states.dtype)
        else:
            n_real = window_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)
            last_idx = (n_real - 1).clamp(min=0)
            batch_idx = torch.arange(B, device=global_states.device)
            final_state = global_states[batch_idx, last_idx, :]

        if squeeze_chunk_axis:
            for key in (
                "logits_token",
                "logits_struct",
                "logits_rvq",
                "logits_meas",
                "logits_medtok",
                "pred_values",
                "pred_dt_next_mu",
                "pred_dt_next_sigma",
                "logits_transition_boundary",
                "logits_boundary_next_window_type",
            ):
                if key in logits_dict and logits_dict[key] is not None:
                    logits_dict[key] = logits_dict[key].squeeze(2)

        return logits_dict, final_state
