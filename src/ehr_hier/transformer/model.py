import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_ORDER, EVENT_PAYLOAD_KIND_TO_ID

from .aggregator import (
    AETGlobalAggregator,
    AETIntraWindowAggregator,
    AETWindowStatePacketBuilder,
)
from .embeddings import (
    AETEmbeddings,
    ContinuousRotaryPositionalEmbedding,
    MultiScaleTimeEmbedding,
)
from .episodic_memory import AETEpisodicMemory, EpisodicMemoryState, PatientMemoryState
from .event_composer import AETEventComposer
from .encoder import AETLocalEncoder
from .global_state import AETLatentHealthState
from .precedent_memory import (
    AETPrecedentMemory,
    build_batch_future_summary_targets,
    build_window_support_flags,
)
from .heads import AETOutputHeads
from .world_model_contract import NUM_SUPPORT_FLAGS, NextWindowHeader, WindowStatePacket


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
        self.use_event_composer = bool(getattr(config, "use_event_composer", True))
        self.event_composer = AETEventComposer(
            d_model=config.d_model,
            dropout=config.dropout,
            num_token_types=int(getattr(config, "num_token_types", 8)),
            max_bundle_slots=int(getattr(config, "event_bundle_slots", 32)),
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
        self.window_state_packet_builder = AETWindowStatePacketBuilder(config)
        self.window_packet_summary_adapter = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.global_context_mode = str(
            getattr(config, "global_context_mode", "transformer")
        ).strip().lower()
        if self.global_context_mode not in {"transformer", "latent_state"}:
            raise ValueError(
                f"Unsupported global_context_mode={self.global_context_mode!r}; expected transformer|latent_state"
            )
        self.global_aggregator = (
            AETGlobalAggregator(config, self.rope)
            if self.global_context_mode == "transformer"
            else None
        )
        self.latent_health_state = (
            AETLatentHealthState(config)
            if self.global_context_mode == "latent_state"
            else None
        )
        self.use_exact_memory = bool(getattr(config, "enable_exact_memory", False))
        if self.use_exact_memory and not self.use_event_composer:
            self.use_exact_memory = False
        self.exact_memory = (
            AETEpisodicMemory(config)
            if self.use_exact_memory
            else None
        )
        self.memory_context_adapter = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_exact_memory
            else None
        )
        self.use_precedent_memory = bool(getattr(config, "enable_precedent_memory", False))
        self.precedent_memory = (
            AETPrecedentMemory(config)
            if self.use_precedent_memory
            else None
        )
        self.latent_query_readout = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_precedent_memory
            else None
        )
        self.precedent_boundary_context_adapter = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_precedent_memory
            else None
        )
        self.precedent_prompt_q = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_precedent_memory
            else None
        )
        self.precedent_prompt_k = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_precedent_memory
            else None
        )
        self.precedent_prompt_v = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_precedent_memory
            else None
        )
        self.precedent_prompt_out = (
            nn.Linear(config.d_model, config.d_model)
            if self.use_precedent_memory
            else None
        )
        self.use_unified_token_head = bool(getattr(config, "use_unified_token_head", True))
        self.emit_switched_heads = bool(getattr(config, "emit_switched_heads", True))
        self.heads = AETOutputHeads(
            config.d_model,
            vocab_config,
            use_unified_token_head=self.use_unified_token_head,
            emit_switched_heads=self.emit_switched_heads,
            emit_event_heads=self.use_event_composer,
            num_event_families=int(getattr(config, "num_token_types", 8)),
            num_event_payloads=int(len(EVENT_PAYLOAD_KIND_ORDER)),
        )
        self.event_value_code_conditioned_head = (
            nn.Sequential(
                nn.Linear(2 * config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if self.use_event_composer
            else None
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
        self.next_window_gap_nll_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_next_window_gap_nll_head", False))
            else None
        )
        self.next_window_gap_nll_min_sigma = float(
            getattr(config, "next_window_gap_nll_min_sigma", 0.1)
        )
        self.next_window_duration_nll_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_next_window_duration_nll_head", True))
            else None
        )
        self.next_window_duration_nll_min_sigma = float(
            getattr(config, "next_window_duration_nll_min_sigma", 0.1)
        )
        self.next_window_support_head = (
            nn.Linear(config.d_model, int(NUM_SUPPORT_FLAGS))
            if bool(getattr(config, "enable_next_window_support_head", True))
            else None
        )
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

    @staticmethod
    def _flatten_event_feature(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            B, W, E = feature.shape
            return feature.reshape(B * W, E)
        if feature.ndim == 4:
            B, W, C, E = feature.shape
            return feature.reshape(B * W, C * E)
        raise ValueError(f"event feature must be 3D or 4D, got shape {tuple(feature.shape)}")

    def _next_numeric_measurement_event_ids(
        self,
        *,
        event_input_ids: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_payload_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_payload_id = int(EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"])
        content_mask = event_attention_mask.to(dtype=torch.bool) & (
            event_type_ids != int(getattr(self.config, "special_type_id", 0))
        )
        numeric_mask = content_mask & (event_payload_ids == numeric_payload_id)

        flat_ids = self._flatten_event_feature(event_input_ids)
        flat_numeric_mask = self._flatten_event_feature(numeric_mask)
        N, S = flat_ids.shape

        next_ids = torch.zeros_like(flat_ids)
        next_exists = torch.zeros((N, S), device=flat_ids.device, dtype=torch.bool)
        last_id = torch.zeros((N,), device=flat_ids.device, dtype=flat_ids.dtype)
        has = torch.zeros((N,), device=flat_ids.device, dtype=torch.bool)
        for i in range(S - 1, -1, -1):
            next_ids[:, i] = last_id
            next_exists[:, i] = has
            cur = flat_numeric_mask[:, i]
            last_id = torch.where(cur, flat_ids[:, i], last_id)
            has = has | cur

        return next_ids.view_as(event_input_ids), next_exists.view_as(event_input_ids)

    def load_precedent_index(
        self,
        path: str,
        *,
        map_location: str | torch.device = "cpu",
    ):
        if self.precedent_memory is None:
            raise RuntimeError("precedent_memory is not enabled for this model.")
        return self.precedent_memory.load_index(path, map_location=map_location)

    @staticmethod
    def _shift_window_tensor(tensor: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(tensor)
        if tensor.shape[1] > 1:
            shifted[:, 1:] = tensor[:, :-1]
        return shifted

    @staticmethod
    def _expected_positive_hours_from_log1p_gaussian(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        max_hours: float = 365.25 * 24.0 * 10.0,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1e6, neginf=1.0).clamp(min=1e-4)
        expected_log1p = (mu + 0.5 * sigma.square()).clamp(max=20.0)
        return torch.expm1(expected_log1p).clamp(min=0.0, max=float(max_hours))

    def _predict_next_window_header(
        self,
        *,
        boundary_context: torch.Tensor,
    ) -> tuple[NextWindowHeader, dict[str, torch.Tensor]]:
        B, W, _ = boundary_context.shape
        device = boundary_context.device
        dtype = boundary_context.dtype

        if self.next_window_type_head is not None:
            logits_type = self.next_window_type_head(boundary_context)
            pred_type_ids = logits_type.argmax(dim=-1).to(dtype=torch.long)
        else:
            logits_type = None
            pred_type_ids = torch.full((B, W), fill_value=-1, device=device, dtype=torch.long)

        if self.next_window_gap_nll_head is not None:
            raw_gap = self.next_window_gap_nll_head(boundary_context)
            pred_gap_mu = raw_gap[..., 0]
            pred_gap_sigma = F.softplus(raw_gap[..., 1]) + float(self.next_window_gap_nll_min_sigma)
            pred_gap_hours = self._expected_positive_hours_from_log1p_gaussian(
                pred_gap_mu,
                pred_gap_sigma,
            )
        else:
            pred_gap_mu = torch.zeros((B, W), device=device, dtype=dtype)
            pred_gap_sigma = torch.ones((B, W), device=device, dtype=dtype)
            pred_gap_hours = torch.zeros((B, W), device=device, dtype=dtype)

        if self.next_window_duration_nll_head is not None:
            raw_duration = self.next_window_duration_nll_head(boundary_context)
            pred_duration_mu = raw_duration[..., 0]
            pred_duration_sigma = (
                F.softplus(raw_duration[..., 1]) + float(self.next_window_duration_nll_min_sigma)
            )
            pred_duration_hours = self._expected_positive_hours_from_log1p_gaussian(
                pred_duration_mu,
                pred_duration_sigma,
            )
        else:
            pred_duration_mu = torch.zeros((B, W), device=device, dtype=dtype)
            pred_duration_sigma = torch.ones((B, W), device=device, dtype=dtype)
            pred_duration_hours = torch.zeros((B, W), device=device, dtype=dtype)

        if self.next_window_support_head is not None:
            logits_support = self.next_window_support_head(boundary_context)
            pred_support_probs = torch.sigmoid(logits_support)
        else:
            logits_support = torch.zeros(
                (B, W, int(NUM_SUPPORT_FLAGS)),
                device=device,
                dtype=dtype,
            )
            pred_support_probs = torch.zeros_like(logits_support)

        return (
            NextWindowHeader(
                window_type_ids=pred_type_ids,
                gap_hours=pred_gap_hours,
                duration_hours=pred_duration_hours,
                support_flags=pred_support_probs,
            ),
            {
                "logits_next_window_type": logits_type,
                "pred_next_window_gap_mu": pred_gap_mu,
                "pred_next_window_gap_sigma": pred_gap_sigma,
                "pred_next_window_duration_mu": pred_duration_mu,
                "pred_next_window_duration_sigma": pred_duration_sigma,
                "logits_next_window_support": logits_support,
                "pred_next_window_support_probs": pred_support_probs,
            },
        )

    def _build_next_window_header(
        self,
        *,
        predicted_header: NextWindowHeader,
        current_support_flags: torch.Tensor | None,
        window_type_ids: torch.Tensor | None,
        window_start_times: torch.Tensor | None,
        semantic_duration_hours: torch.Tensor | None,
        window_mask: torch.Tensor | None,
    ) -> NextWindowHeader:
        next_window_type_ids = predicted_header.window_type_ids.clone()
        next_gap_hours = predicted_header.gap_hours.clone()
        next_duration_hours = predicted_header.duration_hours.clone()
        if predicted_header.support_flags is not None:
            next_support_flags = predicted_header.support_flags.clone()
        else:
            next_support_flags = torch.zeros(
                next_window_type_ids.shape + (int(NUM_SUPPORT_FLAGS),),
                device=next_window_type_ids.device,
                dtype=next_gap_hours.dtype,
            )

        B, W = next_window_type_ids.shape
        dtype = next_gap_hours.dtype

        if (
            window_type_ids is not None
            and window_start_times is not None
            and semantic_duration_hours is not None
            and window_mask is not None
            and W >= 2
        ):
            valid_next = window_mask[:, :-1].to(dtype=torch.bool) & window_mask[:, 1:].to(dtype=torch.bool)
            next_window_type_ids[:, :-1] = window_type_ids[:, 1:].to(dtype=torch.long)
            next_gap_hours[:, :-1] = (
                window_start_times[:, 1:].to(dtype=dtype)
                - (window_start_times[:, :-1].to(dtype=dtype) + semantic_duration_hours[:, :-1].to(dtype=dtype))
            ).clamp(min=0.0)
            next_duration_hours[:, :-1] = semantic_duration_hours[:, 1:].to(dtype=dtype).clamp(min=0.0)
            if current_support_flags is not None:
                next_support_flags[:, :-1] = current_support_flags[:, 1:].to(dtype=dtype)
            next_window_type_ids[:, :-1] = torch.where(
                valid_next,
                next_window_type_ids[:, :-1],
                torch.full_like(next_window_type_ids[:, :-1], fill_value=-1),
            )
            next_gap_hours[:, :-1] = torch.where(
                valid_next,
                next_gap_hours[:, :-1],
                torch.zeros_like(next_gap_hours[:, :-1]),
            )
            next_duration_hours[:, :-1] = torch.where(
                valid_next,
                next_duration_hours[:, :-1],
                torch.zeros_like(next_duration_hours[:, :-1]),
            )
            if current_support_flags is not None:
                next_support_flags[:, :-1] = next_support_flags[:, :-1] * valid_next.unsqueeze(-1).to(dtype=dtype)

        return NextWindowHeader(
            window_type_ids=next_window_type_ids,
            gap_hours=next_gap_hours,
            duration_hours=next_duration_hours,
            support_flags=next_support_flags,
        )

    def _apply_precedent_prompt_tokens(
        self,
        *,
        local_hidden: torch.Tensor,
        prompt_tokens: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            prompt_tokens is None
            or self.precedent_prompt_q is None
            or self.precedent_prompt_k is None
            or self.precedent_prompt_v is None
            or self.precedent_prompt_out is None
        ):
            return torch.zeros_like(local_hidden)
        if prompt_tokens.shape[-1] != local_hidden.shape[-1]:
            raise ValueError(
                f"prompt_tokens hidden dim must match local_hidden; got {tuple(prompt_tokens.shape)} vs {tuple(local_hidden.shape)}"
            )
        q = self.precedent_prompt_q(local_hidden)
        k = self.precedent_prompt_k(prompt_tokens)
        v = self.precedent_prompt_v(prompt_tokens)
        scale = float(max(1, q.shape[-1])) ** -0.5
        scores = torch.einsum("bwcnd,bwpd->bwcnp", q, k) * scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bwcnp,bwpd->bwcnd", attn, v)
        context = self.precedent_prompt_out(context)
        return context * attention_mask.unsqueeze(-1).to(dtype=context.dtype)

    def forward(
        self,
        input_ids,
        time_ids,
        numeric_values,
        token_type_ids,
        attention_mask,
        numeric_mask=None,
        prev_global_state=None,
        prev_memory_state: PatientMemoryState | EpisodicMemoryState | None = None,
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
        token_event_index=None,
        token_event_slot_ids=None,
        event_input_ids=None,
        event_time_ids=None,
        event_numeric_values=None,
        event_numeric_mask=None,
        event_type_ids=None,
        event_payload_ids=None,
        event_demographic_feature_ids=None,
        event_attention_mask=None,
        event_memory_rule_scores=None,
        event_memory_group_ids=None,
        event_memory_first_flags=None,
        event_memory_chronic_flags=None,
        subject_ids=None,
        trajectory_ords=None,
        return_aux_state: bool = False,
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
            if token_event_index is not None:
                token_event_index = token_event_index.unsqueeze(2)
            if token_event_slot_ids is not None:
                token_event_slot_ids = token_event_slot_ids.unsqueeze(2)
            if event_input_ids is not None:
                event_input_ids = event_input_ids.unsqueeze(2)
            if event_time_ids is not None:
                event_time_ids = event_time_ids.unsqueeze(2)
            if event_numeric_values is not None:
                event_numeric_values = event_numeric_values.unsqueeze(2)
            if event_numeric_mask is not None:
                event_numeric_mask = event_numeric_mask.unsqueeze(2)
            if event_type_ids is not None:
                event_type_ids = event_type_ids.unsqueeze(2)
            if event_payload_ids is not None:
                event_payload_ids = event_payload_ids.unsqueeze(2)
            if event_attention_mask is not None:
                event_attention_mask = event_attention_mask.unsqueeze(2)
            if event_demographic_feature_ids is not None:
                event_demographic_feature_ids = event_demographic_feature_ids.unsqueeze(2)
            if event_memory_rule_scores is not None:
                event_memory_rule_scores = event_memory_rule_scores.unsqueeze(2)
            if event_memory_group_ids is not None:
                event_memory_group_ids = event_memory_group_ids.unsqueeze(2)
            if event_memory_first_flags is not None:
                event_memory_first_flags = event_memory_first_flags.unsqueeze(2)
            if event_memory_chronic_flags is not None:
                event_memory_chronic_flags = event_memory_chronic_flags.unsqueeze(2)
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

        x_tokens = self.embeddings(
            input_ids,
            numeric_values,
            numeric_mask=numeric_mask,
            window_type_ids=window_type_ids,
            token_type_ids=token_type_ids,
        )

        use_event_path = bool(
            self.use_event_composer
            and token_event_index is not None
            and token_event_slot_ids is not None
            and event_time_ids is not None
            and event_type_ids is not None
            and event_payload_ids is not None
            and event_attention_mask is not None
        )

        local_x = x_tokens
        local_time_inputs = time_ids
        local_semantic_time_ids = semantic_time_ids
        local_token_type_ids = token_type_ids
        local_attention_mask = attention_mask
        scatter_event_states = False

        if use_event_path:
            if event_numeric_values is None:
                event_numeric_values = torch.zeros(
                    event_attention_mask.shape + (1,),
                    device=x_tokens.device,
                    dtype=x_tokens.dtype,
                )
            if event_numeric_mask is None:
                event_numeric_mask = torch.zeros_like(event_attention_mask, dtype=torch.long)

            event_semantic_time_ids = event_time_ids.clamp(min=0.0) + chunk_start_offsets.unsqueeze(-1)
            event_global_time_ids = event_semantic_time_ids + window_start_times.unsqueeze(-1).unsqueeze(-1)
            local_x = self.event_composer(
                x_tokens,
                token_event_index=token_event_index,
                token_event_slot_ids=token_event_slot_ids,
                attention_mask=attention_mask,
                event_attention_mask=event_attention_mask,
                event_type_ids=event_type_ids,
                event_payload_ids=event_payload_ids,
                event_numeric_values=event_numeric_values,
                event_numeric_mask=event_numeric_mask,
            )
            if self.time_embedding is not None and self.time_embedding_scale is not None:
                local_x = local_x + (
                    self.time_embedding_scale
                    * self.time_embedding(
                        local_time_hours=event_time_ids,
                        semantic_time_hours=event_semantic_time_ids,
                        global_time_hours=event_global_time_ids,
                    )
                )
            local_time_inputs = event_time_ids
            local_semantic_time_ids = event_semantic_time_ids
            local_token_type_ids = event_type_ids
            local_attention_mask = event_attention_mask
            scatter_event_states = True
        elif self.time_embedding is not None and self.time_embedding_scale is not None:
            local_x = local_x + (
                self.time_embedding_scale
                * self.time_embedding(
                    local_time_hours=time_ids,
                    semantic_time_hours=semantic_time_ids,
                    global_time_hours=global_time_ids,
                )
            )

        local_hidden, chunk_summaries = self.local_encoder(
            local_x,
            local_time_inputs,
            local_attention_mask,
            token_type_ids=local_token_type_ids,
        )
        content_mask = local_attention_mask.to(dtype=torch.bool)
        if local_token_type_ids is not None:
            content_mask = content_mask & (
                local_token_type_ids != int(getattr(self.config, "special_type_id", 0))
            )

        if chunk_token_counts is None:
            chunk_token_counts = content_mask.to(dtype=torch.float32).sum(dim=-1)
        if chunk_duration_hours is None:
            neg_inf = torch.tensor(float("-inf"), device=local_time_inputs.device, dtype=local_time_inputs.dtype)
            chunk_time_masked = torch.where(content_mask, local_time_inputs, neg_inf)
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
            neg_inf = torch.tensor(
                float("-inf"),
                device=local_semantic_time_ids.device,
                dtype=local_semantic_time_ids.dtype,
            )
            sem_time_masked = torch.where(content_mask, local_semantic_time_ids, neg_inf)
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
                    torch.log1p(chunk_token_counts.to(dtype=local_x.dtype).clamp(min=0.0)),
                    torch.log1p(chunk_duration_hours.to(dtype=local_x.dtype).clamp(min=0.0)),
                    torch.log1p(chunk_start_offsets.to(dtype=local_x.dtype).clamp(min=0.0)),
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
                    torch.log1p(semantic_token_counts.to(dtype=local_x.dtype).clamp(min=0.0)),
                    torch.log1p(semantic_duration_hours.to(dtype=local_x.dtype).clamp(min=0.0)),
                    torch.log1p(window_start_times.to(dtype=local_x.dtype).clamp(min=0.0)),
                    torch.log1p(gap_prev_h.to(dtype=local_x.dtype).clamp(min=0.0)),
                ],
                dim=-1,
            )
            window_meta_emb = self.window_sequence_meta_proj(window_meta) * self.window_sequence_meta_scale
            semantic_summaries = semantic_summaries + (
                window_meta_emb * window_mask.to(dtype=window_meta_emb.dtype).unsqueeze(-1)
            )

        window_state_packet: WindowStatePacket = self.window_state_packet_builder(
            base_summary=semantic_summaries,
            chunk_states=chunk_states,
            chunk_mask=chunk_mask,
            chunk_start_offsets=chunk_start_offsets,
            window_start_times=window_start_times,
            semantic_duration_hours=semantic_duration_hours,
            chunk_token_counts=chunk_token_counts,
            chunk_duration_hours=chunk_duration_hours,
            window_mask=window_mask,
            window_type_ids=window_type_ids,
        )
        window_packet_summary = self.window_packet_summary_adapter(
            window_state_packet.summary()
        )
        window_packet_summary = window_packet_summary * window_mask.unsqueeze(-1).to(
            dtype=window_packet_summary.dtype
        )

        if self.latent_health_state is not None:
            global_states = self.latent_health_state(
                window_summaries=window_packet_summary,
                window_start_times=window_start_times,
                padding_mask=window_mask,
                semantic_duration_hours=semantic_duration_hours,
                window_type_ids=window_type_ids,
                prev_context_state=prev_global_state,
            )
        else:
            assert self.global_aggregator is not None
            global_states = self.global_aggregator(
                window_packet_summary,
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

        memory_context = torch.zeros_like(shifted_context)
        memory_out = None
        memory_context_by_bank = None
        memory_state_digests_by_bank = None
        if (
            self.exact_memory is not None
            and scatter_event_states
            and event_input_ids is not None
            and event_attention_mask is not None
            and event_type_ids is not None
            and event_payload_ids is not None
            and self.memory_context_adapter is not None
        ):
            memory_out = self.exact_memory(
                event_states=local_hidden,
                event_seed_states=local_x,
                event_input_ids=event_input_ids,
                event_time_ids=event_time_ids,
                event_attention_mask=event_attention_mask,
                event_type_ids=event_type_ids,
                event_payload_ids=event_payload_ids,
                event_demographic_feature_ids=event_demographic_feature_ids,
                query_states=shifted_context,
                window_mask=window_mask,
                window_start_times=window_start_times,
                semantic_duration_hours=semantic_duration_hours,
                prev_memory_state=prev_memory_state,
                event_memory_rule_scores=event_memory_rule_scores,
                event_memory_group_ids=event_memory_group_ids,
                event_memory_first_flags=event_memory_first_flags,
                event_memory_chronic_flags=event_memory_chronic_flags,
            )
            memory_context = self.memory_context_adapter(memory_out.context)
            if memory_out.context_by_bank is not None and self.memory_context_adapter is not None:
                memory_context_by_bank = {
                    str(name): self.memory_context_adapter(bank_ctx)
                    for name, bank_ctx in memory_out.context_by_bank.items()
                }
            if memory_out.state_digest_by_bank is not None:
                memory_state_digests_by_bank = {
                    str(name): bank_ctx
                    for name, bank_ctx in memory_out.state_digest_by_bank.items()
                }
        final_memory_state = memory_out.next_state if memory_out is not None else None

        precedent_out = None
        precedent_generation = None
        precedent_generation_prompt_context = torch.zeros_like(local_hidden)
        precedent_boundary_context = torch.zeros_like(global_states)
        next_window_head_context = global_states
        next_window_header = None
        current_support_flags = None
        precedent_target_future_summary = None
        precedent_target_future_embedding = None
        precedent_target_future_mask = None
        precedent_anchor_item_ids = None
        predicted_next_window_header = None
        predicted_next_window_outputs = {}
        if (
            self.precedent_memory is not None
            and self.latent_query_readout is not None
            and self.precedent_memory.has_index
        ):
            if (
                scatter_event_states
                and event_type_ids is not None
                and event_attention_mask is not None
                and event_payload_ids is not None
            ):
                current_support_flags = build_window_support_flags(
                    event_type_ids=event_type_ids,
                    event_attention_mask=event_attention_mask,
                    event_memory_chronic_flags=event_memory_chronic_flags,
                    event_numeric_values=event_numeric_values,
                    event_numeric_mask=event_numeric_mask,
                ).to(dtype=global_states.dtype)
            else:
                current_support_flags = torch.zeros(
                    (B, W, int(NUM_SUPPORT_FLAGS)),
                    device=global_states.device,
                    dtype=global_states.dtype,
                )

            persistent_digest = (
                memory_state_digests_by_bank.get("persistent", None)
                if memory_state_digests_by_bank is not None
                else None
            )
            if persistent_digest is None:
                persistent_digest = torch.zeros_like(global_states)

            latent_query_state = self.latent_query_readout(global_states)
            precedent_out = self.precedent_memory.query_boundary_prior(
                state_packet=window_state_packet,
                query_state=latent_query_state,
                memory_digest=persistent_digest,
                support_flags=current_support_flags,
            )
            if (
                precedent_out.future_embedding is not None
                and self.precedent_boundary_context_adapter is not None
            ):
                precedent_boundary_context = self.precedent_boundary_context_adapter(
                    precedent_out.future_embedding
                )
                next_window_head_context = global_states + precedent_boundary_context
            else:
                next_window_head_context = global_states

            predicted_next_window_header, predicted_next_window_outputs = (
                self._predict_next_window_header(boundary_context=next_window_head_context)
            )

            if (
                scatter_event_states
                and window_type_ids is not None
                and window_start_times is not None
                and semantic_duration_hours is not None
                and window_mask is not None
                and event_type_ids is not None
                and event_payload_ids is not None
                and event_attention_mask is not None
            ):
                precedent_target_future_summary, precedent_target_future_mask = build_batch_future_summary_targets(
                    num_window_types=int(self.num_window_types),
                    window_type_ids=window_type_ids,
                    window_start_times=window_start_times,
                    semantic_duration_hours=semantic_duration_hours,
                    window_mask=window_mask,
                    event_type_ids=event_type_ids,
                    event_payload_ids=event_payload_ids,
                    event_attention_mask=event_attention_mask,
                    event_memory_chronic_flags=event_memory_chronic_flags,
                    event_numeric_values=event_numeric_values,
                    event_numeric_mask=event_numeric_mask,
                )
                precedent_target_future_embedding = self.precedent_memory.project_future_summaries(
                    precedent_target_future_summary.to(dtype=global_states.dtype)
                )

            next_window_header = self._build_next_window_header(
                predicted_header=predicted_next_window_header,
                current_support_flags=current_support_flags,
                window_type_ids=window_type_ids,
                window_start_times=window_start_times,
                semantic_duration_hours=semantic_duration_hours,
                window_mask=window_mask,
            )
            precedent_generation = self.precedent_memory.query_generation_prompt(
                state_packet=window_state_packet,
                query_state=self.latent_query_readout(next_window_head_context),
                next_window_header=next_window_header,
                memory_digest=persistent_digest,
                support_flags=current_support_flags,
            )
            shifted_prompt_tokens = self._shift_window_tensor(precedent_generation.prompt_tokens)
            precedent_generation_prompt_context = self._apply_precedent_prompt_tokens(
                local_hidden=local_hidden,
                prompt_tokens=shifted_prompt_tokens,
                attention_mask=local_attention_mask,
            )

            if (
                subject_ids is not None
                and trajectory_ords is not None
                and self.precedent_memory.has_index
            ):
                boundary_ords = torch.arange(
                    W,
                    device=global_states.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(B, W)
                precedent_anchor_item_ids = self.precedent_memory.lookup_anchor_item_ids(
                    subject_ids=subject_ids.to(device=global_states.device, dtype=torch.long),
                    trajectory_ords=trajectory_ords.to(device=global_states.device, dtype=torch.long),
                    boundary_ords=boundary_ords,
                )

        chunk_context = self.chunk_context_adapter(chunk_states)
        shifted_chunk_context = torch.zeros_like(chunk_context)
        if C > 1:
            shifted_chunk_context[:, :, 1:, :] = chunk_context[:, :, :-1, :]

        if self.global_fusion_mode == "film" and self.context_film is not None:
            gamma_beta = self.context_film(shifted_context)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma)
            fused_local = local_hidden * (1.0 + gamma.unsqueeze(2).unsqueeze(3)) + beta.unsqueeze(2).unsqueeze(3)
            fused_local = fused_local + shifted_chunk_context.unsqueeze(3)
            fused_local = fused_local + memory_context.unsqueeze(2).unsqueeze(3)
            fused_local = fused_local + precedent_generation_prompt_context
        else:
            fused_local = (
                local_hidden
                + shifted_context.unsqueeze(2).unsqueeze(3)
                + memory_context.unsqueeze(2).unsqueeze(3)
                + shifted_chunk_context.unsqueeze(3)
                + precedent_generation_prompt_context
            )

        if self.exclude_special_from_global_fusion and local_token_type_ids is not None:
            mask = (
                local_token_type_ids != int(getattr(self.config, "special_type_id", 0))
            ).unsqueeze(-1)
            fused_local = torch.where(mask, fused_local, local_hidden)

        fused_local = fused_local * local_attention_mask.unsqueeze(-1)
        fused_local = torch.nan_to_num(
            fused_local,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        event_logits_dict = {}
        if scatter_event_states:
            event_logits_dict = self.heads.forward_event(fused_local)
            if (
                self.event_value_code_conditioned_head is not None
                and event_input_ids is not None
                and event_attention_mask is not None
                and event_type_ids is not None
                and event_payload_ids is not None
            ):
                next_numeric_ids, _ = self._next_numeric_measurement_event_ids(
                    event_input_ids=event_input_ids,
                    event_attention_mask=event_attention_mask,
                    event_type_ids=event_type_ids,
                    event_payload_ids=event_payload_ids,
                )
                safe_next_numeric_ids = next_numeric_ids.clamp(
                    min=0,
                    max=max(0, int(self.embeddings.token_embedding.num_embeddings) - 1),
                )
                next_code_emb = self.embeddings.token_embedding(safe_next_numeric_ids)
                conditioned_input = torch.cat([fused_local, next_code_emb], dim=-1)
                raw_value = self.event_value_code_conditioned_head(conditioned_input)
                event_logits_dict["pred_event_value_mu"] = raw_value[..., 0]
                event_logits_dict["pred_event_value_sigma_raw"] = raw_value[..., 1]

        if scatter_event_states:
            assert token_event_index is not None
            fused_representation = self.event_composer.scatter_to_tokens(
                fused_local,
                token_event_index=token_event_index,
                attention_mask=attention_mask,
            )
        else:
            fused_representation = fused_local

        logits_dict = self.heads(fused_representation)
        logits_dict.update(event_logits_dict)
        if self.transition_boundary_head is not None:
            logits_dict["logits_transition_boundary"] = self.transition_boundary_head(
                fused_representation
            )
        if self.boundary_next_window_type_head is not None:
            logits_dict["logits_boundary_next_window_type"] = (
                self.boundary_next_window_type_head(fused_representation)
            )
        if self.next_window_type_head is not None:
            if "logits_next_window_type" in predicted_next_window_outputs:
                logits_dict["logits_next_window_type"] = predicted_next_window_outputs["logits_next_window_type"]
            else:
                logits_dict["logits_next_window_type"] = self.next_window_type_head(next_window_head_context)

        if self.event_time_nll_head is not None:
            raw = self.event_time_nll_head(fused_representation)
            logits_dict["pred_dt_next_mu"] = raw[..., 0]
            logits_dict["pred_dt_next_sigma"] = F.softplus(raw[..., 1]) + float(self.event_time_nll_min_sigma)
            if scatter_event_states:
                raw_event = self.event_time_nll_head(fused_local)
                logits_dict["pred_event_dt_next_mu"] = raw_event[..., 0]
                logits_dict["pred_event_dt_next_sigma"] = (
                    F.softplus(raw_event[..., 1]) + float(self.event_time_nll_min_sigma)
                )
                if "pred_event_value_sigma_raw" in logits_dict:
                    logits_dict["pred_event_value_sigma"] = (
                        F.softplus(logits_dict.pop("pred_event_value_sigma_raw"))
                        + float(self.event_time_nll_min_sigma)
                    )
        elif "pred_event_value_sigma_raw" in logits_dict:
            logits_dict["pred_event_value_sigma"] = (
                F.softplus(logits_dict.pop("pred_event_value_sigma_raw")) + 0.1
            )

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
        if self.next_window_gap_nll_head is not None:
            if "pred_next_window_gap_mu" in predicted_next_window_outputs:
                logits_dict["pred_next_window_gap_mu"] = predicted_next_window_outputs["pred_next_window_gap_mu"]
                logits_dict["pred_next_window_gap_sigma"] = predicted_next_window_outputs["pred_next_window_gap_sigma"]
            else:
                raw = self.next_window_gap_nll_head(next_window_head_context)
                logits_dict["pred_next_window_gap_mu"] = raw[..., 0]
                logits_dict["pred_next_window_gap_sigma"] = (
                    F.softplus(raw[..., 1]) + float(self.next_window_gap_nll_min_sigma)
                )
        if self.next_window_duration_nll_head is not None:
            if "pred_next_window_duration_mu" in predicted_next_window_outputs:
                logits_dict["pred_next_window_duration_mu"] = predicted_next_window_outputs["pred_next_window_duration_mu"]
                logits_dict["pred_next_window_duration_sigma"] = predicted_next_window_outputs[
                    "pred_next_window_duration_sigma"
                ]
            else:
                raw = self.next_window_duration_nll_head(next_window_head_context)
                logits_dict["pred_next_window_duration_mu"] = raw[..., 0]
                logits_dict["pred_next_window_duration_sigma"] = (
                    F.softplus(raw[..., 1]) + float(self.next_window_duration_nll_min_sigma)
                )
        if self.next_window_support_head is not None:
            if "logits_next_window_support" in predicted_next_window_outputs:
                logits_dict["logits_next_window_support"] = predicted_next_window_outputs["logits_next_window_support"]
                logits_dict["pred_next_window_support_probs"] = predicted_next_window_outputs[
                    "pred_next_window_support_probs"
                ]
            else:
                logits_support = self.next_window_support_head(next_window_head_context)
                logits_dict["logits_next_window_support"] = logits_support
                logits_dict["pred_next_window_support_probs"] = torch.sigmoid(logits_support)

        if precedent_out is not None:
            logits_dict["precedent_summary_prior"] = (
                precedent_out.summary_prior if precedent_out.summary_prior is not None else precedent_out.future_summary
            )
            logits_dict["precedent_prompt_tokens"] = precedent_out.prompt_tokens
            logits_dict["precedent_prompt_summary"] = precedent_out.prompt_summary
            logits_dict["precedent_future_summary"] = precedent_out.future_summary
            logits_dict["precedent_future_embedding"] = precedent_out.future_embedding
            logits_dict["precedent_query_embedding"] = precedent_out.query_embedding
            logits_dict["precedent_retrieval_scores"] = precedent_out.retrieval_scores
            logits_dict["precedent_candidate_weights"] = precedent_out.candidate_weights
            logits_dict["precedent_candidate_prompt_tokens"] = precedent_out.candidate_prompt_tokens
            logits_dict["precedent_candidate_future_summaries"] = precedent_out.candidate_future_summaries
            logits_dict["precedent_candidate_future_embeddings"] = precedent_out.candidate_future_embeddings
            logits_dict["precedent_matched_item_ids"] = precedent_out.matched_item_ids
            if precedent_target_future_summary is not None:
                logits_dict["precedent_target_future_summary"] = precedent_target_future_summary
            if precedent_target_future_embedding is not None:
                logits_dict["precedent_target_future_embedding"] = precedent_target_future_embedding
            if precedent_target_future_mask is not None:
                logits_dict["precedent_target_future_mask"] = precedent_target_future_mask
            if precedent_anchor_item_ids is not None:
                logits_dict["precedent_anchor_item_ids"] = precedent_anchor_item_ids
        if precedent_generation is not None:
            logits_dict["precedent_generation_summary_prior"] = precedent_generation.summary_prior
            logits_dict["precedent_generation_prompt_tokens"] = precedent_generation.prompt_tokens
            logits_dict["precedent_generation_prompt_summary"] = precedent_generation.prompt_summary
            logits_dict["precedent_generation_candidate_weights"] = precedent_generation.candidate_weights
            logits_dict["precedent_generation_matched_item_ids"] = precedent_generation.matched_item_ids
        if next_window_header is not None:
            logits_dict["next_window_header_type_ids"] = next_window_header.window_type_ids
            logits_dict["next_window_header_gap_hours"] = next_window_header.gap_hours
            logits_dict["next_window_header_duration_hours"] = next_window_header.duration_hours
            if next_window_header.support_flags is not None:
                logits_dict["next_window_header_support_flags"] = next_window_header.support_flags

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
                "pred_event_values",
                "pred_event_value_mu",
                "pred_event_value_sigma",
                "pred_dt_next_mu",
                "pred_dt_next_sigma",
                "pred_event_dt_next_mu",
                "pred_event_dt_next_sigma",
                "pred_next_window_gap_mu",
                "pred_next_window_gap_sigma",
                "logits_event_token",
                "logits_event_family",
                "logits_event_payload",
                "logits_event_concept_special",
                "logits_event_concept_measurement",
                "logits_event_concept_diagnosis",
                "logits_event_concept_procedure",
                "logits_event_concept_medication",
                "logits_event_concept_structural",
                "logits_transition_boundary",
                "logits_boundary_next_window_type",
            ):
                if key in logits_dict and logits_dict[key] is not None:
                    if logits_dict[key].ndim >= 3 and logits_dict[key].shape[2] == 1:
                        logits_dict[key] = logits_dict[key].squeeze(2)

        if return_aux_state:
            return logits_dict, {
                "global_state": final_state,
                "window_global_states": global_states,
                "memory_state": final_memory_state,
                "patient_memory_context": memory_out.context if memory_out is not None else None,
                "patient_memory_context_by_bank": memory_context_by_bank,
                "patient_memory_state_digests_by_bank": memory_state_digests_by_bank,
                "precedent_memory": precedent_out,
                "precedent_generation": precedent_generation,
                "precedent_generation_prompt_context": precedent_generation_prompt_context,
                "precedent_target_future_summary": precedent_target_future_summary,
                "precedent_target_future_mask": precedent_target_future_mask,
                "next_window_header": next_window_header,
                "window_state_packet": window_state_packet,
                "window_packet_summary": window_packet_summary,
            }
        return logits_dict, final_state
