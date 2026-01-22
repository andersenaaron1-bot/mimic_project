import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import AETEmbeddings, ContinuousRotaryPositionalEmbedding
from .encoder import AETLocalEncoder
from .aggregator import AETGlobalAggregator
from .heads import AETOutputHeads


class AdaptiveEpisodicTransformer(nn.Module):
    def __init__(self, config, vocab_config):
        """
        Args:
            config: Namespace with d_model, num_layers, etc.
            vocab_config: Dict with vocabulary offsets/sizes.
        """
        super().__init__()
        self.config = config
        self.vocab_config = dict(vocab_config)

        window_markers_cfg = dict(self.vocab_config.get("window_markers", {}) or {})
        self.window_marker_type_offset = int(window_markers_cfg.get("type_token_offset", 0))
        self.window_marker_num_types = int(window_markers_cfg.get("num_types", 0))
        self.window_marker_end_token_id = window_markers_cfg.get("end_token_id", None)
        self.window_marker_end_mode = str(window_markers_cfg.get("end_mode", "end_token"))

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

        # 1. The Time Engine (Shared cRoPE)
        # We share it because Time is Time, whether Local or Global.
        self.rope = ContinuousRotaryPositionalEmbedding(
            dim=config.d_model // config.num_heads,
            max_period=config.rope_max_period
        )

        # 2. Input Embeddings (Discrete + Side-Channel)
        self.embeddings = AETEmbeddings(
            vocab_size=vocab_config['total_size'],
            d_model=config.d_model,
            dropout=config.dropout,
            num_window_types=self.num_window_types,
            special_type_id=int(getattr(config, "special_type_id", 0)),
            exclude_special_from_window_type=True,
        )

        # 3. Local Encoder (The Ribs)
        self.local_encoder = AETLocalEncoder(config, self.rope)

        # 4. Global Aggregator (The Spine)
        self.global_aggregator = AETGlobalAggregator(config, self.rope)

        # 5. Output Heads (The Brain)
        self.heads = AETOutputHeads(config.d_model, vocab_config)

        # 5b. Aux Head: Predict next window type from global state.
        self.next_window_type_head = (
            nn.Linear(config.d_model, self.num_window_types) if self.num_window_types > 0 else None
        )

        # 5c. Aux Head: Predict window duration/density from (history + current type).
        self.window_len_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            if bool(getattr(config, "enable_window_len_head", False))
            else None
        )
        self.window_type_control_embedding = (
            nn.Embedding(self.num_window_types, config.d_model) if self.num_window_types > 0 else None
        )

        # 5d. Global→Local transition bias (optional).
        # This biases the logits of window-transition marker tokens (WIN_END or WIN_<TYPE>)
        # based on predicted window duration/density and next-window-type prior.
        self.enable_transition_bias = bool(getattr(config, "enable_transition_bias", False))
        self.transition_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.transition_hazard_scale = nn.Parameter(torch.tensor(1.0))

        # 6. Global-to-Local Projection (Optional but recommended)
        # Helps adapt the global context before adding it to local tokens
        self.context_adapter = nn.Linear(config.d_model, config.d_model)
        self.global_fusion_mode = str(getattr(config, "global_fusion_mode", "add")).lower()
        if self.global_fusion_mode not in {"add", "film"}:
            raise ValueError(f"Unsupported global_fusion_mode={self.global_fusion_mode!r}; expected 'add' or 'film'.")
        self.exclude_special_from_global_fusion = bool(getattr(config, "exclude_special_from_global_fusion", False))
        self.context_film = nn.Linear(config.d_model, 2 * config.d_model) if self.global_fusion_mode == "film" else None

    def forward(
            self,
            input_ids,  # (B, W, L)
            time_ids,  # (B, W, L) - Float times
            numeric_values,  # (B, W, L, 1)
            token_type_ids,  # (B, W, L)
            attention_mask,  # (B, W, L) - 1 for real, 0 for pad
            prev_global_state=None,  # (B, D) - From previous sequence chunk
            window_start_times=None,  # Optional (B, W) absolute start times
            window_mask=None,  # Optional (B, W) 1 for real, 0 for pad
            window_type_ids=None,  # Optional (B, W) integer window type ids
    ):
        B, W, L = input_ids.shape
        D = self.config.d_model

        # --- PHASE 1: Embed & Local Encode ---

        # A. Input Embedding
        x = self.embeddings(
            input_ids,
            numeric_values,
            window_type_ids=window_type_ids,
            token_type_ids=token_type_ids,
        )  # (B, W, L, D)

        # B. Local Transformer Pass
        # Returns:
        #   local_hidden: (B, W, L, D) - The token-level representations
        #   summaries:    (B, W, D)    - Attention-pooled summary of each window
        local_hidden, summaries = self.local_encoder(x, time_ids, attention_mask, token_type_ids=token_type_ids)

        # --- PHASE 2: Global Aggregation ---

        # A. Prepare Global Inputs
        # Global Times: Prefer absolute window start times if provided by the collator.
        if window_start_times is None:
            # Fallback: relative start time (often 0.0 if window-local times are used)
            window_start_times = time_ids[:, :, 0]  # (B, W)

        # Global Mask: Prefer explicit window mask if provided by the collator.
        if window_mask is None:
            # If a window contains ANY real tokens, token 0 should be real.
            window_mask = attention_mask[:, :, 0]  # (B, W)

        # B. Global Transformer Pass
        # Returns: (B, W, D) - The state of the trajectory at each step
        global_states = self.global_aggregator(
            summaries,
            window_start_times,
            window_mask,
            prev_context_state=prev_global_state
        )

        # --- PHASE 3: Context Injection (Fusion) ---

        # Causality: Window[w] must only receive context from windows < w.
        # Since GlobalAggregator is causal, global_states[:, w-1] contains info about 0..w-1.
        global_context = self.context_adapter(global_states)  # (B, W, D)

        shifted_context = torch.zeros_like(global_context)
        if prev_global_state is not None:
            shifted_context[:, 0, :] = self.context_adapter(prev_global_state)
        if W > 1:
            shifted_context[:, 1:, :] = global_context[:, :-1, :]

        # Late fusion: condition all token predictions in window[w] on context up to w-1.
        if self.global_fusion_mode == "film" and self.context_film is not None:
            gamma_beta = self.context_film(shifted_context)  # (B,W,2D)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma)
            fused_representation = local_hidden * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        else:
            fused_representation = local_hidden + shifted_context.unsqueeze(2)

        if self.exclude_special_from_global_fusion and token_type_ids is not None:
            mask = (token_type_ids != int(getattr(self.config, "special_type_id", 0))).unsqueeze(-1)
            fused_representation = torch.where(mask, fused_representation, local_hidden)

        fused_representation = fused_representation * attention_mask.unsqueeze(-1)

        # --- PHASE 4: Prediction Heads ---

        logits_dict = self.heads(fused_representation)
        if self.next_window_type_head is not None:
            logits_dict["logits_next_window_type"] = self.next_window_type_head(global_states)  # (B, W, K)

        # --- PHASE 4b: Duration/Density prediction + transition bias (optional) ---
        if self.window_len_head is not None:
            control_ctx = shifted_context
            if window_type_ids is not None and self.window_type_control_embedding is not None:
                safe_ids = window_type_ids.clamp(min=0, max=max(0, self.num_window_types - 1))
                control_ctx = control_ctx + self.window_type_control_embedding(safe_ids)

            raw = self.window_len_head(control_ctx)  # (B, W, 2)
            # Positive window length priors
            pred_len_tokens = F.softplus(raw[..., 0]) + 1.0
            pred_len_hours = F.softplus(raw[..., 1]) + 0.25
            logits_dict["pred_window_len_tokens"] = pred_len_tokens  # (B, W)
            logits_dict["pred_window_len_hours"] = pred_len_hours  # (B, W)

        if self.enable_transition_bias and self.size_special > 0:
            # Compute hazard bias if length head is available; otherwise keep hazard at 0.
            if self.window_len_head is not None and "pred_window_len_tokens" in logits_dict and "pred_window_len_hours" in logits_dict:
                pred_len_tokens = logits_dict["pred_window_len_tokens"]  # (B, W)
                pred_len_hours = logits_dict["pred_window_len_hours"]  # (B, W)

                content_mask = attention_mask.to(dtype=torch.bool)
                # Exclude SPECIAL (e.g. summaries, window markers) from length/density counting.
                if token_type_ids is not None:
                    content_mask = content_mask & (token_type_ids != int(getattr(self.config, "special_type_id", 0)))

                pos = torch.cumsum(content_mask.to(dtype=torch.float32), dim=2)  # (B, W, L)
                t_rel = time_ids.clamp(min=0.0)  # (B, W, L), hours since window start

                eps = 1e-6
                prog_tokens = pos / (pred_len_tokens.unsqueeze(-1) + eps)
                prog_time = t_rel / (pred_len_hours.unsqueeze(-1) + eps)
                prog = 0.5 * (prog_tokens + prog_time)

                hazard_logit = self.transition_hazard_scale * (prog - 1.0)  # (B, W, L)
            else:
                hazard_logit = time_ids.new_zeros((B, W, L))

            logits_struct = logits_dict.get("logits_struct", None)
            if logits_struct is not None:
                # Determine which marker tokens to bias.
                end_mode = self.window_marker_end_mode
                if end_mode == "next_type" and self.window_marker_num_types > 0 and self.next_window_type_head is not None:
                    # Bias WIN_<TYPE> token logits (encouraging a transition to a particular next type).
                    K = int(self.window_marker_num_types)
                    # Map WIN_<TYPE> global token ids -> local indices within the SPECIAL head.
                    type_token_ids = self.window_marker_type_offset + torch.arange(K, device=logits_struct.device)
                    type_local = type_token_ids - int(self.special_token_offset)
                    valid = (type_local >= 0) & (type_local < int(self.size_special))
                    if valid.any():
                        prior = logits_dict.get("logits_next_window_type", None)
                        if prior is not None and prior.shape[-1] == K:
                            prior_logp = torch.log_softmax(prior, dim=-1)  # (B, W, K)
                        else:
                            prior_logp = logits_struct.new_zeros((B, W, K))

                        type_local_valid = type_local[valid].to(dtype=torch.long)
                        bias = hazard_logit.unsqueeze(-1) + (self.transition_prior_scale * prior_logp).unsqueeze(2)
                        logits_struct[..., type_local_valid] = logits_struct[..., type_local_valid] + bias[..., valid]

                else:
                    # Bias WIN_END token logit (encouraging an end-of-window transition).
                    end_token_id = (
                        int(self.window_marker_end_token_id)
                        if self.window_marker_end_token_id is not None
                        else int(self.window_marker_type_offset) + int(self.window_marker_num_types)
                    )
                    end_local = end_token_id - int(self.special_token_offset)
                    if 0 <= int(end_local) < int(self.size_special):
                        logits_struct[..., int(end_local)] = logits_struct[..., int(end_local)] + hazard_logit

        # Return final global state for sequence-chunk stitching.
        # Use the last *real* window rather than the padded tail.
        if W == 0:
            final_state = torch.zeros((B, D), device=global_states.device, dtype=global_states.dtype)
        else:
            n_real = window_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)  # (B,)
            last_idx = (n_real - 1).clamp(min=0)  # (B,)
            batch_idx = torch.arange(B, device=global_states.device)
            final_state = global_states[batch_idx, last_idx, :]

        return logits_dict, final_state
