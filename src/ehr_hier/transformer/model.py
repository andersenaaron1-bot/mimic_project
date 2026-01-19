import torch
import torch.nn as nn

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

        num_window_types = int(
            getattr(
                config,
                "num_window_types",
                vocab_config.get("window_markers", {}).get("num_types", 16),
            )
        )
        self.num_window_types = max(0, num_window_types)

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

        # 6. Global-to-Local Projection (Optional but recommended)
        # Helps adapt the global context before adding it to local tokens
        self.context_adapter = nn.Linear(config.d_model, config.d_model)

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
        fused_representation = local_hidden + shifted_context.unsqueeze(2)
        fused_representation = fused_representation * attention_mask.unsqueeze(-1)

        # --- PHASE 4: Prediction Heads ---

        logits_dict = self.heads(fused_representation)
        if self.next_window_type_head is not None:
            logits_dict["logits_next_window_type"] = self.next_window_type_head(global_states)  # (B, W, K)

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
