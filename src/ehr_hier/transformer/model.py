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

        # 1. The Time Engine (Shared cRoPE)
        # We share it because Time is Time, whether Local or Global.
        self.rope = ContinuousRotaryPositionalEmbedding(
            d_model=config.d_model,
            max_period=config.rope_max_period
        )

        # 2. Input Embeddings (Discrete + Side-Channel)
        self.embeddings = AETEmbeddings(
            vocab_size=vocab_config['total_size'],
            d_model=config.d_model,
            dropout=config.dropout
        )

        # 3. Local Encoder (The Ribs)
        self.local_encoder = AETLocalEncoder(config, self.rope)

        # 4. Global Aggregator (The Spine)
        self.global_aggregator = AETGlobalAggregator(config, self.rope)

        # 5. Output Heads (The Brain)
        self.heads = AETOutputHeads(config.d_model, vocab_config)

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
            prev_global_state=None  # (B, D) - From previous sequence chunk
    ):
        B, W, L = input_ids.shape
        D = self.config.d_model

        # --- PHASE 1: Embed & Local Encode ---

        # A. Input Embedding
        x = self.embeddings(input_ids, numeric_values)  # (B, W, L, D)

        # B. Local Transformer Pass
        # Returns:
        #   local_hidden: (B, W, L, D) - The token-level representations
        #   summaries:    (B, W, D)    - The [CLS] token of each window
        local_hidden, summaries = self.local_encoder(x, time_ids, attention_mask)

        # --- PHASE 2: Global Aggregation ---

        # A. Prepare Global Inputs
        # Global Times: We use the start time of each window (Time of token 0)
        # time_ids is (B, W, L), so we take slice [:, :, 0]
        window_start_times = time_ids[:, :, 0]  # (B, W)

        # Global Mask: If a window contains ANY real tokens, it's real.
        # Check if index 0 (CLS) is masked.
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

        # The Goal: Window[i] should know the context of Window[0...i]
        # Since GlobalAggregator is Causal, global_states[i] contains info 0..i

        # We project it to the joint space
        global_context = self.context_adapter(global_states)  # (B, W, D)

        # Broadcast Global Context to all tokens in the window
        # (B, W, D) -> (B, W, 1, D) -> Add to (B, W, L, D)
        # This adds "I am in an ICU Window (Window 5)" to every heartbeat token
        fused_representation = local_hidden + global_context.unsqueeze(2)

        # --- PHASE 4: Prediction Heads ---

        # We can flatten B & W for efficiency in the Linear Layers
        fused_flat = fused_representation.view(B * W * L, D)

        logits_dict = self.heads(fused_flat)

        # Reshape logits back to (B, W, L, Vocab) if needed,
        # but usually Loss function prefers flat or (B, Seq_Len, Vocab).
        # Let's keep them flat or reshape to (B, W, L, ...) depending on Loss implementation.
        # For our Loss module, we likely want (B, S, ...) where S = W*L.

        # Also return global_states[:, -1] to carry over to the next chunk!
        return logits_dict, global_states[:, -1, :]