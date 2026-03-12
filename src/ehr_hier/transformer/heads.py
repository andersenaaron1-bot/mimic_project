import torch
import torch.nn as nn


class AETOutputHeads(nn.Module):
    """
    Output projections for token and side-channel prediction.

    In the current compact-runtime regime, a unified dense token head is
    feasible and is the preferred path for autoregressive next-token
    training. Switched subspace heads remain available as a legacy/ablation
    option.
    """

    def __init__(self, d_model, vocab_config, *, use_unified_token_head: bool = True, emit_switched_heads: bool = True):
        """
        Args:
            d_model: Transformer hidden dimension (e.g., 768).
            vocab_config: A dictionary/object containing the size/offsets of each lane.
                          Expects: {
                              'size_special': int,
                              'size_rvq': int,
                              'size_meds': int, # Includes Meds, Diags, Procs flattened
                              'size_meas_labels': int
                          }
        """
        super().__init__()

        self.use_unified_token_head = bool(use_unified_token_head)
        self.emit_switched_heads = bool(emit_switched_heads)

        self.token_head = (
            nn.Linear(d_model, int(vocab_config["total_size"]))
            if self.use_unified_token_head
            else None
        )

        self.struct_head = (
            nn.Linear(d_model, vocab_config['size_special'])
            if self.emit_switched_heads
            else None
        )
        self.rvq_head = (
            nn.Linear(d_model, vocab_config['size_rvq'])
            if self.emit_switched_heads
            else None
        )
        self.meas_head = (
            nn.Linear(d_model, vocab_config['size_meas_labels'])
            if self.emit_switched_heads
            else None
        )
        self.medtok_head = (
            nn.Linear(d_model, vocab_config['size_meds'])
            if self.emit_switched_heads
            else None
        )

        # 5. Attribute Regression Head (Side-Channel)
        # Predicts log1p(dosage) or log1p(duration)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (Batch, Num_Windows, Window_Len, Dim)
                           or (Batch * Windows, Tokens, Dim)

        Returns:
            A dictionary of logits. We compute ALL of them for every token.
            The Loss function will select the correct one using masking.
        """
        out = {
            "pred_values": self.value_head(hidden_states)
        }
        if self.token_head is not None:
            out["logits_token"] = self.token_head(hidden_states)
        if self.struct_head is not None:
            out["logits_struct"] = self.struct_head(hidden_states)
        if self.rvq_head is not None:
            out["logits_rvq"] = self.rvq_head(hidden_states)
        if self.meas_head is not None:
            out["logits_meas"] = self.meas_head(hidden_states)
        if self.medtok_head is not None:
            out["logits_medtok"] = self.medtok_head(hidden_states)
        return out
