import torch
import torch.nn as nn


class AETOutputHeads(nn.Module):
    """
    The Switched Output Layer.
    Contains separate projections for different vocabulary subspaces to
    avoid the computational bottleneck of a monolithic Softmax(400k).
    """

    def __init__(self, d_model, vocab_config):
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

        # 1. Structural Head (Window Delimiters, Special Tokens)
        # Target Range: 0 - 999 (SPECIAL)
        self.struct_head = nn.Linear(d_model, vocab_config['size_special'])

        # 2. RVQ Head (Physiological Signals)
        # Target Range: 1,000 - 4,999 (RVQ)
        self.rvq_head = nn.Linear(d_model, vocab_config['size_rvq'])

        # 3. Measurement Label Head (e.g., "Heart Rate Identity")
        # Target Range: 10,000 - 29,999 (MEAS)
        self.meas_head = nn.Linear(d_model, vocab_config['size_meas_labels'])

        # 4. Clinical Semantic Head (The Big One)
        # Target Range: 30,000 - End (MED, DIAG, PROC)
        # We group these because they share the same "Ontological" latent space
        self.medtok_head = nn.Linear(d_model, vocab_config['size_meds'])

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
        return {
            "logits_struct": self.struct_head(hidden_states),
            "logits_rvq": self.rvq_head(hidden_states),
            "logits_meas": self.meas_head(hidden_states),
            "logits_medtok": self.medtok_head(hidden_states),
            "pred_values": self.value_head(hidden_states)
        }