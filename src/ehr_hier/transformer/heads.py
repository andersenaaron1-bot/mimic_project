import torch.nn as nn
from typing import Dict


EVENT_CONCEPT_FAMILY_ORDER = (
    "special",
    "measurement",
    "diagnosis",
    "procedure",
    "medication",
    "structural",
)


def build_event_concept_family_sizes(vocab_config: dict) -> Dict[str, int]:
    dense_blocks = vocab_config.get("dense_blocks", [])
    if not isinstance(dense_blocks, list):
        return {name: 0 for name in EVENT_CONCEPT_FAMILY_ORDER}

    block_to_family = {
        "special": "special",
        "measurement_code": "measurement",
        "observation_code": "measurement",
        "diagnosis": "diagnosis",
        "diagnosis_residual": "diagnosis",
        "procedure": "procedure",
        "procedure_residual": "procedure",
        "medication": "medication",
        "medication_residual": "medication",
        "structural": "structural",
    }
    family_sizes = {name: 0 for name in EVENT_CONCEPT_FAMILY_ORDER}
    for block in dense_blocks:
        if not isinstance(block, dict):
            continue
        block_name = str(block.get("name", ""))
        family_name = block_to_family.get(block_name, None)
        if family_name is None:
            continue
        family_sizes[family_name] += int(block.get("dense_size", 0))
    return family_sizes


class AETPrecedentHeads(nn.Module):
    """
    Learned retrieval-space heads for Phase 4 precedent memory.

    These heads sit on top of the offline Phase 3 precedent store:
    - the query head encodes the current predictive state
    - the key projector maps stored keys into the learned space
    - the future projector maps structured future summaries into a comparable
      supervision space
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query_head = nn.Sequential(
            nn.LazyLinear(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.key_projector = nn.Sequential(
            nn.LazyLinear(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.future_summary_projector = nn.Sequential(
            nn.LazyLinear(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode_query(self, features):
        return self.query_head(features)

    def project_keys(self, features):
        return self.key_projector(features)

    def project_future_summaries(self, features):
        return self.future_summary_projector(features)


class AETOutputHeads(nn.Module):
    """
    Output projections for token and side-channel prediction.

    In the current compact-runtime regime, a unified dense token head is
    feasible and is the preferred path for autoregressive next-token
    training. Switched subspace heads remain available as a legacy/ablation
    option.
    """

    def __init__(
        self,
        d_model,
        vocab_config,
        *,
        use_unified_token_head: bool = True,
        emit_switched_heads: bool = True,
        emit_event_heads: bool = False,
        num_event_families: int = 0,
        num_event_payloads: int = 0,
    ):
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
        self.emit_event_heads = bool(emit_event_heads)
        event_concept_family_sizes = build_event_concept_family_sizes(vocab_config)

        self.token_head = (
            nn.Linear(d_model, int(vocab_config["total_size"]))
            if self.use_unified_token_head
            else None
        )
        self.event_family_head = (
            nn.Linear(d_model, int(num_event_families))
            if self.emit_event_heads and int(num_event_families) > 0
            else None
        )
        self.event_payload_head = (
            nn.Linear(d_model, int(num_event_payloads))
            if self.emit_event_heads and int(num_event_payloads) > 0
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
        self.event_concept_heads = nn.ModuleDict(
            {
                family_name: nn.Linear(d_model, int(size))
                for family_name, size in event_concept_family_sizes.items()
                if self.emit_event_heads and int(size) > 0
            }
        )

        # 5. Attribute Regression Head (Side-Channel)
        # Predicts log1p(dosage) or log1p(duration)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.event_value_nll_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 2),
            )
            if self.emit_event_heads
            else None
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

    def forward_event(self, hidden_states):
        out = {
            "pred_event_values": self.value_head(hidden_states)
        }
        if self.event_value_nll_head is not None:
            raw = self.event_value_nll_head(hidden_states)
            out["pred_event_value_mu"] = raw[..., 0]
            out["pred_event_value_sigma_raw"] = raw[..., 1]
        if self.token_head is not None:
            out["logits_event_token"] = self.token_head(hidden_states)
        if self.event_family_head is not None:
            out["logits_event_family"] = self.event_family_head(hidden_states)
        if self.event_payload_head is not None:
            out["logits_event_payload"] = self.event_payload_head(hidden_states)
        for family_name, head in self.event_concept_heads.items():
            out[f"logits_event_concept_{family_name}"] = head(hidden_states)
        return out
