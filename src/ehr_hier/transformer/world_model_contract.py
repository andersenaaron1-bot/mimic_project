from __future__ import annotations

from dataclasses import dataclass

import torch

from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_ORDER
from src.ehr_hier.data.token_types import TokenCategory


STATE_PACKET_SLOT_ORDER: tuple[str, ...] = (
    "end",
    "burden",
    "delta",
    "volatility",
    "intervention",
    "physiology",
)

PRECEDENT_SUPPORT_FLAG_ORDER: tuple[str, ...] = (
    "structural_present",
    "chronic_diagnosis_present",
    "procedure_present",
    "medication_present",
    "measurement_present",
    "extreme_measurement_present",
)

PRECEDENT_ANCHOR_MASK_FLAG_ORDER: tuple[str, ...] = (
    "first_window",
    "long_gap_prev",
    "future_truncated",
    "future_missing",
)

PRECEDENT_TRANSITION_FLAG_ORDER: tuple[str, ...] = (
    "window_type_changed",
    "structural_present",
    "extreme_measurement_present",
    "future_truncated",
)

NUM_TOKEN_CATEGORIES: int = int(max(int(cat) for cat in TokenCategory)) + 1
NUM_EVENT_PAYLOAD_KINDS: int = int(len(EVENT_PAYLOAD_KIND_ORDER))
NUM_SUPPORT_FLAGS: int = int(len(PRECEDENT_SUPPORT_FLAG_ORDER))
NUM_ANCHOR_MASK_FLAGS: int = int(len(PRECEDENT_ANCHOR_MASK_FLAG_ORDER))
NUM_TRANSITION_FLAGS: int = int(len(PRECEDENT_TRANSITION_FLAG_ORDER))
PRECEDENT_INDEX_VERSION: int = 2


@dataclass
class WindowStatePacket:
    """
    Canonical boundary-state object for the dual-memory patient world model.

    The repo currently still uses a single semantic summary vector in the global
    path. This packet is the intended replacement. It separates:

    - compressive state evidence for the latent update
    - exact write candidates for patient-internal memory
    - a predictive-state query substrate for precedent retrieval

    Shapes are left flexible enough to support both batched `(B, W, ...)` and
    per-window `(B, ...)` usage during migration.
    """

    slot_tokens: torch.Tensor
    slot_mask: torch.Tensor | None = None
    query_token: torch.Tensor | None = None
    write_tokens: torch.Tensor | None = None
    write_mask: torch.Tensor | None = None
    window_type_ids: torch.Tensor | None = None
    start_hours: torch.Tensor | None = None
    duration_hours: torch.Tensor | None = None
    gap_prev_hours: torch.Tensor | None = None

    def summary(self) -> torch.Tensor:
        """
        Return the default compressive summary used during migration.

        Preference order:
        1. explicit query token
        2. masked mean over slot tokens
        3. unmasked mean over slot tokens
        """
        if self.query_token is not None:
            return self.query_token
        if self.slot_mask is not None:
            weights = self.slot_mask.to(dtype=self.slot_tokens.dtype).unsqueeze(-1)
            denom = weights.sum(dim=-2).clamp(min=1.0)
            return (self.slot_tokens * weights).sum(dim=-2) / denom
        return self.slot_tokens.mean(dim=-2)


@dataclass
class PatientMemoryReadout:
    """
    Readout from patient-internal memory.

    The final architecture is expected to carry separate static, persistent, and
    episodic banks. This readout is bank-agnostic so the current single-bank
    implementation can migrate without another interface break.
    """

    context_tokens: torch.Tensor
    context_summary: torch.Tensor
    retrieved_event_ids: torch.Tensor | None = None
    bank_ids: torch.Tensor | None = None
    retrieval_scores: torch.Tensor | None = None


@dataclass
class FutureSnippetRef:
    """
    Reference back into a packed precompiled shard for exact continuation access.
    """

    rel_path_id: torch.Tensor
    subject_idx: torch.Tensor
    trajectory_ord: torch.Tensor
    start_boundary_ord: torch.Tensor
    stop_boundary_ord: torch.Tensor


@dataclass
class FutureSummary:
    """
    Structured future summary for one horizon.

    Scalars stay in natural units and are vectorized only when needed so the
    stored representation remains interpretable.
    """

    next_window_type_id: torch.Tensor
    next_window_gap_h: torch.Tensor
    next_window_duration_h: torch.Tensor
    event_family_hist: torch.Tensor
    payload_hist: torch.Tensor
    support_flags: torch.Tensor
    transition_flags: torch.Tensor
    event_count: torch.Tensor
    measurement_count: torch.Tensor
    extreme_measurement_count: torch.Tensor
    numeric_severity: torch.Tensor
    terminal_window_type_id: torch.Tensor
    future_window_count: torch.Tensor

    def to_vector(self, *, num_window_types: int) -> torch.Tensor:
        device = self.event_family_hist.device
        dtype = self.event_family_hist.dtype
        next_type = _safe_one_hot(
            self.next_window_type_id.to(device=device),
            num_classes=max(0, int(num_window_types)),
            dtype=dtype,
        )
        terminal_type = _safe_one_hot(
            self.terminal_window_type_id.to(device=device),
            num_classes=max(0, int(num_window_types)),
            dtype=dtype,
        )
        scalar = torch.stack(
            [
                torch.log1p(self.next_window_gap_h.to(dtype=dtype).clamp(min=0.0)),
                torch.log1p(self.next_window_duration_h.to(dtype=dtype).clamp(min=0.0)),
                torch.log1p(self.event_count.to(dtype=dtype).clamp(min=0.0)),
                torch.log1p(self.measurement_count.to(dtype=dtype).clamp(min=0.0)),
                torch.log1p(self.extreme_measurement_count.to(dtype=dtype).clamp(min=0.0)),
                torch.log1p(self.future_window_count.to(dtype=dtype).clamp(min=0.0)),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                next_type,
                scalar,
                self.event_family_hist.to(dtype=dtype),
                self.payload_hist.to(dtype=dtype),
                self.support_flags.to(dtype=dtype),
                self.transition_flags.to(dtype=dtype),
                self.numeric_severity.to(dtype=dtype),
                terminal_type,
            ],
            dim=-1,
        )


@dataclass
class PrecedentIndexItem:
    """
    One item in the cohort-external precedent index.

    Each item represents the predictive state after one semantic window and
    before generation of the next window.
    """

    item_id: torch.Tensor
    subject_id: torch.Tensor
    trajectory_ord: torch.Tensor
    boundary_ord: torch.Tensor
    anchor_window_ord: torch.Tensor
    current_window_type_id: torch.Tensor
    current_window_start_h: torch.Tensor
    current_window_duration_h: torch.Tensor
    gap_prev_h: torch.Tensor
    support_flags: torch.Tensor
    anchor_mask_flags: torch.Tensor
    key_state: torch.Tensor
    key_packet: torch.Tensor
    key_memory: torch.Tensor
    future_summary_h1: FutureSummary
    future_summary_h2: FutureSummary
    future_summary_h3: FutureSummary
    future_prefix_prompt: torch.Tensor
    future_snippet_ref: FutureSnippetRef


@dataclass
class PrecedentIndexStore:
    """
    Dense searchable precedent store saved by the offline Phase 3 builder.
    """

    version: int
    rel_path_vocab: list[str]
    item_ids: torch.Tensor
    subject_ids: torch.Tensor
    trajectory_ords: torch.Tensor
    boundary_ords: torch.Tensor
    anchor_window_ords: torch.Tensor
    current_window_type_ids: torch.Tensor
    current_window_start_h: torch.Tensor
    current_window_duration_h: torch.Tensor
    gap_prev_h: torch.Tensor
    support_flags: torch.Tensor
    anchor_mask_flags: torch.Tensor
    key_state: torch.Tensor
    key_packet: torch.Tensor
    key_memory: torch.Tensor
    future_h1: torch.Tensor
    future_h2: torch.Tensor
    future_h3: torch.Tensor
    future_prefix_prompt: torch.Tensor
    future_snippet_rel_path_ids: torch.Tensor
    future_snippet_subject_idxs: torch.Tensor
    future_snippet_trajectory_ords: torch.Tensor
    future_snippet_start_boundary_ords: torch.Tensor
    future_snippet_stop_boundary_ords: torch.Tensor
    num_window_types: int


FutureSummaryH1 = FutureSummary
FutureSummaryH2 = FutureSummary
FutureSummaryH3 = FutureSummary


@dataclass
class PrecedentMemoryReadout:
    """
    Readout from cohort-external precedent memory.

    This should expose both a context summary for conditioning and retrieval
    metadata for diagnostics and evaluation.
    """

    context_tokens: torch.Tensor
    context_summary: torch.Tensor
    summary_prior: torch.Tensor | None = None
    prompt_tokens: torch.Tensor | None = None
    prompt_summary: torch.Tensor | None = None
    future_summary: torch.Tensor | None = None
    future_embedding: torch.Tensor | None = None
    query_embedding: torch.Tensor | None = None
    retrieval_scores: torch.Tensor | None = None
    candidate_weights: torch.Tensor | None = None
    candidate_prompt_tokens: torch.Tensor | None = None
    candidate_future_summaries: torch.Tensor | None = None
    candidate_future_embeddings: torch.Tensor | None = None
    matched_item_ids: torch.Tensor | None = None
    matched_subject_ids: torch.Tensor | None = None
    matched_trajectory_ords: torch.Tensor | None = None
    matched_window_ords: torch.Tensor | None = None
    matched_snippet_rel_path_ids: torch.Tensor | None = None
    matched_snippet_subject_idxs: torch.Tensor | None = None
    matched_snippet_start_boundary_ords: torch.Tensor | None = None
    matched_snippet_stop_boundary_ords: torch.Tensor | None = None


@dataclass
class NextWindowHeader:
    """
    Coarse next-window plan used to bridge boundary reasoning and local generation.
    """

    window_type_ids: torch.Tensor
    gap_hours: torch.Tensor
    duration_hours: torch.Tensor
    support_flags: torch.Tensor | None = None
    event_family_prior: torch.Tensor | None = None

    def to_features(self) -> torch.Tensor:
        dtype = self.gap_hours.dtype
        parts = [
            self.window_type_ids.to(dtype=dtype).unsqueeze(-1),
            torch.log1p(self.gap_hours.to(dtype=dtype).clamp(min=0.0)).unsqueeze(-1),
            torch.log1p(self.duration_hours.to(dtype=dtype).clamp(min=0.0)).unsqueeze(-1),
        ]
        if self.support_flags is not None:
            parts.append(self.support_flags.to(dtype=dtype))
        if self.event_family_prior is not None:
            parts.append(self.event_family_prior.to(dtype=dtype))
        return torch.cat(parts, dim=-1)


@dataclass
class PrecedentGenerationReadout:
    """
    Precedent-memory output used directly by next-window generation.
    """

    summary_prior: torch.Tensor
    prompt_tokens: torch.Tensor
    prompt_summary: torch.Tensor
    candidate_weights: torch.Tensor
    matched_item_ids: torch.Tensor
    snippet_rel_path_ids: torch.Tensor | None = None
    snippet_subject_idxs: torch.Tensor | None = None
    snippet_start_boundary_ords: torch.Tensor | None = None
    snippet_stop_boundary_ords: torch.Tensor | None = None


@dataclass
class WindowGenerationConditioning:
    """
    Combined long-range conditioning used to initialize local next-window generation.
    """

    header: NextWindowHeader
    latent_context: torch.Tensor
    patient_memory: PatientMemoryReadout | None = None
    precedent_generation: PrecedentGenerationReadout | None = None


@dataclass
class WorldModelConditioning:
    """
    Combined conditioning surface for the local decoder.

    Intended usage:
    - latent context enters primarily as modulation or compact context tokens
    - patient memory enters as exact retrieval context
    - precedent memory enters as analogical future context
    """

    latent_context: torch.Tensor
    patient_memory: PatientMemoryReadout | None = None
    precedent_memory: PrecedentMemoryReadout | None = None


def _safe_one_hot(
    ids: torch.Tensor,
    *,
    num_classes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_classes <= 0:
        return ids.new_zeros(ids.shape + (0,), dtype=dtype)
    ids_long = ids.to(dtype=torch.long)
    valid = (ids_long >= 0) & (ids_long < int(num_classes))
    clamped = ids_long.clamp(min=0, max=max(0, int(num_classes) - 1))
    out = torch.nn.functional.one_hot(clamped, num_classes=int(num_classes)).to(dtype=dtype)
    return out * valid.unsqueeze(-1).to(dtype=dtype)


def future_summary_vector_dim(*, num_window_types: int) -> int:
    return (
        int(max(0, num_window_types))
        + 6
        + int(NUM_TOKEN_CATEGORIES)
        + int(NUM_EVENT_PAYLOAD_KINDS)
        + int(NUM_SUPPORT_FLAGS)
        + int(NUM_TRANSITION_FLAGS)
        + 2
        + int(max(0, num_window_types))
    )


def empty_future_summary(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FutureSummary:
    return FutureSummary(
        next_window_type_id=torch.tensor(-1, device=device, dtype=torch.long),
        next_window_gap_h=torch.tensor(0.0, device=device, dtype=dtype),
        next_window_duration_h=torch.tensor(0.0, device=device, dtype=dtype),
        event_family_hist=torch.zeros((NUM_TOKEN_CATEGORIES,), device=device, dtype=dtype),
        payload_hist=torch.zeros((NUM_EVENT_PAYLOAD_KINDS,), device=device, dtype=dtype),
        support_flags=torch.zeros((NUM_SUPPORT_FLAGS,), device=device, dtype=dtype),
        transition_flags=torch.zeros((NUM_TRANSITION_FLAGS,), device=device, dtype=dtype),
        event_count=torch.tensor(0.0, device=device, dtype=dtype),
        measurement_count=torch.tensor(0.0, device=device, dtype=dtype),
        extreme_measurement_count=torch.tensor(0.0, device=device, dtype=dtype),
        numeric_severity=torch.zeros((2,), device=device, dtype=dtype),
        terminal_window_type_id=torch.tensor(-1, device=device, dtype=torch.long),
        future_window_count=torch.tensor(0.0, device=device, dtype=dtype),
    )


def compose_precedent_key_state(
    *,
    packet_query: torch.Tensor,
    latent_state: torch.Tensor,
    memory_digest: torch.Tensor,
    window_type_ids: torch.Tensor | None,
    gap_prev_hours: torch.Tensor | None,
    duration_hours: torch.Tensor | None,
) -> torch.Tensor:
    if packet_query.shape != latent_state.shape or packet_query.shape != memory_digest.shape:
        raise ValueError(
            "packet_query, latent_state, and memory_digest must share shape; "
            f"got {tuple(packet_query.shape)}, {tuple(latent_state.shape)}, {tuple(memory_digest.shape)}"
        )
    dtype = packet_query.dtype
    device = packet_query.device
    shape_prefix = packet_query.shape[:-1]
    if window_type_ids is None:
        window_type_feat = torch.zeros(shape_prefix + (1,), device=device, dtype=dtype)
    else:
        window_type_feat = window_type_ids.to(device=device, dtype=dtype).unsqueeze(-1)
    if gap_prev_hours is None:
        gap_feat = torch.zeros(shape_prefix + (1,), device=device, dtype=dtype)
    else:
        gap_feat = torch.log1p(gap_prev_hours.to(device=device, dtype=dtype).clamp(min=0.0)).unsqueeze(-1)
    if duration_hours is None:
        dur_feat = torch.zeros(shape_prefix + (1,), device=device, dtype=dtype)
    else:
        dur_feat = torch.log1p(duration_hours.to(device=device, dtype=dtype).clamp(min=0.0)).unsqueeze(-1)
    return torch.cat(
        [
            packet_query,
            latent_state,
            memory_digest,
            window_type_feat,
            gap_feat,
            dur_feat,
        ],
        dim=-1,
    )
