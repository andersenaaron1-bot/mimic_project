from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
from src.ehr_hier.data.token_types import TokenCategory

from .heads import AETPrecedentHeads
from .world_model_contract import (
    NUM_ANCHOR_MASK_FLAGS,
    NUM_EVENT_PAYLOAD_KINDS,
    NUM_SUPPORT_FLAGS,
    NUM_TOKEN_CATEGORIES,
    NUM_TRANSITION_FLAGS,
    NextWindowHeader,
    PRECEDENT_ANCHOR_MASK_FLAG_ORDER,
    PRECEDENT_INDEX_VERSION,
    PRECEDENT_SUPPORT_FLAG_ORDER,
    PRECEDENT_TRANSITION_FLAG_ORDER,
    FutureSnippetRef,
    FutureSummary,
    PrecedentGenerationReadout,
    PrecedentIndexItem,
    PrecedentIndexStore,
    PrecedentMemoryReadout,
    WindowStatePacket,
    compose_precedent_key_state,
    empty_future_summary,
    future_summary_vector_dim,
)


_EXTREME_MEAS_Z_THRESHOLD = 2.5
_LONG_GAP_THRESHOLD_H = 72.0


def _future_summary_from_payload(payload: dict[str, Any]) -> FutureSummary:
    return FutureSummary(
        next_window_type_id=torch.as_tensor(payload["next_window_type_id"]),
        next_window_gap_h=torch.as_tensor(payload["next_window_gap_h"]),
        next_window_duration_h=torch.as_tensor(payload["next_window_duration_h"]),
        event_family_hist=torch.as_tensor(payload["event_family_hist"]),
        payload_hist=torch.as_tensor(payload["payload_hist"]),
        support_flags=torch.as_tensor(payload["support_flags"]),
        transition_flags=torch.as_tensor(payload["transition_flags"]),
        event_count=torch.as_tensor(payload["event_count"]),
        measurement_count=torch.as_tensor(payload["measurement_count"]),
        extreme_measurement_count=torch.as_tensor(payload["extreme_measurement_count"]),
        numeric_severity=torch.as_tensor(payload["numeric_severity"]),
        terminal_window_type_id=torch.as_tensor(payload["terminal_window_type_id"]),
        future_window_count=torch.as_tensor(payload["future_window_count"]),
    )


def _snippet_ref_from_payload(payload: dict[str, Any]) -> FutureSnippetRef:
    return FutureSnippetRef(
        rel_path_id=torch.as_tensor(payload["rel_path_id"]),
        subject_idx=torch.as_tensor(payload["subject_idx"]),
        trajectory_ord=torch.as_tensor(payload["trajectory_ord"]),
        start_boundary_ord=torch.as_tensor(payload["start_boundary_ord"]),
        stop_boundary_ord=torch.as_tensor(payload["stop_boundary_ord"]),
    )


def compose_future_summary_tensor(
    *,
    future_h1: torch.Tensor,
    future_h2: torch.Tensor,
    future_h3: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([future_h1, future_h2, future_h3], dim=-1)


def save_precedent_index_store(path: str | Path, store: PrecedentIndexStore) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(store), target)


def load_precedent_index_store(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> PrecedentIndexStore:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, PrecedentIndexStore):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported precedent index payload type={type(payload)!r}")
    return PrecedentIndexStore(
        version=int(payload["version"]),
        rel_path_vocab=[str(item) for item in payload["rel_path_vocab"]],
        item_ids=torch.as_tensor(payload["item_ids"]),
        subject_ids=torch.as_tensor(payload["subject_ids"]),
        trajectory_ords=torch.as_tensor(payload["trajectory_ords"]),
        boundary_ords=torch.as_tensor(payload["boundary_ords"]),
        anchor_window_ords=torch.as_tensor(payload["anchor_window_ords"]),
        current_window_type_ids=torch.as_tensor(payload["current_window_type_ids"]),
        current_window_start_h=torch.as_tensor(payload["current_window_start_h"]),
        current_window_duration_h=torch.as_tensor(payload["current_window_duration_h"]),
        gap_prev_h=torch.as_tensor(payload["gap_prev_h"]),
        support_flags=torch.as_tensor(payload["support_flags"]),
        anchor_mask_flags=torch.as_tensor(payload["anchor_mask_flags"]),
        key_state=torch.as_tensor(payload["key_state"]),
        key_packet=torch.as_tensor(payload["key_packet"]),
        key_memory=torch.as_tensor(payload["key_memory"]),
        future_h1=torch.as_tensor(payload["future_h1"]),
        future_h2=torch.as_tensor(payload["future_h2"]),
        future_h3=torch.as_tensor(payload["future_h3"]),
        future_prefix_prompt=torch.as_tensor(
            payload.get(
                "future_prefix_prompt",
                torch.zeros(
                    (
                        int(torch.as_tensor(payload["item_ids"]).shape[0]),
                        1,
                        int(torch.as_tensor(payload["key_packet"]).shape[-1]),
                    ),
                    dtype=torch.float32,
                ),
            )
        ),
        future_snippet_rel_path_ids=torch.as_tensor(payload["future_snippet_rel_path_ids"]),
        future_snippet_subject_idxs=torch.as_tensor(payload["future_snippet_subject_idxs"]),
        future_snippet_trajectory_ords=torch.as_tensor(payload["future_snippet_trajectory_ords"]),
        future_snippet_start_boundary_ords=torch.as_tensor(payload["future_snippet_start_boundary_ords"]),
        future_snippet_stop_boundary_ords=torch.as_tensor(payload["future_snippet_stop_boundary_ords"]),
        num_window_types=int(payload["num_window_types"]),
    )


def build_window_support_flags(
    *,
    event_type_ids: torch.Tensor,
    event_attention_mask: torch.Tensor,
    event_memory_chronic_flags: torch.Tensor | None = None,
    event_numeric_values: torch.Tensor | None = None,
    event_numeric_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = event_attention_mask.to(dtype=torch.bool)
    structural = (event_type_ids == int(TokenCategory.STRUCTURAL)) & mask
    diagnosis = (event_type_ids == int(TokenCategory.DIAGNOSIS)) & mask
    procedure = (event_type_ids == int(TokenCategory.PROCEDURE)) & mask
    medication = (event_type_ids == int(TokenCategory.MEDICATION)) & mask
    measurement = (event_type_ids == int(TokenCategory.MEASUREMENT)) & mask
    chronic = (
        diagnosis
        & event_memory_chronic_flags.to(dtype=torch.bool)
        if event_memory_chronic_flags is not None
        else diagnosis.new_zeros(diagnosis.shape)
    )
    if event_numeric_values is not None and event_numeric_mask is not None:
        extreme_measurement = (
            measurement
            & event_numeric_mask.to(dtype=torch.bool)
            & (event_numeric_values[..., 0].abs() >= float(_EXTREME_MEAS_Z_THRESHOLD))
        )
    else:
        extreme_measurement = measurement.new_zeros(measurement.shape)
    dims = tuple(range(max(0, mask.ndim - 2), mask.ndim))
    return torch.stack(
        [
            structural.any(dim=dims),
            chronic.any(dim=dims),
            procedure.any(dim=dims),
            medication.any(dim=dims),
            measurement.any(dim=dims),
            extreme_measurement.any(dim=dims),
        ],
        dim=-1,
    ).to(dtype=torch.float32)


def build_anchor_mask_flags(
    *,
    boundary_ord: int,
    gap_prev_h: float,
    future_truncated: bool,
    future_missing: bool,
) -> torch.Tensor:
    return torch.tensor(
        [
            1.0 if int(boundary_ord) == 0 else 0.0,
            1.0 if float(gap_prev_h) >= float(_LONG_GAP_THRESHOLD_H) else 0.0,
            1.0 if future_truncated else 0.0,
            1.0 if future_missing else 0.0,
        ],
        dtype=torch.float32,
    )


def select_future_window_indices(
    *,
    window_mask: torch.Tensor,
    window_start_times: torch.Tensor,
    anchor_end_h: float,
    start_idx: int,
    max_windows: int,
    max_hours: float | None,
) -> tuple[list[int], bool]:
    W = int(window_mask.shape[0])
    out: list[int] = []
    truncated = False
    for idx in range(int(start_idx), W):
        if not bool(window_mask[idx].item()):
            break
        if len(out) >= int(max_windows):
            truncated = True
            break
        start_h = float(window_start_times[idx].item())
        if out and max_hours is not None and start_h > float(anchor_end_h + max_hours):
            truncated = True
            break
        out.append(int(idx))
    if out:
        next_idx = out[-1] + 1
        if next_idx < W and bool(window_mask[next_idx].item()):
            if len(out) >= int(max_windows):
                truncated = True
            elif max_hours is not None and float(window_start_times[next_idx].item()) > float(anchor_end_h + max_hours):
                truncated = True
    return out, truncated


def build_future_summary(
    *,
    current_window_type_id: int,
    current_window_end_h: float,
    future_window_indices: Sequence[int],
    future_truncated: bool,
    window_type_ids: torch.Tensor,
    window_start_times: torch.Tensor,
    semantic_duration_hours: torch.Tensor,
    event_type_ids: torch.Tensor,
    event_payload_ids: torch.Tensor,
    event_attention_mask: torch.Tensor,
    event_memory_chronic_flags: torch.Tensor | None = None,
    event_numeric_values: torch.Tensor | None = None,
    event_numeric_mask: torch.Tensor | None = None,
) -> FutureSummary:
    device = window_type_ids.device
    dtype = semantic_duration_hours.dtype
    if not future_window_indices:
        return empty_future_summary(device=device, dtype=dtype)

    idx = torch.tensor(list(future_window_indices), device=device, dtype=torch.long)
    next_idx = int(future_window_indices[0])
    last_idx = int(future_window_indices[-1])
    window_event_types = event_type_ids.index_select(0, idx)
    window_event_payloads = event_payload_ids.index_select(0, idx)
    window_event_mask = event_attention_mask.index_select(0, idx).to(dtype=torch.bool)
    chronic = (
        event_memory_chronic_flags.index_select(0, idx)
        if event_memory_chronic_flags is not None
        else None
    )
    numeric_values = (
        event_numeric_values.index_select(0, idx)
        if event_numeric_values is not None
        else None
    )
    numeric_mask = (
        event_numeric_mask.index_select(0, idx)
        if event_numeric_mask is not None
        else None
    )

    flat_mask = window_event_mask.reshape(-1)
    flat_types = window_event_types.reshape(-1)
    flat_payloads = window_event_payloads.reshape(-1)
    event_count = int(flat_mask.to(dtype=torch.long).sum().item())
    family_hist = torch.zeros((NUM_TOKEN_CATEGORIES,), device=device, dtype=torch.float32)
    payload_hist = torch.zeros((NUM_EVENT_PAYLOAD_KINDS,), device=device, dtype=torch.float32)
    if event_count > 0:
        valid_types = flat_types[flat_mask]
        valid_payloads = flat_payloads[flat_mask]
        family_hist.scatter_add_(
            0,
            valid_types.clamp(min=0, max=max(0, NUM_TOKEN_CATEGORIES - 1)),
            torch.ones_like(valid_types, dtype=family_hist.dtype),
        )
        payload_hist.scatter_add_(
            0,
            valid_payloads.clamp(min=0, max=max(0, NUM_EVENT_PAYLOAD_KINDS - 1)),
            torch.ones_like(valid_payloads, dtype=payload_hist.dtype),
        )
        family_hist = family_hist / family_hist.sum().clamp(min=1.0)
        payload_hist = payload_hist / payload_hist.sum().clamp(min=1.0)

    measurement_mask = (
        (window_event_types == int(TokenCategory.MEASUREMENT)) & window_event_mask
    )
    measurement_count = int(measurement_mask.to(dtype=torch.long).sum().item())
    extreme_count = 0
    numeric_severity = torch.zeros((2,), device=device, dtype=torch.float32)
    if numeric_values is not None and numeric_mask is not None:
        numeric_valid = measurement_mask & numeric_mask.to(dtype=torch.bool)
        if bool(numeric_valid.any().item()):
            numeric_abs = numeric_values[..., 0].abs()
            masked_abs = numeric_abs[numeric_valid]
            extreme_count = int((masked_abs >= float(_EXTREME_MEAS_Z_THRESHOLD)).to(dtype=torch.long).sum().item())
            numeric_severity[0] = masked_abs.mean().to(dtype=torch.float32)
            numeric_severity[1] = masked_abs.max().to(dtype=torch.float32)

    support_flags = build_window_support_flags(
        event_type_ids=window_event_types,
        event_attention_mask=window_event_mask,
        event_memory_chronic_flags=chronic,
        event_numeric_values=numeric_values,
        event_numeric_mask=numeric_mask,
    ).amax(dim=0)

    transition_flags = torch.tensor(
        [
            1.0 if int(window_type_ids[next_idx].item()) != int(current_window_type_id) else 0.0,
            float(support_flags[PRECEDENT_SUPPORT_FLAG_ORDER.index("structural_present")].item()),
            float(support_flags[PRECEDENT_SUPPORT_FLAG_ORDER.index("extreme_measurement_present")].item()),
            1.0 if future_truncated else 0.0,
        ],
        device=device,
        dtype=torch.float32,
    )

    return FutureSummary(
        next_window_type_id=torch.tensor(int(window_type_ids[next_idx].item()), device=device, dtype=torch.long),
        next_window_gap_h=torch.tensor(
            max(0.0, float(window_start_times[next_idx].item()) - float(current_window_end_h)),
            device=device,
            dtype=dtype,
        ),
        next_window_duration_h=torch.tensor(
            float(semantic_duration_hours[next_idx].item()),
            device=device,
            dtype=dtype,
        ),
        event_family_hist=family_hist,
        payload_hist=payload_hist,
        support_flags=support_flags.to(dtype=torch.float32),
        transition_flags=transition_flags,
        event_count=torch.tensor(float(event_count), device=device, dtype=torch.float32),
        measurement_count=torch.tensor(float(measurement_count), device=device, dtype=torch.float32),
        extreme_measurement_count=torch.tensor(float(extreme_count), device=device, dtype=torch.float32),
        numeric_severity=numeric_severity,
        terminal_window_type_id=torch.tensor(int(window_type_ids[last_idx].item()), device=device, dtype=torch.long),
        future_window_count=torch.tensor(float(len(future_window_indices)), device=device, dtype=torch.float32),
    )


def materialize_precedent_index_store(
    *,
    items: Sequence[PrecedentIndexItem],
    rel_path_vocab: Sequence[str],
    num_window_types: int,
) -> PrecedentIndexStore:
    if not items:
        raise ValueError("items must be non-empty")

    def _stack_scalar(name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        values = [getattr(item, name) for item in items]
        out = torch.stack([torch.as_tensor(v) for v in values], dim=0)
        return out.to(dtype=dtype) if dtype is not None else out

    def _stack_future(name: str) -> torch.Tensor:
        vectors = [
            getattr(item, name).to_vector(num_window_types=int(num_window_types)).to(dtype=torch.float32)
            for item in items
        ]
        return torch.stack(vectors, dim=0)

    return PrecedentIndexStore(
        version=int(PRECEDENT_INDEX_VERSION),
        rel_path_vocab=[str(path) for path in rel_path_vocab],
        item_ids=_stack_scalar("item_id", dtype=torch.long),
        subject_ids=_stack_scalar("subject_id", dtype=torch.long),
        trajectory_ords=_stack_scalar("trajectory_ord", dtype=torch.long),
        boundary_ords=_stack_scalar("boundary_ord", dtype=torch.long),
        anchor_window_ords=_stack_scalar("anchor_window_ord", dtype=torch.long),
        current_window_type_ids=_stack_scalar("current_window_type_id", dtype=torch.long),
        current_window_start_h=_stack_scalar("current_window_start_h", dtype=torch.float32),
        current_window_duration_h=_stack_scalar("current_window_duration_h", dtype=torch.float32),
        gap_prev_h=_stack_scalar("gap_prev_h", dtype=torch.float32),
        support_flags=torch.stack([item.support_flags.to(dtype=torch.float32) for item in items], dim=0),
        anchor_mask_flags=torch.stack([item.anchor_mask_flags.to(dtype=torch.float32) for item in items], dim=0),
        key_state=torch.stack([item.key_state.to(dtype=torch.float32) for item in items], dim=0),
        key_packet=torch.stack([item.key_packet.to(dtype=torch.float32) for item in items], dim=0),
        key_memory=torch.stack([item.key_memory.to(dtype=torch.float32) for item in items], dim=0),
        future_h1=_stack_future("future_summary_h1"),
        future_h2=_stack_future("future_summary_h2"),
        future_h3=_stack_future("future_summary_h3"),
        future_prefix_prompt=torch.stack(
            [item.future_prefix_prompt.to(dtype=torch.float32) for item in items], dim=0
        ),
        future_snippet_rel_path_ids=torch.stack(
            [item.future_snippet_ref.rel_path_id.to(dtype=torch.long) for item in items], dim=0
        ),
        future_snippet_subject_idxs=torch.stack(
            [item.future_snippet_ref.subject_idx.to(dtype=torch.long) for item in items], dim=0
        ),
        future_snippet_trajectory_ords=torch.stack(
            [item.future_snippet_ref.trajectory_ord.to(dtype=torch.long) for item in items], dim=0
        ),
        future_snippet_start_boundary_ords=torch.stack(
            [item.future_snippet_ref.start_boundary_ord.to(dtype=torch.long) for item in items], dim=0
        ),
        future_snippet_stop_boundary_ords=torch.stack(
            [item.future_snippet_ref.stop_boundary_ord.to(dtype=torch.long) for item in items], dim=0
        ),
        num_window_types=int(num_window_types),
    )


def build_batch_future_summary_targets(
    *,
    num_window_types: int,
    window_type_ids: torch.Tensor,
    window_start_times: torch.Tensor,
    semantic_duration_hours: torch.Tensor,
    window_mask: torch.Tensor,
    event_type_ids: torch.Tensor,
    event_payload_ids: torch.Tensor,
    event_attention_mask: torch.Tensor,
    event_memory_chronic_flags: torch.Tensor | None = None,
    event_numeric_values: torch.Tensor | None = None,
    event_numeric_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if window_type_ids.ndim != 2:
        raise ValueError(f"window_type_ids must be (B,W), got shape {tuple(window_type_ids.shape)}")
    if window_mask.shape != window_type_ids.shape:
        raise ValueError(
            f"window_mask must match window_type_ids on (B,W); got {tuple(window_mask.shape)} vs {tuple(window_type_ids.shape)}"
        )

    B, W = window_type_ids.shape
    device = window_type_ids.device
    dtype = semantic_duration_hours.dtype
    future_dim = 3 * future_summary_vector_dim(num_window_types=int(num_window_types))
    out = torch.zeros((B, W, future_dim), device=device, dtype=torch.float32)
    valid = torch.zeros((B, W), device=device, dtype=torch.bool)

    for b in range(B):
        sample_window_mask = window_mask[b].to(dtype=torch.bool)
        sample_window_types = window_type_ids[b]
        sample_start_times = window_start_times[b]
        sample_durations = semantic_duration_hours[b]
        sample_event_types = event_type_ids[b]
        sample_event_payloads = event_payload_ids[b]
        sample_event_mask = event_attention_mask[b]
        sample_chronic = (
            event_memory_chronic_flags[b]
            if event_memory_chronic_flags is not None
            else None
        )
        sample_numeric_values = (
            event_numeric_values[b]
            if event_numeric_values is not None
            else None
        )
        sample_numeric_mask = (
            event_numeric_mask[b]
            if event_numeric_mask is not None
            else None
        )

        for w in range(W):
            if not bool(sample_window_mask[w].item()):
                continue
            anchor_end_h = float(sample_start_times[w].item() + sample_durations[w].item())
            future_h1_idx, future_h1_truncated = select_future_window_indices(
                window_mask=sample_window_mask,
                window_start_times=sample_start_times,
                anchor_end_h=anchor_end_h,
                start_idx=w + 1,
                max_windows=1,
                max_hours=None,
            )
            if not future_h1_idx:
                continue
            future_h2_idx, future_h2_truncated = select_future_window_indices(
                window_mask=sample_window_mask,
                window_start_times=sample_start_times,
                anchor_end_h=anchor_end_h,
                start_idx=w + 1,
                max_windows=2,
                max_hours=24.0,
            )
            future_h3_idx, future_h3_truncated = select_future_window_indices(
                window_mask=sample_window_mask,
                window_start_times=sample_start_times,
                anchor_end_h=anchor_end_h,
                start_idx=w + 1,
                max_windows=4,
                max_hours=24.0 * 7.0,
            )
            summary_h1 = build_future_summary(
                current_window_type_id=int(sample_window_types[w].item()),
                current_window_end_h=anchor_end_h,
                future_window_indices=future_h1_idx,
                future_truncated=bool(future_h1_truncated),
                window_type_ids=sample_window_types,
                window_start_times=sample_start_times,
                semantic_duration_hours=sample_durations,
                event_type_ids=sample_event_types,
                event_payload_ids=sample_event_payloads,
                event_attention_mask=sample_event_mask,
                event_memory_chronic_flags=sample_chronic,
                event_numeric_values=sample_numeric_values,
                event_numeric_mask=sample_numeric_mask,
            )
            summary_h2 = build_future_summary(
                current_window_type_id=int(sample_window_types[w].item()),
                current_window_end_h=anchor_end_h,
                future_window_indices=future_h2_idx,
                future_truncated=bool(future_h2_truncated),
                window_type_ids=sample_window_types,
                window_start_times=sample_start_times,
                semantic_duration_hours=sample_durations,
                event_type_ids=sample_event_types,
                event_payload_ids=sample_event_payloads,
                event_attention_mask=sample_event_mask,
                event_memory_chronic_flags=sample_chronic,
                event_numeric_values=sample_numeric_values,
                event_numeric_mask=sample_numeric_mask,
            )
            summary_h3 = build_future_summary(
                current_window_type_id=int(sample_window_types[w].item()),
                current_window_end_h=anchor_end_h,
                future_window_indices=future_h3_idx,
                future_truncated=bool(future_h3_truncated),
                window_type_ids=sample_window_types,
                window_start_times=sample_start_times,
                semantic_duration_hours=sample_durations,
                event_type_ids=sample_event_types,
                event_payload_ids=sample_event_payloads,
                event_attention_mask=sample_event_mask,
                event_memory_chronic_flags=sample_chronic,
                event_numeric_values=sample_numeric_values,
                event_numeric_mask=sample_numeric_mask,
            )
            out[b, w] = compose_future_summary_tensor(
                future_h1=summary_h1.to_vector(num_window_types=int(num_window_types)),
                future_h2=summary_h2.to_vector(num_window_types=int(num_window_types)),
                future_h3=summary_h3.to_vector(num_window_types=int(num_window_types)),
            ).to(device=device, dtype=torch.float32)
            valid[b, w] = True
    return out, valid


class AETPrecedentMemory(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d_model = int(config.d_model)
        self.retrieve_k = max(0, int(getattr(config, "precedent_retrieve_k", 4)))
        self.strict_window_type_match = bool(
            getattr(config, "precedent_strict_window_type_match", True)
        )
        self.support_overlap_bias = float(
            getattr(config, "precedent_support_overlap_bias", 0.25)
        )
        self.score_temperature = float(
            getattr(config, "precedent_score_temperature", 1.0)
        )
        self.learned_heads = AETPrecedentHeads(d_model)
        self.key_context_proj = nn.LazyLinear(d_model)
        self.future_h1_proj = nn.LazyLinear(d_model)
        self.future_h2_proj = nn.LazyLinear(d_model)
        self.future_h3_proj = nn.LazyLinear(d_model)
        self.summary_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.prompt_summary_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._store: PrecedentIndexStore | None = None

    @property
    def has_index(self) -> bool:
        return self._store is not None and int(self._store.key_state.shape[0]) > 0

    def set_store(self, store: PrecedentIndexStore | None) -> None:
        self._store = store

    def load_index(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> PrecedentIndexStore:
        store = load_precedent_index_store(path, map_location=map_location)
        self.set_store(store)
        return store

    def project_future_summaries(self, future_summary: torch.Tensor) -> torch.Tensor:
        return self.learned_heads.project_future_summaries(future_summary)

    def lookup_anchor_item_ids(
        self,
        *,
        subject_ids: torch.Tensor,
        trajectory_ords: torch.Tensor,
        boundary_ords: torch.Tensor,
    ) -> torch.Tensor:
        if not self.has_index:
            return torch.full_like(boundary_ords, fill_value=-1, dtype=torch.long)
        assert self._store is not None
        boundary_shape = boundary_ords.shape
        if subject_ids.shape != boundary_shape:
            if subject_ids.ndim == 1 and subject_ids.shape[0] == boundary_shape[0]:
                subject_ids = subject_ids.unsqueeze(-1).expand(boundary_shape)
            else:
                raise ValueError(
                    f"subject_ids must match boundary_ords or be (B,); got {tuple(subject_ids.shape)} vs {tuple(boundary_shape)}"
                )
        if trajectory_ords.shape != boundary_shape:
            if trajectory_ords.ndim == 1 and trajectory_ords.shape[0] == boundary_shape[0]:
                trajectory_ords = trajectory_ords.unsqueeze(-1).expand(boundary_shape)
            else:
                raise ValueError(
                    f"trajectory_ords must match boundary_ords or be (B,); got {tuple(trajectory_ords.shape)} vs {tuple(boundary_shape)}"
                )
        flat_subject_ids = subject_ids.reshape(-1).to(dtype=torch.long)
        flat_trajectory_ords = trajectory_ords.reshape(-1).to(dtype=torch.long)
        flat_boundary_ords = boundary_ords.reshape(-1).to(dtype=torch.long)

        device = flat_subject_ids.device
        store_subjects = self._store.subject_ids.to(device=device, dtype=torch.long)
        store_trajectories = self._store.trajectory_ords.to(device=device, dtype=torch.long)
        store_boundaries = self._store.boundary_ords.to(device=device, dtype=torch.long)
        store_item_ids = self._store.item_ids.to(device=device, dtype=torch.long)

        out = torch.full((flat_subject_ids.shape[0],), fill_value=-1, dtype=torch.long, device=device)
        for idx in range(flat_subject_ids.shape[0]):
            matches = (
                (store_subjects == flat_subject_ids[idx])
                & (store_trajectories == flat_trajectory_ords[idx])
                & (store_boundaries == flat_boundary_ords[idx])
            )
            if bool(matches.any().item()):
                out[idx] = store_item_ids[matches][0]
        return out.view(boundary_shape)

    def _empty_readout(self, query_state: torch.Tensor) -> PrecedentMemoryReadout:
        zeros = torch.zeros_like(query_state)
        context_tokens = torch.stack([zeros, zeros, zeros, zeros], dim=-2)
        future_dim = 3 * future_summary_vector_dim(num_window_types=int(self._store.num_window_types)) if self._store is not None else 0
        prompt_len = (
            int(self._store.future_prefix_prompt.shape[1])
            if self._store is not None and self._store.future_prefix_prompt.ndim >= 3
            else 1
        )
        return PrecedentMemoryReadout(
            context_tokens=context_tokens,
            context_summary=zeros,
            summary_prior=torch.zeros(
                query_state.shape[:-1] + (future_dim,),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            prompt_tokens=torch.zeros(
                query_state.shape[:-1] + (prompt_len, query_state.shape[-1]),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            prompt_summary=zeros,
            future_summary=torch.zeros(
                query_state.shape[:-1]
                + (future_dim,),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            future_embedding=zeros,
            query_embedding=zeros,
            retrieval_scores=torch.zeros(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            candidate_weights=torch.zeros(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            candidate_prompt_tokens=torch.zeros(
                query_state.shape[:-1] + (max(1, self.retrieve_k), prompt_len, query_state.shape[-1]),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            candidate_future_summaries=torch.zeros(
                query_state.shape[:-1] + (max(1, self.retrieve_k), future_dim),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            candidate_future_embeddings=torch.zeros(
                query_state.shape[:-1] + (max(1, self.retrieve_k), query_state.shape[-1]),
                device=query_state.device,
                dtype=query_state.dtype,
            ),
            matched_item_ids=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_subject_ids=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_trajectory_ords=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_window_ords=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_snippet_rel_path_ids=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_snippet_subject_idxs=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_snippet_start_boundary_ords=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
            matched_snippet_stop_boundary_ords=torch.full(
                query_state.shape[:-1] + (max(1, self.retrieve_k),),
                fill_value=-1,
                device=query_state.device,
                dtype=torch.long,
            ),
        )

    def _compose_header_features(
        self,
        *,
        prefix_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        next_window_header: NextWindowHeader | None,
    ) -> torch.Tensor:
        if next_window_header is None:
            return torch.zeros(
                prefix_shape + (3 + NUM_SUPPORT_FLAGS,),
                device=device,
                dtype=dtype,
            )
        if next_window_header.support_flags is None:
            support_flags = torch.zeros(
                prefix_shape + (NUM_SUPPORT_FLAGS,),
                device=device,
                dtype=dtype,
            )
        else:
            support_flags = next_window_header.support_flags.to(device=device, dtype=dtype)
        return torch.cat(
            [
                next_window_header.window_type_ids.to(device=device, dtype=dtype).unsqueeze(-1),
                torch.log1p(next_window_header.gap_hours.to(device=device, dtype=dtype).clamp(min=0.0)).unsqueeze(-1),
                torch.log1p(next_window_header.duration_hours.to(device=device, dtype=dtype).clamp(min=0.0)).unsqueeze(-1),
                support_flags,
            ],
            dim=-1,
        )

    def _retrieve_readout(
        self,
        *,
        state_packet: WindowStatePacket,
        query_state: torch.Tensor,
        memory_digest: torch.Tensor | None = None,
        support_flags: torch.Tensor | None = None,
        next_window_header: NextWindowHeader | None = None,
    ) -> PrecedentMemoryReadout:
        if not self.has_index or self.retrieve_k <= 0:
            return self._empty_readout(query_state)
        assert self._store is not None

        if memory_digest is None:
            memory_digest = torch.zeros_like(query_state)
        query_packet = state_packet.query_token if state_packet.query_token is not None else state_packet.summary()
        query_key = compose_precedent_key_state(
            packet_query=query_packet,
            latent_state=query_state,
            memory_digest=memory_digest,
            window_type_ids=state_packet.window_type_ids,
            gap_prev_hours=state_packet.gap_prev_hours,
            duration_hours=state_packet.duration_hours,
        )

        prefix_shape = query_state.shape[:-1]
        N = int(torch.tensor(prefix_shape).prod().item()) if prefix_shape else 1
        if support_flags is None:
            support_flags = torch.zeros(
                prefix_shape + (NUM_SUPPORT_FLAGS,),
                device=query_state.device,
                dtype=query_state.dtype,
            )
        header_features = self._compose_header_features(
            prefix_shape=torch.Size(prefix_shape),
            device=query_state.device,
            dtype=query_state.dtype,
            next_window_header=next_window_header,
        )
        flat_query_key = query_key.reshape(N, query_key.shape[-1]).to(dtype=torch.float32)
        flat_query_support = support_flags.reshape(N, support_flags.shape[-1]).to(
            device=query_state.device,
            dtype=torch.float32,
        )
        flat_header_features = header_features.reshape(N, header_features.shape[-1]).to(
            device=query_state.device,
            dtype=torch.float32,
        )
        flat_query_features = torch.cat([flat_query_key, flat_query_support, flat_header_features], dim=-1)
        flat_query = self.learned_heads.encode_query(flat_query_features)

        key_state = self._store.key_state.to(device=flat_query.device, dtype=torch.float32)
        key_support = self._store.support_flags.to(device=flat_query.device, dtype=torch.float32)
        key_features = torch.cat([key_state, key_support], dim=-1)
        projected_keys = self.learned_heads.project_keys(key_features)
        query_norm = F.normalize(flat_query, dim=-1, eps=1e-6)
        key_norm = F.normalize(projected_keys, dim=-1, eps=1e-6)
        scores = torch.matmul(query_norm, key_norm.transpose(0, 1))
        scores = scores / max(1e-6, float(self.score_temperature))

        if self.strict_window_type_match and state_packet.window_type_ids is not None:
            q_type = state_packet.window_type_ids.reshape(N, 1).to(device=scores.device, dtype=torch.long)
            key_type = self._store.current_window_type_ids.to(device=scores.device, dtype=torch.long).view(1, -1)
            typed_scores = scores.masked_fill(q_type != key_type, float("-inf"))
            has_typed = torch.isfinite(typed_scores).any(dim=1, keepdim=True)
            scores = torch.where(has_typed, typed_scores, scores)

        q_flags = flat_query_support.to(device=scores.device, dtype=torch.float32)
        k_flags = self._store.support_flags.to(device=scores.device, dtype=torch.float32)
        overlap = torch.matmul(q_flags, k_flags.transpose(0, 1)) / float(max(1, NUM_SUPPORT_FLAGS))
        scores = scores + (self.support_overlap_bias * overlap)

        topk = min(int(self.retrieve_k), int(scores.shape[1]))
        top_scores, top_idx = torch.topk(scores, k=max(1, topk), dim=1)
        valid_top = torch.isfinite(top_scores)
        safe_scores = torch.where(valid_top, top_scores, torch.full_like(top_scores, -1e9))
        attn = torch.softmax(safe_scores, dim=1)
        attn = attn * valid_top.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)

        gather_index = top_idx
        key_packet = self._store.key_packet.to(device=scores.device, dtype=torch.float32)[gather_index]
        future_h1 = self._store.future_h1.to(device=scores.device, dtype=torch.float32)[gather_index]
        future_h2 = self._store.future_h2.to(device=scores.device, dtype=torch.float32)[gather_index]
        future_h3 = self._store.future_h3.to(device=scores.device, dtype=torch.float32)[gather_index]
        prompt_tokens = self._store.future_prefix_prompt.to(device=scores.device, dtype=torch.float32)[gather_index]
        candidate_future = compose_future_summary_tensor(
            future_h1=future_h1,
            future_h2=future_h2,
            future_h3=future_h3,
        )
        candidate_future_embedding = self.learned_heads.project_future_summaries(candidate_future)

        agg_packet = (attn.unsqueeze(-1) * key_packet).sum(dim=1)
        agg_h1 = (attn.unsqueeze(-1) * future_h1).sum(dim=1)
        agg_h2 = (attn.unsqueeze(-1) * future_h2).sum(dim=1)
        agg_h3 = (attn.unsqueeze(-1) * future_h3).sum(dim=1)
        agg_prompt_tokens = (attn.unsqueeze(-1).unsqueeze(-1) * prompt_tokens).sum(dim=1)

        packet_token = self.key_context_proj(agg_packet)
        h1_token = self.future_h1_proj(agg_h1)
        h2_token = self.future_h2_proj(agg_h2)
        h3_token = self.future_h3_proj(agg_h3)
        context_tokens = torch.stack([packet_token, h1_token, h2_token, h3_token], dim=1)
        context_summary = self.summary_proj(context_tokens.reshape(N, -1))
        future_summary = compose_future_summary_tensor(
            future_h1=agg_h1,
            future_h2=agg_h2,
            future_h3=agg_h3,
        )
        future_embedding = self.learned_heads.project_future_summaries(future_summary)
        prompt_summary = self.prompt_summary_proj(agg_prompt_tokens.mean(dim=1))

        def _gather_meta(name: str) -> torch.Tensor:
            return getattr(self._store, name).to(device=scores.device, dtype=torch.long)[gather_index]

        return PrecedentMemoryReadout(
            context_tokens=context_tokens.reshape(*prefix_shape, 4, context_tokens.shape[-1]).to(dtype=query_state.dtype),
            context_summary=context_summary.reshape(*prefix_shape, context_summary.shape[-1]).to(dtype=query_state.dtype),
            summary_prior=future_summary.reshape(*prefix_shape, future_summary.shape[-1]).to(dtype=query_state.dtype),
            prompt_tokens=agg_prompt_tokens.reshape(*prefix_shape, agg_prompt_tokens.shape[1], agg_prompt_tokens.shape[2]).to(dtype=query_state.dtype),
            prompt_summary=prompt_summary.reshape(*prefix_shape, prompt_summary.shape[-1]).to(dtype=query_state.dtype),
            future_summary=future_summary.reshape(*prefix_shape, future_summary.shape[-1]).to(dtype=query_state.dtype),
            future_embedding=future_embedding.reshape(*prefix_shape, future_embedding.shape[-1]).to(dtype=query_state.dtype),
            query_embedding=flat_query.reshape(*prefix_shape, flat_query.shape[-1]).to(dtype=query_state.dtype),
            retrieval_scores=torch.where(valid_top, top_scores, torch.zeros_like(top_scores)).reshape(*prefix_shape, -1).to(dtype=query_state.dtype),
            candidate_weights=attn.reshape(*prefix_shape, -1).to(dtype=query_state.dtype),
            candidate_prompt_tokens=prompt_tokens.reshape(*prefix_shape, prompt_tokens.shape[1], prompt_tokens.shape[2], prompt_tokens.shape[3]).to(dtype=query_state.dtype),
            candidate_future_summaries=candidate_future.reshape(*prefix_shape, candidate_future.shape[1], candidate_future.shape[2]).to(dtype=query_state.dtype),
            candidate_future_embeddings=candidate_future_embedding.reshape(*prefix_shape, candidate_future_embedding.shape[1], candidate_future_embedding.shape[2]).to(dtype=query_state.dtype),
            matched_item_ids=_gather_meta("item_ids").reshape(*prefix_shape, -1),
            matched_subject_ids=_gather_meta("subject_ids").reshape(*prefix_shape, -1),
            matched_trajectory_ords=_gather_meta("trajectory_ords").reshape(*prefix_shape, -1),
            matched_window_ords=_gather_meta("boundary_ords").reshape(*prefix_shape, -1),
            matched_snippet_rel_path_ids=_gather_meta("future_snippet_rel_path_ids").reshape(*prefix_shape, -1),
            matched_snippet_subject_idxs=_gather_meta("future_snippet_subject_idxs").reshape(*prefix_shape, -1),
            matched_snippet_start_boundary_ords=_gather_meta("future_snippet_start_boundary_ords").reshape(*prefix_shape, -1),
            matched_snippet_stop_boundary_ords=_gather_meta("future_snippet_stop_boundary_ords").reshape(*prefix_shape, -1),
        )

    def query_boundary_prior(
        self,
        *,
        state_packet: WindowStatePacket,
        query_state: torch.Tensor,
        memory_digest: torch.Tensor | None = None,
        support_flags: torch.Tensor | None = None,
    ) -> PrecedentMemoryReadout:
        return self._retrieve_readout(
            state_packet=state_packet,
            query_state=query_state,
            memory_digest=memory_digest,
            support_flags=support_flags,
            next_window_header=None,
        )

    def query_generation_prompt(
        self,
        *,
        state_packet: WindowStatePacket,
        query_state: torch.Tensor,
        next_window_header: NextWindowHeader,
        memory_digest: torch.Tensor | None = None,
        support_flags: torch.Tensor | None = None,
    ) -> PrecedentGenerationReadout:
        readout = self._retrieve_readout(
            state_packet=state_packet,
            query_state=query_state,
            memory_digest=memory_digest,
            support_flags=support_flags,
            next_window_header=next_window_header,
        )
        return PrecedentGenerationReadout(
            summary_prior=readout.summary_prior if readout.summary_prior is not None else readout.future_summary,
            prompt_tokens=readout.prompt_tokens if readout.prompt_tokens is not None else readout.context_tokens,
            prompt_summary=readout.prompt_summary if readout.prompt_summary is not None else readout.context_summary,
            candidate_weights=readout.candidate_weights if readout.candidate_weights is not None else readout.retrieval_scores,
            matched_item_ids=readout.matched_item_ids,
            snippet_rel_path_ids=readout.matched_snippet_rel_path_ids,
            snippet_subject_idxs=readout.matched_snippet_subject_idxs,
            snippet_start_boundary_ords=readout.matched_snippet_start_boundary_ords,
            snippet_stop_boundary_ords=readout.matched_snippet_stop_boundary_ords,
        )

    def forward(
        self,
        *,
        state_packet: WindowStatePacket,
        query_state: torch.Tensor,
        memory_digest: torch.Tensor | None = None,
        support_flags: torch.Tensor | None = None,
    ) -> PrecedentMemoryReadout:
        return self.query_boundary_prior(
            state_packet=state_packet,
            query_state=query_state,
            memory_digest=memory_digest,
            support_flags=support_flags,
        )
