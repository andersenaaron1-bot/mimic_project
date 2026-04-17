from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Literal, Mapping

import torch

from src.ehr_hier.data.event_frames import (
    EventFrame,
    EventPayloadKind,
    build_event_frame,
    clone_event_frame,
    ensure_event_frames,
    flatten_event_frames,
    payload_kind_to_id,
    slice_event_frame,
)
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.data.window_segmentation import (
    SegmentedChunk,
    SegmentedWindow,
    WindowSegmentationConfig,
    chunk_segmented_windows,
    segment_event_tokens,
)
from src.ehr_hier.transformer.memory_rules import (
    classify_event_frame_for_memory,
    update_seen_memory_keys,
)


@dataclass(frozen=True)
class WindowMarkerConfig:
    """
    Controls insertion of per-window marker tokens.

    Marker tokens are inserted *inside* each window sequence during collation:
      [special_tokens...] [WIN_TYPE] [window_tokens...] [WIN_END or WIN_<NEXT_TYPE>]

    Window types are integer class ids in [0, num_types-1]. The WIN_TYPE token id
    is computed as:
        win_type_token_id = type_token_offset + window_type_id
    """

    enabled: bool = True
    # How to terminate a window sequence:
    #  - "end_token": append a dedicated WIN_END token id.
    #  - "next_type": append the *next* window's WIN_<TYPE> token id (last window uses WIN_END).
    end_mode: Literal["end_token", "next_type"] = "end_token"
    # Global token id offset where WIN_<TYPE> tokens start.
    type_token_offset: int = 10
    # Number of supported window types (including UNK=0 if you use it).
    num_types: int = 16
    # Global token id for the end-of-window marker. If None, defaults to
    # type_token_offset + num_types (i.e., directly after the WIN_<TYPE> block).
    end_token_id: int | None = None
    # Global token id for the intra-window continuation marker. If None, defaults to
    # end_token_id + 1 (or directly after the WIN_<TYPE>/WIN_END block).
    continue_token_id: int | None = None
    # Fallback type id when a window type cannot be inferred.
    unk_type_id: int = 0
    # Category assigned to marker tokens (affects pooling masks, not routing).
    marker_category: TokenCategory = TokenCategory.SPECIAL


@dataclass(frozen=True)
class PreparedChunk:
    ids: List[int]
    times: List[float]
    vals: List[float]
    valmask: List[int]
    types: List[int]
    start_abs: float
    start_offset: float
    token_event_index: List[int]
    token_event_slot_ids: List[int]
    event_ids: List[int]
    event_times: List[float]
    event_vals: List[float]
    event_valmask: List[int]
    event_types: List[int]
    event_payloads: List[int]
    event_demographic_feature_ids: List[int]
    event_memory_rule_scores: List[float]
    event_memory_group_ids: List[int]
    event_memory_first_flags: List[int]
    event_memory_chronic_flags: List[int]
    event_med_group_ids: List[int]
    event_med_group_mask: List[int]
    event_med_route_ids: List[int]
    event_med_route_mask: List[int]
    event_med_form_ids: List[int]
    event_med_form_mask: List[int]
    event_med_freq_ids: List[int]
    event_med_freq_mask: List[int]
    event_med_unit_ids: List[int]
    event_med_unit_mask: List[int]
    event_med_marker_ids: List[int]
    event_med_marker_mask: List[int]
    event_med_dosage_values: List[float]
    event_med_dosage_mask: List[int]
    event_med_rate_values: List[float]
    event_med_rate_mask: List[int]
    event_med_duration_values: List[float]
    event_med_duration_mask: List[int]

    def __iter__(self):
        yield self.ids
        yield self.times
        yield self.vals
        yield self.valmask
        yield self.types
        yield self.start_abs
        yield self.start_offset


class AETHierarchicalCollator:
    """
    Collates a batch of event-frame timelines into padded (B, W, L, ...) tensors.

    - Windows are segmented with transfer-driven causal logic.
    - Explicit transition metadata remains the canonical boundary signal.
    - Only transfer-like tokens open typed windows; discharge/death closers move
      later residue into POST_DISCHARGE when configured.
    - token.window_hook remains only as a fallback for older artifacts/tests.
    - Special (window-0) tokens are prefixed to every window.
    - Times are made relative to the start of each window.
    - token_type_ids mirror TokenCategory for now (head mapping handled later).
    """

    def __init__(
        self,
        *,
        max_windows: int = 64,
        max_chunks_per_window: int = 8,
        max_len_per_window: int = 128,
        pad_id: int = 0,
        window_markers: WindowMarkerConfig | None = None,
        segmentation: WindowSegmentationConfig | None = None,
        id_remapper: Any | None = None,
        emit_global_input_ids: bool = False,
    ) -> None:
        self.max_windows = max_windows
        self.max_chunks_per_window = max_chunks_per_window
        self.max_len = max_len_per_window
        self.pad_id = pad_id
        self.window_markers = window_markers or WindowMarkerConfig()
        self.segmentation = segmentation or WindowSegmentationConfig(
            unk_window_type_id=int((window_markers or WindowMarkerConfig()).unk_type_id)
        )
        self.id_remapper = id_remapper
        self.emit_global_input_ids = bool(emit_global_input_ids)

    def __call__(self, batch_timelines: List[Any]) -> Dict[str, Any]:
        normalized_timelines: List[List[EventFrame | EventToken]] = []
        batch_subject_ids: List[int] = []
        batch_trajectory_ords: List[int] = []
        for item in batch_timelines:
            if isinstance(item, Mapping):
                timeline = item.get("timeline", None)
                if not timeline:
                    continue
                normalized_timelines.append(list(timeline))
                batch_subject_ids.append(int(item.get("subject_id", -1)))
                batch_trajectory_ords.append(int(item.get("trajectory_ord", -1)))
            else:
                if not item:
                    continue
                normalized_timelines.append(list(item))
                batch_subject_ids.append(-1)
                batch_trajectory_ords.append(-1)
        if not normalized_timelines:
            raise ValueError("Collator received no non-empty timelines.")

        batch_ids: List[List[List[List[int]]]] = []
        batch_times: List[List[List[List[float]]]] = []
        batch_vals: List[List[List[List[float]]]] = []
        batch_types: List[List[List[List[int]]]] = []
        batch_attn: List[List[List[List[int]]]] = []
        batch_valmask: List[List[List[List[int]]]] = []
        batch_window_types: List[List[int]] = []
        batch_window_start_times: List[List[float]] = []
        batch_chunk_mask: List[List[List[int]]] = []
        batch_chunk_start_offsets: List[List[List[float]]] = []
        batch_chunk_start_times: List[List[List[float]]] = []
        batch_chunk_is_last: List[List[List[int]]] = []
        batch_window_token_counts: List[List[float]] = []
        batch_window_duration_hours: List[List[float]] = []
        batch_chunk_token_counts: List[List[List[float]]] = []
        batch_chunk_duration_hours: List[List[List[float]]] = []
        batch_token_event_index: List[List[List[List[int]]]] = []
        batch_token_event_slot_ids: List[List[List[List[int]]]] = []
        batch_event_ids: List[List[List[List[int]]]] = []
        batch_event_times: List[List[List[List[float]]]] = []
        batch_event_vals: List[List[List[List[float]]]] = []
        batch_event_valmask: List[List[List[List[int]]]] = []
        batch_event_types: List[List[List[List[int]]]] = []
        batch_event_payloads: List[List[List[List[int]]]] = []
        batch_event_demographic_feature_ids: List[List[List[List[int]]]] = []
        batch_event_memory_rule_scores: List[List[List[List[float]]]] = []
        batch_event_memory_group_ids: List[List[List[List[int]]]] = []
        batch_event_memory_first_flags: List[List[List[List[int]]]] = []
        batch_event_memory_chronic_flags: List[List[List[List[int]]]] = []
        batch_event_med_group_ids: List[List[List[List[int]]]] = []
        batch_event_med_group_mask: List[List[List[List[int]]]] = []
        batch_event_med_route_ids: List[List[List[List[int]]]] = []
        batch_event_med_route_mask: List[List[List[List[int]]]] = []
        batch_event_med_form_ids: List[List[List[List[int]]]] = []
        batch_event_med_form_mask: List[List[List[List[int]]]] = []
        batch_event_med_freq_ids: List[List[List[List[int]]]] = []
        batch_event_med_freq_mask: List[List[List[List[int]]]] = []
        batch_event_med_unit_ids: List[List[List[List[int]]]] = []
        batch_event_med_unit_mask: List[List[List[List[int]]]] = []
        batch_event_med_marker_ids: List[List[List[List[int]]]] = []
        batch_event_med_marker_mask: List[List[List[List[int]]]] = []
        batch_event_med_dosage_values: List[List[List[List[float]]]] = []
        batch_event_med_dosage_mask: List[List[List[List[int]]]] = []
        batch_event_med_rate_values: List[List[List[List[float]]]] = []
        batch_event_med_rate_mask: List[List[List[List[int]]]] = []
        batch_event_med_duration_values: List[List[List[List[float]]]] = []
        batch_event_med_duration_mask: List[List[List[List[int]]]] = []
        semantic_windows_total = 0
        semantic_windows_kept = 0
        semantic_windows_dropped = 0
        semantic_tokens_dropped = 0
        semantic_structural_tokens_dropped = 0
        chunks_dropped_by_cap = 0
        chunk_tokens_dropped_by_cap = 0
        chunk_structural_tokens_dropped_by_cap = 0
        subjects_with_overflow = 0

        for timeline in normalized_timelines:
            frame_timeline = ensure_event_frames(timeline)
            special_frames, event_frames = self._split_special(frame_timeline)
            special_tokens = flatten_event_frames(special_frames, clone=False)
            semantic_windows_all = self._segment_windows(event_frames)
            semantic_frame_windows_all = self._align_frames_to_segmented_windows(
                event_frames,
                semantic_windows_all,
            )
            semantic_windows_total += int(len(semantic_windows_all))
            dropped_windows = max(0, int(len(semantic_windows_all) - int(self.max_windows)))
            subject_dropped_tokens = 0
            subject_dropped_structural_tokens = 0
            if dropped_windows > 0:
                semantic_windows_dropped += int(dropped_windows)
                for win in semantic_windows_all[int(self.max_windows) :]:
                    n_tok = int(len(win.tokens))
                    subject_dropped_tokens += n_tok
                    subject_dropped_structural_tokens += int(
                        sum(
                            1
                            for tok in win.tokens
                            if int(tok.category_id) == int(TokenCategory.STRUCTURAL)
                        )
                    )
            semantic_windows = semantic_windows_all[: self.max_windows]
            semantic_windows_kept += int(len(semantic_windows))
            semantic_tokens_dropped += int(subject_dropped_tokens)
            semantic_structural_tokens_dropped += int(subject_dropped_structural_tokens)
            chunked_windows = self._chunk_windows(semantic_windows, special_tokens=special_tokens)
            semantic_frame_windows = semantic_frame_windows_all[: self.max_windows]
            chunked_frame_windows = self._align_frames_to_chunked_windows(
                semantic_frame_windows,
                chunked_windows,
            )
            window_type_ids = [self._clamp_window_type_id(int(window.window_type_id)) for window in chunked_windows]
            window_start_abs_times = [float(window.start_time_hours) for window in chunked_windows]
            subject_dropped_chunks = 0
            subject_dropped_chunk_tokens = 0
            subject_dropped_chunk_structural_tokens = 0
            for window in chunked_windows:
                subject_dropped_chunks += int(getattr(window, "truncated_chunks", 0) or 0)
                subject_dropped_chunk_tokens += int(getattr(window, "truncated_tokens", 0) or 0)
                subject_dropped_chunk_structural_tokens += int(
                    getattr(window, "truncated_structural_tokens", 0) or 0
                )
            chunks_dropped_by_cap += int(subject_dropped_chunks)
            chunk_tokens_dropped_by_cap += int(subject_dropped_chunk_tokens)
            chunk_structural_tokens_dropped_by_cap += int(subject_dropped_chunk_structural_tokens)
            if (
                dropped_windows > 0
                or subject_dropped_chunks > 0
                or subject_dropped_tokens > 0
                or subject_dropped_chunk_tokens > 0
            ):
                subjects_with_overflow += 1

            subj_ids: List[List[List[int]]] = []
            subj_times: List[List[List[float]]] = []
            subj_vals: List[List[List[float]]] = []
            subj_types: List[List[List[int]]] = []
            subj_masks: List[List[List[int]]] = []
            subj_valmask: List[List[List[int]]] = []
            subj_window_types: List[int] = []
            subj_window_start_times: List[float] = []
            subj_chunk_mask: List[List[int]] = []
            subj_chunk_start_offsets: List[List[float]] = []
            subj_chunk_start_times: List[List[float]] = []
            subj_chunk_is_last: List[List[int]] = []
            subj_window_token_counts: List[float] = []
            subj_window_duration_hours: List[float] = []
            subj_chunk_token_counts: List[List[float]] = []
            subj_chunk_duration_hours: List[List[float]] = []
            subj_token_event_index: List[List[List[int]]] = []
            subj_token_event_slot_ids: List[List[List[int]]] = []
            subj_event_ids: List[List[List[int]]] = []
            subj_event_times: List[List[List[float]]] = []
            subj_event_vals: List[List[List[float]]] = []
            subj_event_valmask: List[List[List[int]]] = []
            subj_event_types: List[List[List[int]]] = []
            subj_event_payloads: List[List[List[int]]] = []
            subj_event_demographic_feature_ids: List[List[List[int]]] = []
            subj_event_memory_rule_scores: List[List[List[float]]] = []
            subj_event_memory_group_ids: List[List[List[int]]] = []
            subj_event_memory_first_flags: List[List[List[int]]] = []
            subj_event_memory_chronic_flags: List[List[List[int]]] = []
            subj_event_med_group_ids: List[List[List[int]]] = []
            subj_event_med_group_mask: List[List[List[int]]] = []
            subj_event_med_route_ids: List[List[List[int]]] = []
            subj_event_med_route_mask: List[List[List[int]]] = []
            subj_event_med_form_ids: List[List[List[int]]] = []
            subj_event_med_form_mask: List[List[List[int]]] = []
            subj_event_med_freq_ids: List[List[List[int]]] = []
            subj_event_med_freq_mask: List[List[List[int]]] = []
            subj_event_med_unit_ids: List[List[List[int]]] = []
            subj_event_med_unit_mask: List[List[List[int]]] = []
            subj_event_med_marker_ids: List[List[List[int]]] = []
            subj_event_med_marker_mask: List[List[List[int]]] = []
            subj_event_med_dosage_values: List[List[List[float]]] = []
            subj_event_med_dosage_mask: List[List[List[int]]] = []
            subj_event_med_rate_values: List[List[List[float]]] = []
            subj_event_med_rate_mask: List[List[List[int]]] = []
            subj_event_med_duration_values: List[List[List[float]]] = []
            subj_event_med_duration_mask: List[List[List[int]]] = []
            seen_memory_keys: set[str] = set()

            for wi, window in enumerate(chunked_windows):
                next_type_id = window_type_ids[wi + 1] if wi + 1 < len(window_type_ids) else None
                next_start_abs = window_start_abs_times[wi + 1] if wi + 1 < len(window_start_abs_times) else None
                chunk_ids: List[List[int]] = []
                chunk_times: List[List[float]] = []
                chunk_vals: List[List[float]] = []
                chunk_types: List[List[int]] = []
                chunk_masks: List[List[int]] = []
                chunk_valmask: List[List[int]] = []
                chunk_mask: List[int] = []
                chunk_offsets: List[float] = []
                chunk_start_times: List[float] = []
                chunk_is_last: List[int] = []
                chunk_token_counts: List[float] = []
                chunk_duration_hours: List[float] = []
                chunk_token_event_index: List[List[int]] = []
                chunk_token_event_slot_ids: List[List[int]] = []
                chunk_event_ids: List[List[int]] = []
                chunk_event_times: List[List[float]] = []
                chunk_event_vals: List[List[float]] = []
                chunk_event_valmask: List[List[int]] = []
                chunk_event_types: List[List[int]] = []
                chunk_event_payloads: List[List[int]] = []
                chunk_event_demographic_feature_ids: List[List[int]] = []
                chunk_event_memory_rule_scores: List[List[float]] = []
                chunk_event_memory_group_ids: List[List[int]] = []
                chunk_event_memory_first_flags: List[List[int]] = []
                chunk_event_memory_chronic_flags: List[List[int]] = []
                chunk_event_med_group_ids: List[List[int]] = []
                chunk_event_med_group_mask: List[List[int]] = []
                chunk_event_med_route_ids: List[List[int]] = []
                chunk_event_med_route_mask: List[List[int]] = []
                chunk_event_med_form_ids: List[List[int]] = []
                chunk_event_med_form_mask: List[List[int]] = []
                chunk_event_med_freq_ids: List[List[int]] = []
                chunk_event_med_freq_mask: List[List[int]] = []
                chunk_event_med_unit_ids: List[List[int]] = []
                chunk_event_med_unit_mask: List[List[int]] = []
                chunk_event_med_marker_ids: List[List[int]] = []
                chunk_event_med_marker_mask: List[List[int]] = []
                chunk_event_med_dosage_values: List[List[float]] = []
                chunk_event_med_dosage_mask: List[List[int]] = []
                chunk_event_med_rate_values: List[List[float]] = []
                chunk_event_med_rate_mask: List[List[int]] = []
                chunk_event_med_duration_values: List[List[float]] = []
                chunk_event_med_duration_mask: List[List[int]] = []

                for ci, chunk in enumerate(window.chunks):
                    prepared = self._process_chunk(
                        chunk,
                        special_tokens=special_tokens,
                        special_frames=special_frames,
                        chunk_frames=chunked_frame_windows[wi][ci] if wi < len(chunked_frame_windows) and ci < len(chunked_frame_windows[wi]) else [],
                        w_type_id=window_type_ids[wi],
                        w_start_abs=window_start_abs_times[wi],
                        next_type_id=next_type_id,
                        next_start_abs=next_start_abs,
                        seen_memory_keys=seen_memory_keys,
                    )
                    seq_len = len(prepared.ids)
                    chunk_ids.append(prepared.ids)
                    chunk_times.append(prepared.times)
                    chunk_vals.append(prepared.vals)
                    chunk_types.append(prepared.types)
                    chunk_masks.append([1] * seq_len)
                    chunk_valmask.append(prepared.valmask)
                    chunk_mask.append(1)
                    chunk_offsets.append(prepared.start_offset)
                    chunk_start_times.append(prepared.start_abs)
                    chunk_is_last.append(1 if chunk.is_last_chunk else 0)
                    chunk_token_counts.append(float(len(chunk.tokens)))
                    chunk_token_event_index.append(prepared.token_event_index)
                    chunk_token_event_slot_ids.append(prepared.token_event_slot_ids)
                    chunk_event_ids.append(prepared.event_ids)
                    chunk_event_times.append(prepared.event_times)
                    chunk_event_vals.append(prepared.event_vals)
                    chunk_event_valmask.append(prepared.event_valmask)
                    chunk_event_types.append(prepared.event_types)
                    chunk_event_payloads.append(prepared.event_payloads)
                    chunk_event_demographic_feature_ids.append(
                        prepared.event_demographic_feature_ids
                    )
                    chunk_event_memory_rule_scores.append(prepared.event_memory_rule_scores)
                    chunk_event_memory_group_ids.append(prepared.event_memory_group_ids)
                    chunk_event_memory_first_flags.append(prepared.event_memory_first_flags)
                    chunk_event_memory_chronic_flags.append(prepared.event_memory_chronic_flags)
                    chunk_event_med_group_ids.append(prepared.event_med_group_ids)
                    chunk_event_med_group_mask.append(prepared.event_med_group_mask)
                    chunk_event_med_route_ids.append(prepared.event_med_route_ids)
                    chunk_event_med_route_mask.append(prepared.event_med_route_mask)
                    chunk_event_med_form_ids.append(prepared.event_med_form_ids)
                    chunk_event_med_form_mask.append(prepared.event_med_form_mask)
                    chunk_event_med_freq_ids.append(prepared.event_med_freq_ids)
                    chunk_event_med_freq_mask.append(prepared.event_med_freq_mask)
                    chunk_event_med_unit_ids.append(prepared.event_med_unit_ids)
                    chunk_event_med_unit_mask.append(prepared.event_med_unit_mask)
                    chunk_event_med_marker_ids.append(prepared.event_med_marker_ids)
                    chunk_event_med_marker_mask.append(prepared.event_med_marker_mask)
                    chunk_event_med_dosage_values.append(prepared.event_med_dosage_values)
                    chunk_event_med_dosage_mask.append(prepared.event_med_dosage_mask)
                    chunk_event_med_rate_values.append(prepared.event_med_rate_values)
                    chunk_event_med_rate_mask.append(prepared.event_med_rate_mask)
                    chunk_event_med_duration_values.append(prepared.event_med_duration_values)
                    chunk_event_med_duration_mask.append(prepared.event_med_duration_mask)
                    if chunk.tokens:
                        chunk_duration_hours.append(float(chunk.tokens[-1].t_from_start_hours) - float(chunk.start_time_hours))
                    else:
                        chunk_duration_hours.append(0.0)

                subj_ids.append(chunk_ids)
                subj_times.append(chunk_times)
                subj_vals.append(chunk_vals)
                subj_types.append(chunk_types)
                subj_masks.append(chunk_masks)
                subj_valmask.append(chunk_valmask)
                subj_window_types.append(int(window_type_ids[wi]))
                subj_window_start_times.append(float(window.start_time_hours))
                subj_chunk_mask.append(chunk_mask)
                subj_chunk_start_offsets.append(chunk_offsets)
                subj_chunk_start_times.append(chunk_start_times)
                subj_chunk_is_last.append(chunk_is_last)
                subj_window_token_counts.append(float(len(window.tokens)))
                if window.tokens:
                    subj_window_duration_hours.append(float(window.tokens[-1].t_from_start_hours) - float(window.start_time_hours))
                else:
                    subj_window_duration_hours.append(0.0)
                subj_chunk_token_counts.append(chunk_token_counts)
                subj_chunk_duration_hours.append(chunk_duration_hours)
                subj_token_event_index.append(chunk_token_event_index)
                subj_token_event_slot_ids.append(chunk_token_event_slot_ids)
                subj_event_ids.append(chunk_event_ids)
                subj_event_times.append(chunk_event_times)
                subj_event_vals.append(chunk_event_vals)
                subj_event_valmask.append(chunk_event_valmask)
                subj_event_types.append(chunk_event_types)
                subj_event_payloads.append(chunk_event_payloads)
                subj_event_demographic_feature_ids.append(
                    chunk_event_demographic_feature_ids
                )
                subj_event_memory_rule_scores.append(chunk_event_memory_rule_scores)
                subj_event_memory_group_ids.append(chunk_event_memory_group_ids)
                subj_event_memory_first_flags.append(chunk_event_memory_first_flags)
                subj_event_memory_chronic_flags.append(chunk_event_memory_chronic_flags)
                subj_event_med_group_ids.append(chunk_event_med_group_ids)
                subj_event_med_group_mask.append(chunk_event_med_group_mask)
                subj_event_med_route_ids.append(chunk_event_med_route_ids)
                subj_event_med_route_mask.append(chunk_event_med_route_mask)
                subj_event_med_form_ids.append(chunk_event_med_form_ids)
                subj_event_med_form_mask.append(chunk_event_med_form_mask)
                subj_event_med_freq_ids.append(chunk_event_med_freq_ids)
                subj_event_med_freq_mask.append(chunk_event_med_freq_mask)
                subj_event_med_unit_ids.append(chunk_event_med_unit_ids)
                subj_event_med_unit_mask.append(chunk_event_med_unit_mask)
                subj_event_med_marker_ids.append(chunk_event_med_marker_ids)
                subj_event_med_marker_mask.append(chunk_event_med_marker_mask)
                subj_event_med_dosage_values.append(chunk_event_med_dosage_values)
                subj_event_med_dosage_mask.append(chunk_event_med_dosage_mask)
                subj_event_med_rate_values.append(chunk_event_med_rate_values)
                subj_event_med_rate_mask.append(chunk_event_med_rate_mask)
                subj_event_med_duration_values.append(chunk_event_med_duration_values)
                subj_event_med_duration_mask.append(chunk_event_med_duration_mask)

            batch_ids.append(subj_ids)
            batch_times.append(subj_times)
            batch_vals.append(subj_vals)
            batch_types.append(subj_types)
            batch_attn.append(subj_masks)
            batch_valmask.append(subj_valmask)
            batch_window_types.append(subj_window_types)
            batch_window_start_times.append(subj_window_start_times)
            batch_chunk_mask.append(subj_chunk_mask)
            batch_chunk_start_offsets.append(subj_chunk_start_offsets)
            batch_chunk_start_times.append(subj_chunk_start_times)
            batch_chunk_is_last.append(subj_chunk_is_last)
            batch_window_token_counts.append(subj_window_token_counts)
            batch_window_duration_hours.append(subj_window_duration_hours)
            batch_chunk_token_counts.append(subj_chunk_token_counts)
            batch_chunk_duration_hours.append(subj_chunk_duration_hours)
            batch_token_event_index.append(subj_token_event_index)
            batch_token_event_slot_ids.append(subj_token_event_slot_ids)
            batch_event_ids.append(subj_event_ids)
            batch_event_times.append(subj_event_times)
            batch_event_vals.append(subj_event_vals)
            batch_event_valmask.append(subj_event_valmask)
            batch_event_types.append(subj_event_types)
            batch_event_payloads.append(subj_event_payloads)
            batch_event_demographic_feature_ids.append(
                subj_event_demographic_feature_ids
            )
            batch_event_memory_rule_scores.append(subj_event_memory_rule_scores)
            batch_event_memory_group_ids.append(subj_event_memory_group_ids)
            batch_event_memory_first_flags.append(subj_event_memory_first_flags)
            batch_event_memory_chronic_flags.append(subj_event_memory_chronic_flags)
            batch_event_med_group_ids.append(subj_event_med_group_ids)
            batch_event_med_group_mask.append(subj_event_med_group_mask)
            batch_event_med_route_ids.append(subj_event_med_route_ids)
            batch_event_med_route_mask.append(subj_event_med_route_mask)
            batch_event_med_form_ids.append(subj_event_med_form_ids)
            batch_event_med_form_mask.append(subj_event_med_form_mask)
            batch_event_med_freq_ids.append(subj_event_med_freq_ids)
            batch_event_med_freq_mask.append(subj_event_med_freq_mask)
            batch_event_med_unit_ids.append(subj_event_med_unit_ids)
            batch_event_med_unit_mask.append(subj_event_med_unit_mask)
            batch_event_med_marker_ids.append(subj_event_med_marker_ids)
            batch_event_med_marker_mask.append(subj_event_med_marker_mask)
            batch_event_med_dosage_values.append(subj_event_med_dosage_values)
            batch_event_med_dosage_mask.append(subj_event_med_dosage_mask)
            batch_event_med_rate_values.append(subj_event_med_rate_values)
            batch_event_med_rate_mask.append(subj_event_med_rate_mask)
            batch_event_med_duration_values.append(subj_event_med_duration_values)
            batch_event_med_duration_mask.append(subj_event_med_duration_mask)

        out = self._pad_batch(
            batch_ids,
            batch_times,
            batch_vals,
            batch_types,
            batch_attn,
            batch_valmask,
            batch_window_types,
            batch_window_start_times,
            batch_chunk_mask,
            batch_chunk_start_offsets,
            batch_chunk_start_times,
            batch_chunk_is_last,
            batch_window_token_counts,
            batch_window_duration_hours,
            batch_chunk_token_counts,
            batch_chunk_duration_hours,
            batch_token_event_index,
            batch_token_event_slot_ids,
            batch_event_ids,
            batch_event_times,
            batch_event_vals,
            batch_event_valmask,
            batch_event_types,
            batch_event_payloads,
            batch_event_demographic_feature_ids,
            batch_event_memory_rule_scores,
            batch_event_memory_group_ids,
            batch_event_memory_first_flags,
            batch_event_memory_chronic_flags,
            batch_event_med_group_ids,
            batch_event_med_group_mask,
            batch_event_med_route_ids,
            batch_event_med_route_mask,
            batch_event_med_form_ids,
            batch_event_med_form_mask,
            batch_event_med_freq_ids,
            batch_event_med_freq_mask,
            batch_event_med_unit_ids,
            batch_event_med_unit_mask,
            batch_event_med_marker_ids,
            batch_event_med_marker_mask,
            batch_event_med_dosage_values,
            batch_event_med_dosage_mask,
            batch_event_med_rate_values,
            batch_event_med_rate_mask,
            batch_event_med_duration_values,
            batch_event_med_duration_mask,
        )
        total_windows_base = max(1, int(semantic_windows_total))
        total_subjects_base = max(1, int(len(normalized_timelines)))
        out["overflow_stats"] = {
            "subjects": int(len(normalized_timelines)),
            "subjects_with_overflow": int(subjects_with_overflow),
            "subjects_with_overflow_frac": float(subjects_with_overflow) / float(total_subjects_base),
            "semantic_windows_total": int(semantic_windows_total),
            "semantic_windows_kept": int(semantic_windows_kept),
            "semantic_windows_dropped_by_max_windows": int(semantic_windows_dropped),
            "semantic_windows_dropped_frac": float(semantic_windows_dropped) / float(total_windows_base),
            "semantic_tokens_dropped_by_max_windows": int(semantic_tokens_dropped),
            "semantic_structural_tokens_dropped_by_max_windows": int(semantic_structural_tokens_dropped),
            "chunks_dropped_by_max_chunks": int(chunks_dropped_by_cap),
            "chunk_tokens_dropped_by_max_chunks": int(chunk_tokens_dropped_by_cap),
            "chunk_structural_tokens_dropped_by_max_chunks": int(chunk_structural_tokens_dropped_by_cap),
        }
        out["subject_ids"] = torch.tensor(batch_subject_ids, dtype=torch.long)
        out["trajectory_ords"] = torch.tensor(batch_trajectory_ords, dtype=torch.long)
        return out

    def _split_special(self, timeline: List[EventFrame]) -> tuple[List[EventFrame], List[EventFrame]]:
        specials: List[EventFrame] = []
        events: List[EventFrame] = []
        for frame in timeline:
            if int(frame.category_id) == int(TokenCategory.SPECIAL):
                specials.append(frame)
            else:
                events.append(frame)
        return specials, events

    def _segment_into_windows(self, events: List[EventFrame]) -> List[List[EventToken]]:
        return [window.tokens for window in self._segment_windows(events)]

    def _segment_windows(self, events: List[EventFrame]) -> List[SegmentedWindow]:
        return segment_event_tokens(flatten_event_frames(events, clone=False), config=self.segmentation)

    def _consume_frames_for_token_budget(
        self,
        frames: List[EventFrame],
        *,
        token_count: int,
    ) -> tuple[List[EventFrame], List[EventFrame]]:
        need = max(0, int(token_count))
        if need == 0:
            return [], list(frames)

        remaining = [clone_event_frame(frame) for frame in frames]
        out: List[EventFrame] = []
        while need > 0 and remaining:
            frame = remaining.pop(0)
            if frame.token_count <= need:
                out.append(frame)
                need -= int(frame.token_count)
                continue

            out.append(slice_event_frame(frame, start=0, stop=need))
            remaining.insert(0, slice_event_frame(frame, start=need))
            need = 0

        if need != 0:
            raise ValueError(
                f"Unable to align frame bundles to token budget {int(token_count)}; {int(need)} tokens remain unassigned."
            )
        return out, remaining

    def _align_frames_to_segmented_windows(
        self,
        events: List[EventFrame],
        segmented_windows: List[SegmentedWindow],
    ) -> List[List[EventFrame]]:
        remaining = [clone_event_frame(frame) for frame in events]
        out: List[List[EventFrame]] = []
        for window in segmented_windows:
            aligned, remaining = self._consume_frames_for_token_budget(
                remaining,
                token_count=len(window.tokens),
            )
            out.append(aligned)
        return out

    def _align_frames_to_chunked_windows(
        self,
        semantic_frame_windows: List[List[EventFrame]],
        chunked_windows: List[SegmentedWindow],
    ) -> List[List[List[EventFrame]]]:
        out: List[List[List[EventFrame]]] = []
        for frame_window, chunked_window in zip(semantic_frame_windows, chunked_windows):
            remaining = [clone_event_frame(frame) for frame in frame_window]
            window_chunks: List[List[EventFrame]] = []
            for chunk in chunked_window.chunks:
                aligned, remaining = self._consume_frames_for_token_budget(
                    remaining,
                    token_count=len(chunk.tokens),
                )
                window_chunks.append(aligned)
            out.append(window_chunks)
        return out

    @staticmethod
    def _extract_frame_numeric_scalar(frame: EventFrame) -> tuple[float, int]:
        candidate_keys = ("z", "numeric_value", "value_as_number", "value")
        for key in candidate_keys:
            raw = (frame.num_attrs or {}).get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value, 1
        return 0.0, 0

    @staticmethod
    def _extract_frame_categorical_id(
        frame: EventFrame,
        *,
        attr_name: str,
    ) -> tuple[int, int]:
        raw = (frame.cat_attrs or {}).get(attr_name, None)
        if raw is None:
            return 0, 0
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 0, 0
        return value, 1

    @staticmethod
    def _extract_frame_numeric_attr(
        frame: EventFrame,
        *,
        attr_name: str,
    ) -> tuple[float, int]:
        raw = (frame.num_attrs or {}).get(attr_name, None)
        if raw is None:
            return 0.0, 0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0, 0
        if not math.isfinite(value):
            return 0.0, 0
        return value, 1

    def _build_marker_frame(
        self,
        *,
        token_id: int,
        time_hours: float,
        cat_attrs: Dict[str, int] | None = None,
    ) -> EventFrame:
        return build_event_frame(
            [
                EventToken(
                    value_id=int(token_id),
                    category_id=int(self.window_markers.marker_category),
                    t_from_start_hours=float(time_hours),
                    dt_from_prev_hours=0.0,
                    cat_attrs=dict(cat_attrs or {}),
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.SPECIAL,
            semantic_label="window_marker",
        )

    def _chunk_windows(
        self,
        semantic_windows: List[SegmentedWindow],
        *,
        special_tokens: List[EventToken] | None = None,
    ) -> List[SegmentedWindow]:
        marker_slots = 2 if self.window_markers.enabled else 0
        special_count = len(special_tokens or [])
        max_content_tokens = max(1, int(self.max_len) - int(marker_slots) - int(special_count))
        return chunk_segmented_windows(
            semantic_windows,
            max_content_tokens=max_content_tokens,
            max_chunks_per_window=int(self.max_chunks_per_window),
            config=self.segmentation,
        )

    def _window_end_token_id(self) -> int:
        return (
            int(self.window_markers.end_token_id)
            if self.window_markers.end_token_id is not None
            else int(self.window_markers.type_token_offset) + int(self.window_markers.num_types)
        )

    def _window_continue_token_id(self) -> int:
        if self.window_markers.continue_token_id is not None:
            return int(self.window_markers.continue_token_id)
        return self._window_end_token_id() + 1

    def _process_chunk(
        self,
        chunk: SegmentedChunk,
        special_tokens: List[EventToken],
        *,
        special_frames: List[EventFrame] | None = None,
        chunk_frames: List[EventFrame] | None = None,
        w_type_id: int,
        w_start_abs: float,
        next_type_id: int | None,
        next_start_abs: float | None,
        seen_memory_keys: set[str] | None = None,
    ) -> PreparedChunk:
        if not chunk.tokens:
            return PreparedChunk(
                ids=[],
                times=[],
                vals=[],
                valmask=[],
                types=[],
                start_abs=float(w_start_abs),
                start_offset=0.0,
                token_event_index=[],
                token_event_slot_ids=[],
                event_ids=[],
                event_times=[],
                event_vals=[],
                event_valmask=[],
                event_types=[],
                event_payloads=[],
                event_demographic_feature_ids=[],
                event_memory_rule_scores=[],
                event_memory_group_ids=[],
                event_memory_first_flags=[],
                event_memory_chronic_flags=[],
                event_med_group_ids=[],
                event_med_group_mask=[],
                event_med_route_ids=[],
                event_med_route_mask=[],
                event_med_form_ids=[],
                event_med_form_mask=[],
                event_med_freq_ids=[],
                event_med_freq_mask=[],
                event_med_unit_ids=[],
                event_med_unit_mask=[],
                event_med_marker_ids=[],
                event_med_marker_mask=[],
                event_med_dosage_values=[],
                event_med_dosage_mask=[],
                event_med_rate_values=[],
                event_med_rate_mask=[],
                event_med_duration_values=[],
                event_med_duration_mask=[],
            )

        w_start_abs = float(w_start_abs)
        w_type_id = int(w_type_id)
        chunk_start_abs = float(chunk.start_time_hours)
        chunk_start_offset = max(0.0, chunk_start_abs - w_start_abs)
        special_frames = (
            [clone_event_frame(frame) for frame in special_frames]
            if special_frames is not None
            else ensure_event_frames(special_tokens, payload_kind=EventPayloadKind.SPECIAL)
        )
        chunk_frames = (
            [clone_event_frame(frame) for frame in chunk_frames]
            if chunk_frames is not None
            else ensure_event_frames(chunk.tokens)
        )
        prefix: List[EventToken] = list(special_tokens)
        prefix_frames: List[EventFrame] = [clone_event_frame(frame) for frame in special_frames]
        suffix: List[EventToken] = []
        suffix_frames: List[EventFrame] = []
        if self.window_markers.enabled:
            type_token_id = int(self.window_markers.type_token_offset) + int(w_type_id)
            end_token_id = self._window_end_token_id()
            end_mode = str(getattr(self.window_markers, "end_mode", "end_token"))

            start_marker = self._build_marker_frame(
                token_id=type_token_id,
                time_hours=chunk_start_abs,
                cat_attrs={"window_type_id": int(w_type_id)},
            )
            prefix_frames.append(start_marker)
            prefix.extend(start_marker.token_bundle)
            if not chunk.is_last_chunk:
                continue_marker = self._build_marker_frame(
                    token_id=self._window_continue_token_id(),
                    time_hours=float(chunk.tokens[-1].t_from_start_hours),
                    cat_attrs={"window_type_id": int(w_type_id), "chunk_continue": 1},
                )
                suffix_frames.append(continue_marker)
                suffix.extend(continue_marker.token_bundle)
            elif end_mode == "next_type" and next_type_id is not None:
                next_type_id_int = self._clamp_window_type_id(int(next_type_id))
                next_token_id = int(self.window_markers.type_token_offset) + int(next_type_id_int)
                next_marker = self._build_marker_frame(
                    token_id=next_token_id,
                    time_hours=(
                        float(next_start_abs)
                        if next_start_abs is not None
                        else float(chunk.tokens[-1].t_from_start_hours)
                    ),
                    cat_attrs={"window_type_id": int(next_type_id_int)},
                )
                suffix_frames.append(next_marker)
                suffix.extend(next_marker.token_bundle)
            else:
                end_marker = self._build_marker_frame(
                    token_id=end_token_id,
                    time_hours=float(chunk.tokens[-1].t_from_start_hours),
                )
                suffix_frames.append(end_marker)
                suffix.extend(end_marker.token_bundle)

        budget = max(0, int(self.max_len) - len(prefix) - len(suffix))
        seq: List[EventToken] = prefix + chunk.tokens[:budget] + suffix
        seq_frames: List[EventFrame] = (
            prefix_frames
            + [clone_event_frame(frame) for frame in chunk_frames]
            + suffix_frames
        )

        ids: List[int] = []
        times: List[float] = []
        vals: List[float] = []
        val_mask: List[int] = []
        types: List[int] = []
        token_event_index: List[int] = []
        token_event_slot_ids: List[int] = []
        event_ids: List[int] = []
        event_times: List[float] = []
        event_vals: List[float] = []
        event_valmask: List[int] = []
        event_types: List[int] = []
        event_payloads: List[int] = []
        event_demographic_feature_ids: List[int] = []
        event_memory_rule_scores: List[float] = []
        event_memory_group_ids: List[int] = []
        event_memory_first_flags: List[int] = []
        event_memory_chronic_flags: List[int] = []
        event_med_group_ids: List[int] = []
        event_med_group_mask: List[int] = []
        event_med_route_ids: List[int] = []
        event_med_route_mask: List[int] = []
        event_med_form_ids: List[int] = []
        event_med_form_mask: List[int] = []
        event_med_freq_ids: List[int] = []
        event_med_freq_mask: List[int] = []
        event_med_unit_ids: List[int] = []
        event_med_unit_mask: List[int] = []
        event_med_marker_ids: List[int] = []
        event_med_marker_mask: List[int] = []
        event_med_dosage_values: List[float] = []
        event_med_dosage_mask: List[int] = []
        event_med_rate_values: List[float] = []
        event_med_rate_mask: List[int] = []
        event_med_duration_values: List[float] = []
        event_med_duration_mask: List[int] = []
        seen_exact_keys = seen_memory_keys if seen_memory_keys is not None else set()

        for tok in seq:
            ids.append(int(tok.value_id))
            types.append(int(tok.category_id))
            raw_val = tok.num_attrs.get("numeric_value") if tok.num_attrs is not None else None
            has_val = False
            safe_val = 0.0
            if raw_val is not None:
                try:
                    parsed = float(raw_val)
                    if math.isfinite(parsed):
                        safe_val = parsed
                        has_val = True
                except (TypeError, ValueError):
                    pass
            vals.append(safe_val)
            val_mask.append(1 if has_val else 0)
            if len(times) < len(prefix):
                times.append(0.0)
            else:
                rel_t = max(0.0, float(tok.t_from_start_hours) - chunk_start_abs)
                if not math.isfinite(rel_t):
                    rel_t = 0.0
                times.append(rel_t)

        for event_idx, frame in enumerate(seq_frames):
            event_ids.append(int(frame.token_bundle[0].value_id))
            event_types.append(int(frame.category_id))
            event_payloads.append(payload_kind_to_id(frame.payload_kind))
            frame_cat_attrs = dict(frame.cat_attrs or {})
            demographic_feature_id = 0
            if int(frame.category_id) == int(TokenCategory.SPECIAL):
                if int(frame_cat_attrs.get("global_demographic", 0)) != 0:
                    demographic_feature_id = int(
                        frame_cat_attrs.get("demographic_feature_id", 0) or 0
                    )
            event_demographic_feature_ids.append(int(demographic_feature_id))
            scalar, scalar_mask = self._extract_frame_numeric_scalar(frame)
            event_vals.append(float(scalar))
            event_valmask.append(int(scalar_mask))
            memory_rule = classify_event_frame_for_memory(
                frame=frame,
                seen_exact_keys=seen_exact_keys,
            )
            event_memory_rule_scores.append(float(memory_rule.rule_score))
            event_memory_group_ids.append(int(memory_rule.group_id))
            event_memory_first_flags.append(int(memory_rule.first_occurrence))
            event_memory_chronic_flags.append(int(memory_rule.chronic_flag))
            if int(frame.category_id) == int(TokenCategory.MEDICATION):
                med_group_id, med_group_mask = self._extract_frame_categorical_id(
                    frame,
                    attr_name="med_group",
                )
                med_route_id, med_route_mask = self._extract_frame_categorical_id(
                    frame,
                    attr_name="route",
                )
                med_form_id, med_form_mask = self._extract_frame_categorical_id(
                    frame,
                    attr_name="form",
                )
                med_freq_id, med_freq_mask = self._extract_frame_categorical_id(
                    frame,
                    attr_name="freq",
                )
                med_unit_id, med_unit_mask = self._extract_frame_categorical_id(
                    frame,
                    attr_name="unit",
                )
                med_marker_id, med_marker_mask = self._extract_frame_categorical_id(
                    frame,
                    attr_name="event_marker",
                )
                med_dosage_value, med_dosage_mask = self._extract_frame_numeric_attr(
                    frame,
                    attr_name="dosage",
                )
                med_rate_value, med_rate_mask = self._extract_frame_numeric_attr(
                    frame,
                    attr_name="rate",
                )
                med_duration_value, med_duration_mask = self._extract_frame_numeric_attr(
                    frame,
                    attr_name="duration_hours",
                )
            else:
                med_group_id = med_group_mask = 0
                med_route_id = med_route_mask = 0
                med_form_id = med_form_mask = 0
                med_freq_id = med_freq_mask = 0
                med_unit_id = med_unit_mask = 0
                med_marker_id = med_marker_mask = 0
                med_dosage_value = 0.0
                med_dosage_mask = 0
                med_rate_value = 0.0
                med_rate_mask = 0
                med_duration_value = 0.0
                med_duration_mask = 0
            event_med_group_ids.append(int(med_group_id))
            event_med_group_mask.append(int(med_group_mask))
            event_med_route_ids.append(int(med_route_id))
            event_med_route_mask.append(int(med_route_mask))
            event_med_form_ids.append(int(med_form_id))
            event_med_form_mask.append(int(med_form_mask))
            event_med_freq_ids.append(int(med_freq_id))
            event_med_freq_mask.append(int(med_freq_mask))
            event_med_unit_ids.append(int(med_unit_id))
            event_med_unit_mask.append(int(med_unit_mask))
            event_med_marker_ids.append(int(med_marker_id))
            event_med_marker_mask.append(int(med_marker_mask))
            event_med_dosage_values.append(float(med_dosage_value))
            event_med_dosage_mask.append(int(med_dosage_mask))
            event_med_rate_values.append(float(med_rate_value))
            event_med_rate_mask.append(int(med_rate_mask))
            event_med_duration_values.append(float(med_duration_value))
            event_med_duration_mask.append(int(med_duration_mask))
            if event_idx < len(prefix_frames):
                event_times.append(0.0)
            else:
                rel_event_time = max(0.0, float(frame.t_from_start_hours) - chunk_start_abs)
                if not math.isfinite(rel_event_time):
                    rel_event_time = 0.0
                event_times.append(rel_event_time)
            for slot_idx, _ in enumerate(frame.token_bundle):
                token_event_index.append(int(event_idx))
                token_event_slot_ids.append(int(slot_idx))
            update_seen_memory_keys(seen_exact_keys, frame)

        if len(token_event_index) != len(seq):
            raise ValueError(
                "Chunk event alignment mismatch: token-event map length does not match token sequence length."
            )

        return PreparedChunk(
            ids=ids,
            times=times,
            vals=vals,
            valmask=val_mask,
            types=types,
            start_abs=float(chunk_start_abs),
            start_offset=float(chunk_start_offset),
            token_event_index=token_event_index,
            token_event_slot_ids=token_event_slot_ids,
            event_ids=event_ids,
            event_times=event_times,
            event_vals=event_vals,
            event_valmask=event_valmask,
            event_types=event_types,
            event_payloads=event_payloads,
            event_demographic_feature_ids=event_demographic_feature_ids,
            event_memory_rule_scores=event_memory_rule_scores,
            event_memory_group_ids=event_memory_group_ids,
            event_memory_first_flags=event_memory_first_flags,
            event_memory_chronic_flags=event_memory_chronic_flags,
            event_med_group_ids=event_med_group_ids,
            event_med_group_mask=event_med_group_mask,
            event_med_route_ids=event_med_route_ids,
            event_med_route_mask=event_med_route_mask,
            event_med_form_ids=event_med_form_ids,
            event_med_form_mask=event_med_form_mask,
            event_med_freq_ids=event_med_freq_ids,
            event_med_freq_mask=event_med_freq_mask,
            event_med_unit_ids=event_med_unit_ids,
            event_med_unit_mask=event_med_unit_mask,
            event_med_marker_ids=event_med_marker_ids,
            event_med_marker_mask=event_med_marker_mask,
            event_med_dosage_values=event_med_dosage_values,
            event_med_dosage_mask=event_med_dosage_mask,
            event_med_rate_values=event_med_rate_values,
            event_med_rate_mask=event_med_rate_mask,
            event_med_duration_values=event_med_duration_values,
            event_med_duration_mask=event_med_duration_mask,
        )

    def _process_window(
        self,
        window_tokens: List[EventToken],
        special_tokens: List[EventToken],
        *,
        w_type_id: int,
        w_start_abs: float,
        next_type_id: int | None,
        next_start_abs: float | None,
    ) -> tuple[List[int], List[float], List[float], List[int], List[int], int, float]:
        """
        Backward-compatible helper used by audits/tests that still inspect a single
        semantic window as one local sequence.
        """
        pseudo_chunk = SegmentedChunk(
            tokens=list(window_tokens),
            start_time_hours=float(w_start_abs) if window_tokens else 0.0,
            chunk_index=0,
            is_first_chunk=True,
            is_last_chunk=True,
        )
        prepared = self._process_chunk(
            pseudo_chunk,
            special_tokens=special_tokens,
            special_frames=ensure_event_frames(special_tokens, payload_kind=EventPayloadKind.SPECIAL),
            chunk_frames=ensure_event_frames(window_tokens),
            w_type_id=w_type_id,
            w_start_abs=w_start_abs,
            next_type_id=next_type_id,
            next_start_abs=next_start_abs,
        )
        return (
            prepared.ids,
            prepared.times,
            prepared.vals,
            prepared.valmask,
            prepared.types,
            int(w_type_id),
            float(w_start_abs),
        )

    def _infer_window_type_id(self, window_tokens: List[EventToken]) -> int:
        """
        Best-effort window type inference.

        Default behavior:
          - Use the earliest explicit `window_type_id` / `transition_window_type_id`.
          - Otherwise fall back to UNK.
        """
        if not window_tokens:
            return int(self.window_markers.unk_type_id)

        for tok in window_tokens:
            if tok.cat_attrs is None:
                continue
            for key in ("window_type_id", "transition_window_type_id"):
                if key not in tok.cat_attrs:
                    continue
                try:
                    w = int(tok.cat_attrs[key])
                    return self._clamp_window_type_id(w)
                except Exception:
                    return int(self.window_markers.unk_type_id)

        return int(self.window_markers.unk_type_id)

    def _clamp_window_type_id(self, w: int) -> int:
        if self.window_markers.num_types <= 0:
            return int(self.window_markers.unk_type_id)
        if w < 0 or w >= int(self.window_markers.num_types):
            return int(self.window_markers.unk_type_id)
        return int(w)

    def _pad_batch(
        self,
        batch_ids: List[List[List[List[int]]]],
        batch_times: List[List[List[List[float]]]],
        batch_vals: List[List[List[List[float]]]],
        batch_types: List[List[List[List[int]]]],
        batch_masks: List[List[List[List[int]]]],
        batch_valmask: List[List[List[List[int]]]],
        batch_window_types: List[List[int]],
        batch_window_start_times: List[List[float]],
        batch_chunk_mask: List[List[List[int]]],
        batch_chunk_start_offsets: List[List[List[float]]],
        batch_chunk_start_times: List[List[List[float]]],
        batch_chunk_is_last: List[List[List[int]]],
        batch_window_token_counts: List[List[float]],
        batch_window_duration_hours: List[List[float]],
        batch_chunk_token_counts: List[List[List[float]]],
        batch_chunk_duration_hours: List[List[List[float]]],
        batch_token_event_index: List[List[List[List[int]]]],
        batch_token_event_slot_ids: List[List[List[List[int]]]],
        batch_event_ids: List[List[List[List[int]]]],
        batch_event_times: List[List[List[List[float]]]],
        batch_event_vals: List[List[List[List[float]]]],
        batch_event_valmask: List[List[List[List[int]]]],
        batch_event_types: List[List[List[List[int]]]],
        batch_event_payloads: List[List[List[List[int]]]],
        batch_event_demographic_feature_ids: List[List[List[List[int]]]],
        batch_event_memory_rule_scores: List[List[List[List[float]]]],
        batch_event_memory_group_ids: List[List[List[List[int]]]],
        batch_event_memory_first_flags: List[List[List[List[int]]]],
        batch_event_memory_chronic_flags: List[List[List[List[int]]]],
        batch_event_med_group_ids: List[List[List[List[int]]]],
        batch_event_med_group_mask: List[List[List[List[int]]]],
        batch_event_med_route_ids: List[List[List[List[int]]]],
        batch_event_med_route_mask: List[List[List[List[int]]]],
        batch_event_med_form_ids: List[List[List[List[int]]]],
        batch_event_med_form_mask: List[List[List[List[int]]]],
        batch_event_med_freq_ids: List[List[List[List[int]]]],
        batch_event_med_freq_mask: List[List[List[List[int]]]],
        batch_event_med_unit_ids: List[List[List[List[int]]]],
        batch_event_med_unit_mask: List[List[List[List[int]]]],
        batch_event_med_marker_ids: List[List[List[List[int]]]],
        batch_event_med_marker_mask: List[List[List[List[int]]]],
        batch_event_med_dosage_values: List[List[List[List[float]]]],
        batch_event_med_dosage_mask: List[List[List[List[int]]]],
        batch_event_med_rate_values: List[List[List[List[float]]]],
        batch_event_med_rate_mask: List[List[List[List[int]]]],
        batch_event_med_duration_values: List[List[List[List[float]]]],
        batch_event_med_duration_mask: List[List[List[List[int]]]],
    ) -> Dict[str, Any]:
        B = len(batch_ids)
        W = max((len(x) for x in batch_ids), default=0)
        C = max((len(chunks) for subj in batch_ids for chunks in subj), default=0)
        L = self.max_len
        E = max((len(events) for subj in batch_event_times for win in subj for events in win), default=0)

        input_ids = torch.full((B, W, C, L), self.pad_id, dtype=torch.long)
        time_ids = torch.zeros((B, W, C, L), dtype=torch.float)
        numeric_values = torch.zeros((B, W, C, L, 1), dtype=torch.float)
        token_type_ids = torch.zeros((B, W, C, L), dtype=torch.long)
        attention_mask = torch.zeros((B, W, C, L), dtype=torch.long)
        window_mask = torch.zeros((B, W), dtype=torch.long)
        chunk_mask = torch.zeros((B, W, C), dtype=torch.long)
        numeric_mask = torch.zeros((B, W, C, L), dtype=torch.long)
        window_type_ids = torch.zeros((B, W), dtype=torch.long)
        window_start_times = torch.zeros((B, W), dtype=torch.float)
        chunk_start_offsets = torch.zeros((B, W, C), dtype=torch.float)
        chunk_start_times = torch.zeros((B, W, C), dtype=torch.float)
        chunk_is_last = torch.zeros((B, W, C), dtype=torch.long)
        semantic_token_counts = torch.zeros((B, W), dtype=torch.float)
        semantic_duration_hours = torch.zeros((B, W), dtype=torch.float)
        chunk_token_counts = torch.zeros((B, W, C), dtype=torch.float)
        chunk_duration_hours = torch.zeros((B, W, C), dtype=torch.float)
        token_event_index = torch.full((B, W, C, L), fill_value=-1, dtype=torch.long)
        token_event_slot_ids = torch.zeros((B, W, C, L), dtype=torch.long)
        event_input_ids = torch.full((B, W, C, E), self.pad_id, dtype=torch.long)
        event_time_ids = torch.zeros((B, W, C, E), dtype=torch.float)
        event_numeric_values = torch.zeros((B, W, C, E, 1), dtype=torch.float)
        event_numeric_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_type_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_payload_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_demographic_feature_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_attention_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_memory_rule_scores = torch.zeros((B, W, C, E), dtype=torch.float)
        event_memory_group_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_memory_first_flags = torch.zeros((B, W, C, E), dtype=torch.long)
        event_memory_chronic_flags = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_group_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_group_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_route_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_route_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_form_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_form_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_freq_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_freq_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_unit_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_unit_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_marker_ids = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_marker_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_dosage_values = torch.zeros((B, W, C, E, 1), dtype=torch.float)
        event_med_dosage_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_rate_values = torch.zeros((B, W, C, E, 1), dtype=torch.float)
        event_med_rate_mask = torch.zeros((B, W, C, E), dtype=torch.long)
        event_med_duration_values = torch.zeros((B, W, C, E, 1), dtype=torch.float)
        event_med_duration_mask = torch.zeros((B, W, C, E), dtype=torch.long)

        for b in range(B):
            for w in range(len(batch_ids[b])):
                window_mask[b, w] = 1
                if b < len(batch_window_types) and w < len(batch_window_types[b]):
                    window_type_ids[b, w] = int(batch_window_types[b][w])
                if b < len(batch_window_start_times) and w < len(batch_window_start_times[b]):
                    window_start_times[b, w] = float(batch_window_start_times[b][w])
                if b < len(batch_window_token_counts) and w < len(batch_window_token_counts[b]):
                    semantic_token_counts[b, w] = float(batch_window_token_counts[b][w])
                if b < len(batch_window_duration_hours) and w < len(batch_window_duration_hours[b]):
                    semantic_duration_hours[b, w] = float(batch_window_duration_hours[b][w])

                for c in range(len(batch_ids[b][w])):
                    ids = batch_ids[b][w][c]
                    times = batch_times[b][w][c]
                    vals = batch_vals[b][w][c]
                    types = batch_types[b][w][c]
                    mask = batch_masks[b][w][c]
                    valmask = batch_valmask[b][w][c]
                    tok_to_event = batch_token_event_index[b][w][c]
                    tok_event_slots = batch_token_event_slot_ids[b][w][c]
                    event_ids = batch_event_ids[b][w][c]
                    event_times = batch_event_times[b][w][c]
                    event_vals = batch_event_vals[b][w][c]
                    event_valmask = batch_event_valmask[b][w][c]
                    event_types = batch_event_types[b][w][c]
                    event_payloads = batch_event_payloads[b][w][c]
                    event_demographic_features = batch_event_demographic_feature_ids[b][w][c]
                    event_memory_scores = batch_event_memory_rule_scores[b][w][c]
                    event_memory_groups = batch_event_memory_group_ids[b][w][c]
                    event_memory_first = batch_event_memory_first_flags[b][w][c]
                    event_memory_chronic = batch_event_memory_chronic_flags[b][w][c]
                    med_group_ids = batch_event_med_group_ids[b][w][c]
                    med_group_mask = batch_event_med_group_mask[b][w][c]
                    med_route_ids = batch_event_med_route_ids[b][w][c]
                    med_route_mask = batch_event_med_route_mask[b][w][c]
                    med_form_ids = batch_event_med_form_ids[b][w][c]
                    med_form_mask = batch_event_med_form_mask[b][w][c]
                    med_freq_ids = batch_event_med_freq_ids[b][w][c]
                    med_freq_mask = batch_event_med_freq_mask[b][w][c]
                    med_unit_ids = batch_event_med_unit_ids[b][w][c]
                    med_unit_mask = batch_event_med_unit_mask[b][w][c]
                    med_marker_ids = batch_event_med_marker_ids[b][w][c]
                    med_marker_mask = batch_event_med_marker_mask[b][w][c]
                    med_dosage_values = batch_event_med_dosage_values[b][w][c]
                    med_dosage_mask = batch_event_med_dosage_mask[b][w][c]
                    med_rate_values = batch_event_med_rate_values[b][w][c]
                    med_rate_mask = batch_event_med_rate_mask[b][w][c]
                    med_duration_values = batch_event_med_duration_values[b][w][c]
                    med_duration_mask = batch_event_med_duration_mask[b][w][c]
                    seq_len = min(len(ids), L)
                    event_len = min(len(event_times), E)
                    chunk_mask[b, w, c] = 1

                    input_ids[b, w, c, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
                    time_ids[b, w, c, :seq_len] = torch.tensor(times[:seq_len], dtype=torch.float)
                    token_type_ids[b, w, c, :seq_len] = torch.tensor(types[:seq_len], dtype=torch.long)
                    attention_mask[b, w, c, :seq_len] = torch.tensor(mask[:seq_len], dtype=torch.long)

                    val_slice = torch.tensor(vals[:seq_len], dtype=torch.float).unsqueeze(-1)
                    numeric_values[b, w, c, :seq_len, :] = val_slice
                    numeric_mask[b, w, c, :seq_len] = torch.tensor(valmask[:seq_len], dtype=torch.long)
                    token_event_index[b, w, c, :seq_len] = torch.tensor(tok_to_event[:seq_len], dtype=torch.long)
                    token_event_slot_ids[b, w, c, :seq_len] = torch.tensor(tok_event_slots[:seq_len], dtype=torch.long)
                    if event_len > 0:
                        event_input_ids[b, w, c, :event_len] = torch.tensor(event_ids[:event_len], dtype=torch.long)
                        event_time_ids[b, w, c, :event_len] = torch.tensor(event_times[:event_len], dtype=torch.float)
                        event_numeric_values[b, w, c, :event_len, :] = (
                            torch.tensor(event_vals[:event_len], dtype=torch.float).unsqueeze(-1)
                        )
                        event_numeric_mask[b, w, c, :event_len] = torch.tensor(
                            event_valmask[:event_len],
                            dtype=torch.long,
                        )
                        event_type_ids[b, w, c, :event_len] = torch.tensor(event_types[:event_len], dtype=torch.long)
                        event_payload_ids[b, w, c, :event_len] = torch.tensor(
                            event_payloads[:event_len],
                            dtype=torch.long,
                        )
                        event_demographic_feature_ids[b, w, c, :event_len] = torch.tensor(
                            event_demographic_features[:event_len],
                            dtype=torch.long,
                        )
                        event_attention_mask[b, w, c, :event_len] = 1
                        event_memory_rule_scores[b, w, c, :event_len] = torch.tensor(
                            event_memory_scores[:event_len],
                            dtype=torch.float,
                        )
                        event_memory_group_ids[b, w, c, :event_len] = torch.tensor(
                            event_memory_groups[:event_len],
                            dtype=torch.long,
                        )
                        event_memory_first_flags[b, w, c, :event_len] = torch.tensor(
                            event_memory_first[:event_len],
                            dtype=torch.long,
                        )
                        event_memory_chronic_flags[b, w, c, :event_len] = torch.tensor(
                            event_memory_chronic[:event_len],
                            dtype=torch.long,
                        )
                        event_med_group_ids[b, w, c, :event_len] = torch.tensor(
                            med_group_ids[:event_len],
                            dtype=torch.long,
                        )
                        event_med_group_mask[b, w, c, :event_len] = torch.tensor(
                            med_group_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_route_ids[b, w, c, :event_len] = torch.tensor(
                            med_route_ids[:event_len],
                            dtype=torch.long,
                        )
                        event_med_route_mask[b, w, c, :event_len] = torch.tensor(
                            med_route_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_form_ids[b, w, c, :event_len] = torch.tensor(
                            med_form_ids[:event_len],
                            dtype=torch.long,
                        )
                        event_med_form_mask[b, w, c, :event_len] = torch.tensor(
                            med_form_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_freq_ids[b, w, c, :event_len] = torch.tensor(
                            med_freq_ids[:event_len],
                            dtype=torch.long,
                        )
                        event_med_freq_mask[b, w, c, :event_len] = torch.tensor(
                            med_freq_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_unit_ids[b, w, c, :event_len] = torch.tensor(
                            med_unit_ids[:event_len],
                            dtype=torch.long,
                        )
                        event_med_unit_mask[b, w, c, :event_len] = torch.tensor(
                            med_unit_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_marker_ids[b, w, c, :event_len] = torch.tensor(
                            med_marker_ids[:event_len],
                            dtype=torch.long,
                        )
                        event_med_marker_mask[b, w, c, :event_len] = torch.tensor(
                            med_marker_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_dosage_values[b, w, c, :event_len, :] = torch.tensor(
                            med_dosage_values[:event_len],
                            dtype=torch.float,
                        ).unsqueeze(-1)
                        event_med_dosage_mask[b, w, c, :event_len] = torch.tensor(
                            med_dosage_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_rate_values[b, w, c, :event_len, :] = torch.tensor(
                            med_rate_values[:event_len],
                            dtype=torch.float,
                        ).unsqueeze(-1)
                        event_med_rate_mask[b, w, c, :event_len] = torch.tensor(
                            med_rate_mask[:event_len],
                            dtype=torch.long,
                        )
                        event_med_duration_values[b, w, c, :event_len, :] = torch.tensor(
                            med_duration_values[:event_len],
                            dtype=torch.float,
                        ).unsqueeze(-1)
                        event_med_duration_mask[b, w, c, :event_len] = torch.tensor(
                            med_duration_mask[:event_len],
                            dtype=torch.long,
                        )

                    if b < len(batch_chunk_start_offsets) and w < len(batch_chunk_start_offsets[b]) and c < len(batch_chunk_start_offsets[b][w]):
                        chunk_start_offsets[b, w, c] = float(batch_chunk_start_offsets[b][w][c])
                    if b < len(batch_chunk_start_times) and w < len(batch_chunk_start_times[b]) and c < len(batch_chunk_start_times[b][w]):
                        chunk_start_times[b, w, c] = float(batch_chunk_start_times[b][w][c])
                    if b < len(batch_chunk_is_last) and w < len(batch_chunk_is_last[b]) and c < len(batch_chunk_is_last[b][w]):
                        chunk_is_last[b, w, c] = int(batch_chunk_is_last[b][w][c])
                    if b < len(batch_chunk_token_counts) and w < len(batch_chunk_token_counts[b]) and c < len(batch_chunk_token_counts[b][w]):
                        chunk_token_counts[b, w, c] = float(batch_chunk_token_counts[b][w][c])
                    if b < len(batch_chunk_duration_hours) and w < len(batch_chunk_duration_hours[b]) and c < len(batch_chunk_duration_hours[b][w]):
                        chunk_duration_hours[b, w, c] = float(batch_chunk_duration_hours[b][w][c])

        input_ids_out = input_ids
        event_input_ids_out = event_input_ids
        remap_stats: Dict[str, Any] | None = None
        input_ids_global = input_ids.clone() if self.emit_global_input_ids else None
        if self.id_remapper is not None:
            input_ids_out, remap_stats = self.id_remapper.map_tensor(
                input_ids,
                valid_mask=attention_mask,
            )
            event_input_ids_out, _ = self.id_remapper.map_tensor(
                event_input_ids,
                valid_mask=event_attention_mask,
            )

        out: Dict[str, Any] = {
            "input_ids": input_ids_out,
            "time_ids": time_ids,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "window_mask": window_mask,
            "chunk_mask": chunk_mask,
            "window_type_ids": window_type_ids,
            "window_start_times": window_start_times,
            "chunk_start_offsets": chunk_start_offsets,
            "chunk_start_times": chunk_start_times,
            "chunk_is_last": chunk_is_last,
            "semantic_token_counts": semantic_token_counts,
            "semantic_duration_hours": semantic_duration_hours,
            "chunk_token_counts": chunk_token_counts,
            "chunk_duration_hours": chunk_duration_hours,
            "token_event_index": token_event_index,
            "token_event_slot_ids": token_event_slot_ids,
            "event_input_ids": event_input_ids_out,
            "event_time_ids": event_time_ids,
            "event_numeric_values": event_numeric_values,
            "event_numeric_mask": event_numeric_mask,
            "event_type_ids": event_type_ids,
            "event_payload_ids": event_payload_ids,
            "event_demographic_feature_ids": event_demographic_feature_ids,
            "event_attention_mask": event_attention_mask,
            "event_memory_rule_scores": event_memory_rule_scores,
            "event_memory_group_ids": event_memory_group_ids,
            "event_memory_first_flags": event_memory_first_flags,
            "event_memory_chronic_flags": event_memory_chronic_flags,
            "event_med_group_ids": event_med_group_ids,
            "event_med_group_mask": event_med_group_mask,
            "event_med_route_ids": event_med_route_ids,
            "event_med_route_mask": event_med_route_mask,
            "event_med_form_ids": event_med_form_ids,
            "event_med_form_mask": event_med_form_mask,
            "event_med_freq_ids": event_med_freq_ids,
            "event_med_freq_mask": event_med_freq_mask,
            "event_med_unit_ids": event_med_unit_ids,
            "event_med_unit_mask": event_med_unit_mask,
            "event_med_marker_ids": event_med_marker_ids,
            "event_med_marker_mask": event_med_marker_mask,
            "event_med_dosage_values": event_med_dosage_values,
            "event_med_dosage_mask": event_med_dosage_mask,
            "event_med_rate_values": event_med_rate_values,
            "event_med_rate_mask": event_med_rate_mask,
            "event_med_duration_values": event_med_duration_values,
            "event_med_duration_mask": event_med_duration_mask,
        }
        if input_ids_global is not None:
            out["input_ids_global"] = input_ids_global
        if remap_stats is not None:
            out["id_remap_stats"] = remap_stats
        return out
