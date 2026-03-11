from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Literal

import torch

from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.data.window_segmentation import (
    SegmentedChunk,
    SegmentedWindow,
    WindowSegmentationConfig,
    chunk_segmented_windows,
    segment_event_tokens,
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


class AETHierarchicalCollator:
    """
    Collates a batch of EventToken timelines into padded (B, W, L, ...) tensors.

    - Windows are segmented with bundle-based transition logic.
    - Explicit transition metadata takes precedence over legacy token.window_hook splits.
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

    def __call__(self, batch_timelines: List[List[EventToken]]) -> Dict[str, Any]:
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
        semantic_windows_total = 0
        semantic_windows_kept = 0
        semantic_windows_dropped = 0
        semantic_tokens_dropped = 0
        semantic_structural_tokens_dropped = 0
        chunks_dropped_by_cap = 0
        chunk_tokens_dropped_by_cap = 0
        chunk_structural_tokens_dropped_by_cap = 0
        subjects_with_overflow = 0

        for timeline in batch_timelines:
            special_tokens, events = self._split_special(timeline)
            semantic_windows_all = self._segment_windows(events)
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

                for chunk in window.chunks:
                    ids, times, vals, valmask, types, start_abs, start_offset = self._process_chunk(
                        chunk,
                        special_tokens,
                        w_type_id=window_type_ids[wi],
                        w_start_abs=window_start_abs_times[wi],
                        next_type_id=next_type_id,
                        next_start_abs=next_start_abs,
                    )
                    seq_len = len(ids)
                    chunk_ids.append(ids)
                    chunk_times.append(times)
                    chunk_vals.append(vals)
                    chunk_types.append(types)
                    chunk_masks.append([1] * seq_len)
                    chunk_valmask.append(valmask)
                    chunk_mask.append(1)
                    chunk_offsets.append(start_offset)
                    chunk_start_times.append(start_abs)
                    chunk_is_last.append(1 if chunk.is_last_chunk else 0)
                    chunk_token_counts.append(float(len(chunk.tokens)))
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
        )
        total_windows_base = max(1, int(semantic_windows_total))
        total_subjects_base = max(1, int(len(batch_timelines)))
        out["overflow_stats"] = {
            "subjects": int(len(batch_timelines)),
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
        return out

    def _split_special(self, timeline: List[EventToken]) -> tuple[List[EventToken], List[EventToken]]:
        specials: List[EventToken] = []
        events: List[EventToken] = []
        for tok in timeline:
            if tok.category_id == int(TokenCategory.SPECIAL):
                specials.append(tok)
            else:
                events.append(tok)
        return specials, events

    def _segment_into_windows(self, events: List[EventToken]) -> List[List[EventToken]]:
        return [window.tokens for window in self._segment_windows(events)]

    def _segment_windows(self, events: List[EventToken]) -> List[SegmentedWindow]:
        return segment_event_tokens(events, config=self.segmentation)

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
        w_type_id: int,
        w_start_abs: float,
        next_type_id: int | None,
        next_start_abs: float | None,
    ) -> tuple[List[int], List[float], List[float], List[int], List[int], float, float]:
        if not chunk.tokens:
            return [], [], [], [], [], float(w_start_abs), 0.0

        w_start_abs = float(w_start_abs)
        w_type_id = int(w_type_id)
        chunk_start_abs = float(chunk.start_time_hours)
        chunk_start_offset = max(0.0, chunk_start_abs - w_start_abs)

        prefix: List[EventToken] = list(special_tokens)
        suffix: List[EventToken] = []
        if self.window_markers.enabled:
            type_token_id = int(self.window_markers.type_token_offset) + int(w_type_id)
            end_token_id = self._window_end_token_id()
            end_mode = str(getattr(self.window_markers, "end_mode", "end_token"))

            prefix.append(
                EventToken(
                    value_id=type_token_id,
                    category_id=int(self.window_markers.marker_category),
                    t_from_start_hours=chunk_start_abs,
                    dt_from_prev_hours=0.0,
                    cat_attrs={"window_type_id": int(w_type_id)},
                    num_attrs={},
                )
            )
            if not chunk.is_last_chunk:
                suffix.append(
                    EventToken(
                        value_id=self._window_continue_token_id(),
                        category_id=int(self.window_markers.marker_category),
                        t_from_start_hours=float(chunk.tokens[-1].t_from_start_hours),
                        dt_from_prev_hours=0.0,
                        cat_attrs={"window_type_id": int(w_type_id), "chunk_continue": 1},
                        num_attrs={},
                    )
                )
            elif end_mode == "next_type" and next_type_id is not None:
                next_type_id_int = self._clamp_window_type_id(int(next_type_id))
                next_token_id = int(self.window_markers.type_token_offset) + int(next_type_id_int)
                suffix.append(
                    EventToken(
                        value_id=next_token_id,
                        category_id=int(self.window_markers.marker_category),
                        t_from_start_hours=float(next_start_abs) if next_start_abs is not None else float(chunk.tokens[-1].t_from_start_hours),
                        dt_from_prev_hours=0.0,
                        cat_attrs={"window_type_id": int(next_type_id_int)},
                        num_attrs={},
                    )
                )
            else:
                suffix.append(
                    EventToken(
                        value_id=end_token_id,
                        category_id=int(self.window_markers.marker_category),
                        t_from_start_hours=float(chunk.tokens[-1].t_from_start_hours),
                        dt_from_prev_hours=0.0,
                        cat_attrs={},
                        num_attrs={},
                    )
                )

        budget = max(0, int(self.max_len) - len(prefix) - len(suffix))
        seq: List[EventToken] = prefix + chunk.tokens[:budget] + suffix

        ids: List[int] = []
        times: List[float] = []
        vals: List[float] = []
        val_mask: List[int] = []
        types: List[int] = []

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
            if tok in special_tokens:
                times.append(0.0)
            else:
                rel_t = max(0.0, float(tok.t_from_start_hours) - chunk_start_abs)
                if not math.isfinite(rel_t):
                    rel_t = 0.0
                times.append(rel_t)

        return ids, times, vals, val_mask, types, float(chunk_start_abs), float(chunk_start_offset)

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
        ids, times, vals, valmask, types, _, _ = self._process_chunk(
            pseudo_chunk,
            special_tokens,
            w_type_id=w_type_id,
            w_start_abs=w_start_abs,
            next_type_id=next_type_id,
            next_start_abs=next_start_abs,
        )
        return ids, times, vals, valmask, types, int(w_type_id), float(w_start_abs)

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
    ) -> Dict[str, Any]:
        B = len(batch_ids)
        W = max((len(x) for x in batch_ids), default=0)
        C = max((len(chunks) for subj in batch_ids for chunks in subj), default=0)
        L = self.max_len

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
                    seq_len = min(len(ids), L)
                    chunk_mask[b, w, c] = 1

                    input_ids[b, w, c, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
                    time_ids[b, w, c, :seq_len] = torch.tensor(times[:seq_len], dtype=torch.float)
                    token_type_ids[b, w, c, :seq_len] = torch.tensor(types[:seq_len], dtype=torch.long)
                    attention_mask[b, w, c, :seq_len] = torch.tensor(mask[:seq_len], dtype=torch.long)

                    val_slice = torch.tensor(vals[:seq_len], dtype=torch.float).unsqueeze(-1)
                    numeric_values[b, w, c, :seq_len, :] = val_slice
                    numeric_mask[b, w, c, :seq_len] = torch.tensor(valmask[:seq_len], dtype=torch.long)

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
        remap_stats: Dict[str, Any] | None = None
        input_ids_global = input_ids.clone() if self.emit_global_input_ids else None
        if self.id_remapper is not None:
            input_ids_out, remap_stats = self.id_remapper.map_tensor(
                input_ids,
                valid_mask=attention_mask,
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
        }
        if input_ids_global is not None:
            out["input_ids_global"] = input_ids_global
        if remap_stats is not None:
            out["id_remap_stats"] = remap_stats
        return out
