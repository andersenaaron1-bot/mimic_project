from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from src.ehr_hier.data.token_types import EventToken, TokenCategory


@dataclass(frozen=True)
class WindowMarkerConfig:
    """
    Controls insertion of per-window marker tokens.

    Marker tokens are inserted *inside* each window sequence during collation:
      [special_tokens...] [WIN_TYPE] [window_tokens...] [WIN_END]

    Window types are integer class ids in [0, num_types-1]. The WIN_TYPE token id
    is computed as:
        win_type_token_id = type_token_offset + window_type_id
    """

    enabled: bool = True
    # Global token id offset where WIN_<TYPE> tokens start.
    type_token_offset: int = 10
    # Number of supported window types (including UNK=0 if you use it).
    num_types: int = 16
    # Global token id for the end-of-window marker. If None, defaults to
    # type_token_offset + num_types (i.e., directly after the WIN_<TYPE> block).
    end_token_id: int | None = None
    # Fallback type id when a window type cannot be inferred.
    unk_type_id: int = 0
    # Category assigned to marker tokens (affects pooling masks, not routing).
    marker_category: TokenCategory = TokenCategory.STRUCTURAL


class AETHierarchicalCollator:
    """
    Collates a batch of EventToken timelines into padded (B, W, L, ...) tensors.

    - Windows are segmented on token.window_hook boundaries.
    - Special (window-0) tokens are prefixed to every window.
    - Times are made relative to the start of each window.
    - token_type_ids mirror TokenCategory for now (head mapping handled later).
    """

    def __init__(
        self,
        *,
        max_windows: int = 64,
        max_len_per_window: int = 128,
        pad_id: int = 0,
        window_markers: WindowMarkerConfig | None = None,
    ) -> None:
        self.max_windows = max_windows
        self.max_len = max_len_per_window
        self.pad_id = pad_id
        self.window_markers = window_markers or WindowMarkerConfig()

    def __call__(self, batch_timelines: List[List[EventToken]]) -> Dict[str, torch.Tensor]:
        batch_ids: List[List[List[int]]] = []
        batch_times: List[List[List[float]]] = []
        batch_vals: List[List[List[float]]] = []
        batch_types: List[List[List[int]]] = []
        batch_attn: List[List[List[int]]] = []
        batch_valmask: List[List[List[int]]] = []
        batch_window_types: List[List[int]] = []
        batch_window_start_times: List[List[float]] = []

        for timeline in batch_timelines:
            special_tokens, events = self._split_special(timeline)
            windows = self._segment_into_windows(events)[: self.max_windows]

            subj_ids: List[List[int]] = []
            subj_times: List[List[float]] = []
            subj_vals: List[List[float]] = []
            subj_types: List[List[int]] = []
            subj_masks: List[List[int]] = []
            subj_valmask: List[List[int]] = []
            subj_window_types: List[int] = []
            subj_window_start_times: List[float] = []

            for window in windows:
                w_ids, w_times, w_vals, w_valmask, w_types, w_type_id, w_start_time = self._process_window(
                    window, special_tokens
                )
                seq_len = len(w_ids)
                subj_ids.append(w_ids)
                subj_times.append(w_times)
                subj_vals.append(w_vals)
                subj_types.append(w_types)
                subj_masks.append([1] * seq_len)
                subj_valmask.append(w_valmask)
                subj_window_types.append(w_type_id)
                subj_window_start_times.append(w_start_time)

            batch_ids.append(subj_ids)
            batch_times.append(subj_times)
            batch_vals.append(subj_vals)
            batch_types.append(subj_types)
            batch_attn.append(subj_masks)
            batch_valmask.append(subj_valmask)
            batch_window_types.append(subj_window_types)
            batch_window_start_times.append(subj_window_start_times)

        return self._pad_batch(
            batch_ids,
            batch_times,
            batch_vals,
            batch_types,
            batch_attn,
            batch_valmask,
            batch_window_types,
            batch_window_start_times,
        )

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
        windows: List[List[EventToken]] = []
        current: List[EventToken] = []
        for tok in events:
            if tok.window_hook is not None and current:
                windows.append(current)
                current = [tok]
            else:
                current.append(tok)
        if current:
            windows.append(current)
        return windows

    def _process_window(
        self, window_tokens: List[EventToken], special_tokens: List[EventToken]
    ) -> tuple[List[int], List[float], List[float], List[int], List[int], int, float]:
        if not window_tokens:
            return [], [], [], [], [], int(self.window_markers.unk_type_id), 0.0

        w_start_abs = float(window_tokens[0].t_from_start_hours)

        w_type_id = self._infer_window_type_id(window_tokens)

        # Build sequence with optional window markers while preserving the end marker.
        prefix: List[EventToken] = list(special_tokens)
        suffix: List[EventToken] = []
        if self.window_markers.enabled:
            type_token_id = int(self.window_markers.type_token_offset) + int(w_type_id)
            end_token_id = (
                int(self.window_markers.end_token_id)
                if self.window_markers.end_token_id is not None
                else int(self.window_markers.type_token_offset) + int(self.window_markers.num_types)
            )

            prefix.append(
                EventToken(
                    value_id=type_token_id,
                    category_id=int(self.window_markers.marker_category),
                    t_from_start_hours=w_start_abs,
                    dt_from_prev_hours=0.0,
                    cat_attrs={"window_type_id": int(w_type_id)},
                    num_attrs={},
                )
            )
            suffix.append(
                EventToken(
                    value_id=end_token_id,
                    category_id=int(self.window_markers.marker_category),
                    t_from_start_hours=float(window_tokens[-1].t_from_start_hours),
                    dt_from_prev_hours=0.0,
                    cat_attrs={},
                    num_attrs={},
                )
            )

        budget = max(0, int(self.max_len) - len(prefix) - len(suffix))
        seq: List[EventToken] = prefix + window_tokens[:budget] + suffix

        ids: List[int] = []
        times: List[float] = []
        vals: List[float] = []
        val_mask: List[int] = []
        types: List[int] = []

        for tok in seq:
            ids.append(int(tok.value_id))
            types.append(int(tok.category_id))
            val = tok.num_attrs.get("numeric_value") if tok.num_attrs is not None else None
            has_val = val is not None
            vals.append(float(val) if has_val else 0.0)
            val_mask.append(1 if has_val else 0)
            if tok in special_tokens:
                times.append(0.0)
            else:
                rel_t = max(0.0, float(tok.t_from_start_hours) - w_start_abs)
                times.append(rel_t)

        return ids, times, vals, val_mask, types, int(w_type_id), float(w_start_abs)

    def _infer_window_type_id(self, window_tokens: List[EventToken]) -> int:
        """
        Best-effort window type inference.

        Default behavior:
          - If the first token carries a structural_codebook-derived `struct_label_id`,
            treat it as a window type signal (shifted by +1 so 0 can remain UNK).
          - Otherwise fall back to UNK.
        """
        if not window_tokens:
            return int(self.window_markers.unk_type_id)

        first = window_tokens[0]
        # Explicit override if upstream encoders add it.
        if first.cat_attrs is not None and "window_type_id" in first.cat_attrs:
            try:
                w = int(first.cat_attrs["window_type_id"])
                return self._clamp_window_type_id(w)
            except Exception:
                return int(self.window_markers.unk_type_id)

        # Structural codebook label id (0-based) -> shift to reserve 0 for UNK.
        if first.cat_attrs is not None and "struct_label_id" in first.cat_attrs:
            try:
                w = int(first.cat_attrs["struct_label_id"]) + 1
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
        batch_ids: List[List[List[int]]],
        batch_times: List[List[List[float]]],
        batch_vals: List[List[List[float]]],
        batch_types: List[List[List[int]]],
        batch_masks: List[List[List[int]]],
        batch_valmask: List[List[List[int]]],
        batch_window_types: List[List[int]],
        batch_window_start_times: List[List[float]],
    ) -> Dict[str, torch.Tensor]:
        B = len(batch_ids)
        W = max((len(x) for x in batch_ids), default=0)
        L = self.max_len

        input_ids = torch.full((B, W, L), self.pad_id, dtype=torch.long)
        time_ids = torch.zeros((B, W, L), dtype=torch.float)
        numeric_values = torch.zeros((B, W, L, 1), dtype=torch.float)
        token_type_ids = torch.zeros((B, W, L), dtype=torch.long)
        attention_mask = torch.zeros((B, W, L), dtype=torch.long)
        window_mask = torch.zeros((B, W), dtype=torch.long)
        numeric_mask = torch.zeros((B, W, L), dtype=torch.long)
        window_type_ids = torch.zeros((B, W), dtype=torch.long)
        window_start_times = torch.zeros((B, W), dtype=torch.float)

        for b in range(B):
            for w in range(len(batch_ids[b])):
                ids = batch_ids[b][w]
                times = batch_times[b][w]
                vals = batch_vals[b][w]
                types = batch_types[b][w]
                mask = batch_masks[b][w]
                valmask = batch_valmask[b][w]
                seq_len = min(len(ids), L)
                window_mask[b, w] = 1

                input_ids[b, w, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
                time_ids[b, w, :seq_len] = torch.tensor(times[:seq_len], dtype=torch.float)
                token_type_ids[b, w, :seq_len] = torch.tensor(types[:seq_len], dtype=torch.long)
                attention_mask[b, w, :seq_len] = torch.tensor(mask[:seq_len], dtype=torch.long)

                val_slice = torch.tensor(vals[:seq_len], dtype=torch.float).unsqueeze(-1)
                numeric_values[b, w, :seq_len, :] = val_slice
                numeric_mask[b, w, :seq_len] = torch.tensor(valmask[:seq_len], dtype=torch.long)

                # Per-window metadata
                if b < len(batch_window_types) and w < len(batch_window_types[b]):
                    window_type_ids[b, w] = int(batch_window_types[b][w])
                if b < len(batch_window_start_times) and w < len(batch_window_start_times[b]):
                    window_start_times[b, w] = float(batch_window_start_times[b][w])

        return {
            "input_ids": input_ids,
            "time_ids": time_ids,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "window_mask": window_mask,
            "window_type_ids": window_type_ids,
            "window_start_times": window_start_times,
        }
