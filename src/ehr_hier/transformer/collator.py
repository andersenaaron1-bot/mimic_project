from __future__ import annotations

from typing import Dict, List

import torch

from src.ehr_hier.data.token_types import EventToken, TokenCategory


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
    ) -> None:
        self.max_windows = max_windows
        self.max_len = max_len_per_window
        self.pad_id = pad_id

    def __call__(self, batch_timelines: List[List[EventToken]]) -> Dict[str, torch.Tensor]:
        batch_ids: List[List[List[int]]] = []
        batch_times: List[List[List[float]]] = []
        batch_vals: List[List[List[float]]] = []
        batch_types: List[List[List[int]]] = []
        batch_attn: List[List[List[int]]] = []
        batch_valmask: List[List[List[int]]] = []

        for timeline in batch_timelines:
            special_tokens, events = self._split_special(timeline)
            windows = self._segment_into_windows(events)[: self.max_windows]

            subj_ids: List[List[int]] = []
            subj_times: List[List[float]] = []
            subj_vals: List[List[float]] = []
            subj_types: List[List[int]] = []
            subj_masks: List[List[int]] = []
            subj_valmask: List[List[int]] = []

            for window in windows:
                w_ids, w_times, w_vals, w_valmask, w_types = self._process_window(window, special_tokens)
                seq_len = len(w_ids)
                subj_ids.append(w_ids)
                subj_times.append(w_times)
                subj_vals.append(w_vals)
                subj_types.append(w_types)
                subj_masks.append([1] * seq_len)
                subj_valmask.append(w_valmask)

            batch_ids.append(subj_ids)
            batch_times.append(subj_times)
            batch_vals.append(subj_vals)
            batch_types.append(subj_types)
            batch_attn.append(subj_masks)
            batch_valmask.append(subj_valmask)

        return self._pad_batch(batch_ids, batch_times, batch_vals, batch_types, batch_attn, batch_valmask)

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
    ) -> tuple[List[int], List[float], List[float], List[int], List[int]]:
        if not window_tokens:
            return [], [], [], [], []
        w_start = float(window_tokens[0].t_from_start_hours)

        seq: List[EventToken] = list(special_tokens) + window_tokens

        ids: List[int] = []
        times: List[float] = []
        vals: List[float] = []
        val_mask: List[int] = []
        types: List[int] = []

        for tok in seq[: self.max_len]:
            ids.append(int(tok.value_id))
            types.append(int(tok.category_id))
            val = tok.num_attrs.get("numeric_value") if tok.num_attrs is not None else None
            has_val = val is not None
            vals.append(float(val) if has_val else 0.0)
            val_mask.append(1 if has_val else 0)
            if tok in special_tokens:
                times.append(0.0)
            else:
                rel_t = max(0.0, float(tok.t_from_start_hours) - w_start)
                times.append(rel_t)

        return ids, times, vals, val_mask, types

    def _pad_batch(
        self,
        batch_ids: List[List[List[int]]],
        batch_times: List[List[List[float]]],
        batch_vals: List[List[List[float]]],
        batch_types: List[List[List[int]]],
        batch_masks: List[List[List[int]]],
        batch_valmask: List[List[List[int]]],
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

        return {
            "input_ids": input_ids,
            "time_ids": time_ids,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "window_mask": window_mask,
        }
