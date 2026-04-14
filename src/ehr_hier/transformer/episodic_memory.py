from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.transformer.memory_rules import EventMemoryGroup


class PatientMemoryBank(IntEnum):
    STATIC = 0
    PERSISTENT = 1
    EPISODIC = 2


@dataclass
class EpisodicMemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    event_ids: torch.Tensor
    group_ids: torch.Tensor
    age_bases: torch.Tensor
    rule_scores: torch.Tensor
    valid_mask: torch.Tensor

    def detach(self) -> "EpisodicMemoryState":
        return EpisodicMemoryState(
            keys=self.keys.detach(),
            values=self.values.detach(),
            scores=self.scores.detach(),
            event_ids=self.event_ids.detach(),
            group_ids=self.group_ids.detach(),
            age_bases=self.age_bases.detach(),
            rule_scores=self.rule_scores.detach(),
            valid_mask=self.valid_mask.detach(),
        )

    def to(self, device: torch.device | str) -> "EpisodicMemoryState":
        return EpisodicMemoryState(
            keys=self.keys.to(device),
            values=self.values.to(device),
            scores=self.scores.to(device),
            event_ids=self.event_ids.to(device),
            group_ids=self.group_ids.to(device),
            age_bases=self.age_bases.to(device),
            rule_scores=self.rule_scores.to(device),
            valid_mask=self.valid_mask.to(device),
        )

    @staticmethod
    def empty(
        *,
        batch_size: int,
        slots: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "EpisodicMemoryState":
        return EpisodicMemoryState(
            keys=torch.zeros((batch_size, slots, d_model), device=device, dtype=dtype),
            values=torch.zeros((batch_size, slots, d_model), device=device, dtype=dtype),
            scores=torch.zeros((batch_size, slots), device=device, dtype=dtype),
            event_ids=torch.full((batch_size, slots), -1, device=device, dtype=torch.long),
            group_ids=torch.zeros((batch_size, slots), device=device, dtype=torch.long),
            age_bases=torch.zeros((batch_size, slots), device=device, dtype=dtype),
            rule_scores=torch.zeros((batch_size, slots), device=device, dtype=dtype),
            valid_mask=torch.zeros((batch_size, slots), device=device, dtype=torch.bool),
        )

    @staticmethod
    def stack(
        states: Sequence["EpisodicMemoryState | None"],
        *,
        slots: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "EpisodicMemoryState":
        batch = EpisodicMemoryState.empty(
            batch_size=len(states),
            slots=slots,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        for idx, state in enumerate(states):
            if state is None:
                continue
            sample = state.to(device)
            if sample.keys.ndim == 3:
                sample = sample.select(0)
            valid = sample.valid_mask.to(dtype=torch.bool)
            count = min(int(valid.sum().item()), int(slots))
            if count <= 0:
                continue
            take = torch.nonzero(valid, as_tuple=False).squeeze(-1)[:count]
            batch.keys[idx, :count] = sample.keys[take].to(dtype=dtype)
            batch.values[idx, :count] = sample.values[take].to(dtype=dtype)
            batch.scores[idx, :count] = sample.scores[take].to(dtype=dtype)
            batch.event_ids[idx, :count] = sample.event_ids[take]
            batch.group_ids[idx, :count] = sample.group_ids[take]
            batch.age_bases[idx, :count] = sample.age_bases[take].to(dtype=dtype)
            batch.rule_scores[idx, :count] = sample.rule_scores[take].to(dtype=dtype)
            batch.valid_mask[idx, :count] = True
        return batch

    def select(self, batch_index: int) -> "EpisodicMemoryState":
        if self.keys.ndim == 2:
            return self
        idx = int(batch_index)
        return EpisodicMemoryState(
            keys=self.keys[idx],
            values=self.values[idx],
            scores=self.scores[idx],
            event_ids=self.event_ids[idx],
            group_ids=self.group_ids[idx],
            age_bases=self.age_bases[idx],
            rule_scores=self.rule_scores[idx],
            valid_mask=self.valid_mask[idx],
        )


@dataclass
class PatientMemoryState:
    static: EpisodicMemoryState
    persistent: EpisodicMemoryState
    episodic: EpisodicMemoryState

    def detach(self) -> "PatientMemoryState":
        return PatientMemoryState(
            static=self.static.detach(),
            persistent=self.persistent.detach(),
            episodic=self.episodic.detach(),
        )

    def to(self, device: torch.device | str) -> "PatientMemoryState":
        return PatientMemoryState(
            static=self.static.to(device),
            persistent=self.persistent.to(device),
            episodic=self.episodic.to(device),
        )

    def select(self, batch_index: int) -> "PatientMemoryState":
        return PatientMemoryState(
            static=self.static.select(batch_index),
            persistent=self.persistent.select(batch_index),
            episodic=self.episodic.select(batch_index),
        )

    @staticmethod
    def stack(
        states: Sequence["PatientMemoryState | EpisodicMemoryState | None"],
        *,
        static_slots: int,
        persistent_slots: int,
        episodic_slots: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PatientMemoryState":
        static_rows: list[EpisodicMemoryState | None] = []
        persistent_rows: list[EpisodicMemoryState | None] = []
        episodic_rows: list[EpisodicMemoryState | None] = []
        for state in states:
            if state is None:
                static_rows.append(None)
                persistent_rows.append(None)
                episodic_rows.append(None)
            elif isinstance(state, PatientMemoryState):
                static_rows.append(state.static)
                persistent_rows.append(state.persistent)
                episodic_rows.append(state.episodic)
            elif isinstance(state, EpisodicMemoryState):
                static_rows.append(None)
                persistent_rows.append(None)
                episodic_rows.append(state)
            else:
                raise TypeError(
                    f"Unsupported memory state type={type(state)!r}; expected PatientMemoryState|EpisodicMemoryState|None"
                )
        return PatientMemoryState(
            static=EpisodicMemoryState.stack(
                static_rows,
                slots=static_slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            ),
            persistent=EpisodicMemoryState.stack(
                persistent_rows,
                slots=persistent_slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            ),
            episodic=EpisodicMemoryState.stack(
                episodic_rows,
                slots=episodic_slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            ),
        )

    @staticmethod
    def empty(
        *,
        batch_size: int,
        static_slots: int,
        persistent_slots: int,
        episodic_slots: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PatientMemoryState":
        return PatientMemoryState(
            static=EpisodicMemoryState.empty(
                batch_size=batch_size,
                slots=static_slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            ),
            persistent=EpisodicMemoryState.empty(
                batch_size=batch_size,
                slots=persistent_slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            ),
            episodic=EpisodicMemoryState.empty(
                batch_size=batch_size,
                slots=episodic_slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            ),
        )


@dataclass
class EpisodicMemoryOutput:
    context: torch.Tensor
    retrieval_counts: torch.Tensor
    write_counts: torch.Tensor
    retrieved_event_ids: torch.Tensor
    written_event_ids: torch.Tensor
    next_state: PatientMemoryState | EpisodicMemoryState | None = None
    context_by_bank: dict[str, torch.Tensor] | None = None
    state_digest_by_bank: dict[str, torch.Tensor] | None = None
    retrieval_counts_by_bank: dict[str, torch.Tensor] | None = None
    write_counts_by_bank: dict[str, torch.Tensor] | None = None
    retrieved_event_ids_by_bank: dict[str, torch.Tensor] | None = None
    written_event_ids_by_bank: dict[str, torch.Tensor] | None = None


class AETEpisodicMemory(nn.Module):
    """
    Sparse exact-memory reservoir over prior semantic windows.

    A deeper exact-memory reservoir over prior semantic windows.

    Design goals:
    - exact event writes, not vague summary tokens
    - causal retrieval keyed by prior global context
    - learned salience augmented with explicit clinical and structural priors
    - diversity-aware selection so the bank does not collapse onto one family
    - clear separation from the compressive latent health state
    """

    def __init__(self, config) -> None:
        super().__init__()
        d_model = int(config.d_model)
        self.d_model = d_model
        self.special_type_id = int(getattr(config, "special_type_id", 0))
        self.memory_slots = max(0, int(getattr(config, "exact_memory_slots", 16)))
        self.static_slots = max(
            0, int(getattr(config, "exact_memory_static_slots", 0))
        )
        self.persistent_slots = max(
            0, int(getattr(config, "exact_memory_persistent_slots", self.memory_slots))
        )
        self.episodic_slots = max(
            0, int(getattr(config, "exact_memory_episodic_slots", self.memory_slots))
        )
        self.write_per_window = max(
            0, int(getattr(config, "exact_memory_write_per_window", 2))
        )
        self.retrieve_k = max(0, int(getattr(config, "exact_memory_retrieve_k", 4)))
        self.static_retrieve_k = max(
            0,
            int(
                getattr(
                    config,
                    "exact_memory_static_retrieve_k",
                    min(2, int(self.static_slots)),
                )
            ),
        )
        self.persistent_retrieve_k = max(
            0, int(getattr(config, "exact_memory_persistent_retrieve_k", self.retrieve_k))
        )
        self.episodic_retrieve_k = max(
            0, int(getattr(config, "exact_memory_episodic_retrieve_k", self.retrieve_k))
        )
        self.max_same_group = max(
            1, int(getattr(config, "exact_memory_max_same_group", 2))
        )
        self.age_decay = float(getattr(config, "exact_memory_age_decay", 0.05))
        self.rule_write_scale = float(
            getattr(config, "exact_memory_rule_write_scale", 1.0)
        )
        self.rule_retrieval_scale = float(
            getattr(config, "exact_memory_rule_retrieval_scale", 0.25)
        )
        self.learned_write_scale = float(
            getattr(config, "exact_memory_learned_write_scale", 1.0)
        )
        self.first_occurrence_bonus = float(
            getattr(config, "exact_memory_first_occurrence_bonus", 0.75)
        )
        self.chronic_bonus = float(getattr(config, "exact_memory_chronic_bonus", 1.5))
        self.static_feature_ids = tuple(
            self._normalize_static_feature_ids(
                getattr(config, "exact_memory_static_feature_ids", (1, 2))
            )
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.write_score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    @staticmethod
    def _normalize_static_feature_ids(raw: object) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.split(",")]
            return [int(part) for part in parts if part]
        if isinstance(raw, Iterable):
            out: list[int] = []
            for item in raw:
                try:
                    out.append(int(item))
                except Exception:
                    continue
            return out
        try:
            return [int(raw)]
        except Exception:
            return []

    @staticmethod
    def _bank_name(bank: PatientMemoryBank) -> str:
        return str(bank.name).lower()

    def _bank_slots(self, bank: PatientMemoryBank) -> int:
        if bank == PatientMemoryBank.STATIC:
            return int(self.static_slots)
        if bank == PatientMemoryBank.PERSISTENT:
            return int(self.persistent_slots)
        return int(self.episodic_slots)

    def _bank_retrieve_k(self, bank: PatientMemoryBank) -> int:
        if bank == PatientMemoryBank.STATIC:
            return int(self.static_retrieve_k)
        if bank == PatientMemoryBank.PERSISTENT:
            return int(self.persistent_retrieve_k)
        return int(self.episodic_retrieve_k)

    def _route_group_to_bank(self, group_id: int) -> PatientMemoryBank:
        group = int(group_id)
        if group in {
            int(EventMemoryGroup.STRUCTURAL),
            int(EventMemoryGroup.CHRONIC_DIAGNOSIS),
            int(EventMemoryGroup.PROCEDURE),
        }:
            return PatientMemoryBank.PERSISTENT
        return PatientMemoryBank.EPISODIC

    def _ensure_batched_bank_state(
        self,
        state: EpisodicMemoryState,
        *,
        batch_size: int,
        slots: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> EpisodicMemoryState:
        sample = state.to(device)
        if sample.keys.ndim == 2:
            sample = EpisodicMemoryState.stack(
                [sample] * batch_size,
                slots=slots,
                d_model=d_model,
                device=device,
                dtype=dtype,
            )
        if sample.keys.shape[:2] != (batch_size, int(slots)):
            raise ValueError(
                "bank state must align with batch and slot dimensions; "
                f"got {tuple(sample.keys.shape)} vs {(batch_size, int(slots), d_model)}"
            )
        return sample

    def _coerce_prev_state(
        self,
        prev_state: PatientMemoryState | EpisodicMemoryState | None,
        *,
        batch_size: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PatientMemoryState:
        empty = PatientMemoryState.empty(
            batch_size=batch_size,
            static_slots=int(self.static_slots),
            persistent_slots=int(self.persistent_slots),
            episodic_slots=int(self.episodic_slots),
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        if prev_state is None:
            return empty
        if isinstance(prev_state, PatientMemoryState):
            return PatientMemoryState(
                static=self._ensure_batched_bank_state(
                    prev_state.static,
                    batch_size=batch_size,
                    slots=int(self.static_slots),
                    d_model=d_model,
                    device=device,
                    dtype=dtype,
                ),
                persistent=self._ensure_batched_bank_state(
                    prev_state.persistent,
                    batch_size=batch_size,
                    slots=int(self.persistent_slots),
                    d_model=d_model,
                    device=device,
                    dtype=dtype,
                ),
                episodic=self._ensure_batched_bank_state(
                    prev_state.episodic,
                    batch_size=batch_size,
                    slots=int(self.episodic_slots),
                    d_model=d_model,
                    device=device,
                    dtype=dtype,
                ),
            )
        if isinstance(prev_state, EpisodicMemoryState):
            episodic = self._ensure_batched_bank_state(
                prev_state,
                batch_size=batch_size,
                slots=int(self.episodic_slots),
                d_model=d_model,
                device=device,
                dtype=dtype,
            )
            return PatientMemoryState(
                static=empty.static,
                persistent=empty.persistent,
                episodic=episodic,
            )
        raise TypeError(
            f"Unsupported prev_memory_state type={type(prev_state)!r}; expected PatientMemoryState|EpisodicMemoryState|None"
        )

    def _reset_parameters(self) -> None:
        nn.init.eye_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.eye_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.eye_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        for module in self.write_score_head:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def _base_valid_mask(
        self,
        *,
        event_type_ids: torch.Tensor,
        event_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return event_attention_mask.to(dtype=torch.bool) & (
            event_type_ids != int(self.special_type_id)
        )

    def _write_scores(
        self,
        *,
        event_states: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_memory_rule_scores: torch.Tensor | None,
        event_memory_first_flags: torch.Tensor | None,
        event_memory_chronic_flags: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = self._base_valid_mask(
            event_type_ids=event_type_ids,
            event_attention_mask=event_attention_mask,
        )
        learned = self.write_score_head(event_states).squeeze(-1)
        learned = self.learned_write_scale * learned
        scores = learned
        if event_memory_rule_scores is not None:
            scores = scores + (
                self.rule_write_scale * event_memory_rule_scores.to(dtype=scores.dtype)
            )
        if event_memory_first_flags is not None:
            scores = scores + self.first_occurrence_bonus * event_memory_first_flags.to(
                dtype=scores.dtype
            )
        if event_memory_chronic_flags is not None:
            scores = scores + self.chronic_bonus * event_memory_chronic_flags.to(
                dtype=scores.dtype
            )
        return torch.where(valid, scores, torch.full_like(scores, float("-inf")))

    def _select_diverse(
        self,
        *,
        scores: torch.Tensor,
        event_ids: torch.Tensor,
        group_ids: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")
        order = torch.argsort(scores, descending=True)
        chosen: list[int] = []
        seen_non_measurement_ids: set[int] = set()
        group_counts: dict[int, int] = {}
        for idx in order.tolist():
            if len(chosen) >= int(k):
                break
            if not math.isfinite(float(scores[idx].item())):
                break
            group = int(group_ids[idx].item())
            event_id = int(event_ids[idx].item())
            is_measurement = group == int(EventMemoryGroup.EXTREME_MEASUREMENT)
            if not is_measurement and event_id in seen_non_measurement_ids:
                continue
            if group_counts.get(group, 0) >= int(self.max_same_group):
                continue
            chosen.append(int(idx))
            if not is_measurement:
                seen_non_measurement_ids.add(event_id)
            group_counts[group] = group_counts.get(group, 0) + 1
        if len(chosen) < int(k):
            for idx in order.tolist():
                if len(chosen) >= int(k):
                    break
                if idx in chosen or not math.isfinite(float(scores[idx].item())):
                    continue
                chosen.append(int(idx))
        if not chosen:
            return torch.zeros((0,), device=scores.device, dtype=torch.long)
        return torch.tensor(chosen, device=scores.device, dtype=torch.long)

    @staticmethod
    def _reservoir_from_state(
        state: EpisodicMemoryState,
        *,
        batch_index: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = state.valid_mask[batch_index].to(dtype=torch.bool)
        keys = state.keys[batch_index, valid].to(dtype=dtype)
        values = state.values[batch_index, valid].to(dtype=dtype)
        scores = state.scores[batch_index, valid].to(dtype=dtype)
        event_ids = state.event_ids[batch_index, valid].to(dtype=torch.long)
        group_ids = state.group_ids[batch_index, valid].to(dtype=torch.long)
        age_bases = state.age_bases[batch_index, valid].to(dtype=dtype)
        rule_scores = state.rule_scores[batch_index, valid].to(dtype=dtype)
        return keys, values, scores, event_ids, group_ids, age_bases, rule_scores

    def _retrieve_from_reservoir(
        self,
        *,
        query_state: torch.Tensor,
        reservoir_keys: torch.Tensor,
        reservoir_values: torch.Tensor,
        reservoir_ids: torch.Tensor,
        reservoir_groups: torch.Tensor,
        reservoir_age_bases: torch.Tensor,
        reservoir_rule_scores: torch.Tensor,
        retrieve_k: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if reservoir_values.shape[0] == 0 or int(retrieve_k) <= 0:
            return torch.zeros_like(query_state), torch.zeros((0,), device=query_state.device, dtype=torch.long), 0
        query = F.normalize(self.query_proj(query_state), dim=-1, eps=1e-6)
        keys = F.normalize(reservoir_keys, dim=-1, eps=1e-6)
        sim = torch.matmul(keys, query) * scale
        sim = sim - (self.age_decay * reservoir_age_bases.clamp(min=0.0).to(dtype=sim.dtype))
        sim = sim + (self.rule_retrieval_scale * reservoir_rule_scores.to(dtype=sim.dtype))
        top_idx = self._select_diverse(
            scores=sim,
            event_ids=reservoir_ids,
            group_ids=reservoir_groups,
            k=min(int(retrieve_k), int(sim.shape[0])),
        )
        topk = int(top_idx.numel())
        if topk <= 0:
            return torch.zeros_like(query_state), torch.zeros((0,), device=query_state.device, dtype=torch.long), 0
        top_scores = sim[top_idx]
        attn = torch.softmax(top_scores, dim=0)
        retrieved = (attn.unsqueeze(-1) * reservoir_values[top_idx]).sum(dim=0)
        return retrieved, reservoir_ids[top_idx], topk

    def _prune_reservoir(
        self,
        *,
        reservoir_keys: torch.Tensor,
        reservoir_values: torch.Tensor,
        reservoir_scores: torch.Tensor,
        reservoir_ids: torch.Tensor,
        reservoir_groups: torch.Tensor,
        reservoir_age_bases: torch.Tensor,
        reservoir_rule_scores: torch.Tensor,
        slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if reservoir_keys.shape[0] <= int(slots):
            return (
                reservoir_keys,
                reservoir_values,
                reservoir_scores,
                reservoir_ids,
                reservoir_groups,
                reservoir_age_bases,
                reservoir_rule_scores,
            )
        keep = min(int(slots), int(reservoir_keys.shape[0]))
        persist_scores = reservoir_scores + (self.rule_retrieval_scale * reservoir_rule_scores)
        keep_idx = self._select_diverse(
            scores=persist_scores,
            event_ids=reservoir_ids,
            group_ids=reservoir_groups,
            k=keep,
        )
        return (
            reservoir_keys[keep_idx],
            reservoir_values[keep_idx],
            reservoir_scores[keep_idx],
            reservoir_ids[keep_idx],
            reservoir_groups[keep_idx],
            reservoir_age_bases[keep_idx],
            reservoir_rule_scores[keep_idx],
        )

    @staticmethod
    def _digest_reservoir(
        *,
        reservoir_values: torch.Tensor,
        reservoir_scores: torch.Tensor,
        reservoir_rule_scores: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if reservoir_values.ndim != 2 or reservoir_values.shape[0] == 0:
            return torch.zeros_like(reference)
        scores = reservoir_scores.to(dtype=reference.dtype)
        scores = scores + reservoir_rule_scores.to(dtype=reference.dtype)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * reservoir_values.to(dtype=reference.dtype)).sum(dim=0)

    def _initialize_static_reservoir(
        self,
        *,
        static_reservoir: list[torch.Tensor],
        batch_index: int,
        event_seed_states: torch.Tensor | None,
        event_input_ids: torch.Tensor,
        event_attention_mask: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_demographic_feature_ids: torch.Tensor | None,
        window_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        if (
            self.static_slots <= 0
            or event_seed_states is None
            or event_demographic_feature_ids is None
            or not self.static_feature_ids
        ):
            return static_reservoir

        flat_window_mask = window_mask[batch_index].to(dtype=torch.bool).view(-1, 1, 1)
        candidate_mask = (
            flat_window_mask
            & event_attention_mask[batch_index].to(dtype=torch.bool)
            & (event_type_ids[batch_index] == int(self.special_type_id))
            & (event_demographic_feature_ids[batch_index] > 0)
        )
        flat_mask = candidate_mask.reshape(-1)
        if not bool(flat_mask.any().item()):
            return static_reservoir

        allowed_features = set(int(fid) for fid in self.static_feature_ids)
        existing_event_ids = set(int(x) for x in static_reservoir[3].tolist())
        flat_ids = event_input_ids[batch_index].reshape(-1)
        flat_features = event_demographic_feature_ids[batch_index].reshape(-1)
        flat_seed_states = event_seed_states[batch_index].reshape(-1, event_seed_states.shape[-1])

        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        new_scores: list[torch.Tensor] = []
        new_event_ids: list[int] = []
        new_group_ids: list[int] = []
        new_age_bases: list[torch.Tensor] = []
        new_rule_scores: list[torch.Tensor] = []

        seen_features: set[int] = set()
        for idx in torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).tolist():
            feature_id = int(flat_features[idx].item())
            if feature_id not in allowed_features or feature_id in seen_features:
                continue
            event_id = int(flat_ids[idx].item())
            if event_id in existing_event_ids:
                seen_features.add(feature_id)
                continue
            seed_state = flat_seed_states[idx : idx + 1]
            new_keys.append(self.key_proj(seed_state).squeeze(0))
            new_values.append(self.value_proj(seed_state).squeeze(0))
            new_scores.append(seed_state.new_tensor(1.0))
            new_event_ids.append(event_id)
            new_group_ids.append(10_000 + feature_id)
            new_age_bases.append(seed_state.new_tensor(0.0))
            new_rule_scores.append(seed_state.new_tensor(1.0))
            existing_event_ids.add(event_id)
            seen_features.add(feature_id)
            if len(existing_event_ids) >= int(self.static_slots):
                break

        if not new_keys:
            return static_reservoir

        static_reservoir[0] = torch.cat([static_reservoir[0], torch.stack(new_keys, dim=0)], dim=0)
        static_reservoir[1] = torch.cat([static_reservoir[1], torch.stack(new_values, dim=0)], dim=0)
        static_reservoir[2] = torch.cat([static_reservoir[2], torch.stack(new_scores, dim=0)], dim=0)
        static_reservoir[3] = torch.cat(
            [
                static_reservoir[3],
                torch.tensor(new_event_ids, device=static_reservoir[3].device, dtype=torch.long),
            ],
            dim=0,
        )
        static_reservoir[4] = torch.cat(
            [
                static_reservoir[4],
                torch.tensor(new_group_ids, device=static_reservoir[4].device, dtype=torch.long),
            ],
            dim=0,
        )
        static_reservoir[5] = torch.cat([static_reservoir[5], torch.stack(new_age_bases, dim=0)], dim=0)
        static_reservoir[6] = torch.cat([static_reservoir[6], torch.stack(new_rule_scores, dim=0)], dim=0)

        if static_reservoir[0].shape[0] > int(self.static_slots):
            static_reservoir = list(
                self._prune_reservoir(
                    reservoir_keys=static_reservoir[0],
                    reservoir_values=static_reservoir[1],
                    reservoir_scores=static_reservoir[2],
                    reservoir_ids=static_reservoir[3],
                    reservoir_groups=static_reservoir[4],
                    reservoir_age_bases=static_reservoir[5],
                    reservoir_rule_scores=static_reservoir[6],
                    slots=int(self.static_slots),
                )
            )
        return static_reservoir

    def forward(
        self,
        *,
        event_states: torch.Tensor,
        event_seed_states: torch.Tensor | None = None,
        event_input_ids: torch.Tensor,
        event_time_ids: torch.Tensor | None,
        event_attention_mask: torch.Tensor,
        event_type_ids: torch.Tensor,
        event_payload_ids: torch.Tensor,
        event_demographic_feature_ids: torch.Tensor | None = None,
        query_states: torch.Tensor,
        window_mask: torch.Tensor,
        window_start_times: torch.Tensor | None = None,
        semantic_duration_hours: torch.Tensor | None = None,
        prev_memory_state: PatientMemoryState | EpisodicMemoryState | None = None,
        event_memory_rule_scores: torch.Tensor | None = None,
        event_memory_group_ids: torch.Tensor | None = None,
        event_memory_first_flags: torch.Tensor | None = None,
        event_memory_chronic_flags: torch.Tensor | None = None,
    ) -> EpisodicMemoryOutput:
        if event_states.ndim != 5:
            raise ValueError(
                f"event_states must be (B,W,C,E,D), got shape {tuple(event_states.shape)}"
            )
        if event_input_ids.shape != event_attention_mask.shape:
            raise ValueError(
                "event_input_ids and event_attention_mask must match; "
                f"got {tuple(event_input_ids.shape)} vs {tuple(event_attention_mask.shape)}"
            )
        if event_seed_states is not None and event_seed_states.shape != event_states.shape:
            raise ValueError(
                "event_seed_states must match event_states; "
                f"got {tuple(event_seed_states.shape)} vs {tuple(event_states.shape)}"
            )
        if (
            event_type_ids.shape != event_input_ids.shape
            or event_payload_ids.shape != event_input_ids.shape
        ):
            raise ValueError("event type/payload tensors must match event_input_ids")
        if (
            event_demographic_feature_ids is not None
            and event_demographic_feature_ids.shape != event_input_ids.shape
        ):
            raise ValueError("event_demographic_feature_ids must match event_input_ids")
        if event_time_ids is not None and event_time_ids.shape != event_input_ids.shape:
            raise ValueError("event_time_ids must match event_input_ids")
        if query_states.ndim != 3 or query_states.shape[:2] != event_states.shape[:2]:
            raise ValueError(
                "query_states must be (B,W,D) aligned with event_states; "
                f"got {tuple(query_states.shape)} vs {tuple(event_states.shape[:2])}"
            )
        if window_mask.shape != event_states.shape[:2]:
            raise ValueError(
                "window_mask must be (B,W) aligned with event_states; "
                f"got {tuple(window_mask.shape)} vs {tuple(event_states.shape[:2])}"
            )
        B, W, C, E, D = event_states.shape
        if window_start_times is None:
            window_start_times = event_states.new_zeros((B, W))
        elif window_start_times.shape != event_states.shape[:2]:
            raise ValueError(
                "window_start_times must be (B,W) aligned with event_states; "
                f"got {tuple(window_start_times.shape)} vs {tuple(event_states.shape[:2])}"
            )
        if semantic_duration_hours is None:
            semantic_duration_hours = event_states.new_zeros((B, W))
        elif semantic_duration_hours.shape != event_states.shape[:2]:
            raise ValueError(
                "semantic_duration_hours must be (B,W) aligned with event_states; "
                f"got {tuple(semantic_duration_hours.shape)} vs {tuple(event_states.shape[:2])}"
            )
        device = event_states.device
        context = event_states.new_zeros((B, W, D))
        retrieval_counts = torch.zeros((B, W), device=device, dtype=torch.long)
        write_counts = torch.zeros((B, W), device=device, dtype=torch.long)
        retrieval_counts_by_bank = {
            self._bank_name(PatientMemoryBank.STATIC): torch.zeros((B, W), device=device, dtype=torch.long),
            self._bank_name(PatientMemoryBank.PERSISTENT): torch.zeros((B, W), device=device, dtype=torch.long),
            self._bank_name(PatientMemoryBank.EPISODIC): torch.zeros((B, W), device=device, dtype=torch.long),
        }
        context_by_bank = {
            self._bank_name(PatientMemoryBank.STATIC): event_states.new_zeros((B, W, D)),
            self._bank_name(PatientMemoryBank.PERSISTENT): event_states.new_zeros((B, W, D)),
            self._bank_name(PatientMemoryBank.EPISODIC): event_states.new_zeros((B, W, D)),
        }
        state_digest_by_bank = {
            self._bank_name(PatientMemoryBank.STATIC): event_states.new_zeros((B, W, D)),
            self._bank_name(PatientMemoryBank.PERSISTENT): event_states.new_zeros((B, W, D)),
            self._bank_name(PatientMemoryBank.EPISODIC): event_states.new_zeros((B, W, D)),
        }
        write_counts_by_bank = {
            self._bank_name(PatientMemoryBank.STATIC): torch.zeros((B, W), device=device, dtype=torch.long),
            self._bank_name(PatientMemoryBank.PERSISTENT): torch.zeros((B, W), device=device, dtype=torch.long),
            self._bank_name(PatientMemoryBank.EPISODIC): torch.zeros((B, W), device=device, dtype=torch.long),
        }
        total_retrieve_slots = max(
            1,
            int(self.static_retrieve_k) + int(self.persistent_retrieve_k) + int(self.episodic_retrieve_k),
        )
        retrieved_event_ids = torch.full(
            (B, W, total_retrieve_slots),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )
        written_event_ids = torch.full(
            (B, W, max(1, self.write_per_window)),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )
        retrieved_event_ids_by_bank = {
            self._bank_name(PatientMemoryBank.STATIC): torch.full(
                (B, W, max(1, int(self.static_retrieve_k))),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            self._bank_name(PatientMemoryBank.PERSISTENT): torch.full(
                (B, W, max(1, int(self.persistent_retrieve_k))),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            self._bank_name(PatientMemoryBank.EPISODIC): torch.full(
                (B, W, max(1, int(self.episodic_retrieve_k))),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
        }
        written_event_ids_by_bank = {
            self._bank_name(PatientMemoryBank.STATIC): torch.full(
                (B, W, 1),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            self._bank_name(PatientMemoryBank.PERSISTENT): torch.full(
                (B, W, max(1, int(self.write_per_window))),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            self._bank_name(PatientMemoryBank.EPISODIC): torch.full(
                (B, W, max(1, int(self.write_per_window))),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
        }
        next_state = PatientMemoryState.empty(
            batch_size=B,
            static_slots=int(self.static_slots),
            persistent_slots=int(self.persistent_slots),
            episodic_slots=int(self.episodic_slots),
            d_model=D,
            device=device,
            dtype=event_states.dtype,
        )

        if (
            W == 0
            or (
                self.static_slots <= 0
                and self.persistent_slots <= 0
                and self.episodic_slots <= 0
            )
            or total_retrieve_slots <= 0
        ):
            return EpisodicMemoryOutput(
                context=context,
                retrieval_counts=retrieval_counts,
                write_counts=write_counts,
                retrieved_event_ids=retrieved_event_ids,
                written_event_ids=written_event_ids,
                next_state=next_state,
                context_by_bank=context_by_bank,
                state_digest_by_bank=state_digest_by_bank,
                retrieval_counts_by_bank=retrieval_counts_by_bank,
                write_counts_by_bank=write_counts_by_bank,
                retrieved_event_ids_by_bank=retrieved_event_ids_by_bank,
                written_event_ids_by_bank=written_event_ids_by_bank,
            )

        prev_memory_state = self._coerce_prev_state(
            prev_memory_state,
            batch_size=B,
            d_model=D,
            device=device,
            dtype=event_states.dtype,
        )

        if event_memory_group_ids is None:
            event_memory_group_ids = torch.where(
                event_type_ids == int(TokenCategory.STRUCTURAL),
                torch.full_like(event_type_ids, int(EventMemoryGroup.STRUCTURAL)),
                torch.where(
                    event_payload_ids
                    == int(EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]),
                    torch.full_like(
                        event_type_ids, int(EventMemoryGroup.EXTREME_MEASUREMENT)
                    ),
                    torch.full_like(event_type_ids, int(EventMemoryGroup.SYMBOLIC)),
                ),
            )

        write_scores = self._write_scores(
            event_states=event_states,
            event_type_ids=event_type_ids,
            event_attention_mask=event_attention_mask,
            event_memory_rule_scores=event_memory_rule_scores,
            event_memory_first_flags=event_memory_first_flags,
            event_memory_chronic_flags=event_memory_chronic_flags,
        )

        event_states_flat = event_states.view(B, W, C * E, D)
        event_keys_flat = self.key_proj(event_states).view(B, W, C * E, D)
        event_values_flat = self.value_proj(event_states).view(B, W, C * E, D)
        event_ids_flat = event_input_ids.view(B, W, C * E)
        write_scores_flat = write_scores.view(B, W, C * E)
        group_ids_flat = event_memory_group_ids.view(B, W, C * E)
        valid_flat = torch.isfinite(write_scores_flat)
        scale = 1.0 / math.sqrt(float(max(1, D)))

        for b in range(B):
            static_reservoir = list(
                self._reservoir_from_state(
                    prev_memory_state.static,
                    batch_index=b,
                    dtype=event_states.dtype,
                )
            )
            static_reservoir = self._initialize_static_reservoir(
                static_reservoir=static_reservoir,
                batch_index=b,
                event_seed_states=event_seed_states,
                event_input_ids=event_input_ids,
                event_attention_mask=event_attention_mask,
                event_type_ids=event_type_ids,
                event_demographic_feature_ids=event_demographic_feature_ids,
                window_mask=window_mask,
            )
            persistent_reservoir = list(
                self._reservoir_from_state(
                    prev_memory_state.persistent,
                    batch_index=b,
                    dtype=event_states.dtype,
                )
            )
            episodic_reservoir = list(
                self._reservoir_from_state(
                    prev_memory_state.episodic,
                    batch_index=b,
                    dtype=event_states.dtype,
                )
            )

            for w in range(W):
                if not bool(window_mask[b, w].item()):
                    continue

                current_query_h = float(window_start_times[b, w].item())
                retrieved_contexts: list[torch.Tensor] = []
                retrieved_ids_agg: list[torch.Tensor] = []
                for bank, reservoir in (
                    (PatientMemoryBank.STATIC, static_reservoir),
                    (PatientMemoryBank.PERSISTENT, persistent_reservoir),
                    (PatientMemoryBank.EPISODIC, episodic_reservoir),
                ):
                    retrieved_vec, bank_ids, bank_count = self._retrieve_from_reservoir(
                        query_state=query_states[b, w],
                        reservoir_keys=reservoir[0],
                        reservoir_values=reservoir[1],
                        reservoir_ids=reservoir[3],
                        reservoir_groups=reservoir[4],
                        reservoir_age_bases=(
                            torch.zeros_like(reservoir[5])
                            if bank == PatientMemoryBank.STATIC
                            else (current_query_h - reservoir[5]).clamp(min=0.0)
                        ),
                        reservoir_rule_scores=reservoir[6],
                        retrieve_k=self._bank_retrieve_k(bank),
                        scale=scale,
                    )
                    bank_name = self._bank_name(bank)
                    retrieval_counts_by_bank[bank_name][b, w] = int(bank_count)
                    context_by_bank[bank_name][b, w] = retrieved_vec
                    if bank_count > 0:
                        retrieved_contexts.append(retrieved_vec)
                        retrieved_ids_agg.append(bank_ids)
                        retrieved_event_ids_by_bank[bank_name][b, w, :bank_count] = bank_ids

                if retrieved_contexts:
                    combined_retrieved = torch.stack(retrieved_contexts, dim=0).sum(dim=0)
                    fused = self.fusion_proj(
                        torch.cat([query_states[b, w], combined_retrieved], dim=-1)
                    )
                    context[b, w] = self.fusion_norm(query_states[b, w] + fused)
                    flat_ids = torch.cat(retrieved_ids_agg, dim=0)
                    topk = min(int(flat_ids.numel()), int(retrieved_event_ids.shape[-1]))
                    retrieval_counts[b, w] = int(topk)
                    retrieved_event_ids[b, w, :topk] = flat_ids[:topk]

                valid_idx = torch.nonzero(valid_flat[b, w], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                if self.write_per_window <= 0:
                    continue

                candidate_scores = write_scores_flat[b, w, valid_idx]
                candidate_ids = event_ids_flat[b, w, valid_idx]
                candidate_groups = group_ids_flat[b, w, valid_idx]
                top_idx_local = self._select_diverse(
                    scores=candidate_scores,
                    event_ids=candidate_ids,
                    group_ids=candidate_groups,
                    k=min(int(self.write_per_window), int(candidate_scores.numel())),
                )
                top_write = int(top_idx_local.numel())
                top_idx = valid_idx[top_idx_local]
                top_scores = write_scores_flat[b, w, top_idx]
                new_keys = event_keys_flat[b, w, top_idx]
                new_values = event_values_flat[b, w, top_idx]
                new_ids = event_ids_flat[b, w, top_idx]
                new_groups = group_ids_flat[b, w, top_idx]
                new_rule_scores = (
                    event_memory_rule_scores.view(B, W, C * E)[b, w, top_idx]
                    if event_memory_rule_scores is not None
                    else top_scores.new_zeros((top_write,))
                )

                write_counts[b, w] = int(top_write)
                written_event_ids[b, w, :top_write] = new_ids
                persistent_write_pos = 0
                episodic_write_pos = 0
                for write_idx in range(top_write):
                    bank = self._route_group_to_bank(int(new_groups[write_idx].item()))
                    if bank == PatientMemoryBank.PERSISTENT and self.persistent_slots <= 0:
                        bank = PatientMemoryBank.EPISODIC
                    if bank == PatientMemoryBank.EPISODIC and self.episodic_slots <= 0:
                        bank = PatientMemoryBank.PERSISTENT
                    if bank == PatientMemoryBank.STATIC or self._bank_slots(bank) <= 0:
                        continue

                    if bank == PatientMemoryBank.PERSISTENT:
                        target = persistent_reservoir
                        write_pos = persistent_write_pos
                        persistent_write_pos += 1
                    else:
                        target = episodic_reservoir
                        write_pos = episodic_write_pos
                        episodic_write_pos += 1

                    target[0] = torch.cat([target[0], new_keys[write_idx : write_idx + 1]], dim=0)
                    target[1] = torch.cat([target[1], new_values[write_idx : write_idx + 1]], dim=0)
                    target[2] = torch.cat([target[2], top_scores[write_idx : write_idx + 1]], dim=0)
                    target[3] = torch.cat([target[3], new_ids[write_idx : write_idx + 1]], dim=0)
                    target[4] = torch.cat([target[4], new_groups[write_idx : write_idx + 1]], dim=0)
                    current_window_end_h = float(window_start_times[b, w].item()) + float(
                        semantic_duration_hours[b, w].item()
                    )
                    target[5] = torch.cat(
                        [target[5], top_scores.new_full((1,), fill_value=current_window_end_h)],
                        dim=0,
                    )
                    target[6] = torch.cat(
                        [target[6], new_rule_scores[write_idx : write_idx + 1].to(dtype=top_scores.dtype)],
                        dim=0,
                    )

                    bank_name = self._bank_name(bank)
                    write_counts_by_bank[bank_name][b, w] += 1
                    written_event_ids_by_bank[bank_name][b, w, write_pos] = new_ids[write_idx]

                persistent_reservoir = list(
                    self._prune_reservoir(
                        reservoir_keys=persistent_reservoir[0],
                        reservoir_values=persistent_reservoir[1],
                        reservoir_scores=persistent_reservoir[2],
                        reservoir_ids=persistent_reservoir[3],
                        reservoir_groups=persistent_reservoir[4],
                        reservoir_age_bases=persistent_reservoir[5],
                        reservoir_rule_scores=persistent_reservoir[6],
                        slots=int(self.persistent_slots),
                    )
                )
                episodic_reservoir = list(
                    self._prune_reservoir(
                        reservoir_keys=episodic_reservoir[0],
                        reservoir_values=episodic_reservoir[1],
                        reservoir_scores=episodic_reservoir[2],
                        reservoir_ids=episodic_reservoir[3],
                        reservoir_groups=episodic_reservoir[4],
                        reservoir_age_bases=episodic_reservoir[5],
                        reservoir_rule_scores=episodic_reservoir[6],
                        slots=int(self.episodic_slots),
                    )
                )

                for bank, reservoir in (
                    (PatientMemoryBank.STATIC, static_reservoir),
                    (PatientMemoryBank.PERSISTENT, persistent_reservoir),
                    (PatientMemoryBank.EPISODIC, episodic_reservoir),
                ):
                    bank_name = self._bank_name(bank)
                    state_digest_by_bank[bank_name][b, w] = self._digest_reservoir(
                        reservoir_values=reservoir[1],
                        reservoir_scores=reservoir[2],
                        reservoir_rule_scores=reservoir[6],
                        reference=query_states[b, w],
                    )

            valid_windows = int(window_mask[b].to(dtype=torch.long).sum().item())
            for bank, reservoir, bank_state in (
                (PatientMemoryBank.STATIC, static_reservoir, next_state.static),
                (PatientMemoryBank.PERSISTENT, persistent_reservoir, next_state.persistent),
                (PatientMemoryBank.EPISODIC, episodic_reservoir, next_state.episodic),
            ):
                slots = self._bank_slots(bank)
                if slots <= 0 or reservoir[0].shape[0] <= 0:
                    continue
                final_count = min(int(reservoir[0].shape[0]), int(slots))
                keep_idx = self._select_diverse(
                    scores=reservoir[2] + (self.rule_retrieval_scale * reservoir[6]),
                    event_ids=reservoir[3],
                    group_ids=reservoir[4],
                    k=final_count,
                )
                keep_count = int(keep_idx.numel())
                bank_state.keys[b, :keep_count] = reservoir[0][keep_idx]
                bank_state.values[b, :keep_count] = reservoir[1][keep_idx]
                bank_state.scores[b, :keep_count] = reservoir[2][keep_idx]
                bank_state.event_ids[b, :keep_count] = reservoir[3][keep_idx]
                bank_state.group_ids[b, :keep_count] = reservoir[4][keep_idx]
                bank_state.age_bases[b, :keep_count] = reservoir[5][keep_idx]
                bank_state.rule_scores[b, :keep_count] = reservoir[6][keep_idx]
                bank_state.valid_mask[b, :keep_count] = True

        return EpisodicMemoryOutput(
            context=context,
            retrieval_counts=retrieval_counts,
            write_counts=write_counts,
            retrieved_event_ids=retrieved_event_ids,
            written_event_ids=written_event_ids,
            next_state=next_state,
            context_by_bank=context_by_bank,
            state_digest_by_bank=state_digest_by_bank,
            retrieval_counts_by_bank=retrieval_counts_by_bank,
            write_counts_by_bank=write_counts_by_bank,
            retrieved_event_ids_by_bank=retrieved_event_ids_by_bank,
            written_event_ids_by_bank=written_event_ids_by_bank,
        )
