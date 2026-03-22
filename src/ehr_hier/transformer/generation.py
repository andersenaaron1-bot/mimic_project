from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Sequence

import torch

from src.ehr_hier.data.token_types import TokenCategory


class GenerationState(str, Enum):
    INSIDE_CHUNK = "inside_chunk"
    CHUNK_END = "chunk_end"
    WINDOW_END = "window_end"


@dataclass(frozen=True)
class WindowGenerationGrammar:
    """
    Lightweight state-machine grammar for constrained autoregressive rollout.

    Token classes:
    - WIN_<TYPE> markers: [type_start, type_end_excl)
    - WIN_END marker: end_id
    - WIN_CONTINUE marker: continue_id
    """

    type_start: int
    type_end_excl: int
    end_id: int
    continue_id: int

    @classmethod
    def from_vocab_config(cls, vocab_config: dict) -> "WindowGenerationGrammar":
        offsets = vocab_config.get("offsets", {})
        offsets = offsets if isinstance(offsets, dict) else {}
        special_offset = int(offsets.get("SPECIAL", 0))
        markers = vocab_config.get("window_markers", {})
        markers = markers if isinstance(markers, dict) else {}

        type_rel = int(markers.get("type_token_offset", 0))
        num_types = int(markers.get("num_types", 0))
        end_rel = int(markers.get("end_token_id", type_rel + num_types))
        continue_rel = int(markers.get("continue_token_id", end_rel + 1))
        return cls(
            type_start=special_offset + type_rel,
            type_end_excl=special_offset + type_rel + max(0, num_types),
            end_id=special_offset + end_rel,
            continue_id=special_offset + continue_rel,
        )

    @property
    def type_token_ids(self) -> tuple[int, ...]:
        return tuple(range(int(self.type_start), int(self.type_end_excl)))

    def is_type_token(self, token_id: int) -> bool:
        tid = int(token_id)
        return int(self.type_start) <= tid < int(self.type_end_excl)

    def is_marker_token(self, token_id: int) -> bool:
        tid = int(token_id)
        return (
            self.is_type_token(tid)
            or tid == int(self.end_id)
            or tid == int(self.continue_id)
        )

    def legal_token_mask(
        self,
        *,
        vocab_size: int,
        state: GenerationState | str,
        boundary_candidate: bool = False,
        allow_window_open: bool = True,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Returns a bool mask over token ids [0, vocab_size).
        """
        if int(vocab_size) <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

        st = state if isinstance(state, GenerationState) else GenerationState(str(state))
        mask = torch.zeros((int(vocab_size),), dtype=torch.bool, device=device)

        if st == GenerationState.INSIDE_CHUNK:
            mask[:] = True
            # Marker tokens are reserved for transition control steps.
            if 0 <= int(self.end_id) < int(vocab_size):
                mask[int(self.end_id)] = False
            if 0 <= int(self.continue_id) < int(vocab_size):
                mask[int(self.continue_id)] = False
            if int(self.type_end_excl) > int(self.type_start):
                lo = max(0, int(self.type_start))
                hi = min(int(vocab_size), int(self.type_end_excl))
                if hi > lo:
                    mask[lo:hi] = False
            return mask

        if st == GenerationState.CHUNK_END:
            if 0 <= int(self.continue_id) < int(vocab_size):
                mask[int(self.continue_id)] = True
            if boundary_candidate:
                if 0 <= int(self.end_id) < int(vocab_size):
                    mask[int(self.end_id)] = True
                if int(self.type_end_excl) > int(self.type_start):
                    lo = max(0, int(self.type_start))
                    hi = min(int(vocab_size), int(self.type_end_excl))
                    if hi > lo:
                        mask[lo:hi] = True
            return mask

        # WINDOW_END state: caller must choose next semantic-window type marker.
        if st == GenerationState.WINDOW_END:
            if allow_window_open and int(self.type_end_excl) > int(self.type_start):
                lo = max(0, int(self.type_start))
                hi = min(int(vocab_size), int(self.type_end_excl))
                if hi > lo:
                    mask[lo:hi] = True
            return mask

        raise ValueError(f"Unsupported generation state: {state!r}")

    def is_legal_token(
        self,
        *,
        token_id: int,
        vocab_size: int,
        state: GenerationState | str,
        boundary_candidate: bool = False,
        allow_window_open: bool = True,
    ) -> bool:
        mask = self.legal_token_mask(
            vocab_size=vocab_size,
            state=state,
            boundary_candidate=boundary_candidate,
            allow_window_open=allow_window_open,
        )
        tid = int(token_id)
        if tid < 0 or tid >= int(vocab_size):
            return False
        return bool(mask[tid].item())

    def apply_legal_mask(
        self,
        logits: torch.Tensor,
        *,
        state: GenerationState | str,
        boundary_candidate: bool = False,
        allow_window_open: bool = True,
    ) -> torch.Tensor:
        """
        Apply hard legal-token masking to logits (..., V).
        """
        if logits.ndim < 1:
            raise ValueError(f"logits must have at least 1 dim, got shape {tuple(logits.shape)}")
        vocab_size = int(logits.shape[-1])
        legal = self.legal_token_mask(
            vocab_size=vocab_size,
            state=state,
            boundary_candidate=boundary_candidate,
            allow_window_open=allow_window_open,
            device=logits.device,
        )
        return logits.masked_fill(~legal, float("-inf"))

    def illegal_token_rate(
        self,
        *,
        token_ids: Sequence[int] | torch.Tensor,
        states: Sequence[GenerationState | str],
        vocab_size: int,
        boundary_candidates: Sequence[bool] | None = None,
        allow_window_open: Sequence[bool] | None = None,
    ) -> float:
        if isinstance(token_ids, torch.Tensor):
            toks = [int(x) for x in token_ids.detach().cpu().flatten().tolist()]
        else:
            toks = [int(x) for x in token_ids]
        sts = [s if isinstance(s, GenerationState) else GenerationState(str(s)) for s in states]
        if len(toks) != len(sts):
            raise ValueError(f"token_ids and states must match length, got {len(toks)} vs {len(sts)}")
        if len(toks) == 0:
            return 0.0

        if boundary_candidates is None:
            boundary_candidates = [False] * len(toks)
        if allow_window_open is None:
            allow_window_open = [True] * len(toks)
        if len(boundary_candidates) != len(toks) or len(allow_window_open) != len(toks):
            raise ValueError("boundary_candidates and allow_window_open must match token_ids length")

        illegal = 0
        total = len(toks)
        for tid, st, bc, awo in zip(toks, sts, boundary_candidates, allow_window_open):
            if not self.is_legal_token(
                token_id=int(tid),
                vocab_size=int(vocab_size),
                state=st,
                boundary_candidate=bool(bc),
                allow_window_open=bool(awo),
            ):
                illegal += 1
        return float(illegal) / float(total)


@dataclass(frozen=True)
class RolloutConfig:
    max_new_tokens: int = 64
    max_new_windows: int = 1
    max_chunks_per_window: int = 4
    max_content_tokens_per_chunk: int = 96
    min_content_tokens_per_chunk: int = 1
    default_content_dt_hours: float = 1.0
    boundary_logit_margin: float = 0.0
    temperature: float = 1.0
    top_k: int | None = None
    sample: bool = False
    stop_after_end_token: bool = False
    trim_trailing_marker: bool = True


@dataclass(frozen=True)
class RolloutStep:
    step_index: int
    window_index: int
    chunk_index: int
    position_in_chunk: int
    token_id: int
    token_kind: str
    generation_state: str
    content_score: float | None
    boundary_score: float | None
    inserted_prefix_token_ids: tuple[int, ...] = ()
    opened_window_type_id: int | None = None
    sparse_global_id: int | None = None
    block_name: str | None = None
    category_id: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "window_index": int(self.window_index),
            "chunk_index": int(self.chunk_index),
            "position_in_chunk": int(self.position_in_chunk),
            "token_id": int(self.token_id),
            "token_kind": str(self.token_kind),
            "generation_state": str(self.generation_state),
            "content_score": float(self.content_score) if self.content_score is not None else None,
            "boundary_score": float(self.boundary_score) if self.boundary_score is not None else None,
            "inserted_prefix_token_ids": [int(x) for x in self.inserted_prefix_token_ids],
            "opened_window_type_id": int(self.opened_window_type_id) if self.opened_window_type_id is not None else None,
            "sparse_global_id": int(self.sparse_global_id) if self.sparse_global_id is not None else None,
            "block_name": self.block_name,
            "category_id": int(self.category_id) if self.category_id is not None else None,
        }


def _block_name_to_category_id(name: str) -> int:
    name_l = str(name).lower()
    if name_l == "special":
        return int(TokenCategory.SPECIAL)
    if name_l == "structural":
        return int(TokenCategory.STRUCTURAL)
    if name_l in {"rvq", "measurement_code", "measurement_value", "observation_code", "observation_value"}:
        return int(TokenCategory.MEASUREMENT)
    if name_l in {"diagnosis", "diagnosis_residual"}:
        return int(TokenCategory.DIAGNOSIS)
    if name_l in {"procedure", "procedure_residual"}:
        return int(TokenCategory.PROCEDURE)
    if name_l in {"medication", "medication_residual"}:
        return int(TokenCategory.MEDICATION)
    return int(TokenCategory.OTHER)


def build_dense_token_metadata(vocab_config: dict) -> List[Dict[str, Any]]:
    total_size = int(vocab_config.get("total_size", 0))
    dense_blocks = vocab_config.get("dense_blocks", [])
    meta: List[Dict[str, Any]] = [
        {
            "block_name": None,
            "category_id": int(TokenCategory.OTHER),
            "sparse_global_id": None,
        }
        for _ in range(max(0, total_size))
    ]
    if not isinstance(dense_blocks, list):
        return meta

    for block in dense_blocks:
        if not isinstance(block, dict):
            continue
        dense_offset = int(block.get("dense_offset", 0))
        dense_size = int(block.get("dense_size", 0))
        global_offset = int(block.get("global_offset", 0))
        sparse_ids = block.get("sparse_global_ids", None)
        block_name = str(block.get("name", ""))
        category_id = _block_name_to_category_id(block_name)
        for local_idx in range(max(0, dense_size)):
            dense_id = dense_offset + local_idx
            if dense_id < 0 or dense_id >= len(meta):
                continue
            if isinstance(sparse_ids, list) and local_idx < len(sparse_ids):
                sparse_global_id = int(sparse_ids[local_idx])
            else:
                sparse_global_id = int(global_offset) + int(local_idx)
            meta[dense_id] = {
                "block_name": block_name,
                "category_id": int(category_id),
                "sparse_global_id": int(sparse_global_id),
            }
    return meta


@dataclass
class RolloutSubjectState:
    input_ids: List[List[List[int]]]
    time_ids: List[List[List[float]]]
    numeric_values: List[List[List[float]]]
    numeric_mask: List[List[List[int]]]
    token_type_ids: List[List[List[int]]]
    window_type_ids: List[int]
    window_start_times: List[float]
    chunk_start_offsets: List[List[float]]
    chunk_start_times: List[List[float]]
    chunk_is_last: List[List[int]]
    global_special_ids: List[int]
    pad_id: int = 0

    @classmethod
    def from_collated_batch(
        cls,
        batch: Dict[str, torch.Tensor],
        *,
        sample_idx: int = 0,
        grammar: WindowGenerationGrammar | None = None,
        trim_trailing_marker: bool = True,
        pad_id: int = 0,
    ) -> "RolloutSubjectState":
        required = [
            "input_ids",
            "time_ids",
            "numeric_values",
            "numeric_mask",
            "token_type_ids",
            "attention_mask",
            "window_type_ids",
            "window_start_times",
            "chunk_start_offsets",
            "chunk_start_times",
            "chunk_is_last",
            "window_mask",
            "chunk_mask",
        ]
        for key in required:
            if key not in batch:
                raise KeyError(f"collated batch missing required key {key!r}")

        attention_mask = batch["attention_mask"][int(sample_idx)]
        input_ids_t = batch["input_ids"][int(sample_idx)]
        time_ids_t = batch["time_ids"][int(sample_idx)]
        numeric_values_t = batch["numeric_values"][int(sample_idx)]
        numeric_mask_t = batch["numeric_mask"][int(sample_idx)]
        token_type_ids_t = batch["token_type_ids"][int(sample_idx)]
        window_type_ids_t = batch["window_type_ids"][int(sample_idx)]
        window_start_times_t = batch["window_start_times"][int(sample_idx)]
        chunk_start_offsets_t = batch["chunk_start_offsets"][int(sample_idx)]
        chunk_start_times_t = batch["chunk_start_times"][int(sample_idx)]
        chunk_is_last_t = batch["chunk_is_last"][int(sample_idx)]
        window_mask_t = batch["window_mask"][int(sample_idx)]
        chunk_mask_t = batch["chunk_mask"][int(sample_idx)]

        windows_ids: List[List[List[int]]] = []
        windows_times: List[List[List[float]]] = []
        windows_vals: List[List[List[float]]] = []
        windows_valmask: List[List[List[int]]] = []
        windows_types: List[List[List[int]]] = []
        window_types: List[int] = []
        window_start_times: List[float] = []
        chunk_start_offsets: List[List[float]] = []
        chunk_start_times: List[List[float]] = []
        chunk_is_last: List[List[int]] = []

        for w in range(int(window_mask_t.shape[0])):
            if int(window_mask_t[w].item()) == 0:
                continue
            chunk_ids_w: List[List[int]] = []
            chunk_times_w: List[List[float]] = []
            chunk_vals_w: List[List[float]] = []
            chunk_valmask_w: List[List[int]] = []
            chunk_types_w: List[List[List[int]]] | List[List[int]] = []
            chunk_offsets_w: List[float] = []
            chunk_start_abs_w: List[float] = []
            chunk_last_w: List[int] = []
            for c in range(int(chunk_mask_t[w].shape[0])):
                if int(chunk_mask_t[w, c].item()) == 0:
                    continue
                seq_len = int(attention_mask[w, c].to(dtype=torch.long).sum().item())
                if seq_len <= 0:
                    continue
                chunk_ids_w.append([int(x) for x in input_ids_t[w, c, :seq_len].detach().cpu().tolist()])
                chunk_times_w.append([float(x) for x in time_ids_t[w, c, :seq_len].detach().cpu().tolist()])
                chunk_vals_w.append([float(x) for x in numeric_values_t[w, c, :seq_len, 0].detach().cpu().tolist()])
                chunk_valmask_w.append([int(x) for x in numeric_mask_t[w, c, :seq_len].detach().cpu().tolist()])
                chunk_types_w.append([int(x) for x in token_type_ids_t[w, c, :seq_len].detach().cpu().tolist()])
                chunk_offsets_w.append(float(chunk_start_offsets_t[w, c].detach().cpu().item()))
                chunk_start_abs_w.append(float(chunk_start_times_t[w, c].detach().cpu().item()))
                chunk_last_w.append(int(chunk_is_last_t[w, c].detach().cpu().item()))

            if not chunk_ids_w:
                continue
            windows_ids.append(chunk_ids_w)
            windows_times.append(chunk_times_w)
            windows_vals.append(chunk_vals_w)
            windows_valmask.append(chunk_valmask_w)
            windows_types.append(chunk_types_w)  # type: ignore[arg-type]
            window_types.append(int(window_type_ids_t[w].detach().cpu().item()))
            window_start_times.append(float(window_start_times_t[w].detach().cpu().item()))
            chunk_start_offsets.append(chunk_offsets_w)
            chunk_start_times.append(chunk_start_abs_w)
            chunk_is_last.append(chunk_last_w)

        if not windows_ids:
            raise ValueError("Cannot build rollout state from empty collated batch sample.")

        global_special_ids: List[int] = []
        first_chunk_ids = windows_ids[0][0]
        first_chunk_types = windows_types[0][0]
        for tid, cat in zip(first_chunk_ids, first_chunk_types):
            if int(cat) != int(TokenCategory.SPECIAL):
                break
            if grammar is not None and grammar.is_marker_token(int(tid)):
                break
            global_special_ids.append(int(tid))

        state = cls(
            input_ids=windows_ids,
            time_ids=windows_times,
            numeric_values=windows_vals,
            numeric_mask=windows_valmask,
            token_type_ids=windows_types,  # type: ignore[arg-type]
            window_type_ids=window_types,
            window_start_times=window_start_times,
            chunk_start_offsets=chunk_start_offsets,
            chunk_start_times=chunk_start_times,
            chunk_is_last=chunk_is_last,
            global_special_ids=global_special_ids,
            pad_id=int(pad_id),
        )
        if trim_trailing_marker and grammar is not None:
            state.trim_final_suffix_marker(grammar)
        return state

    def copy(self) -> "RolloutSubjectState":
        return RolloutSubjectState(
            input_ids=[[list(seq) for seq in window] for window in self.input_ids],
            time_ids=[[list(seq) for seq in window] for window in self.time_ids],
            numeric_values=[[list(seq) for seq in window] for window in self.numeric_values],
            numeric_mask=[[list(seq) for seq in window] for window in self.numeric_mask],
            token_type_ids=[[list(seq) for seq in window] for window in self.token_type_ids],
            window_type_ids=list(self.window_type_ids),
            window_start_times=list(self.window_start_times),
            chunk_start_offsets=[list(x) for x in self.chunk_start_offsets],
            chunk_start_times=[list(x) for x in self.chunk_start_times],
            chunk_is_last=[list(x) for x in self.chunk_is_last],
            global_special_ids=list(self.global_special_ids),
            pad_id=int(self.pad_id),
        )

    def active_indices(self) -> tuple[int, int]:
        return len(self.input_ids) - 1, len(self.input_ids[-1]) - 1

    def last_position(self) -> int:
        w, c = self.active_indices()
        return len(self.input_ids[w][c]) - 1

    def active_chunk_content_count(self) -> int:
        w, c = self.active_indices()
        return sum(
            1
            for cat in self.token_type_ids[w][c]
            if int(cat) != int(TokenCategory.SPECIAL)
        )

    def current_window_chunk_count(self) -> int:
        w, _ = self.active_indices()
        return len(self.input_ids[w])

    def current_absolute_time_hours(self) -> float:
        w, c = self.active_indices()
        if not self.time_ids[w][c]:
            return float(self.chunk_start_times[w][c])
        return float(self.chunk_start_times[w][c]) + float(self.time_ids[w][c][-1])

    def trim_final_suffix_marker(self, grammar: WindowGenerationGrammar) -> None:
        w, c = self.active_indices()
        seq = self.input_ids[w][c]
        if not seq:
            return
        if not grammar.is_marker_token(int(seq[-1])):
            return
        if len(seq) < 2:
            return
        self.input_ids[w][c].pop()
        self.time_ids[w][c].pop()
        self.numeric_values[w][c].pop()
        self.numeric_mask[w][c].pop()
        self.token_type_ids[w][c].pop()
        self.chunk_is_last[w][c] = 1

    def truncate_to_window_prefix(
        self,
        num_windows: int,
        *,
        grammar: WindowGenerationGrammar | None = None,
        trim_trailing_marker: bool = True,
    ) -> None:
        keep = max(1, min(int(num_windows), len(self.input_ids)))
        self.input_ids = self.input_ids[:keep]
        self.time_ids = self.time_ids[:keep]
        self.numeric_values = self.numeric_values[:keep]
        self.numeric_mask = self.numeric_mask[:keep]
        self.token_type_ids = self.token_type_ids[:keep]
        self.window_type_ids = self.window_type_ids[:keep]
        self.window_start_times = self.window_start_times[:keep]
        self.chunk_start_offsets = self.chunk_start_offsets[:keep]
        self.chunk_start_times = self.chunk_start_times[:keep]
        self.chunk_is_last = self.chunk_is_last[:keep]
        if trim_trailing_marker and grammar is not None:
            self.trim_final_suffix_marker(grammar)

    def append_content_token(self, token_id: int, *, category_id: int, dt_hours: float) -> None:
        w, c = self.active_indices()
        last_t = float(self.time_ids[w][c][-1]) if self.time_ids[w][c] else 0.0
        next_t = max(0.0, last_t + float(dt_hours))
        self.input_ids[w][c].append(int(token_id))
        self.time_ids[w][c].append(float(next_t))
        self.numeric_values[w][c].append(0.0)
        self.numeric_mask[w][c].append(0)
        self.token_type_ids[w][c].append(int(category_id))

    def append_marker_token(self, token_id: int) -> None:
        w, c = self.active_indices()
        last_t = float(self.time_ids[w][c][-1]) if self.time_ids[w][c] else 0.0
        self.input_ids[w][c].append(int(token_id))
        self.time_ids[w][c].append(float(last_t))
        self.numeric_values[w][c].append(0.0)
        self.numeric_mask[w][c].append(0)
        self.token_type_ids[w][c].append(int(TokenCategory.SPECIAL))

    def open_new_chunk(self, *, window_type_id: int, grammar: WindowGenerationGrammar) -> List[int]:
        w, c = self.active_indices()
        current_abs = self.current_absolute_time_hours()
        self.chunk_is_last[w][c] = 0
        prefix = [int(x) for x in self.global_special_ids]
        prefix.append(int(grammar.type_start) + int(window_type_id))
        self.input_ids[w].append(list(prefix))
        self.time_ids[w].append([0.0] * len(prefix))
        self.numeric_values[w].append([0.0] * len(prefix))
        self.numeric_mask[w].append([0] * len(prefix))
        self.token_type_ids[w].append([int(TokenCategory.SPECIAL)] * len(prefix))
        self.chunk_start_offsets[w].append(float(current_abs - float(self.window_start_times[w])))
        self.chunk_start_times[w].append(float(current_abs))
        self.chunk_is_last[w].append(1)
        return prefix

    def open_new_window(self, *, window_type_id: int, grammar: WindowGenerationGrammar) -> List[int]:
        current_abs = self.current_absolute_time_hours()
        prefix = [int(x) for x in self.global_special_ids]
        prefix.append(int(grammar.type_start) + int(window_type_id))
        self.input_ids.append([list(prefix)])
        self.time_ids.append([[0.0] * len(prefix)])
        self.numeric_values.append([[0.0] * len(prefix)])
        self.numeric_mask.append([[0] * len(prefix)])
        self.token_type_ids.append([[int(TokenCategory.SPECIAL)] * len(prefix)])
        self.window_type_ids.append(int(window_type_id))
        self.window_start_times.append(float(current_abs))
        self.chunk_start_offsets.append([0.0])
        self.chunk_start_times.append([float(current_abs)])
        self.chunk_is_last.append([1])
        return prefix

    def to_model_batch(self, *, device: torch.device | None = None) -> Dict[str, torch.Tensor]:
        W = len(self.input_ids)
        C = max((len(chunks) for chunks in self.input_ids), default=0)
        L = max((len(seq) for chunks in self.input_ids for seq in chunks), default=1)
        device = device or torch.device("cpu")

        input_ids = torch.full((1, W, C, L), int(self.pad_id), dtype=torch.long, device=device)
        time_ids = torch.zeros((1, W, C, L), dtype=torch.float32, device=device)
        numeric_values = torch.zeros((1, W, C, L, 1), dtype=torch.float32, device=device)
        numeric_mask = torch.zeros((1, W, C, L), dtype=torch.long, device=device)
        token_type_ids = torch.zeros((1, W, C, L), dtype=torch.long, device=device)
        attention_mask = torch.zeros((1, W, C, L), dtype=torch.long, device=device)
        window_mask = torch.zeros((1, W), dtype=torch.long, device=device)
        chunk_mask = torch.zeros((1, W, C), dtype=torch.long, device=device)
        window_type_ids = torch.zeros((1, W), dtype=torch.long, device=device)
        window_start_times = torch.zeros((1, W), dtype=torch.float32, device=device)
        chunk_start_offsets = torch.zeros((1, W, C), dtype=torch.float32, device=device)
        chunk_start_times = torch.zeros((1, W, C), dtype=torch.float32, device=device)
        chunk_is_last = torch.zeros((1, W, C), dtype=torch.long, device=device)

        for w in range(W):
            window_mask[0, w] = 1
            window_type_ids[0, w] = int(self.window_type_ids[w])
            window_start_times[0, w] = float(self.window_start_times[w])
            for c in range(len(self.input_ids[w])):
                seq_ids = self.input_ids[w][c]
                seq_len = len(seq_ids)
                chunk_mask[0, w, c] = 1
                input_ids[0, w, c, :seq_len] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                time_ids[0, w, c, :seq_len] = torch.tensor(self.time_ids[w][c], dtype=torch.float32, device=device)
                numeric_values[0, w, c, :seq_len, 0] = torch.tensor(
                    self.numeric_values[w][c], dtype=torch.float32, device=device
                )
                numeric_mask[0, w, c, :seq_len] = torch.tensor(
                    self.numeric_mask[w][c], dtype=torch.long, device=device
                )
                token_type_ids[0, w, c, :seq_len] = torch.tensor(
                    self.token_type_ids[w][c], dtype=torch.long, device=device
                )
                attention_mask[0, w, c, :seq_len] = 1
                chunk_start_offsets[0, w, c] = float(self.chunk_start_offsets[w][c])
                chunk_start_times[0, w, c] = float(self.chunk_start_times[w][c])
                chunk_is_last[0, w, c] = int(self.chunk_is_last[w][c])

        return {
            "input_ids": input_ids,
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
        }

    def export_sequences(self, *, dense_token_meta: Sequence[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for w, chunks in enumerate(self.input_ids):
            win_entry: Dict[str, Any] = {
                "window_index": int(w),
                "window_type_id": int(self.window_type_ids[w]),
                "window_start_time_hours": float(self.window_start_times[w]),
                "chunks": [],
            }
            for c, seq_ids in enumerate(chunks):
                seq_items: List[Dict[str, Any]] = []
                for i, tid in enumerate(seq_ids):
                    item = {
                        "token_id": int(tid),
                        "token_type_id": int(self.token_type_ids[w][c][i]),
                        "time_id": float(self.time_ids[w][c][i]),
                        "numeric_value": float(self.numeric_values[w][c][i]),
                        "numeric_mask": int(self.numeric_mask[w][c][i]),
                    }
                    if dense_token_meta is not None and 0 <= int(tid) < len(dense_token_meta):
                        item.update(
                            {
                                "sparse_global_id": dense_token_meta[int(tid)].get("sparse_global_id"),
                                "block_name": dense_token_meta[int(tid)].get("block_name"),
                                "category_id_meta": dense_token_meta[int(tid)].get("category_id"),
                            }
                        )
                    seq_items.append(item)
                win_entry["chunks"].append(
                    {
                        "chunk_index": int(c),
                        "chunk_start_offset_hours": float(self.chunk_start_offsets[w][c]),
                        "chunk_start_time_hours": float(self.chunk_start_times[w][c]),
                        "chunk_is_last": int(self.chunk_is_last[w][c]),
                        "sequence": seq_items,
                    }
                )
            out.append(win_entry)
        return out


def _mask_noncontent_special_tokens(
    logits: torch.Tensor,
    *,
    grammar: WindowGenerationGrammar,
    dense_token_meta: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    masked = logits.clone()
    max_idx = min(int(masked.shape[-1]), len(dense_token_meta))
    for tid in range(max_idx):
        if grammar.is_marker_token(tid):
            continue
        if int(dense_token_meta[tid].get("category_id", int(TokenCategory.OTHER))) == int(TokenCategory.SPECIAL):
            masked[..., tid] = float("-inf")
    return masked


def _pick_from_logits(
    logits: torch.Tensor,
    *,
    sample: bool,
    temperature: float,
    top_k: int | None,
) -> tuple[int, float]:
    logits = logits.detach()
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D for token selection, got shape {tuple(logits.shape)}")
    finite_mask = torch.isfinite(logits)
    if not finite_mask.any():
        raise ValueError("No finite logits available for generation step.")

    if not sample:
        idx = int(torch.argmax(logits).item())
        return idx, float(logits[idx].item())

    temp = max(float(temperature), 1e-6)
    work = logits / temp
    if top_k is not None and int(top_k) > 0 and int(top_k) < int(work.shape[0]):
        top_vals, top_idx = torch.topk(work, k=int(top_k))
        probs = torch.softmax(top_vals, dim=-1)
        choice = int(torch.multinomial(probs, num_samples=1).item())
        idx = int(top_idx[choice].item())
        return idx, float(logits[idx].item())

    probs = torch.softmax(work, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return idx, float(logits[idx].item())


def _select_next_window_type(
    *,
    head_outputs: Dict[str, torch.Tensor],
    window_index: int,
    chunk_index: int,
    position_in_chunk: int,
) -> int:
    score: torch.Tensor | None = None
    logits_local = head_outputs.get("logits_boundary_next_window_type", None)
    if logits_local is not None:
        score = logits_local[0, int(window_index), int(chunk_index), int(position_in_chunk), :].detach()
    logits_global = head_outputs.get("logits_next_window_type", None)
    if logits_global is not None:
        global_score = logits_global[0, int(window_index), :].detach()
        score = global_score if score is None else score + global_score
    if score is None:
        return 0
    return int(torch.argmax(score).item())


def rollout_subject_with_model(
    *,
    model: Any,
    vocab_config: dict,
    subject_state: RolloutSubjectState,
    config: RolloutConfig | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    cfg = config or RolloutConfig()
    grammar = WindowGenerationGrammar.from_vocab_config(vocab_config)
    dense_token_meta = build_dense_token_metadata(vocab_config)
    state = subject_state.copy()
    if bool(cfg.trim_trailing_marker):
        state.trim_final_suffix_marker(grammar)

    model_device = device
    if model_device is None:
        try:
            model_device = next(model.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            model_device = torch.device("cpu")

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    steps: List[RolloutStep] = []
    generated_tokens = 0
    opened_windows = 0
    stop_reason = "max_new_tokens"

    with torch.no_grad():
        while generated_tokens < int(cfg.max_new_tokens):
            batch = state.to_model_batch(device=model_device)
            head_outputs, _ = model(
                input_ids=batch["input_ids"],
                time_ids=batch["time_ids"],
                numeric_values=batch["numeric_values"],
                token_type_ids=batch["token_type_ids"],
                attention_mask=batch["attention_mask"],
                window_start_times=batch["window_start_times"],
                window_mask=batch["window_mask"],
                window_type_ids=batch["window_type_ids"],
                chunk_mask=batch["chunk_mask"],
                chunk_start_offsets=batch["chunk_start_offsets"],
                chunk_is_last=batch["chunk_is_last"],
            )

            w_idx, c_idx = state.active_indices()
            pos = state.last_position()
            token_logits = head_outputs["logits_token"][0, w_idx, c_idx, pos, :].detach()
            token_logits = _mask_noncontent_special_tokens(
                token_logits,
                grammar=grammar,
                dense_token_meta=dense_token_meta,
            )

            content_logits = grammar.apply_legal_mask(
                token_logits,
                state=GenerationState.INSIDE_CHUNK,
            )
            content_token_id, content_score = _pick_from_logits(
                content_logits,
                sample=bool(cfg.sample),
                temperature=float(cfg.temperature),
                top_k=cfg.top_k,
            )

            content_count = state.active_chunk_content_count()
            chunk_limit_reached = content_count >= int(cfg.max_content_tokens_per_chunk)
            can_emit_boundary = content_count >= int(cfg.min_content_tokens_per_chunk)

            boundary_token_id: int | None = None
            boundary_score: float | None = None
            if can_emit_boundary:
                boundary_logits = grammar.apply_legal_mask(
                    token_logits,
                    state=GenerationState.CHUNK_END,
                    boundary_candidate=True,
                )
                if state.current_window_chunk_count() >= int(cfg.max_chunks_per_window):
                    boundary_logits[int(grammar.continue_id)] = float("-inf")

                logits_tb = head_outputs.get("logits_transition_boundary", None)
                if logits_tb is not None:
                    tb = torch.log_softmax(logits_tb[0, w_idx, c_idx, pos, :].detach(), dim=-1)
                    if 0 <= int(grammar.continue_id) < int(boundary_logits.shape[0]):
                        boundary_logits[int(grammar.continue_id)] += float(tb[0].item())
                    end_bias = float(tb[1].item())
                    if 0 <= int(grammar.end_id) < int(boundary_logits.shape[0]) and torch.isfinite(
                        boundary_logits[int(grammar.end_id)]
                    ):
                        boundary_logits[int(grammar.end_id)] += end_bias
                    for tid in grammar.type_token_ids:
                        if 0 <= int(tid) < int(boundary_logits.shape[0]) and torch.isfinite(boundary_logits[int(tid)]):
                            boundary_logits[int(tid)] += end_bias

                if torch.isfinite(boundary_logits).any():
                    boundary_token_id, boundary_score = _pick_from_logits(
                        boundary_logits,
                        sample=bool(cfg.sample),
                        temperature=float(cfg.temperature),
                        top_k=cfg.top_k,
                    )

            use_boundary = False
            if boundary_token_id is not None:
                if chunk_limit_reached:
                    use_boundary = True
                else:
                    use_boundary = float(boundary_score) > (float(content_score) + float(cfg.boundary_logit_margin))

            if use_boundary and boundary_token_id is not None:
                state.append_marker_token(boundary_token_id)
                inserted_prefix: List[int] = []
                opened_window_type_id: int | None = None
                token_kind = "window_end"

                if int(boundary_token_id) == int(grammar.continue_id):
                    token_kind = "chunk_continue"
                    inserted_prefix = state.open_new_chunk(
                        window_type_id=int(state.window_type_ids[w_idx]),
                        grammar=grammar,
                    )
                elif grammar.is_type_token(int(boundary_token_id)):
                    token_kind = "next_window_type_suffix"
                    opened_window_type_id = int(boundary_token_id) - int(grammar.type_start)
                    if opened_windows >= int(cfg.max_new_windows):
                        stop_reason = "max_new_windows"
                    else:
                        inserted_prefix = state.open_new_window(
                            window_type_id=int(opened_window_type_id),
                            grammar=grammar,
                        )
                        opened_windows += 1
                else:
                    if bool(cfg.stop_after_end_token) or opened_windows >= int(cfg.max_new_windows):
                        stop_reason = "end_token_stop"
                    else:
                        opened_window_type_id = _select_next_window_type(
                            head_outputs=head_outputs,
                            window_index=w_idx,
                            chunk_index=c_idx,
                            position_in_chunk=pos,
                        )
                        inserted_prefix = state.open_new_window(
                            window_type_id=int(opened_window_type_id),
                            grammar=grammar,
                        )
                        opened_windows += 1

                meta = dense_token_meta[int(boundary_token_id)] if 0 <= int(boundary_token_id) < len(dense_token_meta) else {}
                steps.append(
                    RolloutStep(
                        step_index=len(steps),
                        window_index=int(w_idx),
                        chunk_index=int(c_idx),
                        position_in_chunk=int(pos) + 1,
                        token_id=int(boundary_token_id),
                        token_kind=token_kind,
                        generation_state=GenerationState.CHUNK_END.value,
                        content_score=float(content_score) if content_score is not None else None,
                        boundary_score=float(boundary_score) if boundary_score is not None else None,
                        inserted_prefix_token_ids=tuple(int(x) for x in inserted_prefix),
                        opened_window_type_id=int(opened_window_type_id) if opened_window_type_id is not None else None,
                        sparse_global_id=meta.get("sparse_global_id"),
                        block_name=meta.get("block_name"),
                        category_id=int(TokenCategory.SPECIAL),
                    )
                )
                generated_tokens += 1
                if token_kind in {"window_end", "next_window_type_suffix"} and not inserted_prefix:
                    break
                continue

            meta = dense_token_meta[int(content_token_id)] if 0 <= int(content_token_id) < len(dense_token_meta) else {}
            state.append_content_token(
                int(content_token_id),
                category_id=int(meta.get("category_id", int(TokenCategory.OTHER))),
                dt_hours=float(cfg.default_content_dt_hours),
            )
            steps.append(
                RolloutStep(
                    step_index=len(steps),
                    window_index=int(w_idx),
                    chunk_index=int(c_idx),
                    position_in_chunk=int(pos) + 1,
                    token_id=int(content_token_id),
                    token_kind="content",
                    generation_state=GenerationState.INSIDE_CHUNK.value,
                    content_score=float(content_score),
                    boundary_score=float(boundary_score) if boundary_score is not None else None,
                    sparse_global_id=meta.get("sparse_global_id"),
                    block_name=meta.get("block_name"),
                    category_id=int(meta.get("category_id", int(TokenCategory.OTHER))),
                )
            )
            generated_tokens += 1
        else:
            stop_reason = "max_new_tokens"

    if was_training and hasattr(model, "train"):
        model.train()

    return {
        "stop_reason": str(stop_reason),
        "num_generated_tokens": int(generated_tokens),
        "num_opened_windows": int(opened_windows),
        "steps": [step.to_dict() for step in steps],
        "final_state": state.export_sequences(dense_token_meta=dense_token_meta),
    }
