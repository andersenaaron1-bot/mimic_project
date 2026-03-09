from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import torch


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
