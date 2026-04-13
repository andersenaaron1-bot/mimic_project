from __future__ import annotations

import zlib
from datetime import datetime
from typing import Any, Optional

from src.ehr_hier.data.event_frames import EventFrame, EventPayloadKind, build_event_frame
from src.ehr_hier.data.observation_vocab import OBS_RESERVED_VALUE_IDS, observation_surfaces
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab

OBS_SPECIAL_VALUE_IDS = {
    "UNK": int(OBS_RESERVED_VALUE_IDS["UNK"]),
    "N/A": int(OBS_RESERVED_VALUE_IDS["N/A"]),
    "NA": int(OBS_RESERVED_VALUE_IDS["N/A"]),
    "NONE": int(OBS_RESERVED_VALUE_IDS["NONE"]),
    "NULL": int(OBS_RESERVED_VALUE_IDS["NONE"]),
    "": int(OBS_RESERVED_VALUE_IDS[""]),
}


def _stable_local_id(raw: str, *, modulo: int) -> int:
    data = str(raw).encode("utf-8", errors="ignore")
    return 1 + (zlib.crc32(data) % max(1, int(modulo)))


class QualitativeObservationFrameCodec:
    category = TokenCategory.MEASUREMENT

    def __init__(
        self,
        *,
        code_offset: int = 2_300_000,
        value_offset: int = 2_320_000,
        value_vocab_size: int = 80_000,
        code_vocab: CategoryVocab | None = None,
        value_vocab: CategoryVocab | None = None,
        tail_policy: str = "drop",
    ) -> None:
        self.code_offset = int(code_offset)
        self.value_offset = int(value_offset)
        self.value_vocab_size = int(value_vocab_size)
        self.code_vocab = code_vocab
        self.value_vocab = value_vocab
        self.tail_policy = str(tail_policy)

    def reset_state(self) -> None:
        return None

    def encode_event(self, ev: Any, dt_hours: float) -> list[EventToken]:
        code_value = getattr(ev, "code", None)
        if code_value is None:
            return []
        surfaces = observation_surfaces(ev, code_value=str(code_value))
        if surfaces is None:
            return []

        obs_code_lane = max(16, int(self.value_offset) - int(self.code_offset) - 1)
        obs_value_lane = max(16, int(self.value_vocab_size))

        code_exact = False
        obs_code_gid: Optional[int] = None
        local_code_id: Optional[int] = None
        if self.code_vocab is not None:
            obs_code_gid = self.code_vocab.maybe_encode(surfaces.code_surface)
            if obs_code_gid is not None:
                local_code_id = int(obs_code_gid) - int(self.code_vocab.offset)
                code_exact = True
            elif self.tail_policy.lower() == "drop":
                return []
        if obs_code_gid is None:
            local_code_id = _stable_local_id(surfaces.code_surface, modulo=obs_code_lane)
            obs_code_gid = int(self.code_offset) + int(local_code_id)

        value_exact = False
        obs_val_text = surfaces.value_surface
        obs_val_gid: Optional[int] = None
        obs_val_local: Optional[int] = None
        if self.value_vocab is not None:
            obs_val_gid = self.value_vocab.maybe_encode(obs_val_text)
            if obs_val_gid is not None:
                obs_val_local = int(obs_val_gid) - int(self.value_vocab.offset)
                value_exact = True
            elif self.tail_policy.lower() == "drop":
                return []
        if obs_val_gid is None:
            obs_val_upper = obs_val_text.upper()
            special_local = OBS_SPECIAL_VALUE_IDS.get(obs_val_upper)
            if special_local is not None and int(special_local) <= int(obs_value_lane):
                obs_val_local = int(special_local)
            else:
                obs_val_local = _stable_local_id(f"OBS_VAL::{obs_val_text}", modulo=obs_value_lane)
            obs_val_gid = int(self.value_offset) + int(obs_val_local)

        if local_code_id is None or obs_val_local is None:
            return []

        obs_stage_exact = int(code_exact and value_exact)
        obs_stage_hash = int(not code_exact and not value_exact)
        raw_time = getattr(ev, "time", None)
        if not isinstance(raw_time, datetime):
            raw_time = None

        return [
            EventToken(
                value_id=int(obs_code_gid),
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=0.0,
                dt_from_prev_hours=float(dt_hours),
                cat_attrs={
                    "obs_bundle_pos": 1,
                    "obs_code_local_id": int(local_code_id),
                    "obs_code_exact": int(code_exact),
                    "obs_stage_exact": obs_stage_exact,
                    "obs_stage_hash": obs_stage_hash,
                },
                num_attrs={},
                raw_time=raw_time,
                window_hook=None,
            ),
            EventToken(
                value_id=int(obs_val_gid),
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=0.0,
                dt_from_prev_hours=0.0,
                cat_attrs={
                    "obs_bundle_pos": 2,
                    "obs_value_local_id": int(obs_val_local),
                    "obs_value_exact": int(value_exact),
                    "obs_stage_exact": obs_stage_exact,
                    "obs_stage_hash": obs_stage_hash,
                },
                num_attrs={},
                raw_time=raw_time,
                window_hook=None,
            ),
        ]

    def encode_frame(self, ev: Any, dt_hours: float) -> list[EventFrame]:
        tokens = self.encode_event(ev, dt_hours)
        if not tokens:
            return []
        code = getattr(ev, "code", None)
        code_str = None if code is None else str(code)
        return [
            build_event_frame(
                tokens,
                payload_kind=EventPayloadKind.QUALITATIVE_OBSERVATION,
                source_code=code_str,
                concept_code=code_str,
            )
        ]
