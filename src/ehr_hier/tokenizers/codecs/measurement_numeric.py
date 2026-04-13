from __future__ import annotations

from typing import Any

from src.ehr_hier.data.event_frames import EventFrame, EventPayloadKind, build_event_frame
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.codecs.qualitative_observation import QualitativeObservationFrameCodec
from src.ehr_hier.tokenizers.measurement_encoder import MeasurementEncoderConfig, MeasurementTokenEncoder
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab


class MeasurementEventFrameCodec:
    category = TokenCategory.MEASUREMENT

    def __init__(
        self,
        cfg: MeasurementEncoderConfig,
        *,
        numeric_token_encoder: Any | None = None,
        qual_obs_code_offset: int = 2_300_000,
        qual_obs_value_offset: int = 2_320_000,
        qual_obs_value_vocab_size: int = 80_000,
        qual_obs_code_vocab: CategoryVocab | None = None,
        qual_obs_value_vocab: CategoryVocab | None = None,
        qual_obs_tail_policy: str = "drop",
    ) -> None:
        self.numeric_codec = numeric_token_encoder or MeasurementTokenEncoder(cfg)
        self.obs_codec = QualitativeObservationFrameCodec(
            code_offset=qual_obs_code_offset,
            value_offset=qual_obs_value_offset,
            value_vocab_size=qual_obs_value_vocab_size,
            code_vocab=qual_obs_code_vocab,
            value_vocab=qual_obs_value_vocab,
            tail_policy=qual_obs_tail_policy,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.numeric_codec, name)

    def reset_state(self) -> None:
        self.numeric_codec.reset_state()
        self.obs_codec.reset_state()

    def encode_event(self, ev: Any, dt_hours: float):
        return self.numeric_codec.encode_event(ev, dt_hours)

    def encode_frame(self, ev: Any, dt_hours: float) -> list[EventFrame]:
        tokens = self.numeric_codec.encode_event(ev, dt_hours)
        code = getattr(ev, "code", None)
        code_str = None if code is None else str(code)
        if tokens:
            return [
                build_event_frame(
                    tokens,
                    payload_kind=EventPayloadKind.NUMERIC_MEASUREMENT,
                    source_code=code_str,
                    concept_code=code_str,
                )
            ]
        return self.obs_codec.encode_frame(ev, dt_hours)
