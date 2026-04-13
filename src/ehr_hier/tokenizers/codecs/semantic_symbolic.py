from __future__ import annotations

from typing import Any

from src.ehr_hier.data.event_frames import EventPayloadKind
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.codecs.base import TokenBundleFrameCodec


class SymbolicEventFrameCodec(TokenBundleFrameCodec):
    def __init__(self, *, category: TokenCategory, token_encoder: Any) -> None:
        payload_kind = (
            EventPayloadKind.STRUCTURAL
            if int(category) == int(TokenCategory.STRUCTURAL)
            else EventPayloadKind.SYMBOLIC_CODE
        )
        super().__init__(
            category=category,
            token_encoder=token_encoder,
            payload_kind=payload_kind,
            concept_code_fn=self._resolve_concept_code,
        )

    def _resolve_concept_code(self, ev: Any) -> str | None:
        resolve = getattr(self.token_encoder, "resolve_event", None)
        if callable(resolve):
            resolution = resolve(ev)
            matched = getattr(resolution, "matched_code", None)
            if matched is not None:
                return str(matched)
            source = getattr(resolution, "source_code", None)
            if source is not None:
                return str(source)
        code = getattr(ev, "code", None)
        return None if code is None else str(code)


class OtherNoOpFrameCodec:
    category = TokenCategory.OTHER

    def __init__(self, *, token_encoder: Any) -> None:
        self.token_encoder = token_encoder

    def __getattr__(self, name: str) -> Any:
        return getattr(self.token_encoder, name)

    def reset_state(self) -> None:
        reset = getattr(self.token_encoder, "reset_state", None)
        if callable(reset):
            reset()

    def encode_event(self, ev: Any, dt_hours: float):
        return self.token_encoder.encode_event(ev, dt_hours)

    def encode_frame(self, ev: Any, dt_hours: float):
        return []
