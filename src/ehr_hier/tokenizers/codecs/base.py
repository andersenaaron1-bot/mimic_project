from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from src.ehr_hier.data.event_frames import EventFrame, EventPayloadKind, build_event_frame
from src.ehr_hier.data.token_types import TokenCategory


@runtime_checkable
class EventFrameCodec(Protocol):
    category: TokenCategory

    def reset_state(self) -> None:
        ...

    def encode_frame(self, ev: Any, dt_hours: float) -> list[EventFrame]:
        ...


class TokenBundleFrameCodec:
    """
    Adapter that turns an internal token-bundle encoder into a frame-native codec.

    This keeps `EventFrame` as the canonical timeline object while allowing legacy
    atomic token encoders to remain the low-level payload implementation.
    """

    def __init__(
        self,
        *,
        category: TokenCategory,
        token_encoder: Any,
        payload_kind: EventPayloadKind | str,
        concept_code_fn: Callable[[Any], str | None] | None = None,
        semantic_label_fn: Callable[[Any], str | None] | None = None,
    ) -> None:
        self.category = category
        self.token_encoder = token_encoder
        self.payload_kind = payload_kind
        self._concept_code_fn = concept_code_fn
        self._semantic_label_fn = semantic_label_fn

    def __getattr__(self, name: str) -> Any:
        return getattr(self.token_encoder, name)

    def reset_state(self) -> None:
        reset = getattr(self.token_encoder, "reset_state", None)
        if callable(reset):
            reset()

    def encode_event(self, ev: Any, dt_hours: float):
        return self.token_encoder.encode_event(ev, dt_hours)

    def _source_code(self, ev: Any) -> str | None:
        code = getattr(ev, "code", None)
        return None if code is None else str(code)

    def _concept_code(self, ev: Any) -> str | None:
        if self._concept_code_fn is not None:
            return self._concept_code_fn(ev)
        return self._source_code(ev)

    def _semantic_label(self, ev: Any) -> str | None:
        if self._semantic_label_fn is not None:
            return self._semantic_label_fn(ev)
        return None

    def encode_frame(self, ev: Any, dt_hours: float) -> list[EventFrame]:
        tokens = self.encode_event(ev, dt_hours)
        if not tokens:
            return []
        return [
            build_event_frame(
                tokens,
                payload_kind=self.payload_kind,
                source_code=self._source_code(ev),
                concept_code=self._concept_code(ev),
                semantic_label=self._semantic_label(ev),
            )
        ]
