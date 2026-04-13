from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable, Optional, Sequence

from src.ehr_hier.data.token_types import EventToken, TokenCategory


class EventPayloadKind(str, Enum):
    SPECIAL = "special"
    DEMOGRAPHIC = "demographic"
    NUMERIC_MEASUREMENT = "numeric_measurement"
    QUALITATIVE_OBSERVATION = "qualitative_observation"
    SYMBOLIC_CODE = "symbolic_code"
    STRUCTURAL = "structural"
    PROCESS = "process"


EVENT_PAYLOAD_KIND_ORDER: tuple[str, ...] = tuple(kind.value for kind in EventPayloadKind)
EVENT_PAYLOAD_KIND_TO_ID: dict[str, int] = {
    name: idx for idx, name in enumerate(EVENT_PAYLOAD_KIND_ORDER)
}


def clone_event_token(
    tok: EventToken,
    *,
    force_window_type_id: Optional[int] = None,
    force_window_site_id: Optional[int] = None,
    time_offset_hours: float = 0.0,
    force_dt_from_prev_hours: Optional[float] = None,
) -> EventToken:
    cat_attrs = dict(tok.cat_attrs or {})
    if force_window_type_id is not None:
        cat_attrs["window_type_id"] = int(force_window_type_id)
    if force_window_site_id is not None and int(force_window_site_id) > 0:
        cat_attrs["window_site_id"] = int(force_window_site_id)
    return EventToken(
        value_id=int(tok.value_id),
        category_id=int(tok.category_id),
        t_from_start_hours=max(0.0, float(tok.t_from_start_hours) - float(time_offset_hours)),
        dt_from_prev_hours=(
            float(force_dt_from_prev_hours)
            if force_dt_from_prev_hours is not None
            else float(tok.dt_from_prev_hours)
        ),
        cat_attrs=cat_attrs,
        num_attrs=dict(tok.num_attrs or {}),
        raw_time=tok.raw_time,
        window_hook=tok.window_hook,
    )


def infer_payload_kind(tokens: Sequence[EventToken]) -> EventPayloadKind:
    if not tokens:
        raise ValueError("tokens must be non-empty")
    category = TokenCategory(int(tokens[0].category_id))
    if category == TokenCategory.SPECIAL:
        if any(int((tok.cat_attrs or {}).get("global_demographic", 0)) != 0 for tok in tokens):
            return EventPayloadKind.DEMOGRAPHIC
        return EventPayloadKind.SPECIAL
    if category == TokenCategory.STRUCTURAL:
        if any("struct_process_action_id" in (tok.cat_attrs or {}) for tok in tokens):
            return EventPayloadKind.PROCESS
        return EventPayloadKind.STRUCTURAL
    if category == TokenCategory.MEASUREMENT:
        if any(int((tok.cat_attrs or {}).get("obs_bundle_pos", 0)) > 0 for tok in tokens):
            return EventPayloadKind.QUALITATIVE_OBSERVATION
        return EventPayloadKind.NUMERIC_MEASUREMENT
    return EventPayloadKind.SYMBOLIC_CODE


@dataclass
class EventFrame:
    token_bundle: list[EventToken]
    category_id: int
    payload_kind: str
    t_from_start_hours: float
    dt_from_prev_hours: float
    cat_attrs: dict[str, int] = field(default_factory=dict)
    num_attrs: dict[str, Optional[float]] = field(default_factory=dict)
    raw_time: Optional[datetime] = None
    window_hook: Optional[str] = None
    source_code: Optional[str] = None
    concept_code: Optional[str] = None
    semantic_label: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.token_bundle:
            raise ValueError("EventFrame.token_bundle must be non-empty")

    @property
    def value_id(self) -> int:
        return int(self.token_bundle[0].value_id)

    @property
    def token_count(self) -> int:
        return int(len(self.token_bundle))


def build_event_frame(
    tokens: Sequence[EventToken],
    *,
    payload_kind: EventPayloadKind | str | None = None,
    source_code: Optional[str] = None,
    concept_code: Optional[str] = None,
    semantic_label: Optional[str] = None,
) -> EventFrame:
    if not tokens:
        raise ValueError("tokens must be non-empty")
    bundle = [clone_event_token(tok) for tok in tokens]
    first = bundle[0]
    kind = payload_kind or infer_payload_kind(bundle)
    return EventFrame(
        token_bundle=bundle,
        category_id=int(first.category_id),
        payload_kind=str(kind.value if isinstance(kind, EventPayloadKind) else kind),
        t_from_start_hours=float(first.t_from_start_hours),
        dt_from_prev_hours=float(first.dt_from_prev_hours),
        cat_attrs=dict(first.cat_attrs or {}),
        num_attrs=dict(first.num_attrs or {}),
        raw_time=first.raw_time,
        window_hook=first.window_hook,
        source_code=source_code,
        concept_code=concept_code,
        semantic_label=semantic_label,
    )


def clone_event_frame(
    frame: EventFrame,
    *,
    force_window_type_id: Optional[int] = None,
    force_window_site_id: Optional[int] = None,
    time_offset_hours: float = 0.0,
    force_dt_from_prev_hours: Optional[float] = None,
) -> EventFrame:
    bundle = [
        clone_event_token(
            tok,
            force_window_type_id=force_window_type_id,
            force_window_site_id=force_window_site_id,
            time_offset_hours=time_offset_hours,
            force_dt_from_prev_hours=(force_dt_from_prev_hours if idx == 0 else None),
        )
        for idx, tok in enumerate(frame.token_bundle)
    ]
    first = bundle[0]
    return EventFrame(
        token_bundle=bundle,
        category_id=int(frame.category_id),
        payload_kind=str(frame.payload_kind),
        t_from_start_hours=float(first.t_from_start_hours),
        dt_from_prev_hours=float(first.dt_from_prev_hours),
        cat_attrs=dict(first.cat_attrs or {}),
        num_attrs=dict(first.num_attrs or {}),
        raw_time=first.raw_time,
        window_hook=first.window_hook,
        source_code=frame.source_code,
        concept_code=frame.concept_code,
        semantic_label=frame.semantic_label,
    )


def ensure_event_frames(
    timeline: Iterable[EventFrame | EventToken],
    *,
    payload_kind: EventPayloadKind | str | None = None,
) -> list[EventFrame]:
    out: list[EventFrame] = []
    for item in timeline:
        if isinstance(item, EventFrame):
            out.append(clone_event_frame(item))
        else:
            out.append(build_event_frame([item], payload_kind=payload_kind))
    return out


def flatten_event_frames(
    timeline: Iterable[EventFrame | EventToken],
    *,
    clone: bool = True,
) -> list[EventToken]:
    out: list[EventToken] = []
    for item in timeline:
        if isinstance(item, EventFrame):
            for tok in item.token_bundle:
                out.append(clone_event_token(tok) if clone else tok)
        else:
            out.append(clone_event_token(item) if clone else item)
    return out


def payload_kind_to_id(payload_kind: EventPayloadKind | str | None) -> int:
    if payload_kind is None:
        return int(EVENT_PAYLOAD_KIND_TO_ID[EventPayloadKind.SYMBOLIC_CODE.value])
    key = (
        payload_kind.value
        if isinstance(payload_kind, EventPayloadKind)
        else str(payload_kind).strip().lower()
    )
    return int(
        EVENT_PAYLOAD_KIND_TO_ID.get(
            key,
            EVENT_PAYLOAD_KIND_TO_ID[EventPayloadKind.SYMBOLIC_CODE.value],
        )
    )


def slice_event_frame(
    frame: EventFrame,
    *,
    start: int = 0,
    stop: int | None = None,
) -> EventFrame:
    token_count = int(len(frame.token_bundle))
    lo = max(0, int(start))
    hi = token_count if stop is None else max(lo, min(token_count, int(stop)))
    if lo >= hi:
        raise ValueError("slice_event_frame produced an empty slice")

    sliced = frame.token_bundle[lo:hi]
    adjusted_tokens = [
        clone_event_token(
            tok,
            force_dt_from_prev_hours=(
                float(tok.dt_from_prev_hours)
                if idx == 0 and lo == 0
                else (0.0 if idx == 0 else None)
            ),
        )
        for idx, tok in enumerate(sliced)
    ]
    return EventFrame(
        token_bundle=adjusted_tokens,
        category_id=int(frame.category_id),
        payload_kind=str(frame.payload_kind),
        t_from_start_hours=float(adjusted_tokens[0].t_from_start_hours),
        dt_from_prev_hours=float(adjusted_tokens[0].dt_from_prev_hours),
        cat_attrs=dict(adjusted_tokens[0].cat_attrs or {}),
        num_attrs=dict(adjusted_tokens[0].num_attrs or {}),
        raw_time=adjusted_tokens[0].raw_time,
        window_hook=adjusted_tokens[0].window_hook,
        source_code=frame.source_code,
        concept_code=frame.concept_code,
        semantic_label=frame.semantic_label,
    )
