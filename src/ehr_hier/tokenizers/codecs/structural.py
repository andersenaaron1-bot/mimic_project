from __future__ import annotations

import zlib
from datetime import datetime

from src.ehr_hier.data.event_frames import EventFrame, EventPayloadKind, build_event_frame
from src.ehr_hier.data.token_types import EventToken, TokenCategory

PROCESS_ACTION_TO_ID = {"START": 1, "END": 2, "STOP": 3}
PROCESS_DOMAIN_TO_ID = {"MEDICATION": 1, "PROCEDURE": 2, "INFUSION": 3}


def _stable_local_id(raw: str, *, modulo: int = 900_000) -> int:
    data = str(raw).encode("utf-8", errors="ignore")
    return 1 + (zlib.crc32(data) % max(1, int(modulo)))


def parse_process_transition(code_value: str | None) -> tuple[str, str, str] | None:
    if code_value is None:
        return None
    code_norm = str(code_value).strip()
    if not code_norm:
        return None
    upper = code_norm.upper()
    if upper.startswith("INFUSION_START//"):
        return ("START", "INFUSION", code_norm.split("//", 1)[1])
    if upper.startswith("INFUSION_END//"):
        return ("END", "INFUSION", code_norm.split("//", 1)[1])

    parts = [p.strip() for p in code_norm.split("//")]
    if len(parts) >= 3 and parts[0].upper() in {"MEDICATION", "PROCEDURE"}:
        marker = parts[1].upper()
        if marker in {"START", "END", "STOP"}:
            entity = "//".join(parts[2:]).strip()
            if entity:
                return (marker, parts[0].upper(), entity)
    return None


def emit_process_transition_frame(
    *,
    transition: tuple[str, str, str],
    struct_action_offset: int,
    struct_entity_offset: int,
    raw_time: datetime | None = None,
) -> EventFrame | None:
    action, domain, entity = transition
    action_id = PROCESS_ACTION_TO_ID.get(action.upper(), 0)
    domain_id = PROCESS_DOMAIN_TO_ID.get(domain.upper(), 0)
    if action_id <= 0 or domain_id <= 0:
        return None

    entity_local = int(entity) if entity.isdigit() else _stable_local_id(f"{domain.upper()}::{entity}")
    tokens = [
        EventToken(
            value_id=int(struct_action_offset) + int(action_id),
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "struct_process_action_id": int(action_id),
                "struct_process_domain_id": int(domain_id),
            },
            num_attrs={},
            raw_time=raw_time,
            window_hook=None,
        ),
        EventToken(
            value_id=int(struct_entity_offset) + int(entity_local),
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "struct_process_action_id": int(action_id),
                "struct_process_domain_id": int(domain_id),
            },
            num_attrs={},
            raw_time=raw_time,
            window_hook=None,
        ),
    ]
    return build_event_frame(
        tokens,
        payload_kind=EventPayloadKind.PROCESS,
        semantic_label=f"{domain.upper()}_{action.upper()}",
    )
