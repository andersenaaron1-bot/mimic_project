from __future__ import annotations

import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.ehr_hier.data.token_types import EventToken

PRECOMPILED_STORAGE_FORMAT_LEGACY = "legacy_subject_pt"
PRECOMPILED_STORAGE_FORMAT_PACKED_V2 = "packed_shard_v2"
PRECOMPILED_SHARD_SUFFIX = ".ptz"
PRECOMPILED_PAYLOAD_VERSION = 2

_EPOCH = datetime(1970, 1, 1)
_NONE_TIME_SENTINEL = -1


def _datetime_to_epoch_us(value: datetime | None) -> int:
    if value is None:
        return _NONE_TIME_SENTINEL
    delta = value - _EPOCH
    return int(delta.days) * 86_400_000_000 + int(delta.seconds) * 1_000_000 + int(delta.microseconds)


def _epoch_us_to_datetime(value: int) -> datetime | None:
    if int(value) < 0:
        return None
    return _EPOCH + timedelta(microseconds=int(value))


def serialize_timeline_compact(timeline: Sequence[EventToken]) -> dict[str, Any]:
    token_count = int(len(timeline))
    value_ids = torch.empty(token_count, dtype=torch.int32)
    category_ids = torch.empty(token_count, dtype=torch.int16)
    t_from_start_hours = torch.empty(token_count, dtype=torch.float64)
    dt_from_prev_hours = torch.empty(token_count, dtype=torch.float64)
    window_hook_ids = torch.full((token_count,), -1, dtype=torch.int16)
    raw_time_epoch_us = torch.full((token_count,), _NONE_TIME_SENTINEL, dtype=torch.int64)

    window_hook_vocab: list[str] = []
    window_hook_to_id: dict[str, int] = {}

    cat_attr_keys: list[str] = []
    cat_attr_key_to_id: dict[str, int] = {}
    cat_attr_offsets = torch.zeros(token_count + 1, dtype=torch.int32)
    cat_attr_key_ids: list[int] = []
    cat_attr_values: list[int] = []

    num_attr_keys: list[str] = []
    num_attr_key_to_id: dict[str, int] = {}
    num_attr_offsets = torch.zeros(token_count + 1, dtype=torch.int32)
    num_attr_key_ids: list[int] = []
    num_attr_values: list[float] = []

    cat_pos = 0
    num_pos = 0
    for idx, tok in enumerate(timeline):
        value_ids[idx] = int(tok.value_id)
        category_ids[idx] = int(tok.category_id)
        t_from_start_hours[idx] = float(tok.t_from_start_hours)
        dt_from_prev_hours[idx] = float(tok.dt_from_prev_hours)
        raw_time_epoch_us[idx] = _datetime_to_epoch_us(tok.raw_time)

        if tok.window_hook is not None:
            hook = str(tok.window_hook)
            hook_id = window_hook_to_id.get(hook)
            if hook_id is None:
                hook_id = len(window_hook_vocab)
                window_hook_vocab.append(hook)
                window_hook_to_id[hook] = hook_id
            window_hook_ids[idx] = int(hook_id)

        cat_items = sorted((str(k), int(v)) for k, v in (tok.cat_attrs or {}).items())
        cat_attr_offsets[idx] = int(cat_pos)
        for key, value in cat_items:
            key_id = cat_attr_key_to_id.get(key)
            if key_id is None:
                key_id = len(cat_attr_keys)
                cat_attr_keys.append(key)
                cat_attr_key_to_id[key] = key_id
            cat_attr_key_ids.append(int(key_id))
            cat_attr_values.append(int(value))
            cat_pos += 1

        num_items = sorted((str(k), v) for k, v in (tok.num_attrs or {}).items())
        num_attr_offsets[idx] = int(num_pos)
        for key, value in num_items:
            key_id = num_attr_key_to_id.get(key)
            if key_id is None:
                key_id = len(num_attr_keys)
                num_attr_keys.append(key)
                num_attr_key_to_id[key] = key_id
            num_attr_key_ids.append(int(key_id))
            num_attr_values.append(float("nan") if value is None else float(value))
            num_pos += 1

    cat_attr_offsets[token_count] = int(cat_pos)
    num_attr_offsets[token_count] = int(num_pos)

    return {
        "version": PRECOMPILED_PAYLOAD_VERSION,
        "token_count": token_count,
        "value_ids": value_ids,
        "category_ids": category_ids,
        "t_from_start_hours": t_from_start_hours,
        "dt_from_prev_hours": dt_from_prev_hours,
        "window_hook_vocab": window_hook_vocab,
        "window_hook_ids": window_hook_ids,
        "raw_time_epoch_us": raw_time_epoch_us,
        "cat_attr_keys": cat_attr_keys,
        "cat_attr_offsets": cat_attr_offsets,
        "cat_attr_key_ids": torch.tensor(cat_attr_key_ids, dtype=torch.int16),
        "cat_attr_values": torch.tensor(cat_attr_values, dtype=torch.int64),
        "num_attr_keys": num_attr_keys,
        "num_attr_offsets": num_attr_offsets,
        "num_attr_key_ids": torch.tensor(num_attr_key_ids, dtype=torch.int16),
        "num_attr_values": torch.tensor(num_attr_values, dtype=torch.float64),
    }


def deserialize_timeline_compact(payload: Mapping[str, Any]) -> list[EventToken]:
    value_ids = payload["value_ids"].tolist()
    category_ids = payload["category_ids"].tolist()
    t_from_start_hours = payload["t_from_start_hours"].tolist()
    dt_from_prev_hours = payload["dt_from_prev_hours"].tolist()
    window_hook_vocab = [str(x) for x in payload.get("window_hook_vocab", [])]
    window_hook_ids = payload["window_hook_ids"].tolist()
    raw_time_epoch_us = payload["raw_time_epoch_us"].tolist()

    cat_attr_keys = [str(x) for x in payload.get("cat_attr_keys", [])]
    cat_attr_offsets = payload["cat_attr_offsets"].tolist()
    cat_attr_key_ids = payload["cat_attr_key_ids"].tolist()
    cat_attr_values = payload["cat_attr_values"].tolist()

    num_attr_keys = [str(x) for x in payload.get("num_attr_keys", [])]
    num_attr_offsets = payload["num_attr_offsets"].tolist()
    num_attr_key_ids = payload["num_attr_key_ids"].tolist()
    num_attr_values = payload["num_attr_values"].tolist()

    timeline: list[EventToken] = []
    token_count = int(payload.get("token_count", len(value_ids)))
    for idx in range(token_count):
        cat_attrs: dict[str, int] = {}
        cat_start = int(cat_attr_offsets[idx])
        cat_end = int(cat_attr_offsets[idx + 1])
        for entry_idx in range(cat_start, cat_end):
            key = cat_attr_keys[int(cat_attr_key_ids[entry_idx])]
            cat_attrs[key] = int(cat_attr_values[entry_idx])

        num_attrs: dict[str, float | None] = {}
        num_start = int(num_attr_offsets[idx])
        num_end = int(num_attr_offsets[idx + 1])
        for entry_idx in range(num_start, num_end):
            key = num_attr_keys[int(num_attr_key_ids[entry_idx])]
            value = float(num_attr_values[entry_idx])
            num_attrs[key] = None if value != value else value

        hook_id = int(window_hook_ids[idx])
        hook = window_hook_vocab[hook_id] if hook_id >= 0 else None
        timeline.append(
            EventToken(
                value_id=int(value_ids[idx]),
                category_id=int(category_ids[idx]),
                t_from_start_hours=float(t_from_start_hours[idx]),
                dt_from_prev_hours=float(dt_from_prev_hours[idx]),
                cat_attrs=cat_attrs,
                num_attrs=num_attrs,
                raw_time=_epoch_us_to_datetime(int(raw_time_epoch_us[idx])),
                window_hook=hook,
            )
        )
    return timeline


def save_packed_shard(
    path: str | Path,
    *,
    subject_ids: Sequence[int],
    serialized_timelines: Sequence[Mapping[str, Any]],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PRECOMPILED_PAYLOAD_VERSION,
        "storage_format": PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
        "subject_ids": [int(sid) for sid in subject_ids],
        "timelines": list(serialized_timelines),
    }
    with gzip.open(out_path, "wb", compresslevel=6) as handle:
        torch.save(payload, handle)


def load_packed_shard(path: str | Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)
