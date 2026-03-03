from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from src.ehr_hier.data.token_types import EventToken, TokenCategory


EVENT_MARKER_NAMES = {
    1: "START",
    2: "END",
    3: "STOP",
}


def invert_code2id(code2id: Mapping[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in code2id.items()}


def _is_measurement_code_token(
    tok: EventToken,
    *,
    code_token_offset: int,
    rvq_token_offset: int | None = None,
) -> bool:
    if tok.category_id != int(TokenCategory.MEASUREMENT):
        return False
    if tok.cat_attrs and "codebook" in tok.cat_attrs:
        return False
    value_id = int(tok.value_id)
    if value_id < int(code_token_offset):
        return False
    if rvq_token_offset is not None and value_id >= int(rvq_token_offset):
        return False
    return True


def _is_measurement_rvq_token(tok: EventToken, *, rvq_token_offset: int) -> bool:
    if tok.category_id != int(TokenCategory.MEASUREMENT):
        return False
    if tok.cat_attrs and "codebook" in tok.cat_attrs:
        return True
    return int(tok.value_id) >= int(rvq_token_offset)


def decode_measurement_tokens(
    tokens: List[EventToken],
    *,
    code_token_offset: int,
    rvq_token_offset: int,
    rvq_codebook_stride: int,
    num_codebooks: int | None = None,
    code2name: Dict[int, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Best-effort decoder for measurement token bundles emitted as:
      - one code token (var_id)
      - one token per RVQ codebook

    If `num_codebooks` is provided, it is used to keep adjacent measurement
    events from being over-grouped.
    """
    decoded: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not _is_measurement_code_token(
            tok,
            code_token_offset=code_token_offset,
            rvq_token_offset=rvq_token_offset,
        ):
            i += 1
            continue

        var_id = int(tok.value_id) - int(code_token_offset)
        var_name = code2name.get(var_id) if code2name else None
        rvq_indices: List[int] = []
        j = i + 1
        while j < len(tokens):
            if num_codebooks is not None and len(rvq_indices) >= int(num_codebooks):
                break
            t_j = tokens[j]
            if not _is_measurement_rvq_token(t_j, rvq_token_offset=rvq_token_offset):
                break
            idx = (int(t_j.value_id) - int(rvq_token_offset)) % int(rvq_codebook_stride)
            rvq_indices.append(idx)
            j += 1

        decoded.append(
            {
                "kind": "measurement_bundle",
                "var_id": var_id,
                "var_name": var_name,
                "rvq_indices": rvq_indices,
                "z_norm": tok.num_attrs.get("z"),
                "t_from_start_hours": float(tok.t_from_start_hours),
                "dt_from_prev_hours": float(tok.dt_from_prev_hours),
                "window_hook": tok.window_hook,
                "token_span": [i, max(i, j - 1)],
            }
        )
        i = max(j, i + 1)
    return decoded


def _decode_vocab_gid(
    value_id: int,
    *,
    offset: int,
    id2code: Mapping[int, str] | None,
) -> str | None:
    if id2code is None:
        return None
    raw_id = int(value_id) - int(offset)
    return id2code.get(raw_id)


def decode_timeline_tokens(
    tokens: Iterable[EventToken],
    *,
    code_token_offset: int | None = None,
    rvq_token_offset: int | None = None,
    rvq_codebook_stride: int | None = None,
    measurement_num_codebooks: int | None = None,
    measurement_code2name: Mapping[int, str] | None = None,
    diagnosis_offset: int | None = None,
    diagnosis_id2code: Mapping[int, str] | None = None,
    procedure_offset: int | None = None,
    procedure_id2code: Mapping[int, str] | None = None,
    medication_offset: int | None = None,
    medication_id2code: Mapping[int, str] | None = None,
    structural_offset: int | None = None,
    structural_id2label: Mapping[int, str] | None = None,
    structural_id2code: Mapping[int, str] | None = None,
    special_id2name: Mapping[int, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Decode a timeline of EventTokens into a more human-readable event list.

    This is intentionally lossy. It aims to preserve semantic interpretability
    rather than exact round-trip reconstruction.
    """
    toks = list(tokens)
    out: List[Dict[str, Any]] = []
    i = 0

    while i < len(toks):
        tok = toks[i]
        category = TokenCategory(int(tok.category_id))

        if (
            category == TokenCategory.MEASUREMENT
            and code_token_offset is not None
            and rvq_token_offset is not None
            and rvq_codebook_stride is not None
            and _is_measurement_code_token(
                tok,
                code_token_offset=int(code_token_offset),
                rvq_token_offset=int(rvq_token_offset),
            )
        ):
            decoded = decode_measurement_tokens(
                toks[i:],
                code_token_offset=int(code_token_offset),
                rvq_token_offset=int(rvq_token_offset),
                rvq_codebook_stride=int(rvq_codebook_stride),
                num_codebooks=measurement_num_codebooks,
                code2name=dict(measurement_code2name) if measurement_code2name is not None else None,
            )
            if decoded:
                first = decoded[0]
                span_len = int(first["token_span"][1]) - int(first["token_span"][0]) + 1
                first["token_span"] = [i, i + span_len - 1]
                out.append(first)
                i += span_len
                continue

        entry: Dict[str, Any] = {
            "kind": "token",
            "category": category.name,
            "value_id": int(tok.value_id),
            "t_from_start_hours": float(tok.t_from_start_hours),
            "dt_from_prev_hours": float(tok.dt_from_prev_hours),
            "window_hook": tok.window_hook,
            "cat_attrs": dict(tok.cat_attrs),
            "num_attrs": dict(tok.num_attrs),
        }

        if category == TokenCategory.SPECIAL and special_id2name is not None:
            entry["label"] = special_id2name.get(int(tok.value_id), f"SPECIAL::{int(tok.value_id)}")
        elif category == TokenCategory.DIAGNOSIS and diagnosis_offset is not None:
            entry["label"] = _decode_vocab_gid(
                int(tok.value_id),
                offset=int(diagnosis_offset),
                id2code=diagnosis_id2code,
            )
        elif category == TokenCategory.PROCEDURE and procedure_offset is not None:
            entry["label"] = _decode_vocab_gid(
                int(tok.value_id),
                offset=int(procedure_offset),
                id2code=procedure_id2code,
            )
        elif category == TokenCategory.MEDICATION and medication_offset is not None:
            marker_id = tok.cat_attrs.get("event_marker") if tok.cat_attrs else None
            if marker_id is not None:
                entry["label"] = f"MED_MARKER::{EVENT_MARKER_NAMES.get(int(marker_id), int(marker_id))}"
            else:
                entry["label"] = _decode_vocab_gid(
                    int(tok.value_id),
                    offset=int(medication_offset),
                    id2code=medication_id2code,
                )
        elif category == TokenCategory.STRUCTURAL and structural_offset is not None:
            if tok.cat_attrs and "struct_label_id" in tok.cat_attrs and structural_id2label is not None:
                entry["label"] = structural_id2label.get(int(tok.cat_attrs["struct_label_id"]))
            elif structural_id2code is not None:
                entry["label"] = structural_id2code.get(int(tok.value_id) - int(structural_offset))
                entry["raw_code"] = entry.get("label")

        out.append(entry)
        i += 1

    return out
