from __future__ import annotations
from typing import Dict, List, Tuple

from src.ehr_hier.data.token_types import EventToken, TokenCategory


def decode_measurement_tokens(
    tokens: List[EventToken],
    *,
    code_token_offset: int,
    rvq_token_offset: int,
    rvq_codebook_stride: int,
    code2name: Dict[int, str] | None = None,
) -> List[Dict]:
    """
    Best-effort decoder for measurement token bundles emitted as:
      - one code token (var_id) + L RVQ tokens (one per codebook)
    Returns a list of dicts with var_id, var_name (if available), and rvq_indices.
    """
    decoded = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.category_id != int(TokenCategory.MEASUREMENT):
            i += 1
            continue
        # expect a code token first
        var_id = tok.value_id - code_token_offset
        var_name = code2name.get(var_id) if code2name else None
        rvq_indices: List[int] = []
        j = i + 1
        while j < len(tokens):
            t_j = tokens[j]
            if t_j.category_id != int(TokenCategory.MEASUREMENT):
                break
            idx = (t_j.value_id - rvq_token_offset) % rvq_codebook_stride
            rvq_indices.append(idx)
            j += 1
        decoded.append(
            {
                "var_id": var_id,
                "var_name": var_name,
                "rvq_indices": rvq_indices,
                "z_norm": tok.num_attrs.get("z"),
            }
        )
        i = j
    return decoded
