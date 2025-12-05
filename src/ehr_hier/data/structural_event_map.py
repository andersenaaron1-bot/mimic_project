from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .event_router import classify_code_to_category
from .token_types import TokenCategory


@dataclass
class StructuralEventMap:
    """
    Lightweight container for structural/procedural boundary events.

    codes      : mapping code -> frequency (or weight) used for provenance.
    top_k      : number of codes kept.
    percentile : percentile cutoff applied when deriving the map.
    """
    codes: Dict[str, int]
    top_k: int
    percentile: float

    def as_dict(self) -> Dict[str, int]:
        return dict(self.codes)

    def __contains__(self, code: object) -> bool:
        if code is None:
            return False
        return str(code) in self.codes


def tally_structural_procedural_events(
    db,
    *,
    categories: Iterable[TokenCategory] | None = None,
) -> Dict[str, int]:
    """
    Count structural/procedural events across a meds_reader SubjectDatabase (or compatible).

    Parameters
    ----------
    db : meds_reader.SubjectDatabase-like
    categories : Iterable[TokenCategory], optional
        Defaults to PROCEDURE + STRUCTURAL if not provided.
    """
    cats = set(categories) if categories is not None else {
        TokenCategory.PROCEDURE,
        TokenCategory.STRUCTURAL,
    }
    counts: Dict[str, int] = {}
    for sid in db:
        subj = db[int(sid)]
        for ev in subj.events:
            code = getattr(ev, "code", None)
            cat = classify_code_to_category(code)
            if cat not in cats:
                continue
            code_str = str(code)
            counts[code_str] = counts.get(code_str, 0) + 1
    return counts


def topk_by_percentile(
    counts: Dict[str, int],
    *,
    top_k: int = 50,
    percentile: float = 0.9,
) -> Dict[str, int]:
    """
    Select top_k codes whose frequency is above a percentile cutoff.
    """
    if not counts:
        return {}
    values = sorted(counts.values())
    pct = max(0.0, min(1.0, float(percentile)))
    idx = int(math.floor(pct * (len(values) - 1))) if values else 0
    idx = max(0, min(len(values) - 1, idx))
    cutoff = values[idx]

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    filtered = [(c, n) for c, n in items if n >= cutoff]
    return dict(filtered[: top_k if top_k > 0 else len(filtered)])


def build_structural_event_map(
    db,
    *,
    top_k: int = 50,
    percentile: float = 0.9,
) -> StructuralEventMap:
    counts = tally_structural_procedural_events(db)
    return StructuralEventMap(
        codes=topk_by_percentile(counts, top_k=top_k, percentile=percentile),
        top_k=top_k,
        percentile=percentile,
    )


def load_structural_event_map(json_fp: str, *, top_k: Optional[int] = None, percentile: Optional[float] = None) -> StructuralEventMap:
    """
    Load a structural event map JSON of shape {"code": count, ...}
    """
    with open(json_fp, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Structural event map must be a dict, got {type(payload)}")
    codes = {str(k): int(v) for k, v in payload.items()}
    return StructuralEventMap(
        codes=codes,
        top_k=top_k if top_k is not None else len(codes),
        percentile=percentile if percentile is not None else 1.0,
    )


def save_structural_event_map(map_obj: StructuralEventMap | Dict[str, int], json_fp: str) -> None:
    """
    Persist a structural event map to JSON for reuse during training/inference.
    """
    codes = map_obj.codes if isinstance(map_obj, StructuralEventMap) else map_obj
    with open(json_fp, "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=2, sort_keys=True)
