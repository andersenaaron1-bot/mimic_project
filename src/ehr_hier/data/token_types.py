# src/ehr_hier/data/token_types.py
from enum import IntEnum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


class TokenCategory(IntEnum):
    """
    Coarse token categories for type embeddings / routing.
    """
    SPECIAL     = 0
    MEASUREMENT = 1   # CVAE+RVQ-based value tokens
    DIAGNOSIS   = 2   # ICD/MEDTOK-style tokens
    PROCEDURE   = 3   # procedures (ICD, ICU procedures)
    MEDICATION  = 4   # meds, infusions
    STRUCTURAL  = 5   # admissions, transfers, ICU stays, birth/death
    OTHER       = 6   # anything else


class SpecialToken(IntEnum):
    """
    Global special tokens that live in the same vocabulary as everything else.
    """
    PAD   = 0
    PT_CLS = 1   # patient/episode-level CLS token
    SEP   = 2   # separator between segments (e.g. summary vs timeline)
    MASK  = 3   # for masked modeling, if you use it


@dataclass
class EventToken:
    """
    Rich per-token bundle for transformer batching.

    Fields
    ------
    value_id : int
        Global token id (measurement RVQ, MedTok code, procedure id, etc.).
    category_id : int
        Coarse category (TokenCategory.*) for type embeddings / routing.
    t_from_start_hours : float
        Time from the start of the subject timeline (hours).
    dt_from_prev_hours : float
        Time since the previous emitted token (hours).
    cat_attrs : Dict[str, int]
        Optional categorical attributes (already mapped to global ids).
    num_attrs : Dict[str, Optional[float]]
        Optional numeric attributes (raw/normalized scalars to be folded later).
        Missing values can be represented as None for downstream masking.
    raw_time : Optional[datetime]
        Raw event timestamp for debugging or downstream windowing.
    window_hook : Optional[str]
        Optional hook label to drive window segmentation/boundaries.
    """

    value_id: int
    category_id: int
    t_from_start_hours: float
    dt_from_prev_hours: float
    cat_attrs: Dict[str, int] = field(default_factory=dict)
    num_attrs: Dict[str, Optional[float]] = field(default_factory=dict)
    raw_time: Optional[datetime] = None
    window_hook: Optional[str] = None
