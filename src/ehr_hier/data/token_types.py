# src/ehr_hier/data/token_types.py
from enum import IntEnum
from dataclasses import dataclass


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
    OTHER       = 6   # anything else you keep


class SpecialToken(IntEnum):
    """
    Global special tokens that live in the same vocabulary as everything else.
    You’ll decide their actual IDs in your global vocab, but this is a common scheme.
    """
    PAD   = 0
    PT_CLS = 1   # patient/episode-level CLS token
    SEP   = 2   # separator between segments (e.g. summary vs timeline)
    MASK  = 3   # for masked modeling, if you use it


@dataclass
class TokenTriplet:
    """
    Minimal per-token representation for the transformer input.

    value_id     : integer ID into the *global* token vocabulary
                   (value RVQ codes, MEDTOK IDs, procedure IDs, etc.)
    category_id  : coarse token category (for type embeddings)
    dt_hours     : time since previous *emitted token* in hours (continuous feature)
    """
    value_id: int
    category_id: int   # should be one of TokenCategory.*
    dt_hours: float
