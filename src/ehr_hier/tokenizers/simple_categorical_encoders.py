from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Set
from collections import OrderedDict

from src.ehr_hier.data.token_types import TokenTriplet, TokenCategory


@dataclass
class CategoryVocab:
    """
    A tiny per-category codebook living in a reserved global ID range.

    Global ID = offset + index
      - index=1 is reserved for <UNK>
      - known codes start at index=2
    """
    offset: int
    codes2idx: Dict[str, int]           # must NOT include index 1; we add UNK
    unk_token: str = "<UNK>"
    unk_index: int = 1

    def with_unk(self) -> "CategoryVocab":
        if self.unk_token in self.codes2idx:
            return self
        # shift nothing; we assume user began indexing at 2
        merged = OrderedDict([(self.unk_token, self.unk_index)])
        merged.update(self.codes2idx)
        self.codes2idx = merged
        return self

    def encode(self, code: Optional[str]) -> int:
        idx = self.codes2idx.get(str(code), self.unk_index)
        return self.offset + idx


class SimpleCategoricalEncoder:
    """
    Minimal EventTokenEncoder for non-measurement categories.
    Emits ONE token per event; unknown codes map to category-specific UNK.

    Use distinct offsets per category to avoid clashes with measurement tokens.
    """
    def __init__(self, category: TokenCategory, vocab: CategoryVocab):
        self.category = category
        self.vocab = vocab.with_unk()

    def reset_state(self) -> None:
        return None  # stateless

    def encode_event(self, ev, dt_hours: float) -> List[TokenTriplet]:
        code = getattr(ev, "code", None)
        gid = self.vocab.encode(code)
        return [TokenTriplet(value_id=gid,
                             category_id=int(self.category),
                             dt_hours=float(dt_hours))]


class OtherNoOpEncoder:
    """
    Drops events routed to TokenCategory.OTHER (unrecognized).
    Helpful when you want counts but no tokens.
    """
    category = TokenCategory.OTHER
    def reset_state(self) -> None:
        return None
    def encode_event(self, ev, dt_hours: float) -> List[TokenTriplet]:
        return []


# src/ehr_hier/tokenizers/simple_categorical_encoders.py (append)

import meds_reader as mr
from src.ehr_hier.data.event_router import classify_code_to_category

def build_category_vocab_from_db(
    db: mr.SubjectDatabase,
    target_category: TokenCategory,
    offset: int,
    max_codes: int = 5000,
) -> CategoryVocab:
    """
    Quick & deterministic: collect up to max_codes codes of a category
    by scanning the DB once, assign indices starting at 2, UNK=1.
    """
    seen: Set[str] = set()
    for sid in db:
        for ev in db[int(sid)].events:
            code = getattr(ev, "code", None)
            if classify_code_to_category(code) != target_category:
                continue
            cs = str(code)
            if cs not in seen:
                seen.add(cs)
                if len(seen) >= max_codes:
                    break
        if len(seen) >= max_codes:
            break

    # stable order
    codes_sorted = sorted(seen)
    codes2idx = {c: i + 2 for i, c in enumerate(codes_sorted)}  # start at 2
    return CategoryVocab(offset=offset, codes2idx=codes2idx).with_unk()
