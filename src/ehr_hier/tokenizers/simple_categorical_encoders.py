from __future__ import annotations
from typing import Dict, List, Optional, Set

from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab


class SimpleCategoricalEncoder:
    """
    Minimal EventTokenEncoder for non-measurement categories.
    Emits ONE token per event; unknown codes map to category-specific UNK.
    """
    def __init__(self, category: TokenCategory, vocab: CategoryVocab):
        self.category = category
        self.vocab = vocab

    def reset_state(self) -> None:
        return None  # stateless

    def encode_event(self, ev, dt_hours: float) -> List[EventToken]:
        code = getattr(ev, "code", None)
        gid = self.vocab.encode(code)
        return [
            EventToken(
                value_id=gid,
                category_id=int(self.category),
                t_from_start_hours=0.0,
                dt_from_prev_hours=float(dt_hours),
                cat_attrs={},
                num_attrs={},
            )
        ]


class OtherNoOpEncoder:
    """
    Drops events routed to TokenCategory.OTHER (unrecognized).
    Helpful when you want counts but no tokens.
    """
    category = TokenCategory.OTHER

    def reset_state(self) -> None:
        return None

    def encode_event(self, ev, dt_hours: float) -> List[EventToken]:
        return []


# Optional helper when scanning a meds_reader DB ---------------------------
try:
    import meds_reader as mr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mr = None
from src.ehr_hier.data.event_router import classify_code_to_category


def build_category_vocab_from_db(
    db: mr.SubjectDatabase,
    target_category: TokenCategory,
    offset: int,
    max_codes: int = 5000,
) -> CategoryVocab:
    """
    Quick & deterministic: collect up to max_codes codes of a category,
    assign indices starting at 1 (reserve 0 for UNK).
    """
    if mr is None:
        raise ImportError("meds_reader is required to build category vocab from DB")
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

    codes_sorted = sorted(seen)
    code2id: Dict[str, int] = {c: i + 1 for i, c in enumerate(codes_sorted)}
    code2id["<UNK>"] = 0
    return CategoryVocab(
        name=target_category.name.lower(),
        offset=offset,
        code2id=code2id,
    )
