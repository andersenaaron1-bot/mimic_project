from __future__ import annotations
from typing import Dict, List, Any, Optional, Iterable, Callable

from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig
from src.ehr_hier.tokenizers.medtok_canonicalize import ensure_list


class MedTokenWithAttrsEncoder:
    """
    Emits a single EventToken with MedTok base id and bundled metadata.
    - base token id comes from MedTok vocab (with canonicalization fallback)
    - categorical metadata stored under EventToken.cat_attrs
    - numeric metadata stored under EventToken.num_attrs (normalized to [0,1])
    """

    def __init__(
        self,
        category: TokenCategory,
        base_vocab: CategoryVocab,
        *,
        categorical_attrs: Dict[str, CategoryVocab] | None = None,
        numeric_attrs: Dict[str, NumericBinConfig] | None = None,
        canonicalize_fn: Optional[Callable[[Any], Iterable[str]]] = None,
        drop_unknowns: bool = False,
        fallback_to_raw: bool = True,
    ):
        self.category = category
        self.base_vocab = base_vocab
        self.categorical_attrs = categorical_attrs or {}
        self.numeric_attrs = numeric_attrs or {}
        self.canonicalize_fn = canonicalize_fn
        self.drop_unknowns = drop_unknowns
        self.fallback_to_raw = fallback_to_raw
        self.unk_gid = self.base_vocab.offset + self.base_vocab.unk_id
        self._cache: Dict[str, Optional[int]] = {}

    def reset_state(self) -> None:
        return None  # stateless

    def _candidate_codes(self, raw_code: Optional[str]) -> List[str]:
        if raw_code is None:
            return []
        candidates: List[str] = []
        if self.canonicalize_fn:
            canonicalized = self.canonicalize_fn(raw_code)
            candidates.extend(ensure_list(canonicalized))
        if self.fallback_to_raw:
            candidates.append(str(raw_code))
        # dedupe while preserving order
        seen = set()
        uniq = []
        for c in candidates:
            if c in seen:
                continue
            uniq.append(c)
            seen.add(c)
        return uniq

    def _encode_base(self, raw_code: Optional[str]) -> Optional[int]:
        if raw_code is None:
            return None if self.drop_unknowns else self.unk_gid

        cache_key = str(raw_code)
        if cache_key in self._cache:
            return self._cache[cache_key]

        for cand in self._candidate_codes(raw_code):
            gid = self.base_vocab.maybe_encode(cand)
            if gid is not None:
                self._cache[cache_key] = gid
                return gid

        if self.drop_unknowns:
            self._cache[cache_key] = None
            return None

        self._cache[cache_key] = self.unk_gid
        return self.unk_gid

    def _encode_categorical_attrs(self, ev: Any) -> Dict[str, int]:
        cat_attrs: Dict[str, int] = {}
        for attr_name, vocab in self.categorical_attrs.items():
            val = getattr(ev, attr_name, None)
            cat_attrs[attr_name] = vocab.encode(val)
        return cat_attrs

    def _encode_numeric_attrs(self, ev: Any) -> Dict[str, float]:
        num_attrs: Dict[str, float] = {}
        for attr_name, cfg in self.numeric_attrs.items():
            val = getattr(ev, attr_name, None)
            num_attrs[attr_name] = cfg.normalize(val)
        return num_attrs

    def encode_event(self, ev: Any, dt_hours: float) -> List[EventToken]:
        code = getattr(ev, "code", None)
        base_gid = self._encode_base(code)
        if base_gid is None:
            return []
        cat_attrs = self._encode_categorical_attrs(ev)
        num_attrs = self._encode_numeric_attrs(ev)

        return [
            EventToken(
                value_id=base_gid,
                category_id=int(self.category),
                t_from_start_hours=0.0,
                dt_from_prev_hours=float(dt_hours),
                cat_attrs=cat_attrs,
                num_attrs=num_attrs,
            )
        ]
