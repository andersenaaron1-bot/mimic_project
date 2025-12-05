from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from pathlib import Path


@dataclass
class CategoryVocab:
    """
    Per-category MedTok-style vocabulary bound to a global offset.

    Global ID = offset + raw_id (with raw_id==unk_id for unknowns).
    """
    name: str
    offset: int
    code2id: Dict[str, int]
    unk_token: str = "<UNK>"
    unk_id: int = 0
    _sealed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Ensure UNK is present and consistent
        if self.unk_token not in self.code2id:
            self.code2id[self.unk_token] = self.unk_id
        self.code2id[self.unk_token] = int(self.code2id[self.unk_token])
        self._sealed = True

    def encode(self, code: Optional[str]) -> int:
        raw_id = self.code2id.get(str(code), self.unk_id)
        return self.offset + int(raw_id)

    def maybe_encode(self, code: Optional[str]) -> Optional[int]:
        """
        Return global ID if present in vocab, else None.
        """
        if code is None:
            return None
        if str(code) not in self.code2id:
            return None
        return self.offset + int(self.code2id[str(code)])


def load_medtok_vocab(json_fp: str, *, offset: int, name: str) -> CategoryVocab:
    """
    Load a tiny MedTok vocabulary JSON of the form:
      { "<MEDS_CODE>": <int>, "<UNK>": 0 }
    """
    with open(json_fp, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Normalize keys/values
    code2id = {str(k): int(v) for k, v in payload.items()}
    return CategoryVocab(name=name, offset=offset, code2id=code2id)


def load_attr_vocab(json_fp: str, *, offset: int, name: str) -> CategoryVocab:
    """
    Loader alias for categorical attribute vocabularies (route/form/freq/unit).
    Same JSON shape as MedTok vocab.
    """
    return load_medtok_vocab(json_fp, offset=offset, name=name)


def load_code_embeddings(json_fp: str) -> Dict[str, list]:
    """
    Load MedTok code2embeddings.json mapping {<code>: [float...]}.
    """
    with open(json_fp, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(k): v for k, v in payload.items()}


def build_vocab_from_code2embeddings(
    json_fp: str,
    *,
    offset: int,
    name: str,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> CategoryVocab:
    """
    Build CategoryVocab from code2embeddings.json.

    - Uses keys from the embeddings file as the canonical code list.
    - Ensures <UNK> exists at id 0 (adds if missing).
    - Applies optional filter_fn(code)->bool to keep a subset (e.g., DIAG/PROC/MED).
    - Assigns deterministic raw IDs by sorting codes (except <UNK> which is 0).
    """
    path = Path(json_fp)
    if not path.exists():
        raise FileNotFoundError(f"code2embeddings file not found: {json_fp}")
    code2emb = load_code_embeddings(json_fp)

    codes = list(code2emb.keys())
    if filter_fn:
        codes = [c for c in codes if filter_fn(c)]

    # Ensure UNK
    if "<UNK>" not in codes:
        codes.append("<UNK>")

    codes_uniq = []
    seen = set()
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        codes_uniq.append(c)

    # UNK gets 0, rest sorted
    rest = sorted([c for c in codes_uniq if c != "<UNK>"])
    ordered = ["<UNK>"] + rest
    code2id = {c: i for i, c in enumerate(ordered)}
    return CategoryVocab(name=name, offset=offset, code2id=code2id)
