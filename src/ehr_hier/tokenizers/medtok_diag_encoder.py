from __future__ import annotations
from dataclasses import dataclass

from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_attr_encoder import MedTokenWithAttrsEncoder
from src.ehr_hier.tokenizers.medtok_canonicalize import canonicalize_diagnosis_code, diagnosis_filter
from src.ehr_hier.tokenizers.medtok_loader import build_vocab_from_code2embeddings, CategoryVocab


@dataclass
class MedTokDiagnosisConfig:
    code2embeddings_fp: str = "artifacts/medtok/code2embeddings.json"
    offset: int = 1_000_000       # put DIAG tokens far from measurement range
    drop_unknowns: bool = True    # drop if not in MedTok
    fallback_to_raw: bool = True  # still allow raw code lookup if canonicalization fails
    name: str = "diagnosis"


class MedTokDiagnosisEncoder:
    """
    MedTok-backed diagnosis encoder using local code2embeddings.json
    """
    category = TokenCategory.DIAGNOSIS

    def __init__(self, cfg: MedTokDiagnosisConfig):
        self.cfg = cfg
        self.vocab: CategoryVocab = build_vocab_from_code2embeddings(
            cfg.code2embeddings_fp,
            offset=cfg.offset,
            name=cfg.name,
            filter_fn=diagnosis_filter,
        )
        self.encoder = MedTokenWithAttrsEncoder(
            self.category,
            self.vocab,
            canonicalize_fn=canonicalize_diagnosis_code,
            drop_unknowns=cfg.drop_unknowns,
            fallback_to_raw=cfg.fallback_to_raw,
        )

    def reset_state(self) -> None:
        return self.encoder.reset_state()

    def encode_event(self, ev, dt_hours: float) -> list[EventToken]:
        return self.encoder.encode_event(ev, dt_hours)
