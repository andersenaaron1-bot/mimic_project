from __future__ import annotations
from typing import Dict, Optional
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.tokenizers.measurement_encoder import (
    MeasurementTokenEncoder, MeasurementEncoderConfig
)
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.tokenizers.simple_categorical_encoders import (
    SimpleCategoricalEncoder, OtherNoOpEncoder,
)
from src.ehr_hier.tokenizers.medtok_attr_encoder import MedTokenWithAttrsEncoder
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_procedure_code,
    canonicalize_medication_code,
)
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig

def build_base_encoders(
    meas_cfg: MeasurementEncoderConfig,
    *,
    diag_vocab: CategoryVocab,
    proc_vocab: CategoryVocab,
    med_vocab: CategoryVocab,
    struct_vocab: CategoryVocab,
    med_attr_vocabs: Optional[Dict[str, CategoryVocab]] = None,
    med_numeric_attrs: Optional[Dict[str, NumericBinConfig]] = None,
    include_other_noop: bool = True,
    drop_unknowns: bool = False,
) -> Dict[TokenCategory, EventTokenEncoder]:
    """
    Build the default set of EventTokenEncoders for each TokenCategory.

    drop_unknowns controls whether MedTok-based encoders drop OOV codes (True)
    or emit UNK tokens (False, default for broader coverage).
    """
    encoders: Dict[TokenCategory, EventTokenEncoder] = {}

    encoders[TokenCategory.MEASUREMENT] = MeasurementTokenEncoder(meas_cfg)
    encoders[TokenCategory.DIAGNOSIS]   = MedTokenWithAttrsEncoder(
        TokenCategory.DIAGNOSIS,
        diag_vocab,
        canonicalize_fn=canonicalize_diagnosis_code,
        drop_unknowns=drop_unknowns,
    )
    encoders[TokenCategory.PROCEDURE]   = MedTokenWithAttrsEncoder(
        TokenCategory.PROCEDURE,
        proc_vocab,
        canonicalize_fn=canonicalize_procedure_code,
        drop_unknowns=drop_unknowns,
    )
    if med_attr_vocabs or med_numeric_attrs:
        encoders[TokenCategory.MEDICATION] = MedTokenWithAttrsEncoder(
            TokenCategory.MEDICATION,
            med_vocab,
            categorical_attrs=med_attr_vocabs or {},
            numeric_attrs=med_numeric_attrs or {},
            canonicalize_fn=canonicalize_medication_code,
        )
    else:
        encoders[TokenCategory.MEDICATION]  = MedTokenWithAttrsEncoder(
            TokenCategory.MEDICATION,
            med_vocab,
            canonicalize_fn=canonicalize_medication_code,
        )
    encoders[TokenCategory.STRUCTURAL]  = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL,  struct_vocab)
    if include_other_noop:
        encoders[TokenCategory.OTHER] = OtherNoOpEncoder()
    return encoders
