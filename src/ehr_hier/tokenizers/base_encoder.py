from __future__ import annotations
from typing import Dict
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.tokenizers.measurement_encoder import (
    MeasurementTokenEncoder, MeasurementEncoderConfig
)
from src.ehr_hier.tokenizers.simple_categorical_encoders import (
    SimpleCategoricalEncoder, CategoryVocab, OtherNoOpEncoder,
)

def build_base_encoders(
    meas_cfg: MeasurementEncoderConfig,
    *,
    diag_vocab: CategoryVocab,
    proc_vocab: CategoryVocab,
    med_vocab: CategoryVocab,
    struct_vocab: CategoryVocab,
    include_other_noop: bool = True,
) -> Dict[TokenCategory, EventTokenEncoder]:
    encoders: Dict[TokenCategory, EventTokenEncoder] = {}

    encoders[TokenCategory.MEASUREMENT] = MeasurementTokenEncoder(meas_cfg)
    encoders[TokenCategory.DIAGNOSIS]   = SimpleCategoricalEncoder(TokenCategory.DIAGNOSIS,   diag_vocab)
    encoders[TokenCategory.PROCEDURE]   = SimpleCategoricalEncoder(TokenCategory.PROCEDURE,   proc_vocab)
    encoders[TokenCategory.MEDICATION]  = SimpleCategoricalEncoder(TokenCategory.MEDICATION,  med_vocab)
    encoders[TokenCategory.STRUCTURAL]  = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL,  struct_vocab)
    if include_other_noop:
        encoders[TokenCategory.OTHER] = OtherNoOpEncoder()
    return encoders

