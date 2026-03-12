from __future__ import annotations
from typing import Dict, Optional, List
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
    medtok_parent_lookup: Optional[Dict[str, List[str]]] = None,
    medtok_crosswalks: Optional[Dict[str, Dict[str, str]]] = None,
    residual_fallback_vocabs: Optional[Dict[str, CategoryVocab]] = None,
    enable_residual_fallback: bool = False,
    residual_fallback_buckets: int = 40_000,
    residual_fallback_offsets: Optional[Dict[str, int]] = None,
    residual_tail_policy: Optional[str] = None,
    include_other_noop: bool = True,
    drop_unknowns: bool = False,
) -> Dict[TokenCategory, EventTokenEncoder]:
    """
    Build the default set of EventTokenEncoders for each TokenCategory.

    drop_unknowns controls whether MedTok-based encoders drop OOV codes (True)
    or emit UNK tokens (False, default for broader coverage).
    """
    encoders: Dict[TokenCategory, EventTokenEncoder] = {}

    diag_residual_offset = None
    proc_residual_offset = None
    med_residual_offset = None
    residual_vocabs = (residual_fallback_vocabs or {}) if enable_residual_fallback else {}
    if enable_residual_fallback:
        if residual_fallback_offsets:
            if "diagnosis" in residual_fallback_offsets:
                diag_residual_offset = int(residual_fallback_offsets["diagnosis"])
            if "procedure" in residual_fallback_offsets:
                proc_residual_offset = int(residual_fallback_offsets["procedure"])
            if "medication" in residual_fallback_offsets:
                med_residual_offset = int(residual_fallback_offsets["medication"])

        # If not explicitly provided, carve residual ranges from in-band slack
        # between diagnosis->procedure and procedure->medication offsets.
        if diag_residual_offset is None:
            diag_max_local = max(diag_vocab.code2id.values()) if diag_vocab.code2id else 0
            candidate = int(diag_vocab.offset) + int(diag_max_local) + 1_000
            if candidate + int(residual_fallback_buckets) < int(proc_vocab.offset):
                diag_residual_offset = candidate
        if proc_residual_offset is None:
            proc_max_local = max(proc_vocab.code2id.values()) if proc_vocab.code2id else 0
            candidate = int(proc_vocab.offset) + int(proc_max_local) + 1_000
            if candidate + int(residual_fallback_buckets) < int(med_vocab.offset):
                proc_residual_offset = candidate
        if med_residual_offset is None:
            med_max_local = max(med_vocab.code2id.values()) if med_vocab.code2id else 0
            candidate = int(med_vocab.offset) + int(med_max_local) + 1_000
            if candidate + int(residual_fallback_buckets) < int(struct_vocab.offset):
                med_residual_offset = candidate

    if "diagnosis" in residual_vocabs:
        diag_residual_offset = int(residual_vocabs["diagnosis"].offset)
    if "procedure" in residual_vocabs:
        proc_residual_offset = int(residual_vocabs["procedure"].offset)
    if "medication" in residual_vocabs:
        med_residual_offset = int(residual_vocabs["medication"].offset)

    encoders[TokenCategory.MEASUREMENT] = MeasurementTokenEncoder(meas_cfg)
    encoders[TokenCategory.DIAGNOSIS]   = MedTokenWithAttrsEncoder(
        TokenCategory.DIAGNOSIS,
        diag_vocab,
        canonicalize_fn=canonicalize_diagnosis_code,
        parent_lookup=medtok_parent_lookup,
        crosswalk_lookup=(medtok_crosswalks or {}).get("diagnosis"),
        residual_exact_vocab=residual_vocabs.get("diagnosis"),
        residual_fallback_offset=diag_residual_offset,
        residual_fallback_buckets=residual_fallback_buckets,
        residual_tail_policy=residual_tail_policy,
        drop_unknowns=drop_unknowns,
    )
    encoders[TokenCategory.PROCEDURE]   = MedTokenWithAttrsEncoder(
        TokenCategory.PROCEDURE,
        proc_vocab,
        canonicalize_fn=canonicalize_procedure_code,
        parent_lookup=medtok_parent_lookup,
        crosswalk_lookup=(medtok_crosswalks or {}).get("procedure"),
        residual_exact_vocab=residual_vocabs.get("procedure"),
        residual_fallback_offset=proc_residual_offset,
        residual_fallback_buckets=residual_fallback_buckets,
        residual_tail_policy=residual_tail_policy,
        drop_unknowns=drop_unknowns,
    )
    if med_attr_vocabs or med_numeric_attrs:
        encoders[TokenCategory.MEDICATION] = MedTokenWithAttrsEncoder(
            TokenCategory.MEDICATION,
            med_vocab,
            categorical_attrs=med_attr_vocabs or {},
            numeric_attrs=med_numeric_attrs or {},
            canonicalize_fn=canonicalize_medication_code,
            parent_lookup=medtok_parent_lookup,
            crosswalk_lookup=(medtok_crosswalks or {}).get("medication"),
            residual_exact_vocab=residual_vocabs.get("medication"),
            residual_fallback_offset=med_residual_offset,
            residual_fallback_buckets=residual_fallback_buckets,
            residual_tail_policy=residual_tail_policy,
            drop_unknowns=drop_unknowns,
        )
    else:
        encoders[TokenCategory.MEDICATION]  = MedTokenWithAttrsEncoder(
            TokenCategory.MEDICATION,
            med_vocab,
            canonicalize_fn=canonicalize_medication_code,
            parent_lookup=medtok_parent_lookup,
            crosswalk_lookup=(medtok_crosswalks or {}).get("medication"),
            residual_exact_vocab=residual_vocabs.get("medication"),
            residual_fallback_offset=med_residual_offset,
            residual_fallback_buckets=residual_fallback_buckets,
            residual_tail_policy=residual_tail_policy,
            drop_unknowns=drop_unknowns,
        )
    encoders[TokenCategory.STRUCTURAL]  = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL,  struct_vocab)
    if include_other_noop:
        encoders[TokenCategory.OTHER] = OtherNoOpEncoder()
    return encoders
