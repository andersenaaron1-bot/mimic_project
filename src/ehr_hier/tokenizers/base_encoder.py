from __future__ import annotations

from typing import Dict, List, Optional

from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.codecs.measurement_numeric import MeasurementEventFrameCodec
from src.ehr_hier.tokenizers.codecs.semantic_symbolic import (
    OtherNoOpFrameCodec,
    SymbolicEventFrameCodec,
)
from src.ehr_hier.tokenizers.interfaces import EventFrameEncoder
from src.ehr_hier.tokenizers.measurement_encoder import (
    MeasurementEncoderConfig,
    MeasurementTokenEncoder,
)
from src.ehr_hier.tokenizers.medtok_attr_encoder import MedTokenWithAttrsEncoder
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_medication_code,
    canonicalize_procedure_code,
)
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig
from src.ehr_hier.tokenizers.simple_categorical_encoders import (
    OtherNoOpEncoder,
    SimpleCategoricalEncoder,
)


def build_base_codecs(
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
    residual_tail_policies: Optional[Dict[str, str]] = None,
    include_other_noop: bool = True,
    drop_unknowns: bool = False,
    qual_obs_code_vocab: CategoryVocab | None = None,
    qual_obs_value_vocab: CategoryVocab | None = None,
    qual_obs_tail_policy: str = "drop",
    qual_obs_code_offset: int = 2_300_000,
    qual_obs_value_offset: int = 2_320_000,
    qual_obs_value_vocab_size: int = 80_000,
) -> Dict[TokenCategory, EventFrameEncoder]:
    """
    Build the default frame-native codecs for each token family.

    Atomic token encoders are still used internally where appropriate, but
    `EventFrame` is the canonical timeline interface returned by these codecs.
    """

    encoders: Dict[TokenCategory, EventFrameEncoder] = {}

    def _resolve_tail_policy(family: str) -> Optional[str]:
        if residual_tail_policies and family in residual_tail_policies:
            return str(residual_tail_policies[family])
        return residual_tail_policy

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

    encoders[TokenCategory.MEASUREMENT] = MeasurementEventFrameCodec(
        meas_cfg,
        numeric_token_encoder=MeasurementTokenEncoder(meas_cfg),
        qual_obs_code_offset=int(qual_obs_code_offset),
        qual_obs_value_offset=int(qual_obs_value_offset),
        qual_obs_value_vocab_size=int(qual_obs_value_vocab_size),
        qual_obs_code_vocab=qual_obs_code_vocab,
        qual_obs_value_vocab=qual_obs_value_vocab,
        qual_obs_tail_policy=str(qual_obs_tail_policy),
    )
    encoders[TokenCategory.DIAGNOSIS] = SymbolicEventFrameCodec(
        category=TokenCategory.DIAGNOSIS,
        token_encoder=MedTokenWithAttrsEncoder(
            TokenCategory.DIAGNOSIS,
            diag_vocab,
            canonicalize_fn=canonicalize_diagnosis_code,
            parent_lookup=medtok_parent_lookup,
            crosswalk_lookup=(medtok_crosswalks or {}).get("diagnosis"),
            residual_exact_vocab=residual_vocabs.get("diagnosis"),
            residual_fallback_offset=diag_residual_offset,
            residual_fallback_buckets=residual_fallback_buckets,
            residual_tail_policy=_resolve_tail_policy("diagnosis"),
            drop_unknowns=drop_unknowns,
        ),
    )
    encoders[TokenCategory.PROCEDURE] = SymbolicEventFrameCodec(
        category=TokenCategory.PROCEDURE,
        token_encoder=MedTokenWithAttrsEncoder(
            TokenCategory.PROCEDURE,
            proc_vocab,
            canonicalize_fn=canonicalize_procedure_code,
            parent_lookup=medtok_parent_lookup,
            crosswalk_lookup=(medtok_crosswalks or {}).get("procedure"),
            residual_exact_vocab=residual_vocabs.get("procedure"),
            residual_fallback_offset=proc_residual_offset,
            residual_fallback_buckets=residual_fallback_buckets,
            residual_tail_policy=_resolve_tail_policy("procedure"),
            drop_unknowns=drop_unknowns,
        ),
    )
    encoders[TokenCategory.MEDICATION] = SymbolicEventFrameCodec(
        category=TokenCategory.MEDICATION,
        token_encoder=MedTokenWithAttrsEncoder(
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
            residual_tail_policy=_resolve_tail_policy("medication"),
            drop_unknowns=drop_unknowns,
        ),
    )
    encoders[TokenCategory.STRUCTURAL] = SymbolicEventFrameCodec(
        category=TokenCategory.STRUCTURAL,
        token_encoder=SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab),
    )
    if include_other_noop:
        encoders[TokenCategory.OTHER] = OtherNoOpFrameCodec(
            token_encoder=OtherNoOpEncoder(),
        )
    return encoders


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
    residual_tail_policies: Optional[Dict[str, str]] = None,
    include_other_noop: bool = True,
    drop_unknowns: bool = False,
    qual_obs_code_vocab: CategoryVocab | None = None,
    qual_obs_value_vocab: CategoryVocab | None = None,
    qual_obs_tail_policy: str = "drop",
    qual_obs_code_offset: int = 2_300_000,
    qual_obs_value_offset: int = 2_320_000,
    qual_obs_value_vocab_size: int = 80_000,
) -> Dict[TokenCategory, EventFrameEncoder]:
    """
    Backward-compatible factory name for the new frame-native codec set.
    """

    return build_base_codecs(
        meas_cfg,
        diag_vocab=diag_vocab,
        proc_vocab=proc_vocab,
        med_vocab=med_vocab,
        struct_vocab=struct_vocab,
        med_attr_vocabs=med_attr_vocabs,
        med_numeric_attrs=med_numeric_attrs,
        medtok_parent_lookup=medtok_parent_lookup,
        medtok_crosswalks=medtok_crosswalks,
        residual_fallback_vocabs=residual_fallback_vocabs,
        enable_residual_fallback=enable_residual_fallback,
        residual_fallback_buckets=residual_fallback_buckets,
        residual_fallback_offsets=residual_fallback_offsets,
        residual_tail_policy=residual_tail_policy,
        residual_tail_policies=residual_tail_policies,
        include_other_noop=include_other_noop,
        drop_unknowns=drop_unknowns,
        qual_obs_code_vocab=qual_obs_code_vocab,
        qual_obs_value_vocab=qual_obs_value_vocab,
        qual_obs_tail_policy=qual_obs_tail_policy,
        qual_obs_code_offset=qual_obs_code_offset,
        qual_obs_value_offset=qual_obs_value_offset,
        qual_obs_value_vocab_size=qual_obs_value_vocab_size,
    )
