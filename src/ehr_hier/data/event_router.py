# src/ehr_hier/data/event_router.py
from typing import Optional
from .token_types import TokenCategory


_MEASUREMENT_OTHER_ALIASES = {
    "BLOOD PRESSURE",
}

_DEMOGRAPHIC_OTHER_ALIASES = {
    "BMI (KG/M2)",
    "WEIGHT (LBS)",
    "HEIGHT (INCHES)",
}

# Lightweight clinically informed signifiers to avoid dropping rare but
# high-acuity transitions that often land in OTHER on raw MEDS exports.
_RARE_CRITICAL_PATTERNS = (
    "CARDIAC ARREST",
    "CODE BLUE",
    "CPR",
    "DEFIB",
    "ROSC",
    "SEPSIS BUNDLE",
    "MASSIVE TRANSFUSION",
    "REINTUBATION",
    "SEPTIC SHOCK",
    "STROKE ALERT",
)


def _normalize_code(code: object) -> str:
    return str(code).strip().upper()


def _matches_alias(code_norm: str, aliases: set[str]) -> bool:
    if code_norm in aliases:
        return True
    return any(code_norm.endswith(f"//{alias}") for alias in aliases)


def is_rare_critical_other_code(code: Optional[str]) -> bool:
    if code is None:
        return False
    code_norm = _normalize_code(code)
    if not code_norm:
        return False
    # Do not reclassify explicitly modeled families.
    explicit_prefixes = {
        "LAB",
        "VITAL",
        "MEAS",
        "OMR",
        "MEDICATION",
        "INFUSION_START",
        "INFUSION_END",
        "PROCEDURE",
        "DIAGNOSIS",
        "ICD",
        "CPT",
        "HCPCS",
    }
    prefix = code_norm.split("//", 1)[0]
    if prefix in explicit_prefixes:
        return False
    return any(pattern in code_norm for pattern in _RARE_CRITICAL_PATTERNS)


def classify_code_to_category(code: Optional[str]) -> TokenCategory:
    """
    Map a MEDS event `code` string to a coarse TokenCategory.

    This is a prefix-based heuristic aligned to MEDS naming conventions.
    """
    if code is None:
        return TokenCategory.OTHER

    code_norm = _normalize_code(code)
    if _matches_alias(code_norm, _MEASUREMENT_OTHER_ALIASES):
        return TokenCategory.MEASUREMENT

    # Keep these in OTHER so they can be injected as stable global demographics
    # instead of high-frequency per-event timeline tokens.
    if _matches_alias(code_norm, _DEMOGRAPHIC_OTHER_ALIASES):
        return TokenCategory.OTHER

    if is_rare_critical_other_code(code_norm):
        return TokenCategory.STRUCTURAL

    prefix = code_norm.split("//", 1)[0]

    measurement_prefixes = {
        "LAB",
        "VITAL",
        "MEAS",
        "SUBJECT_FLUID_OUTPUT",
        "SUBJECT_WEIGHT_AT_INFUSION",
        "OMR",
    }
    if prefix in measurement_prefixes:
        return TokenCategory.MEASUREMENT

    if prefix.startswith("RXNORM") or prefix.startswith("NDC") or prefix.startswith("MEDICATION") or prefix.startswith("INFUSION"):
        return TokenCategory.MEDICATION

    if prefix.startswith("ICD10PCS") or prefix.startswith("ICD9PROC") or prefix.startswith("CPT") or prefix.startswith("PROCEDURE") or prefix.startswith("HCPCS"):
        return TokenCategory.PROCEDURE

    if prefix.startswith("ICD10CM") or prefix.startswith("ICD9CM") or prefix.startswith("DIAGNOSIS") or prefix.startswith("ICD"):
        return TokenCategory.DIAGNOSIS

    if prefix in {
        "MEDS_BIRTH",
        "MEDS_DEATH",
        "HOSPITAL_ADMISSION",
        "HOSPITAL_DISCHARGE",
        "ICU_ADMISSION",
        "ICU_DISCHARGE",
        "TRANSFER_TO",
        "ED_REGISTRATION",
        "ED_OUT",
    }:
        return TokenCategory.STRUCTURAL

    return TokenCategory.OTHER
