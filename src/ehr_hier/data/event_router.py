# src/ehr_hier/data/event_router.py
from typing import Optional
from .token_types import TokenCategory


def classify_code_to_category(code: Optional[str]) -> TokenCategory:
    """
    Map a MEDS event `code` string to a coarse TokenCategory.

    This is a prefix-based heuristic aligned to MEDS naming conventions.
    """
    if code is None:
        return TokenCategory.OTHER

    prefix = str(code).split("//", 1)[0].upper()

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
