# src/ehr_hier/data/event_router.py
from typing import Optional
from .token_types import TokenCategory


def classify_code_to_category(code: Optional[str]) -> TokenCategory:
    """
    Map a MEDS event `code` string to a coarse TokenCategory.

    This is a heuristic based on MIMIC_IV_MEDS naming conventions.
    """
    if code is None:
        return TokenCategory.OTHER

    prefix = str(code).split("//", 1)[0].upper()

    # Numeric / measurement-like
    if prefix in {
        "LAB",
        "VITAL",
        "MEAS",
        "SUBJECT_FLUID_OUTPUT",
        "SUBJECT_WEIGHT_AT_INFUSION",
        "OMR",  # often numeric or categorical measurements
    }:
        return TokenCategory.MEASUREMENT

    # Diagnoses (ICD, etc.)
    if prefix.startswith("DIAGNOSIS") or prefix.startswith("ICD"):
        return TokenCategory.DIAGNOSIS

    # Procedures
    if prefix.startswith("PROCEDURE"):
        return TokenCategory.PROCEDURE

    # Medications and infusions
    if prefix.startswith("MEDICATION") or prefix.startswith("INFUSION"):
        return TokenCategory.MEDICATION

    # Structural / segment markers
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
