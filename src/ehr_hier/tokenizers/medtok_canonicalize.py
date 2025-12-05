from __future__ import annotations
import re
from typing import Iterable, List, Optional, Callable

"""
Canonicalization helpers for MedTok-backed categories.

These functions try to recover standardized code strings that match MedTok
`code2tokens.json` entries. They emit multiple candidates to maximize hit rate
before falling back to raw codes upstream.
"""

# Diagnosis (ICD) -------------------------------------------------------------

_ICD10_RE = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b")   # ICD-10-CM (no 'U')
_ICD9_RE = re.compile(r"\b([0-9]{3}(?:\.[0-9A-Z]{1,2})?)\b")                  # ICD-9-CM


def extract_icd_from_meds_code(code_str: str) -> Optional[str]:
    """
    Try to recover a clean ICD-10/ICD-9 code from a MEDS code string.
    Returns e.g. 'E11.9' or '250.00' if found, else None.
    """
    s = code_str.upper().replace("ICD-10", "ICD10").replace("ICD-9", "ICD9")
    # Prefer explicit markers if present
    m = re.search(r"ICD10\w*[:/\\|-]*([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)", s)
    if m:
        return m.group(1)
    m = re.search(r"ICD9\w*[:/\\|-]*([0-9]{3}(?:\.[0-9A-Z]{1,2})?)", s)
    if m:
        return m.group(1)
    # Generic fallbacks
    m = _ICD10_RE.search(s)
    if m:
        return m.group(1)
    m = _ICD9_RE.search(s)
    if m:
        return m.group(1)
    return None


def canonicalize_diagnosis_code(code: Optional[str]) -> List[str]:
    """
    Yield MedTok-style diagnosis codes. Order matters for lookup.
    """
    if code is None:
        return []
    s = str(code).upper()
    # MEDS-style prefixes: DIAGNOSIS//ICD//9//<code> or DIAGNOSIS//ICD//10//<code>
    if s.startswith("DIAGNOSIS//ICD//9//"):
        icd_raw = s.split("//")[-1]
        icd = icd_raw
    elif s.startswith("DIAGNOSIS//ICD//10//"):
        icd_raw = s.split("//")[-1]
        icd = icd_raw
    else:
        icd = extract_icd_from_meds_code(s)
    if not icd:
        return []
    icd_up = icd.upper()
    no_dot = icd_up.replace(".", "")
    dotted = icd_up if "." in icd_up else (icd_up[:3] + "." + icd_up[3:] if len(icd_up) > 3 else icd_up)

    # MedTok keys commonly look like ICD10CM//A123 or ICD9CM//25000
    if icd_up[0].isalpha():  # ICD-10
        return list(dict.fromkeys([  # dedupe while preserving order
            f"ICD10CM//{icd_up}",
            f"ICD10CM//{no_dot}",
            f"ICD10CM//{dotted}",
            icd_up,
            no_dot,
            dotted,
        ]))
    # ICD-9
    return list(dict.fromkeys([
        f"ICD9CM//{icd_up}",
        f"ICD9CM//{no_dot}",
        f"ICD9CM//{dotted}",
        icd_up,
        no_dot,
        dotted,
    ]))


# Procedure -------------------------------------------------------------------

_ICD10PCS_RE = re.compile(r"\b([0-9A-HJ-NP-Z]{7})\b")  # 7 chars, excludes I/O
_CPT_RE = re.compile(r"\b(\d{4,5}[A-Z]?)\b")          # 4-5 digits plus optional letter
_ICD9PROC_RE = re.compile(r"\b(\d{2}\.\d{1,2}|\d{3,4})\b")


def canonicalize_procedure_code(code: Optional[str]) -> List[str]:
    """
    Emit likely MedTok procedure codes (ICD10PCS, CPT, ICD9PROC).
    """
    if code is None:
        return []
    s = str(code).upper()
    # MEDS-style prefixes: PROCEDURE//ICD//9//<code>, PROCEDURE//ICD//10//<code>, PROCEDURE//CPT//<code>
    if s.startswith("PROCEDURE//ICD//9//"):
        icd9 = s.split("//")[-1]
        no_dot = icd9.replace(".", "")
        return [f"ICD9PROC//{icd9}", f"ICD9PROC//{no_dot}", icd9, no_dot]
    if s.startswith("PROCEDURE//ICD//10//"):
        pcs = s.split("//")[-1]
        return [f"ICD10PCS//{pcs}", pcs]
    if s.startswith("PROCEDURE//CPT//"):
        cpt = s.split("//")[-1]
        return [f"CPT//{cpt}", cpt]

    cands: List[str] = []
    m = _ICD10PCS_RE.search(s)
    if m:
        pcs = m.group(1)
        cands.extend([f"ICD10PCS//{pcs}", pcs])
    m = _CPT_RE.search(s)
    if m:
        cpt = m.group(1)
        cands.extend([f"CPT//{cpt}", cpt])
    m = _ICD9PROC_RE.search(s)
    if m:
        icd9 = m.group(1)
        no_dot = icd9.replace(".", "")
        cands.extend([f"ICD9PROC//{icd9}", f"ICD9PROC//{no_dot}", icd9, no_dot])
    return list(dict.fromkeys(cands))  # dedupe, preserve order


# Medication ------------------------------------------------------------------

_RXNORM_RE = re.compile(r"RXNORM\W*([0-9]{3,9})", re.IGNORECASE)
_NDC_RE = re.compile(r"NDC\W*([0-9]{4,5}-?[0-9]{3,4}-?[0-9]{1,2})", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\b([0-9]{3,11})\b")


def _normalize_ndc(ndc: str) -> str:
    """Normalize NDC to hyphen-less 11 digits when possible."""
    digits = re.sub(r"\D", "", ndc)
    if len(digits) == 10:  # pad to 11 using FDA segments (simple heuristic)
        return digits[0].zfill(5) + digits[1:5].zfill(4) + digits[5:].zfill(2)
    return digits


def canonicalize_medication_code(code: Optional[str]) -> List[str]:
    """
    Emit likely MedTok medication codes (RXNORM/NDC).
    """
    if code is None:
        return []
    s = str(code).upper()
    cands: List[str] = []

    m = _RXNORM_RE.search(s)
    if m:
        rx = m.group(1)
        cands.extend([f"RXNORM//{rx}", rx])

    m = _NDC_RE.search(s)
    if m:
        ndc_raw = m.group(1)
        ndc_norm = _normalize_ndc(ndc_raw)
        cands.extend(
            [
                f"NDC//{ndc_raw}",
                f"NDC//{ndc_norm}",
                ndc_raw,
                ndc_norm,
            ]
        )

    # As a fallback, capture standalone numeric codes that might be RxNorm
    if not cands:
        m = _NUMERIC_RE.search(s)
        if m:
            num = m.group(1)
            cands.extend([f"RXNORM//{num}", num])

    return list(dict.fromkeys(cands))


# Utility ---------------------------------------------------------------------

def ensure_list(maybe_iter: Optional[Iterable[str]]) -> List[str]:
    if maybe_iter is None:
        return []
    if isinstance(maybe_iter, str):
        return [maybe_iter] if maybe_iter else []
    return [c for c in maybe_iter if c]


# Filtering helpers for building vocabs --------------------------------------

def diagnosis_filter(code: str) -> bool:
    c = code.upper()
    if c.startswith("DIAGNOSIS//ICD//") or c.startswith("ICD10CM") or c.startswith("ICD9CM"):
        return True
    if any(prefix in c for prefix in ("MEDICATION//", "INFUSION", "PROCEDURE//", "DRG//", "CPT//", "HCPCS")):
        return False
    if c.split("//", 1)[0] in {"LAB", "VITAL", "MEAS", "SUBJECT_FLUID_OUTPUT", "SUBJECT_WEIGHT_AT_INFUSION", "OMR"}:
        return False
    seg = c.split("//")[-1] if "//" in c else c
    if _ICD10_RE.fullmatch(seg) or _ICD10_RE.search(seg):
        return True
    if _ICD9_RE.fullmatch(seg) or _ICD9_RE.search(seg):
        return True
    return False


def procedure_filter(code: str) -> bool:
    c = code.upper()
    if c.startswith(("PROCEDURE//", "ICD10PCS", "ICD9PROC", "CPT", "HCPCS")):
        return True
    if any(prefix in c for prefix in ("MEDICATION//", "INFUSION", "DRG//")):
        return False
    if c.split("//", 1)[0] in {"LAB", "VITAL", "MEAS", "SUBJECT_FLUID_OUTPUT", "SUBJECT_WEIGHT_AT_INFUSION", "OMR"}:
        return False
    seg = c.split("//")[-1] if "//" in c else c
    if not any(ch.isdigit() for ch in seg):
        return False
    if _ICD10PCS_RE.fullmatch(seg):
        return True
    if _ICD9PROC_RE.fullmatch(seg):
        return True
    return False


def medication_filter(code: str) -> bool:
    c = code.upper()
    if c.startswith(("RXNORM", "NDC", "MEDICATION//", "INFUSION")):
        return True
    if c.split("//", 1)[0] in {"LAB", "VITAL", "MEAS", "SUBJECT_FLUID_OUTPUT", "SUBJECT_WEIGHT_AT_INFUSION", "OMR", "PROCEDURE"}:
        return False
    seg = c.split("//")[-1] if "//" in c else c
    if _NDC_RE.fullmatch(seg) or _NDC_RE.search(seg):
        return True
    if _NUMERIC_RE.fullmatch(seg):
        return True
    return False
