from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import re

from src.ehr_hier.data.event_frames import EventFrame
from src.ehr_hier.data.token_types import TokenCategory


class EventMemoryGroup(IntEnum):
    NONE = 0
    STRUCTURAL = 1
    CHRONIC_DIAGNOSIS = 2
    PROCEDURE = 3
    MEDICATION = 4
    EXTREME_MEASUREMENT = 5
    SYMBOLIC = 6


@dataclass(frozen=True)
class MemoryRuleFeatures:
    group_id: int
    rule_score: float
    first_occurrence: int
    chronic_flag: int
    structural_flag: int
    numeric_extreme_flag: int


_ICD10_CHRONIC_PREFIXES: tuple[str, ...] = (
    "B18",
    "C",
    "E10",
    "E11",
    "E13",
    "E66",
    "F20",
    "F25",
    "F31",
    "F32",
    "F33",
    "G20",
    "G30",
    "G35",
    "G40",
    "I10",
    "I11",
    "I12",
    "I13",
    "I20",
    "I21",
    "I22",
    "I23",
    "I24",
    "I25",
    "I42",
    "I48",
    "I50",
    "J44",
    "J45",
    "J47",
    "K50",
    "K51",
    "K74",
    "M05",
    "M06",
    "N18",
)

_ICD9_CHRONIC_PREFIXES: tuple[str, ...] = (
    "07022",
    "07023",
    "07032",
    "07033",
    "07044",
    "07054",
    "140",
    "141",
    "142",
    "143",
    "144",
    "145",
    "146",
    "147",
    "148",
    "149",
    "150",
    "151",
    "152",
    "153",
    "154",
    "155",
    "156",
    "157",
    "158",
    "159",
    "160",
    "161",
    "162",
    "163",
    "164",
    "165",
    "170",
    "171",
    "172",
    "174",
    "175",
    "176",
    "179",
    "180",
    "181",
    "182",
    "183",
    "184",
    "185",
    "186",
    "187",
    "188",
    "189",
    "190",
    "191",
    "192",
    "193",
    "194",
    "195",
    "196",
    "197",
    "198",
    "199",
    "200",
    "201",
    "202",
    "203",
    "204",
    "205",
    "206",
    "207",
    "208",
    "209",
    "250",
    "2780",
    "295",
    "296",
    "311",
    "332",
    "3310",
    "340",
    "345",
    "401",
    "402",
    "403",
    "404",
    "405",
    "410",
    "411",
    "412",
    "413",
    "414",
    "425",
    "42731",
    "428",
    "493",
    "496",
    "5712",
    "5715",
    "5716",
    "585",
    "714",
)

_STRUCTURAL_KEYWORDS: tuple[str, ...] = (
    "ADMISSION",
    "DISCHARGE",
    "TRANSFER",
    "ICU",
    "OR_",
    "OPERATING",
    "VENT",
    "MECH",
    "INTUB",
    "EXTUB",
    "RRT",
    "DIAL",
    "CPR",
    "SHOCK",
    "SEPSIS",
    "PRESSOR",
    "VASO",
    "CODE_STATUS",
    "TRACH",
    "ECMO",
)


def _canonical_code(text: str | None) -> str:
    if text is None:
        return ""
    code = str(text).strip().upper()
    if not code:
        return ""
    if "//" in code:
        code = code.split("//", 1)[1]
    if ":" in code:
        code = code.split(":", 1)[-1]
    code = re.sub(r"[^A-Z0-9.]", "", code)
    return code


def _code_key(frame: EventFrame) -> str:
    if frame.concept_code:
        return _canonical_code(frame.concept_code)
    if frame.source_code:
        return _canonical_code(frame.source_code)
    if frame.semantic_label:
        return str(frame.semantic_label).strip().upper()
    return str(int(frame.value_id))


def _is_icd10_chronic(code: str) -> bool:
    if not code or not code[0].isalpha():
        return False
    body = code.replace(".", "")
    return any(body.startswith(prefix) for prefix in _ICD10_CHRONIC_PREFIXES)


def _is_icd9_chronic(code: str) -> bool:
    if not code or not code[0].isdigit():
        return False
    body = code.replace(".", "")
    return any(body.startswith(prefix) for prefix in _ICD9_CHRONIC_PREFIXES)


def _structural_label(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).strip().upper()


def classify_event_frame_for_memory(
    *,
    frame: EventFrame,
    seen_exact_keys: set[str],
    numeric_extreme_threshold: float = 2.0,
) -> MemoryRuleFeatures:
    category = int(frame.category_id)
    exact_key = f"{category}:{_code_key(frame)}"
    first_occurrence = 1 if exact_key not in seen_exact_keys else 0

    chronic_flag = 0
    structural_flag = 0
    numeric_extreme_flag = 0
    group_id = int(EventMemoryGroup.NONE)
    score = 0.0

    label = _structural_label(frame.semantic_label or frame.source_code or frame.concept_code)
    code = _canonical_code(frame.concept_code or frame.source_code)
    if category == int(TokenCategory.STRUCTURAL):
        structural_flag = 1
        group_id = int(EventMemoryGroup.STRUCTURAL)
        score = 4.0
        if any(keyword in label for keyword in _STRUCTURAL_KEYWORDS):
            score += 0.75
        if first_occurrence:
            score += 0.5
    elif category == int(TokenCategory.DIAGNOSIS):
        if _is_icd10_chronic(code) or _is_icd9_chronic(code):
            chronic_flag = 1
            group_id = int(EventMemoryGroup.CHRONIC_DIAGNOSIS)
            score = 4.0
            if first_occurrence:
                score += 1.5
        else:
            group_id = int(EventMemoryGroup.SYMBOLIC)
            score = 1.75 + (1.0 if first_occurrence else 0.0)
    elif category == int(TokenCategory.PROCEDURE):
        group_id = int(EventMemoryGroup.PROCEDURE)
        score = 2.5 + (1.0 if first_occurrence else 0.0)
    elif category == int(TokenCategory.MEDICATION):
        group_id = int(EventMemoryGroup.MEDICATION)
        score = 1.75 + (0.75 if first_occurrence else 0.0)
    elif category == int(TokenCategory.MEASUREMENT):
        scalar = None
        for key in ("z", "numeric_value", "value_z", "value_std"):
            raw = frame.num_attrs.get(key) if frame.num_attrs is not None else None
            if raw is None:
                continue
            try:
                scalar = float(raw)
                break
            except (TypeError, ValueError):
                continue
        if scalar is not None and abs(scalar) >= float(numeric_extreme_threshold):
            numeric_extreme_flag = 1
            group_id = int(EventMemoryGroup.EXTREME_MEASUREMENT)
            score = 1.5 + min(2.5, abs(float(scalar)) / max(1e-6, float(numeric_extreme_threshold)))

    return MemoryRuleFeatures(
        group_id=int(group_id),
        rule_score=float(score),
        first_occurrence=int(first_occurrence),
        chronic_flag=int(chronic_flag),
        structural_flag=int(structural_flag),
        numeric_extreme_flag=int(numeric_extreme_flag),
    )


def update_seen_memory_keys(seen_exact_keys: set[str], frame: EventFrame) -> None:
    category = int(frame.category_id)
    if category == int(TokenCategory.SPECIAL):
        return
    seen_exact_keys.add(f"{category}:{_code_key(frame)}")
