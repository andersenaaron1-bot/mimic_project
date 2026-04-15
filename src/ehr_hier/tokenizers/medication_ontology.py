from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab


MEDICATION_GROUP_PREFIX = "MED_GROUP::"
DEFAULT_MED_GROUP_OFFSET = 1_670_000

MEDICATION_CODE_SYSTEM_TO_ID: Dict[str, int] = {
    "UNK": 0,
    "MEDTOK": 1,
    "FORMULARY": 2,
    "PRODUCT_CODE": 3,
    "NDC": 4,
    "GSN": 5,
    "GENERIC": 6,
    "RESIDUAL_SURFACE": 7,
    "MEDICATION_SURFACE": 8,
    "INFUSION_SURFACE": 9,
}

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")
_STRENGTH_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:MG|MCG|G|GM|GRAM|GRAMS|ML|L|MEQ|MMOL|UNITS?|UNIT|%)\b"
)
_FORM_RE = re.compile(
    r"\b(?:TAB(?:LET)?S?|CAPS?(?:ULE)?S?|INJ(?:ECTION)?S?|SYR(?:INGE)?S?|SOL(?:UTION)?S?|"
    r"SUSP(?:ENSION)?S?|PATCH(?:ES)?|CREAM|OINT(?:MENT)?|ELIXIR|DROP(?:S)?|SPRAY|POWDER|BOLUS|BAGS?)\b"
)
_ROUTE_RE = re.compile(
    r"\b(?:PO|IV|IM|SC|SQ|SL|PR|TOPICAL|INHAL(?:ED|ATION)?|INTRAVENOUS|ORAL|NASAL|OPHTHALMIC|OTIC)\b"
)
_ACTION_SUFFIX_RE = re.compile(
    r"\b(?:ADMINISTERED|NOT ADMINISTERED|NOT GIVEN(?: PER SLIDING SCALE)?|CONFIRMED|FLUSHED|"
    r"NOT FLUSHED|STARTED|STOPPED|HELD|REFUSED|PAUSED|RESUMED|START|STOP|END)\b"
)
_TRAILING_SEP_RE = re.compile(r"(?:\s+|//)+$")

_GROUP_TEXT_FIELD_ALIASES: tuple[tuple[str, ...], ...] = (
    ("drug_name_generic", "generic_name", "generic_drug_name"),
    ("medication", "drug", "drug_name", "label", "name"),
    ("formulary_drug_cd", "product_code"),
    ("ndc",),
    ("gsn",),
)
_CATEGORICAL_ATTR_ALIASES: Mapping[str, tuple[str, ...]] = {
    "route": ("route",),
    "form": ("form", "dosage_form"),
    "freq": ("freq", "frequency"),
    "unit": ("unit", "dose_unit", "amountuom", "rate_unit"),
}
_NUMERIC_ATTR_ALIASES: Mapping[str, tuple[str, ...]] = {
    "dosage": ("dosage", "dose", "dose_val_rx", "dose_value", "amount", "amount_value"),
    "rate": ("rate", "infusion_rate", "rate_value", "order_rate"),
    "duration_hours": (
        "duration_hours",
        "duration_h",
        "duration_hrs",
        "duration",
        "infusion_duration_hours",
        "hours",
    ),
}


@dataclass(frozen=True)
class MedicationSemanticDescriptor:
    exact_concept_code: Optional[str]
    group_code: Optional[str]
    semantic_label: Optional[str]
    code_system: str
    categorical_attrs: Dict[str, str]
    numeric_attrs: Dict[str, float]


def _strip_marker_and_action(raw: object) -> str:
    text = str(raw).strip()
    upper = text.upper()
    if upper.startswith("INFUSION_START//"):
        return f"INFUSION//{text[len('INFUSION_START//'):]}"
    if upper.startswith("INFUSION_END//"):
        return f"INFUSION//{text[len('INFUSION_END//'):]}"
    if upper.startswith("MEDICATION//START//"):
        return f"MEDICATION//{text[len('MEDICATION//START//'):]}"
    if upper.startswith("MEDICATION//END//"):
        return f"MEDICATION//{text[len('MEDICATION//END//'):]}"
    if upper.startswith("MEDICATION//STOP//"):
        return f"MEDICATION//{text[len('MEDICATION//STOP//'):]}"
    text = re.sub(r"//(?:START|END|STOP)//", "//", text, flags=re.IGNORECASE)
    text = _ACTION_SUFFIX_RE.sub("", text)
    return _TRAILING_SEP_RE.sub("", text).strip()


def _iter_text_candidates(text: object) -> List[str]:
    raw = _strip_marker_and_action(text)
    if not raw:
        return []
    parts = [part.strip() for part in raw.split("//") if part.strip()]
    candidates: List[str] = [raw]
    if parts:
        candidates.append(parts[-1])
        if len(parts) >= 2 and parts[0].upper() in {"MEDICATION", "INFUSION"}:
            candidates.append(parts[1])
    seen: set[str] = set()
    out: List[str] = []
    for item in candidates:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def normalize_medication_group_surface(value: object) -> str:
    text = str(value).strip().upper()
    if not text:
        return ""
    if text.startswith(MEDICATION_GROUP_PREFIX):
        text = text[len(MEDICATION_GROUP_PREFIX) :]
    for candidate in _iter_text_candidates(text):
        cleaned = _NON_ALNUM_RE.sub(" ", candidate.upper())
        cleaned = _STRENGTH_RE.sub(" ", cleaned)
        cleaned = _FORM_RE.sub(" ", cleaned)
        cleaned = _ROUTE_RE.sub(" ", cleaned)
        cleaned = _ACTION_SUFFIX_RE.sub(" ", cleaned)
        cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
        if cleaned and any(ch.isalpha() for ch in cleaned):
            return cleaned
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def medication_group_code_from_surface(value: object) -> str:
    normalized = normalize_medication_group_surface(value)
    if not normalized:
        normalized = _NON_ALNUM_RE.sub(" ", str(value).strip().upper()).strip()
    return f"{MEDICATION_GROUP_PREFIX}{normalized}" if normalized else ""


def build_medication_group_vocab(
    *,
    med_vocab: CategoryVocab,
    residual_vocab: CategoryVocab | None = None,
    offset: int = DEFAULT_MED_GROUP_OFFSET,
) -> CategoryVocab:
    codes: List[str] = []
    for vocab in (med_vocab, residual_vocab):
        if vocab is None:
            continue
        codes.extend(str(code) for code in vocab.code2id.keys() if str(code) != vocab.unk_token)
    group_codes = sorted(
        {
            group_code
            for code in codes
            for group_code in medication_group_codes_from_strings([code])
            if group_code
        }
    )
    code2id = {"<UNK>": 0}
    for idx, code in enumerate(group_codes, start=1):
        code2id[str(code)] = int(idx)
    return CategoryVocab(name="med_group", offset=int(offset), code2id=code2id)


def medication_group_codes_from_strings(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        for candidate in _iter_text_candidates(value):
            group_code = medication_group_code_from_surface(candidate)
            if not group_code or group_code in seen:
                continue
            seen.add(group_code)
            out.append(group_code)
    return out


def build_medication_group_vocab_from_files(
    *,
    med_vocab_json: str | Path | None,
    residual_vocab_json: str | Path | None = None,
    offset: int = DEFAULT_MED_GROUP_OFFSET,
) -> CategoryVocab:
    def _load_codes(path: str | Path | None) -> List[str]:
        if path is None:
            return []
        fp = Path(path)
        if not fp.exists():
            return []
        payload = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        return [str(code) for code in payload.keys() if str(code) != "<UNK>"]

    codes = _load_codes(med_vocab_json) + _load_codes(residual_vocab_json)
    group_codes = sorted(
        {
            group_code
            for group_code in medication_group_codes_from_strings(codes)
            if group_code
        }
    )
    code2id = {"<UNK>": 0}
    for idx, code in enumerate(group_codes, start=1):
        code2id[str(code)] = int(idx)
    return CategoryVocab(name="med_group", offset=int(offset), code2id=code2id)


def medication_group_size_from_vocab_files(
    *,
    med_vocab_json: str | Path | None,
    residual_vocab_json: str | Path | None = None,
) -> int:
    vocab = build_medication_group_vocab_from_files(
        med_vocab_json=med_vocab_json,
        residual_vocab_json=residual_vocab_json,
    )
    return int(max(vocab.code2id.values()) + 1 if vocab.code2id else 0)


def _safe_getattr(ev: Any, aliases: Sequence[str]) -> Optional[Any]:
    for name in aliases:
        if hasattr(ev, name):
            value = getattr(ev, name)
            if value is not None:
                return value
    return None


def _as_nonempty_str(value: object | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_finite_float(value: object | None) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _infer_code_system(
    *,
    matched_code: Optional[str],
    source_code: Optional[str],
    resolution_stage: Optional[str],
) -> str:
    code = str(matched_code or source_code or "").strip().upper()
    if resolution_stage in {"exact", "canonicalized", "parent_lookup", "crosswalk_lookup", "lexical_bridge"}:
        if code.startswith("FORMULARY::"):
            return "FORMULARY"
        if code.startswith("PRODUCT_CODE::"):
            return "PRODUCT_CODE"
        if code.startswith("NDC//") or code.startswith("NDC::"):
            return "NDC"
        if code.startswith("GSN//") or code.startswith("GSN::"):
            return "GSN"
        if code.startswith("GENERIC::"):
            return "GENERIC"
        return "MEDTOK"
    if resolution_stage == "residual_exact":
        return "RESIDUAL_SURFACE"
    if code.startswith("MEDICATION//"):
        return "MEDICATION_SURFACE"
    if code.startswith("INFUSION//"):
        return "INFUSION_SURFACE"
    return "UNK"


def build_medication_semantic_descriptor(
    *,
    ev: Any,
    matched_code: Optional[str],
    source_code: Optional[str],
    resolution_stage: Optional[str],
    marker: Optional[str],
    med_group_vocab: CategoryVocab | None,
    categorical_attr_vocabs: Mapping[str, CategoryVocab],
    numeric_attr_cfgs: Mapping[str, NumericBinConfig],
) -> MedicationSemanticDescriptor:
    exact_concept_code = (
        _as_nonempty_str(matched_code)
        or _as_nonempty_str(source_code)
        or _as_nonempty_str(getattr(ev, "code", None))
    )
    code_system = _infer_code_system(
        matched_code=exact_concept_code,
        source_code=source_code,
        resolution_stage=resolution_stage,
    )

    group_candidates: List[str] = []
    for aliases in _GROUP_TEXT_FIELD_ALIASES:
        raw = _safe_getattr(ev, aliases)
        if raw is None:
            continue
        group_candidates.extend(medication_group_codes_from_strings([str(raw)]))
    if exact_concept_code is not None:
        group_candidates.extend(medication_group_codes_from_strings([str(exact_concept_code)]))
    if source_code is not None:
        group_candidates.extend(medication_group_codes_from_strings([str(source_code)]))
    if getattr(ev, "code", None) is not None:
        group_candidates.extend(medication_group_codes_from_strings([str(getattr(ev, "code"))]))

    group_code: Optional[str] = None
    if med_group_vocab is not None:
        for candidate in group_candidates:
            if candidate in med_group_vocab.code2id:
                group_code = candidate
                break
    if group_code is None and group_candidates:
        group_code = group_candidates[0]
    semantic_label = None
    if group_code is not None and group_code.startswith(MEDICATION_GROUP_PREFIX):
        semantic_label = group_code[len(MEDICATION_GROUP_PREFIX) :]
    elif exact_concept_code is not None:
        semantic_label = normalize_medication_group_surface(exact_concept_code) or exact_concept_code

    categorical_attrs: Dict[str, str] = {}
    for attr_name, aliases in _CATEGORICAL_ATTR_ALIASES.items():
        raw = _safe_getattr(ev, aliases)
        text = _as_nonempty_str(raw)
        if text is not None:
            categorical_attrs[attr_name] = text.upper()

    numeric_attrs: Dict[str, float] = {}
    for attr_name, cfg in numeric_attr_cfgs.items():
        aliases = _NUMERIC_ATTR_ALIASES.get(attr_name, (attr_name,))
        raw = _safe_getattr(ev, aliases)
        value = _as_finite_float(raw)
        if value is None:
            continue
        numeric_attrs[attr_name] = float(cfg.normalize(value))

    if marker is not None:
        categorical_attrs["event_marker_label"] = str(marker).upper()

    return MedicationSemanticDescriptor(
        exact_concept_code=exact_concept_code,
        group_code=group_code,
        semantic_label=semantic_label,
        code_system=code_system,
        categorical_attrs=categorical_attrs,
        numeric_attrs=numeric_attrs,
    )
