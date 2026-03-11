from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_medication_code,
    canonicalize_procedure_code,
    ensure_list,
)

_ALIAS_RE = re.compile(r"[^A-Z0-9]+")
_PARENS_RE = re.compile(r"\(([^)]+)\)")
_MED_ACTION_SUFFIXES = {
    "ADMINISTERED",
    "NOT ADMINISTERED",
    "NOT GIVEN",
    "NOT GIVEN PER SLIDING SCALE",
    "CONFIRMED",
    "FLUSHED",
    "NOT FLUSHED",
    "STARTED",
    "STOPPED",
    "HELD",
    "REFUSED",
    "PAUSED",
    "RESUMED",
}
_MED_DROP_WORDS = {
    "IV",
    "IVPB",
    "PO",
    "IM",
    "SC",
    "SQ",
    "SUBCUT",
    "SUBCUTANEOUS",
    "ORAL",
    "INJECTION",
    "INJECTABLE",
    "SOLUTION",
    "SOLN",
    "TABLET",
    "TABLETS",
    "CAPSULE",
    "CAPSULES",
    "SUSPENSION",
    "SYRUP",
    "ELIXIR",
    "VIAL",
    "BAG",
    "DRIP",
    "PUSH",
    "BOLUS",
    "FLUSH",
    "CONCENTRATE",
    "IMMEDIATE",
    "RELEASE",
    "EXTENDED",
    "ER",
    "XR",
    "SR",
    "DR",
}
_DIGIT_TOKEN_RE = re.compile(r"^\d+(?:[A-Z]+)?$")


def normalize_crosswalk_key(raw: object) -> str:
    s = _ALIAS_RE.sub("", str(raw).upper())
    if len(s) < 4:
        return ""
    if not any(ch.isalpha() for ch in s):
        return ""
    return s


def _dedupe(values: Iterable[object]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        s = str(value).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _medication_surface_parts(raw: object) -> Tuple[str, Optional[str], Optional[str]]:
    parts = [p.strip() for p in str(raw).split("//")]
    if not parts:
        return "", None, None

    head = parts[0].upper()
    tail = parts[1:]
    action: Optional[str] = None
    if head == "INFUSION_START":
        return "INFUSION", ("//".join(tail).strip() or None), "START"
    if head == "INFUSION_END":
        return "INFUSION", ("//".join(tail).strip() or None), "END"

    if head in {"MEDICATION", "INFUSION"}:
        if tail and tail[0].upper() in {"START", "END", "STOP"}:
            action = tail[0].upper()
            tail = tail[1:]
        elif len(tail) >= 2 and tail[-1].upper() in _MED_ACTION_SUFFIXES:
            action = tail[-1].upper()
            tail = tail[:-1]
        entity = "//".join(tail).strip() if tail else ""
        return head, (entity or None), action
    return head, None, None


def _variant_strings(raw: object) -> List[str]:
    text = str(raw).strip()
    if not text:
        return []
    out = [text]
    inner = [m.group(1).strip() for m in _PARENS_RE.finditer(text) if m.group(1).strip()]
    if inner:
        out.extend(inner)
        stripped = _PARENS_RE.sub(" ", text)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            out.append(stripped)
    return _dedupe(out)


def _simplify_medication_text(raw: object) -> Optional[str]:
    tokens = re.findall(r"[A-Z0-9]+", str(raw).upper())
    keep: List[str] = []
    for tok in tokens:
        if tok in _MED_DROP_WORDS:
            continue
        if _DIGIT_TOKEN_RE.fullmatch(tok):
            continue
        keep.append(tok)
    if not keep:
        return None
    simplified = " ".join(keep).strip()
    return simplified or None


def medication_alias_strings(raw: object) -> List[str]:
    text = str(raw).strip()
    if not text:
        return []
    out: List[str] = [text]
    domain, entity, _action = _medication_surface_parts(text)
    if entity:
        out.append(entity)
        if domain == "MEDICATION":
            out.append(f"MEDICATION//{entity}")
        elif domain == "INFUSION":
            out.append(f"INFUSION//{entity}")
    for value in list(out):
        out.extend(_variant_strings(value))
    simplified: List[str] = []
    for value in out:
        simp = _simplify_medication_text(value)
        if simp:
            simplified.append(simp)
    out.extend(simplified)
    return _dedupe(out)


def procedure_alias_strings(raw: object) -> List[str]:
    text = str(raw).strip()
    if not text:
        return []
    out = [text]
    upper = text.upper()
    if upper.startswith("PROCEDURE//"):
        rest = text.split("//", 1)[1]
        out.append(rest)
        parts = [p.strip() for p in rest.split("//") if p.strip()]
        if parts:
            if parts[0].upper() in {"START", "END", "STOP"} and len(parts) > 1:
                out.append("//".join(parts[1:]))
            out.append(parts[-1])
    for value in list(out):
        out.extend(_variant_strings(value))
    return _dedupe(out)


def crosswalk_alias_strings(category: str | TokenCategory, raw: object) -> List[str]:
    if isinstance(category, TokenCategory):
        family = category.name.lower()
    else:
        family = str(category).strip().lower()
    if family == "medication":
        return medication_alias_strings(raw)
    if family == "procedure":
        return procedure_alias_strings(raw)
    return _dedupe([raw])


def crosswalk_candidate_keys(
    category: str | TokenCategory,
    *values: object,
) -> List[str]:
    out: List[str] = []
    for value in values:
        for alias in crosswalk_alias_strings(category, value):
            key = normalize_crosswalk_key(alias)
            if key:
                out.append(key)
    return _dedupe(out)


def _family_name(category: str | TokenCategory) -> str:
    if isinstance(category, TokenCategory):
        return category.name.lower()
    return str(category).strip().lower()


def load_crosswalk_candidate_map(
    json_fp: str | Path | None,
    family: str | TokenCategory,
) -> Dict[str, List[str]]:
    if json_fp is None:
        return {}
    fp = Path(str(json_fp))
    if not fp.exists():
        raise FileNotFoundError(f"MedTok crosswalk artifact not found: {fp}")
    payload = json.loads(fp.read_text(encoding="utf-8"))
    families = payload.get("families", {})
    fam = families.get(_family_name(family), {})
    aliases = fam.get("aliases", {})
    out: Dict[str, List[str]] = {}
    for alias_key, target in aliases.items():
        key = str(alias_key).strip().upper()
        if not key:
            continue
        if isinstance(target, str):
            cands = [target]
        elif isinstance(target, Sequence):
            cands = [str(x).strip() for x in target if str(x).strip()]
        else:
            continue
        if cands:
            out[key] = _dedupe(cands)
    return out


def resolve_crosswalk_target(
    *,
    family: str | TokenCategory,
    candidate_map: Mapping[str, Sequence[str]],
    available_codes: Optional[Iterable[str]] = None,
    allow_unvalidated_fallback: bool = False,
    values: Sequence[object],
) -> Tuple[Optional[str], Optional[str]]:
    if not candidate_map:
        return None, None
    allowed = {str(x) for x in available_codes} if available_codes is not None else None
    for key in crosswalk_candidate_keys(family, *values):
        targets = candidate_map.get(key)
        if not targets:
            continue
        if allowed is not None:
            for target in targets:
                if str(target) in allowed:
                    return str(target), key
        if allow_unvalidated_fallback:
            return str(targets[0]), key
    return None, None


def load_resolved_crosswalk_lookup(
    json_fp: str | Path | None,
    family: str | TokenCategory,
    *,
    available_codes: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    candidate_map = load_crosswalk_candidate_map(json_fp, family)
    if not candidate_map:
        return {}
    allowed = {str(x) for x in available_codes} if available_codes is not None else None
    out: Dict[str, str] = {}
    for key, candidates in candidate_map.items():
        if allowed is None:
            out[key] = str(candidates[0])
            continue
        for candidate in candidates:
            if str(candidate) in allowed:
                out[key] = str(candidate)
                break
    return out


def _parse_parent_cell(v: object) -> List[str]:
    vals: List[str] = []

    def _append(x: object) -> None:
        if x is None:
            return
        s = str(x).strip()
        if s:
            vals.append(s)

    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        for x in v:
            _append(x)
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        parsed = None
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None
        if isinstance(parsed, (list, tuple, set)):
            for x in parsed:
                _append(x)
        else:
            _append(s)
    else:
        _append(v)
    return _dedupe(vals)


def target_candidates_from_system(vocab_id: object, concept_code: object) -> List[str]:
    system = str(vocab_id).strip().upper()
    code = str(concept_code).strip()
    if not system or not code:
        return []
    if system == "RXNORM":
        return _dedupe([code, f"RXNORM//{code}"])
    if system == "SNOMED":
        return _dedupe([code, f"SNOMED//{code}", f"SNOMED/{code}"])
    if system == "NDC":
        return _dedupe([f"NDC//{code}", code])
    if system == "CPT4":
        return _dedupe([f"CPT//{code}", code])
    if system == "HCPCS":
        return _dedupe([f"HCPCS//{code}", code])
    if system == "ICD10PCS":
        return _dedupe([f"ICD10PCS//{code}", code])
    if system in {"ICD9PROC", "ICD9_PROC"}:
        return _dedupe([f"ICD9PROC//{code}", code, code.replace(".", "")])
    return []


def target_candidates_from_parent_code(parent_code: object, family: str | TokenCategory) -> List[str]:
    fam = _family_name(family)
    raw = str(parent_code).strip()
    if not raw:
        return []
    candidates: List[str] = [raw]
    if fam == "medication":
        candidates.extend(ensure_list(canonicalize_medication_code(raw)))
    elif fam == "procedure":
        candidates.extend(ensure_list(canonicalize_procedure_code(raw)))
    return _dedupe(candidates)


def _record_alias(
    alias_targets: Dict[str, set[str]],
    alias_payloads: Dict[str, List[str]],
    *,
    family: str,
    alias_values: Iterable[object],
    semantic_id: str,
    target_candidates: Sequence[str],
) -> None:
    cleaned_targets = _dedupe(target_candidates)
    if not cleaned_targets:
        return
    for alias in alias_values:
        for alias_value in crosswalk_alias_strings(family, alias):
            key = normalize_crosswalk_key(alias_value)
            if not key:
                continue
            alias_targets[key].add(str(semantic_id))
            alias_payloads.setdefault(key, cleaned_targets)


def build_medtok_crosswalk_artifact(
    *,
    concept_map_dir: str | Path | None = None,
    codes_parquet: str | Path | None = None,
) -> Dict[str, Any]:
    concept_dir = Path(concept_map_dir) if concept_map_dir is not None else None
    alias_targets: Dict[str, Dict[str, set[str]]] = {
        "medication": defaultdict(set),
        "procedure": defaultdict(set),
    }
    alias_payloads: Dict[str, Dict[str, List[str]]] = {
        "medication": {},
        "procedure": {},
    }
    source_rows = defaultdict(int)

    def _add_family_entry(
        family: str,
        *,
        alias_values: Iterable[object],
        semantic_id: str,
        target_candidates: Sequence[str],
    ) -> None:
        _record_alias(
            alias_targets[family],
            alias_payloads[family],
            family=family,
            alias_values=alias_values,
            semantic_id=semantic_id,
            target_candidates=target_candidates,
        )

    if concept_dir is not None and concept_dir.exists():
        med_fp = concept_dir / "inputevents_to_rxnorm.csv"
        if med_fp.exists():
            with med_fp.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = str(row.get("label", "")).strip()
                    concept_name = str(row.get("omop_concept_name", "")).strip()
                    concept_code = str(row.get("omop_concept_code", "")).strip()
                    vocab_id = str(row.get("omop_vocabulary_id", "")).strip()
                    targets = target_candidates_from_system(vocab_id, concept_code)
                    if not label or not targets:
                        continue
                    semantic_id = f"{vocab_id}:{concept_code}"
                    _add_family_entry(
                        "medication",
                        alias_values=[label, concept_name],
                        semantic_id=semantic_id,
                        target_candidates=targets,
                    )
                    source_rows["mimic_inputevents_to_rxnorm"] += 1

        for source_name in ("proc_itemid.csv", "proc_datetimeevents.csv"):
            proc_fp = concept_dir / source_name
            if not proc_fp.exists():
                continue
            with proc_fp.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = str(row.get("label", "")).strip()
                    concept_name = str(row.get("omop_concept_name", "")).strip()
                    concept_code = str(row.get("omop_concept_code", "")).strip()
                    vocab_id = str(row.get("omop_vocabulary_id", "")).strip()
                    targets = target_candidates_from_system(vocab_id, concept_code)
                    if not label or not targets:
                        continue
                    semantic_id = f"{vocab_id}:{concept_code}"
                    _add_family_entry(
                        "procedure",
                        alias_values=[label, concept_name],
                        semantic_id=semantic_id,
                        target_candidates=targets,
                    )
                    source_rows[f"mimic_{source_name}"] += 1

    if codes_parquet is not None:
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise ImportError("pyarrow is required to read codes.parquet for MedTok crosswalks") from exc

        tbl = pq.read_table(
            str(codes_parquet),
            columns=["code", "description", "parent_codes"],
        )
        codes = tbl.column("code").to_pylist()
        descriptions = tbl.column("description").to_pylist()
        parents = tbl.column("parent_codes").to_pylist()
        for code, description, parent_cell in zip(codes, descriptions, parents):
            if code is None:
                continue
            category = classify_code_to_category(code)
            family = ""
            if category == TokenCategory.MEDICATION:
                family = "medication"
            elif category == TokenCategory.PROCEDURE:
                family = "procedure"
            if not family:
                continue
            parent_codes = _parse_parent_cell(parent_cell)
            targets: List[str] = []
            semantic_id: Optional[str] = None
            for parent_code in parent_codes:
                cand = target_candidates_from_parent_code(parent_code, family)
                if not cand:
                    continue
                targets = cand
                semantic_id = f"parent:{parent_code}"
                break
            if not targets or semantic_id is None:
                continue
            _add_family_entry(
                family,
                alias_values=[code, description],
                semantic_id=semantic_id,
                target_candidates=targets,
            )
            source_rows["codes_parquet"] += 1

    families: Dict[str, Any] = {}
    for family in ("medication", "procedure"):
        kept_aliases: Dict[str, List[str]] = {}
        ambiguous = 0
        for alias_key, semantic_ids in alias_targets[family].items():
            if len(semantic_ids) != 1:
                ambiguous += 1
                continue
            kept_aliases[alias_key] = list(alias_payloads[family][alias_key])
        families[family] = {
            "aliases": dict(sorted(kept_aliases.items())),
            "summary": {
                "unique_aliases": int(len(kept_aliases)),
                "ambiguous_aliases_dropped": int(ambiguous),
            },
        }

    return {
        "version": 1,
        "families": families,
        "sources": dict(sorted(source_rows.items())),
    }
