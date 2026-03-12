from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Set

import yaml

from src.ehr_hier.data.token_types import TokenCategory


TRANSITION_ACTION_TO_ID: Dict[str, int] = {
    "open_next": 1,
    "close_current": 2,
    "close_open": 3,
    "suppress": 4,
}
TRANSITION_ACTION_FROM_ID: Dict[int, str] = {v: k for k, v in TRANSITION_ACTION_TO_ID.items()}


@dataclass
class StructuralCodebook:
    """
    Minimal structural codebook used by the timeline builder to optionally emit
    structural tokens alongside (or instead of) the routed category tokens.

    code2label:    mapping from raw MEDS code -> structural label string
    label2id_map:  mapping from label string -> local id (0-based)
    offset:        global vocab offset for structural tokens
    structural_only: codes that should only emit structural tokens
    keep_original: codes that should emit both structural + routed tokens
    boundary_labels: optional set of structural labels that define window boundaries
    boundary_codes: optional set of raw codes that define window boundaries
    transition_map: optional mapping from raw code / prefix / label -> transition action

    If neither boundary_labels nor boundary_codes are provided, all structural
    codebook hits are treated as window boundaries (backward compatible).
    """

    code2label: Dict[str, str]
    label2id_map: Dict[str, int] = field(default_factory=dict)
    offset: int = 0
    structural_only: Set[str] = field(default_factory=set)
    keep_original: Set[str] = field(default_factory=set)
    boundary_labels: Optional[Set[str]] = None
    boundary_codes: Optional[Set[str]] = None
    transition_map: Dict[str, str] = field(default_factory=dict)
    window_type2id_map: Dict[str, int] = field(default_factory=dict)
    window_type_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-derive label2id_map if not provided
        if not self.label2id_map:
            labels = sorted(set(self.code2label.values()))
            self.label2id_map = {lbl: i for i, lbl in enumerate(labels)}

        if self.boundary_labels is not None:
            self.boundary_labels = {str(x) for x in self.boundary_labels}
        if self.boundary_codes is not None:
            self.boundary_codes = {str(x) for x in self.boundary_codes}
        self.transition_map = {str(k): str(v) for k, v in self.transition_map.items()}
        if not self.window_type2id_map:
            self.window_type2id_map = {"UNK": 0}
        else:
            self.window_type2id_map = {str(k): int(v) for k, v in self.window_type2id_map.items()}
            self.window_type2id_map.setdefault("UNK", 0)
        self.window_type_map = {str(k): str(v) for k, v in self.window_type_map.items()}

    def label2id(self) -> Dict[str, int]:
        # Return a copy to avoid external mutation
        return dict(self.label2id_map)

    def is_window_boundary(self, *, code: str, label: str) -> bool:
        """
        Decide whether a structural token should create a new window.

        """
        code_str = str(code)
        label_str = str(label)

        if self.boundary_codes is None and self.boundary_labels is None:
            return True
        if self.boundary_codes is not None and code_str in self.boundary_codes:
            return True
        if self.boundary_labels is not None and label_str in self.boundary_labels:
            return True
        return False

    def transition_action(self, *, code: str | None = None, label: str | None = None) -> Optional[str]:
        """
        Resolve a transition action from a raw code or semantic label.

        Matching precedence:
          1. exact raw code
          2. raw prefix before "//"
          3. exact label
        """
        if not self.transition_map:
            return None

        candidates = []
        if code is not None:
            code_str = str(code)
            candidates.append(code_str)
            candidates.append(code_str.split("//", 1)[0])
        if label is not None:
            candidates.append(str(label))

        for candidate in candidates:
            if candidate in self.transition_map:
                return self.transition_map[candidate]
        return None

    def transition_action_id(self, *, code: str | None = None, label: str | None = None) -> Optional[int]:
        action = self.transition_action(code=code, label=label)
        if action is None:
            return None
        return TRANSITION_ACTION_TO_ID.get(action)

    def window_type2id(self) -> Dict[str, int]:
        return dict(self.window_type2id_map)

    def window_type_name(
        self,
        *,
        code: str | None = None,
        label: str | None = None,
        action: str | None = None,
    ) -> Optional[str]:
        candidates = []
        code_str = None
        if code is not None:
            code_str = str(code)
            candidates.append(code_str)
        if label is not None:
            candidates.append(str(label))
        for candidate in candidates:
            if candidate in self.window_type_map:
                return self.window_type_map[candidate]

        if code_str is not None:
            upper = code_str.upper()
            prefix = upper.split("//", 1)[0]
            if prefix in {"HOSPITAL_ADMISSION", "ADMISSION", "CAREUNIT_CHANGE"} and prefix in self.window_type_map:
                return self.window_type_map[prefix]
            if prefix in {"ICU_ADMISSION", "ICU_DISCHARGE"} and prefix in self.window_type_map:
                return self.window_type_map[prefix]
            if prefix == "TRANSFER_TO":
                if "//ED//" in upper or "EMERGENCY DEPARTMENT" in upper:
                    return "ED"
                if "ICU" in upper:
                    return "ICU"
                if "OPERATING ROOM" in upper or "//OR//" in upper:
                    return "OR"
                if prefix in self.window_type_map:
                    return self.window_type_map[prefix]
            if "EMERGENCY DEPARTMENT" in upper or "EMERGENCY ROOM" in upper or upper.startswith("ED_"):
                return "ED"
            if "ICU" in upper:
                return "ICU"
            if "OPERATING ROOM" in upper or upper.startswith("OR_") or "//OR//" in upper:
                return "OR"
            if prefix in self.window_type_map:
                return self.window_type_map[prefix]

        if label is not None:
            label_str = str(label).upper()
            if "ICU" in label_str:
                return "ICU"
            if "OR" in label_str:
                return "OR"

        if action == "suppress" and code_str == "MEDS_BIRTH":
            return "PROLOGUE"
        return None

    def window_type_id(
        self,
        *,
        code: str | None = None,
        label: str | None = None,
        action: str | None = None,
    ) -> Optional[int]:
        name = self.window_type_name(code=code, label=label, action=action)
        if name is None:
            return None
        return self.window_type2id_map.get(name)

    def __contains__(self, code: object) -> bool:
        return code is not None and str(code) in self.code2label


def structural_surface_code(
    code: object,
    *,
    codebook: Optional[StructuralCodebook] = None,
    routed_category: Optional[TokenCategory | int] = None,
) -> Optional[str]:
    """
    Return the structural token surface for an event.

    Rules:
    - exact codebook-hit raw codes stay exact
    - otherwise, routed structural events normalize to their raw prefix
    - everything else returns None
    """
    if code is None:
        return None
    code_str = str(code).strip()
    if not code_str:
        return None
    if codebook is not None and code_str in codebook.code2label:
        return code_str
    if routed_category is None:
        return None
    try:
        cat_value = int(routed_category)
    except Exception:
        return None
    if cat_value != int(TokenCategory.STRUCTURAL):
        return None
    return code_str.split("//", 1)[0].upper()


def structural_surface_vocab_codes(
    codebook: Optional[StructuralCodebook],
) -> Set[str]:
    """
    Deterministic seed set for the structural family vocabulary.

    This is intentionally based on actual emitted structural surfaces:
    - exact raw codes that the codebook maps directly
    - raw transition/window-type keys that represent routed structural prefixes
    """
    out: Set[str] = set()
    if codebook is None:
        return out

    out.update(str(k) for k in codebook.code2label.keys())
    semantic_labels = set(str(x) for x in codebook.label2id().keys())

    for mapping in (codebook.transition_map, codebook.window_type_map):
        for raw_key in mapping.keys():
            key = str(raw_key).strip()
            if not key or key in semantic_labels:
                continue
            surface = structural_surface_code(
                key,
                codebook=codebook,
                routed_category=TokenCategory.STRUCTURAL,
            )
            if surface:
                out.add(surface)
    return out


def serialize_structural_codebook(
    codebook: Optional[StructuralCodebook],
) -> Dict[str, Any]:
    """
    Serialize the live structural codebook into a builder/runtime-friendly payload.

    This is used by the generated sparse vocab contract so the tokenization-v1
    artifact can show the exact structural surfaces, transitions, and window
    type mappings that are active in the current code path.
    """
    if codebook is None:
        return {}
    return {
        "offset": int(codebook.offset),
        "code2label": {str(k): str(v) for k, v in codebook.code2label.items()},
        "label2id": {str(k): int(v) for k, v in codebook.label2id().items()},
        "structural_only": sorted(str(x) for x in codebook.structural_only),
        "keep_original": sorted(str(x) for x in codebook.keep_original),
        "boundary_labels": (
            sorted(str(x) for x in codebook.boundary_labels)
            if codebook.boundary_labels is not None
            else None
        ),
        "boundary_codes": (
            sorted(str(x) for x in codebook.boundary_codes)
            if codebook.boundary_codes is not None
            else None
        ),
        "transition_map": {str(k): str(v) for k, v in codebook.transition_map.items()},
        "window_types": {str(k): int(v) for k, v in codebook.window_type2id().items()},
        "window_type_map": {str(k): str(v) for k, v in codebook.window_type_map.items()},
        "surface_vocab_codes": sorted(structural_surface_vocab_codes(codebook)),
        "active_transition_actions": sorted(str(x) for x in TRANSITION_ACTION_TO_ID.keys()),
    }


def structural_codebook_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    default_offset: int = 0,
) -> Optional[StructuralCodebook]:
    """
    Reconstruct a StructuralCodebook from a serialized payload.
    """
    if not isinstance(payload, Mapping):
        return None
    code2label_raw = payload.get("code2label", None)
    if not isinstance(code2label_raw, Mapping) or not code2label_raw:
        return None
    return StructuralCodebook(
        code2label={str(k): str(v) for k, v in code2label_raw.items()},
        label2id_map={
            str(k): int(v)
            for k, v in (payload.get("label2id", {}) or {}).items()
        },
        offset=int(payload.get("offset", default_offset)),
        structural_only={str(x) for x in (payload.get("structural_only", []) or [])},
        keep_original={str(x) for x in (payload.get("keep_original", []) or [])},
        boundary_labels=(
            {str(x) for x in payload.get("boundary_labels", [])}
            if payload.get("boundary_labels", None) is not None
            else None
        ),
        boundary_codes=(
            {str(x) for x in payload.get("boundary_codes", [])}
            if payload.get("boundary_codes", None) is not None
            else None
        ),
        transition_map={
            str(k): str(v)
            for k, v in (payload.get("transition_map", {}) or {}).items()
        },
        window_type2id_map={
            str(k): int(v)
            for k, v in (payload.get("window_types", {}) or {}).items()
        },
        window_type_map={
            str(k): str(v)
            for k, v in (payload.get("window_type_map", {}) or {}).items()
        },
    )


def load_structural_codebook_yaml(yaml_fp: str, *, default_offset: int = 0) -> StructuralCodebook:
    """
    Load a StructuralCodebook from a YAML file.

    Expected YAML keys (all optional unless noted):
      - structural_map (required): mapping raw_code -> label
      - structural_only: list of raw codes that emit ONLY structural tokens
      - keep_original: list of raw codes that emit structural + routed tokens
      - offset: optional global vocab offset fallback
      - window_boundary_labels / window_boundaries / boundary_labels: labels that split windows
      - window_boundary_codes / boundary_codes: raw codes that split windows
      - transition_map: mapping raw code / prefix / label -> transition action
      - window_types: mapping semantic window type -> integer id
      - window_type_map: mapping raw code / prefix / label -> semantic window type
      - soft_signifiers: list of codes to emit structural markers for (defaults to label "SOFT::<code>")
    """
    with open(yaml_fp, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Structural codebook YAML must be a dict, got {type(payload)}")

    structural_map = payload.get("structural_map", None)
    if not isinstance(structural_map, dict):
        raise TypeError("Structural codebook YAML must include 'structural_map' as a dict[code -> label].")

    code2label = {str(k): str(v) for k, v in structural_map.items()}

    def _as_code_set(key: str) -> Set[str]:
        v = payload.get(key, None)
        if v is None:
            return set()
        if not isinstance(v, list):
            raise TypeError(f"YAML key '{key}' must be a list, got {type(v)}")
        return {str(x) for x in v}

    structural_only = _as_code_set("structural_only")
    keep_original = _as_code_set("keep_original")

    # Soft signifiers: ensure they emit a structural marker token but do not automatically
    # make them boundaries unless boundary_labels/codes specify it.
    if "soft_signifiers" in payload and payload.get("soft_signifiers") is not None:
        soft = payload.get("soft_signifiers")
        if not isinstance(soft, list):
            raise TypeError(f"YAML key 'soft_signifiers' must be a list, got {type(soft)}")
        for code in soft:
            code_str = str(code)
            code2label.setdefault(code_str, f"SOFT::{code_str}")

    offset = int(payload.get("offset", default_offset))

    boundary_labels: Optional[Set[str]] = None
    for key in ("window_boundary_labels", "window_boundaries", "boundary_labels"):
        if key in payload:
            boundary_labels = _as_code_set(key)
            break

    boundary_codes: Optional[Set[str]] = None
    for key in ("window_boundary_codes", "boundary_codes"):
        if key in payload:
            boundary_codes = _as_code_set(key)
            break

    transition_map = payload.get("transition_map", {})
    if not isinstance(transition_map, dict):
        raise TypeError(f"YAML key 'transition_map' must be a dict, got {type(transition_map)}")

    window_types = payload.get("window_types", {})
    if not isinstance(window_types, dict):
        raise TypeError(f"YAML key 'window_types' must be a dict, got {type(window_types)}")

    window_type_map = payload.get("window_type_map", {})
    if not isinstance(window_type_map, dict):
        raise TypeError(f"YAML key 'window_type_map' must be a dict, got {type(window_type_map)}")

    return StructuralCodebook(
        code2label=code2label,
        structural_only=structural_only,
        keep_original=keep_original,
        offset=offset,
        boundary_labels=boundary_labels,
        boundary_codes=boundary_codes,
        transition_map={str(k): str(v) for k, v in transition_map.items()},
        window_type2id_map={str(k): int(v) for k, v in window_types.items()},
        window_type_map={str(k): str(v) for k, v in window_type_map.items()},
    )
