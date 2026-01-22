from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import yaml


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

    def __post_init__(self) -> None:
        # Auto-derive label2id_map if not provided
        if not self.label2id_map:
            labels = sorted(set(self.code2label.values()))
            self.label2id_map = {lbl: i for i, lbl in enumerate(labels)}

        if self.boundary_labels is not None:
            self.boundary_labels = {str(x) for x in self.boundary_labels}
        if self.boundary_codes is not None:
            self.boundary_codes = {str(x) for x in self.boundary_codes}

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

    def __contains__(self, code: object) -> bool:
        return code is not None and str(code) in self.code2label


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

    return StructuralCodebook(
        code2label=code2label,
        structural_only=structural_only,
        keep_original=keep_original,
        offset=offset,
        boundary_labels=boundary_labels,
        boundary_codes=boundary_codes,
    )
