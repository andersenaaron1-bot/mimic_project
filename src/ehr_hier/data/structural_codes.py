from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set


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
    """

    code2label: Dict[str, str]
    label2id_map: Dict[str, int] = field(default_factory=dict)
    offset: int = 0
    structural_only: Set[str] = field(default_factory=set)
    keep_original: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Auto-derive label2id_map if not provided
        if not self.label2id_map:
            labels = sorted(set(self.code2label.values()))
            self.label2id_map = {lbl: i for i, lbl in enumerate(labels)}

    def label2id(self) -> Dict[str, int]:
        # Return a copy to avoid external mutation
        return dict(self.label2id_map)

    def __contains__(self, code: object) -> bool:
        return code is not None and str(code) in self.code2label
