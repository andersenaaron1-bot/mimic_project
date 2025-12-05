# src/ehr_hier/tokenizers/interfaces.py
from __future__ import annotations
from typing import Protocol, List, Any, runtime_checkable
from src.ehr_hier.data.token_types import EventToken, TokenCategory

@runtime_checkable
class EventTokenEncoder(Protocol):
    """Minimal contract for all event encoders used by the timeline builder."""

    # Optional but useful for debugging/logging
    category: TokenCategory

    def reset_state(self) -> None:
        """Called at the start of each subject (e.g., to clear dt_prev caches)."""
        ...

    def encode_event(self, ev: Any, dt_hours: float) -> List[EventToken]:
        """
        Convert a single MEDS event into one or more tokens for this category.
        Return [] to skip.
        """
        ...
