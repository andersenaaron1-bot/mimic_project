# src/ehr_hier/tokenizers/interfaces.py
from __future__ import annotations
from typing import Protocol, List, Any, runtime_checkable
from src.ehr_hier.data.event_frames import EventFrame
from src.ehr_hier.data.token_types import EventToken, TokenCategory

@runtime_checkable
class EventTokenEncoder(Protocol):
    """
    Minimal internal codec contract used by the timeline builder.

    Implementations emit ordered `EventToken` atoms for one semantic event. The
    public timeline interface is `EventFrame`; the builder wraps emitted atoms
    into frames after routing and timing adjustments.
    """

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


@runtime_checkable
class EventFrameEncoder(EventTokenEncoder, Protocol):
    """
    Canonical codec contract for timeline building.

    Implementations emit semantic `EventFrame` objects directly. They may still
    expose `encode_event()` for atomic-token compatibility, but builder and
    precompile paths should treat `encode_frame()` as the primary interface.
    """

    def encode_frame(self, ev: Any, dt_hours: float) -> List[EventFrame]:
        ...
