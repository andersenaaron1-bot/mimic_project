from __future__ import annotations

import math
from typing import Any, Iterable, Optional


def _parse_sex_to_float(val: Any) -> Optional[float]:
    """
    Normalize sex/gender inputs to {0.0, 1.0} (female/non-male -> 0, male -> 1).
    Returns None if the value cannot be interpreted.
    """
    if val is None:
        return None

    if isinstance(val, bool):
        return 1.0 if val else 0.0

    if isinstance(val, (int, float)):
        f = float(val)
        if not math.isfinite(f):
            return None
        return 1.0 if f >= 0.5 else 0.0

    if isinstance(val, str):
        s = val.strip().lower()
        if not s:
            return None
        if s.startswith("m"):
            return 1.0
        if s.startswith("f"):
            return 0.0
        return None

    return None


def infer_subject_sex(events: Iterable[Any], *, default: float = 0.0) -> float:
    """
    Infer a subject-level sex signal from a stream of MEDS-like events.

    Priority:
      1) explicit `sex` or `gender` attributes on any event (numeric or string)
      2) MEDS code convention: "GENDER//M" / "GENDER//F"
      3) fallback to default (0.0)
    """
    # 1) explicit attributes
    for ev in events:
        for attr in ("sex", "gender"):
            parsed = _parse_sex_to_float(getattr(ev, attr, None))
            if parsed is not None:
                return parsed

    # 2) MEDS codes
    for ev in events:
        code = getattr(ev, "code", None)
        if not isinstance(code, str):
            continue
        if not code.upper().startswith("GENDER//"):
            continue
        suffix = code.split("//")[-1]
        parsed = _parse_sex_to_float(suffix)
        if parsed is not None:
            return parsed

    return float(default)


def infer_birth_timestamp(events: Iterable[Any]) -> Optional[float]:
    """
    Infer a subject birth timestamp (seconds since epoch) from MEDS-like events.

    Uses MEDS convention: code=="MEDS_BIRTH" with a datetime-like `time` supporting
    `.timestamp()`.
    """
    for ev in events:
        code = getattr(ev, "code", None)
        if str(code) != "MEDS_BIRTH":
            continue
        t = getattr(ev, "time", None)
        if t is None or not hasattr(t, "timestamp"):
            continue
        try:
            ts = float(t.timestamp())
        except Exception:
            continue
        if math.isfinite(ts):
            return ts
    return None


def age_years_from_timestamps(event_ts: Optional[float], birth_ts: Optional[float]) -> float:
    """
    Compute age in years at event time from timestamps, else 0.0.
    """
    if event_ts is None or birth_ts is None:
        return 0.0
    if not (math.isfinite(event_ts) and math.isfinite(birth_ts)):
        return 0.0
    return max(0.0, (event_ts - birth_ts) / (3600.0 * 24.0 * 365.25))


def infer_event_age_years(ev: Any, *, birth_ts: Optional[float]) -> float:
    """
    Infer an event-level age in years.

    Priority:
      1) explicit `age_years` attribute on the event, if finite
      2) compute from (event.time.timestamp() - birth_ts)
      3) fallback to 0.0
    """
    age_attr = getattr(ev, "age_years", None)
    if age_attr is not None:
        try:
            a = float(age_attr)
        except (TypeError, ValueError):
            a = None
        if a is not None and math.isfinite(a):
            return max(0.0, a)

    t = getattr(ev, "time", None)
    if t is None or not hasattr(t, "timestamp"):
        return 0.0
    try:
        event_ts = float(t.timestamp())
    except Exception:
        return 0.0
    return age_years_from_timestamps(event_ts, birth_ts)
