from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


OBS_PREFIX_ALLOWLIST = {
    "LAB",
    "VITAL",
    "MEAS",
    "SUBJECT_FLUID_OUTPUT",
    "SUBJECT_WEIGHT_AT_INFUSION",
    "OMR",
    "BLOOD PRESSURE",
}

# These values stay stable across exact and fallback OBS vocab generation.
OBS_RESERVED_VALUE_IDS = {
    "UNK": 1,
    "N/A": 2,
    "NONE": 3,
    "": 4,
}


@dataclass(frozen=True)
class ObservationSurfaces:
    code_surface: str
    value_surface: str


def code_parts(code_value: object) -> list[str]:
    if code_value is None:
        return []
    return [p.strip() for p in str(code_value).split("//")]


def normalize_obs_value(raw_value: object) -> str:
    if raw_value is None:
        return "UNK"
    val = str(raw_value).strip()
    if not val:
        return "UNK"
    val = " ".join(val.split())
    upper = val.upper()
    if upper in {"UNKNOWN", "UNK"}:
        return "UNK"
    if upper in {"N/A", "NA", "NOT APPLICABLE"}:
        return "N/A"
    if upper in {"NONE", "NULL"}:
        return "NONE"
    return val[:96]


def extract_obs_value(ev_obj: object, *, code_value: Optional[str]) -> str:
    for attr in (
        "text_value",
        "value",
        "value_as_string",
        "value_text",
        "result_value",
        "status",
    ):
        if hasattr(ev_obj, attr):
            raw = getattr(ev_obj, attr)
            if raw is not None and str(raw).strip():
                return normalize_obs_value(raw)
    if hasattr(ev_obj, "numeric_value"):
        try:
            nval = float(getattr(ev_obj, "numeric_value"))
        except (TypeError, ValueError):
            nval = None
        if nval is not None and math.isfinite(nval):
            return normalize_obs_value(f"{nval:.3f}".rstrip("0").rstrip("."))
    parts = code_parts(code_value)
    if len(parts) >= 3:
        return normalize_obs_value(parts[2])
    return "UNK"


def observation_code_surface(code_value: Optional[str]) -> Optional[str]:
    if code_value is None:
        return None
    parts = code_parts(code_value)
    if not parts:
        return None
    prefix = parts[0].upper()
    if prefix not in OBS_PREFIX_ALLOWLIST:
        return None
    item_or_code = parts[1] if len(parts) >= 2 else str(code_value)
    return f"{prefix}::{item_or_code}"


def observation_surfaces(ev_obj: object, *, code_value: Optional[str]) -> Optional[ObservationSurfaces]:
    code_surface = observation_code_surface(code_value)
    if code_surface is None:
        return None
    return ObservationSurfaces(
        code_surface=code_surface,
        value_surface=extract_obs_value(ev_obj, code_value=code_value),
    )

