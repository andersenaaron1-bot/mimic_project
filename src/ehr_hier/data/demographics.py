from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from src.ehr_hier.data.token_types import EventToken, TokenCategory


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


def _normalize_code_value(code_value: object) -> str:
    return str(code_value).strip().upper()


def _matches_code_alias(code_value: object, aliases: set[str]) -> bool:
    code_norm = _normalize_code_value(code_value)
    if not code_norm:
        return False
    return code_norm in aliases or any(code_norm.endswith(f"//{alias}") for alias in aliases)


def _extract_finite_numeric(ev_obj: object) -> Optional[float]:
    for attr in ("numeric_value", "value_as_number", "value", "result_value"):
        if not hasattr(ev_obj, attr):
            continue
        raw = getattr(ev_obj, attr)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return float(value)
    return None


def collect_subject_demographic_metadata(
    events: Iterable[Any],
    *,
    timeline_start_ts: Optional[float],
    sex_default: float = 0.0,
) -> dict[str, Any]:
    event_list = list(events)
    sex_value = infer_subject_sex(event_list, default=float(sex_default))
    birth_ts = infer_birth_timestamp(event_list)

    obs: dict[str, list[dict[str, float]]] = {
        "BMI": [],
        "HEIGHT_CM": [],
        "WEIGHT_KG": [],
    }
    alias_to_kind = {
        "BMI (KG/M2)": "BMI",
        "HEIGHT (INCHES)": "HEIGHT_CM",
        "WEIGHT (LBS)": "WEIGHT_KG",
    }
    alias_sets = {
        alias: {_normalize_code_value(alias)}
        for alias in alias_to_kind
    }

    for ev in event_list:
        t = getattr(ev, "time", None)
        if t is None or not hasattr(t, "timestamp"):
            continue
        try:
            event_ts = float(t.timestamp())
        except Exception:
            continue
        if not math.isfinite(event_ts):
            continue
        if timeline_start_ts is None or not math.isfinite(float(timeline_start_ts)):
            t_from_start_hours = 0.0
        else:
            t_from_start_hours = max(0.0, (event_ts - float(timeline_start_ts)) / 3600.0)

        code_value = getattr(ev, "code", None)
        raw_numeric = _extract_finite_numeric(ev)
        if raw_numeric is None:
            continue
        for alias, kind in alias_to_kind.items():
            if not _matches_code_alias(code_value, alias_sets[alias]):
                continue
            value = float(raw_numeric)
            if kind == "HEIGHT_CM":
                value = value * 2.54
            elif kind == "WEIGHT_KG":
                value = value * 0.45359237
            obs[str(kind)].append(
                {
                    "t_from_start_hours": float(t_from_start_hours),
                    "value": float(value),
                }
            )

    return {
        "sex_value": float(sex_value),
        "birth_ts": float(birth_ts) if birth_ts is not None and math.isfinite(float(birth_ts)) else None,
        "timeline_start_ts": float(timeline_start_ts) if timeline_start_ts is not None and math.isfinite(float(timeline_start_ts)) else None,
        "observations": obs,
    }


def _latest_demographic_value_at_or_before(
    metadata: dict[str, Any],
    *,
    kind: str,
    anchor_time_hours: float,
) -> Optional[float]:
    obs = ((metadata or {}).get("observations", {}) or {}).get(str(kind), []) or []
    latest_time: Optional[float] = None
    latest_value: Optional[float] = None
    for item in obs:
        try:
            t_val = float(item.get("t_from_start_hours"))
            value = float(item.get("value"))
        except Exception:
            continue
        if not (math.isfinite(t_val) and math.isfinite(value)):
            continue
        if t_val > float(anchor_time_hours):
            continue
        if latest_time is None or t_val >= latest_time:
            latest_time = t_val
            latest_value = value
    return latest_value


def build_global_demographic_special_tokens(
    *,
    metadata: dict[str, Any],
    anchor_time_hours: float,
    anchor_time_epoch_s: Optional[float],
    token_ids: dict[str, int],
    special_token_offset: int = 0,
) -> list[EventToken]:
    out: list[EventToken] = []
    seen_ids: set[int] = set()

    def _age_bucket_token_id(age_years: float) -> int:
        if age_years < 18.0:
            return int(token_ids["AGE_0_17"])
        if age_years < 40.0:
            return int(token_ids["AGE_18_39"])
        if age_years < 65.0:
            return int(token_ids["AGE_40_64"])
        if age_years < 80.0:
            return int(token_ids["AGE_65_79"])
        return int(token_ids["AGE_80P"])

    def _bmi_bucket_token_id(bmi_value: float) -> int:
        if bmi_value < 18.5:
            return int(token_ids["BMI_UNDER"])
        if bmi_value < 25.0:
            return int(token_ids["BMI_NORMAL"])
        if bmi_value < 30.0:
            return int(token_ids["BMI_OVER"])
        if bmi_value < 35.0:
            return int(token_ids["BMI_OBESE_1"])
        if bmi_value < 40.0:
            return int(token_ids["BMI_OBESE_2"])
        return int(token_ids["BMI_OBESE_3"])

    def _emit(local_id: int, *, feature_id: int, numeric_value: Optional[float] = None) -> None:
        gid = int(special_token_offset) + int(local_id)
        if gid in seen_ids:
            return
        seen_ids.add(gid)
        num_attrs = {}
        if numeric_value is not None and math.isfinite(float(numeric_value)):
            num_attrs["numeric_value"] = float(numeric_value)
        out.append(
            EventToken(
                value_id=int(gid),
                category_id=int(TokenCategory.SPECIAL),
                t_from_start_hours=0.0,
                dt_from_prev_hours=0.0,
                cat_attrs={
                    "global_demographic": 1,
                    "demographic_feature_id": int(feature_id),
                    "demographic_numeric": 1 if numeric_value is not None else 0,
                },
                num_attrs=num_attrs,
                raw_time=None,
                window_hook=None,
            )
        )

    sex_value = (metadata or {}).get("sex_value", None)
    if sex_value is not None:
        try:
            sex_float = float(sex_value)
        except Exception:
            sex_float = None
        if sex_float is not None and math.isfinite(sex_float):
            _emit(int(token_ids["SEX_M"] if sex_float >= 0.5 else token_ids["SEX_F"]), feature_id=1)

    birth_ts = (metadata or {}).get("birth_ts", None)
    if birth_ts is not None and anchor_time_epoch_s is not None:
        try:
            age_at_anchor = age_years_from_timestamps(float(anchor_time_epoch_s), float(birth_ts))
        except Exception:
            age_at_anchor = 0.0
        if math.isfinite(age_at_anchor) and age_at_anchor >= 0.0:
            _emit(_age_bucket_token_id(age_at_anchor), feature_id=2)

    bmi_val = _latest_demographic_value_at_or_before(metadata, kind="BMI", anchor_time_hours=float(anchor_time_hours))
    if bmi_val is None:
        weight_kg = _latest_demographic_value_at_or_before(metadata, kind="WEIGHT_KG", anchor_time_hours=float(anchor_time_hours))
        height_cm = _latest_demographic_value_at_or_before(metadata, kind="HEIGHT_CM", anchor_time_hours=float(anchor_time_hours))
        if weight_kg is not None and height_cm is not None and height_cm > 0.0:
            bmi_val = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)
    if bmi_val is not None and math.isfinite(float(bmi_val)) and float(bmi_val) > 0.0:
        _emit(_bmi_bucket_token_id(float(bmi_val)), feature_id=3)

    height_cm = _latest_demographic_value_at_or_before(metadata, kind="HEIGHT_CM", anchor_time_hours=float(anchor_time_hours))
    if height_cm is not None and math.isfinite(float(height_cm)) and float(height_cm) > 0.0:
        _emit(int(token_ids["HEIGHT_AT_ADMISSION"]), feature_id=4, numeric_value=float(height_cm))

    weight_kg = _latest_demographic_value_at_or_before(metadata, kind="WEIGHT_KG", anchor_time_hours=float(anchor_time_hours))
    if weight_kg is not None and math.isfinite(float(weight_kg)) and float(weight_kg) > 0.0:
        _emit(int(token_ids["WEIGHT_AT_ADMISSION"]), feature_id=5, numeric_value=float(weight_kg))

    return out
