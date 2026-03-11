from typing import Dict, List, Optional, Set
from datetime import datetime
import math
import zlib

import meds_reader as mr   # pip install meds_reader

from .token_types import EventToken, TokenCategory
from .event_router import classify_code_to_category
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.data.structural_codes import StructuralCodebook
from src.ehr_hier.data.demographics import (
    age_years_from_timestamps,
    infer_subject_sex,
    infer_birth_timestamp,
    infer_event_age_years,
)


class _EventWithDemographics:
    """
    Lightweight view over a meds_reader event that injects per-event demographics.
    Encoders can access `age_years` and `sex` without relying on the raw event schema.
    """

    __slots__ = ("_ev", "age_years", "sex")

    def __init__(self, ev: object, *, age_years: float, sex: float) -> None:
        self._ev = ev
        self.age_years = float(age_years)
        self.sex = float(sex)

    def __getattr__(self, name: str):
        return getattr(self._ev, name)


ACTIVE_TRANSITION_ACTIONS = {"open_next", "close_current", "close_open"}

PROCESS_ACTION_TO_ID = {"START": 1, "END": 2, "STOP": 3}
PROCESS_DOMAIN_TO_ID = {"MEDICATION": 1, "PROCEDURE": 2, "INFUSION": 3}
OBS_SPECIAL_VALUE_IDS = {
    "UNK": 1,
    "N/A": 2,
    "NA": 2,
    "NONE": 3,
    "NULL": 3,
    "": 4,
}

GLOBAL_DEMOGRAPHIC_TOKEN_IDS = {
    "SEX_F": 30,
    "SEX_M": 31,
    "AGE_0_17": 32,
    "AGE_18_39": 33,
    "AGE_40_64": 34,
    "AGE_65_79": 35,
    "AGE_80P": 36,
    "BMI_UNDER": 37,
    "BMI_NORMAL": 38,
    "BMI_OVER": 39,
    "BMI_OBESE_1": 40,
    "BMI_OBESE_2": 41,
    "BMI_OBESE_3": 42,
}

DEMOGRAPHIC_ANCHOR_PREFIXES = {
    "ED_REGISTRATION",
    "ADMISSION",
    "HOSPITAL_ADMISSION",
    "ICU_ADMISSION",
    "TRANSFER_TO",
}

LEGACY_STRUCTURAL_BOUNDARY_PREFIXES = {
    "MEDS_DEATH",
    "ADMISSION",
    "DISCHARGE",
    "CAREUNIT_CHANGE",
    "HOSPITAL_ADMISSION",
    "HOSPITAL_DISCHARGE",
    "ICU_ADMISSION",
    "ICU_DISCHARGE",
    "TRANSFER_TO",
    "ED_REGISTRATION",
    "ED_OUT",
}


def build_subject_timeline(
    db: mr.SubjectDatabase,
    subject_id: int,
    encoders: Dict[TokenCategory, EventTokenEncoder],
    add_summary_tokens: Optional[List[EventToken]] = None,
    structural_event_map: Optional[Dict[str, float]] = None,
    window_hook_label: str = "window_boundary",
    attach_med_numeric: bool = True,
    structural_codebook: Optional[StructuralCodebook] = None,
    qual_obs_code_offset: int = 2_300_000,
    qual_obs_value_offset: int = 2_320_000,
    struct_action_offset: int = 2_400_000,
    struct_entity_offset: int = 2_420_000,
    emit_process_struct_tokens: bool = False,
    drop_original_process_marker_tokens: bool = False,
    emit_global_demographic_tokens: bool = True,
    special_token_offset: int = 0,
) -> List[EventToken]:
    """
    Build a flat token timeline for a single subject using rich EventToken bundles.
    This function emits only semantic tokens; special CLS/SEP markers are added later
    during batching/windowing. Any metadata emitted by encoders (cat_attrs/num_attrs)
    is preserved on the returned EventToken objects for collation-time projection.

    Steps:
      1. Reset encoder state (per subject).
      2. Optionally add summary/window-0 tokens (t_from_start=0).
      3. Walk through MEDS events in time order:
           - classify by code -> category
           - get corresponding encoder
           - compute dt_hours since previous emitted token
           - extend the token list with encoder outputs, carrying t_from_start
           - attach medication numeric_value (if available) and window hooks

    Parameters
    ----------
    structural_event_map : Optional[Dict[str, float]]
        Map of structural/procedural codes to keep as window boundaries.
    window_hook_label : str
        Label attached to EventToken.window_hook when an event hits the map.
    attach_med_numeric : bool
        If True, copy ev.numeric_value into EventToken.num_attrs["numeric_value"] for meds.
    emit_global_demographic_tokens : bool
        If True, emit lightweight global demographic tokens (sex, age bucket, BMI bucket)
        as SPECIAL tokens. Collation prefixes SPECIAL tokens into each window/chunk,
        making these globally attendable without creating extra timeline events.
    structural_codebook : Optional[StructuralCodebook]
        Optional codebook to force structural tokens for specific codes. Codes listed
        in structural_only will skip medtok tokens; codes listed in keep_original will
        also emit their original category tokens. If the codebook specifies
        boundary_labels/boundary_codes, only those structural tokens will carry
        window hooks (i.e., create new windows); the remainder act as in-window
        overlay markers.

    Returns
    -------
    tokens : List[EventToken]
        Event-level tokens ready for downstream windowing/collation.
    """
    subj = db[int(subject_id)]
    events = list(subj.events)  # already time-sorted per MEDS spec

    # Subject-level demographics used by the measurement value tokenizer (CVAE conditions).
    sex_val = infer_subject_sex(events, default=0.0)
    birth_ts = infer_birth_timestamp(events)

    # establish timeline start (earliest event timestamp if present)
    timeline_start: Optional[datetime] = None
    for ev in events:
        t_ev = getattr(ev, "time", None)
        if isinstance(t_ev, datetime):
            timeline_start = t_ev
            break

    # Demographic age token should anchor on first clinical event, not on MEDS_BIRTH.
    first_clinical_time: Optional[datetime] = None
    for ev in events:
        code_norm = str(getattr(ev, "code", "")).strip().upper()
        if code_norm == "MEDS_BIRTH" or code_norm.startswith("GENDER//"):
            continue
        t_ev = getattr(ev, "time", None)
        if isinstance(t_ev, datetime):
            first_clinical_time = t_ev
            break

    demographic_anchor_time: Optional[datetime] = None
    for ev in events:
        code_norm = str(getattr(ev, "code", "")).strip().upper()
        prefix = code_norm.split("//", 1)[0] if code_norm else ""
        if prefix not in DEMOGRAPHIC_ANCHOR_PREFIXES:
            continue
        t_ev = getattr(ev, "time", None)
        if isinstance(t_ev, datetime):
            demographic_anchor_time = t_ev
            break
    if demographic_anchor_time is None:
        demographic_anchor_time = first_clinical_time

    # 1) Reset per-subject state in encoders (dt_prev etc.)
    for enc in encoders.values():
        if hasattr(enc, "reset_state"):
            enc.reset_state()

    tokens: List[EventToken] = []
    if structural_event_map is None:
        structural_codes = set()
    elif hasattr(structural_event_map, "codes"):
        structural_codes = {str(k) for k in getattr(structural_event_map, "codes").keys()}
    else:
        structural_codes = {str(k) for k in structural_event_map}

    # Structural codebook helpers (direct structural token emission)
    struct_label2id: Dict[str, int] = {}
    struct_offset = 0
    struct_only: Set[str] = set()
    struct_keep_orig: Set[str] = set()
    if structural_codebook is not None:
        struct_label2id = structural_codebook.label2id()
        struct_only = structural_codebook.structural_only
        struct_keep_orig = structural_codebook.keep_original
        # derive offset from the provided STRUCTURAL encoder if present
        struct_enc = encoders.get(TokenCategory.STRUCTURAL)
        struct_offset = getattr(getattr(struct_enc, "vocab", None), "offset", 0) or structural_codebook.offset

    def _transition_attrs_for_event(*, code_str: Optional[str], label: Optional[str]) -> tuple[Dict[str, int], Optional[str]]:
        if structural_codebook is None or code_str is None:
            return {}, None
        transition_action = structural_codebook.transition_action(code=code_str, label=label)
        transition_action_id = structural_codebook.transition_action_id(code=code_str, label=label)
        transition_window_type_id = structural_codebook.window_type_id(
            code=code_str,
            label=label,
            action=transition_action,
        )
        attrs: Dict[str, int] = {}
        code_prefix = str(code_str).split("//", 1)[0].upper()
        if transition_action_id is not None:
            attrs["transition_action_id"] = int(transition_action_id)
        if transition_window_type_id is not None:
            attrs["transition_window_type_id"] = int(transition_window_type_id)
            if transition_action in {"open_next", "close_open"}:
                attrs["window_type_id"] = int(transition_window_type_id)
        if code_prefix in {
            "TRANSFER_TO",
            "HOSPITAL_ADMISSION",
            "ADMISSION",
            "ICU_ADMISSION",
            "ED_REGISTRATION",
            "STRUCT_START_ADM",
            "STRUCT_CAREUNIT_CHANGE",
            "STRUCT_START_OR",
        }:
            attrs["transition_admission_like"] = 1
        if code_prefix == "TRANSFER_TO":
            attrs["transition_transfer_like"] = 1
        if code_prefix in {
            "HOSPITAL_DISCHARGE",
            "DISCHARGE",
            "ICU_DISCHARGE",
            "ED_OUT",
            "STRUCT_END_ADM",
            "STRUCT_END_OR",
        }:
            attrs["transition_discharge_like"] = 1
        if code_prefix == "MEDS_DEATH":
            attrs["transition_death_like"] = 1
        return attrs, transition_action

    def _extract_med_numeric(ev: object) -> Optional[float]:
        """
        Pull raw numeric_value from MEDS medication events, guarding for NaN/None.
        """
        if not attach_med_numeric:
            return None
        if not hasattr(ev, "numeric_value"):
            return None
        try:
            val = float(getattr(ev, "numeric_value"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        return val

    def _t_from_start_hours(t: Optional[datetime]) -> float:
        if timeline_start is None or not isinstance(t, datetime):
            return 0.0
        return max(0.0, (t - timeline_start).total_seconds() / 3600.0)

    def _stable_local_id(raw: str, *, modulo: int = 900_000) -> int:
        data = str(raw).encode("utf-8", errors="ignore")
        mod = max(1, int(modulo))
        return 1 + (zlib.crc32(data) % mod)

    def _code_parts(code_value: Optional[str]) -> List[str]:
        if code_value is None:
            return []
        return [p.strip() for p in str(code_value).split("//")]

    def _normalize_obs_value(raw_value: object) -> str:
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
        return val[:96]

    def _extract_obs_value(ev_obj: object, *, code_value: Optional[str]) -> str:
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
                    return _normalize_obs_value(raw)
        if hasattr(ev_obj, "numeric_value"):
            try:
                nval = float(getattr(ev_obj, "numeric_value"))
            except (TypeError, ValueError):
                nval = None
            if nval is not None and math.isfinite(nval):
                return _normalize_obs_value(f"{nval:.3f}".rstrip("0").rstrip("."))
        parts = _code_parts(code_value)
        if len(parts) >= 3:
            return _normalize_obs_value(parts[2])
        return "UNK"

    def _emit_qual_obs_tokens(
        *,
        code_value: Optional[str],
        t_value: Optional[datetime],
        dt_value: float,
    ) -> List[EventToken]:
        if code_value is None:
            return []
        parts = _code_parts(code_value)
        if not parts:
            return []
        prefix = parts[0].upper()
        if prefix not in {
            "LAB",
            "VITAL",
            "MEAS",
            "SUBJECT_FLUID_OUTPUT",
            "SUBJECT_WEIGHT_AT_INFUSION",
            "OMR",
            "BLOOD PRESSURE",
        }:
            return []

        obs_code_lane = max(16, int(qual_obs_value_offset) - int(qual_obs_code_offset) - 1)
        obs_value_lane = max(16, int(struct_action_offset) - int(qual_obs_value_offset) - 1)

        item_or_code = parts[1] if len(parts) >= 2 else code_value
        local_code_id = _stable_local_id(f"{prefix}::{item_or_code}", modulo=obs_code_lane)
        obs_code_gid = int(qual_obs_code_offset) + int(local_code_id)

        obs_val_text = _extract_obs_value(ev_view, code_value=code_value)
        obs_val_upper = obs_val_text.upper()
        special_local = OBS_SPECIAL_VALUE_IDS.get(obs_val_upper)
        if special_local is not None and int(special_local) <= int(obs_value_lane):
            obs_val_local = int(special_local)
        else:
            obs_val_local = _stable_local_id(f"OBS_VAL::{obs_val_text}", modulo=obs_value_lane)
        obs_val_gid = int(qual_obs_value_offset) + int(obs_val_local)
        t_from_start = _t_from_start_hours(t_value) if isinstance(t_value, datetime) else 0.0

        return [
            EventToken(
                value_id=int(obs_code_gid),
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=t_from_start,
                dt_from_prev_hours=float(dt_value),
                cat_attrs={
                    "obs_bundle_pos": 1,
                    "obs_code_local_id": int(local_code_id),
                },
                num_attrs={},
                raw_time=t_value if isinstance(t_value, datetime) else None,
                window_hook=None,
            ),
            EventToken(
                value_id=int(obs_val_gid),
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=t_from_start,
                dt_from_prev_hours=0.0,
                cat_attrs={
                    "obs_bundle_pos": 2,
                    "obs_value_local_id": int(obs_val_local),
                },
                num_attrs={},
                raw_time=t_value if isinstance(t_value, datetime) else None,
                window_hook=None,
            ),
        ]

    def _parse_process_transition(code_value: Optional[str]) -> Optional[tuple[str, str, str]]:
        if code_value is None:
            return None
        code_norm = str(code_value).strip()
        if not code_norm:
            return None
        upper = code_norm.upper()
        if upper.startswith("INFUSION_START//"):
            return ("START", "INFUSION", code_norm.split("//", 1)[1])
        if upper.startswith("INFUSION_END//"):
            return ("END", "INFUSION", code_norm.split("//", 1)[1])

        parts = _code_parts(code_norm)
        if len(parts) >= 3 and parts[0].upper() in {"MEDICATION", "PROCEDURE"}:
            marker = parts[1].upper()
            if marker in {"START", "END", "STOP"}:
                entity = "//".join(parts[2:]).strip()
                if entity:
                    return (marker, parts[0].upper(), entity)
        return None

    def _emit_process_struct_tokens(
        *,
        transition: tuple[str, str, str],
        t_value: Optional[datetime],
        dt_value: float,
    ) -> List[EventToken]:
        action, domain, entity = transition
        action_id = PROCESS_ACTION_TO_ID.get(action.upper(), 0)
        domain_id = PROCESS_DOMAIN_TO_ID.get(domain.upper(), 0)
        if action_id <= 0 or domain_id <= 0:
            return []
        entity_local: int
        if entity.isdigit():
            entity_local = int(entity)
        else:
            entity_local = _stable_local_id(f"{domain.upper()}::{entity}")
        t_from_start = _t_from_start_hours(t_value) if isinstance(t_value, datetime) else 0.0
        return [
            EventToken(
                value_id=int(struct_action_offset) + int(action_id),
                category_id=int(TokenCategory.STRUCTURAL),
                t_from_start_hours=t_from_start,
                dt_from_prev_hours=float(dt_value),
                cat_attrs={
                    "struct_process_action_id": int(action_id),
                    "struct_process_domain_id": int(domain_id),
                },
                num_attrs={},
                raw_time=t_value if isinstance(t_value, datetime) else None,
                window_hook=None,
            ),
            EventToken(
                value_id=int(struct_entity_offset) + int(entity_local),
                category_id=int(TokenCategory.STRUCTURAL),
                t_from_start_hours=t_from_start,
                dt_from_prev_hours=0.0,
                cat_attrs={
                    "struct_process_action_id": int(action_id),
                    "struct_process_domain_id": int(domain_id),
                },
                num_attrs={},
                raw_time=t_value if isinstance(t_value, datetime) else None,
                window_hook=None,
            ),
        ]

    def _normalize_code(code_value: Optional[str]) -> str:
        if code_value is None:
            return ""
        return str(code_value).strip().upper()

    def _code_matches_alias(code_value: Optional[str], alias: str) -> bool:
        code_norm = _normalize_code(code_value)
        alias_norm = str(alias).strip().upper()
        return code_norm == alias_norm or code_norm.endswith(f"//{alias_norm}")

    def _extract_finite_numeric(ev_obj: object) -> Optional[float]:
        for attr in ("numeric_value", "value_as_number", "value", "result_value"):
            if not hasattr(ev_obj, attr):
                continue
            raw = getattr(ev_obj, attr)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                return float(val)
        return None

    def _latest_numeric_for_aliases_at_or_before(
        aliases: Set[str],
        *,
        anchor_time: Optional[datetime],
    ) -> Optional[float]:
        if anchor_time is None:
            return None
        latest_val: Optional[float] = None
        latest_ts: Optional[datetime] = None
        alias_norm = {str(a).strip().upper() for a in aliases}
        for ev_obj in events:
            code_value = getattr(ev_obj, "code", None)
            if not any(_code_matches_alias(code_value, alias) for alias in alias_norm):
                continue
            val = _extract_finite_numeric(ev_obj)
            if val is None:
                continue
            t_ev = getattr(ev_obj, "time", None)
            if not isinstance(t_ev, datetime):
                continue
            if t_ev > anchor_time:
                continue
            if latest_ts is None or t_ev >= latest_ts:
                latest_ts = t_ev
                latest_val = val
        return latest_val

    def _parse_sex_to_float(val: object) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float)):
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(f):
                return None
            return 1.0 if f >= 0.5 else 0.0
        s = str(val).strip().lower()
        if not s:
            return None
        if s.startswith("m"):
            return 1.0
        if s.startswith("f"):
            return 0.0
        return None

    def _infer_causal_sex_at_or_before(anchor_time: Optional[datetime]) -> Optional[float]:
        if anchor_time is None:
            return None
        for ev_obj in events:
            t_ev = getattr(ev_obj, "time", None)
            if not isinstance(t_ev, datetime) or t_ev > anchor_time:
                continue
            sex_attr = _parse_sex_to_float(getattr(ev_obj, "sex", None))
            if sex_attr is not None:
                return sex_attr
            gender_attr = _parse_sex_to_float(getattr(ev_obj, "gender", None))
            if gender_attr is not None:
                return gender_attr
            code_norm = _normalize_code(getattr(ev_obj, "code", None))
            if code_norm.startswith("GENDER//"):
                suffix = code_norm.split("//")[-1]
                parsed = _parse_sex_to_float(suffix)
                if parsed is not None:
                    return parsed
        return None

    def _age_bucket_token_id(age_years: float) -> int:
        if age_years < 18.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["AGE_0_17"]
        if age_years < 40.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["AGE_18_39"]
        if age_years < 65.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["AGE_40_64"]
        if age_years < 80.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["AGE_65_79"]
        return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["AGE_80P"]

    def _bmi_bucket_token_id(bmi_value: float) -> int:
        if bmi_value < 18.5:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["BMI_UNDER"]
        if bmi_value < 25.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["BMI_NORMAL"]
        if bmi_value < 30.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["BMI_OVER"]
        if bmi_value < 35.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["BMI_OBESE_1"]
        if bmi_value < 40.0:
            return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["BMI_OBESE_2"]
        return GLOBAL_DEMOGRAPHIC_TOKEN_IDS["BMI_OBESE_3"]

    def _build_global_demographic_tokens() -> List[EventToken]:
        if not emit_global_demographic_tokens:
            return []

        out: List[EventToken] = []
        seen_ids: Set[int] = set()

        def _emit_demog_token(local_id: int, *, feature_id: int) -> None:
            gid = int(special_token_offset) + int(local_id)
            if gid in seen_ids:
                return
            seen_ids.add(gid)
            out.append(
                EventToken(
                    value_id=int(gid),
                    category_id=int(TokenCategory.SPECIAL),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,
                    cat_attrs={"global_demographic": 1, "demographic_feature_id": int(feature_id)},
                    num_attrs={},
                    raw_time=None,
                    window_hook=None,
                )
            )

        # 1) Sex token (only if observed at/before admission anchor).
        causal_sex = _infer_causal_sex_at_or_before(demographic_anchor_time)
        if causal_sex is not None:
            sex_tok = GLOBAL_DEMOGRAPHIC_TOKEN_IDS["SEX_M"] if float(causal_sex) >= 0.5 else GLOBAL_DEMOGRAPHIC_TOKEN_IDS["SEX_F"]
            _emit_demog_token(sex_tok, feature_id=1)

        # 2) Age bucket at admission anchor (if birth timestamp is known).
        if birth_ts is not None and demographic_anchor_time is not None:
            age_at_start = age_years_from_timestamps(float(demographic_anchor_time.timestamp()), float(birth_ts))
            if math.isfinite(age_at_start) and age_at_start >= 0.0:
                _emit_demog_token(_age_bucket_token_id(float(age_at_start)), feature_id=2)

        # 3) BMI bucket from values known at/before admission anchor.
        bmi_val = _latest_numeric_for_aliases_at_or_before({"BMI (KG/M2)"}, anchor_time=demographic_anchor_time)
        if bmi_val is None:
            wt_lbs = _latest_numeric_for_aliases_at_or_before({"WEIGHT (LBS)"}, anchor_time=demographic_anchor_time)
            ht_in = _latest_numeric_for_aliases_at_or_before({"HEIGHT (INCHES)"}, anchor_time=demographic_anchor_time)
            if wt_lbs is not None and ht_in is not None and ht_in > 0.0:
                bmi_val = float(wt_lbs) * 703.0 / (float(ht_in) ** 2)
        if bmi_val is not None and math.isfinite(bmi_val) and bmi_val > 0.0:
            _emit_demog_token(_bmi_bucket_token_id(float(bmi_val)), feature_id=3)

        return out

    # 2) Optional global summary/window-0 tokens (coerced to EventToken, t=0)
    if add_summary_tokens:
        for tok in add_summary_tokens:
            tokens.append(
                EventToken(
                    value_id=int(tok.value_id),
                    category_id=int(getattr(tok, "category_id", TokenCategory.SPECIAL)),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,
                    cat_attrs=dict(tok.cat_attrs),
                    num_attrs=dict(tok.num_attrs),
                    raw_time=None,
                    window_hook=None,
                )
            )
    tokens.extend(_build_global_demographic_tokens())

    # 3) Timeline tokens from events
    last_emitted_time: Optional[datetime] = None

    for ev in events:
        # Wrap raw event with per-event demographics for downstream encoders.
        age_years = infer_event_age_years(ev, birth_ts=birth_ts)
        ev_view = _EventWithDemographics(ev, age_years=age_years, sex=sex_val)

        code = getattr(ev_view, "code", None)
        code_str = str(code) if code is not None else None
        category = classify_code_to_category(code)
        encoder = encoders.get(category)

        t = getattr(ev_view, "time", None)
        # dt relative to previous emitted token with a timestamp
        dt_hours = 0.0
        if isinstance(t, datetime) and isinstance(last_emitted_time, datetime):
            dt_hours = max(0.0, (t - last_emitted_time).total_seconds() / 3600.0)

        # Structural codebook: emit structural token regardless of routing
        emitted_for_event: List[EventToken] = []
        process_transition = (
            _parse_process_transition(code_str) if emit_process_struct_tokens else None
        )
        if process_transition is not None:
            emitted_for_event.extend(
                _emit_process_struct_tokens(
                    transition=process_transition,
                    t_value=t if isinstance(t, datetime) else None,
                    dt_value=dt_hours,
                )
            )
        struct_hit = structural_codebook is not None and code_str in structural_codebook.code2label
        routed_transition_attrs: Dict[str, int] = {}
        routed_transition_action: Optional[str] = None
        if category == TokenCategory.STRUCTURAL:
            routed_transition_attrs, routed_transition_action = _transition_attrs_for_event(
                code_str=code_str,
                label=None,
            )
        if struct_hit:
            label = structural_codebook.code2label.get(code_str, "")
            label_id = struct_label2id.get(label, 0)
            val_id = struct_offset + label_id
            struct_transition_attrs, _ = _transition_attrs_for_event(code_str=code_str, label=label)
            is_boundary = bool(window_hook_label) and structural_codebook.is_window_boundary(code=code_str, label=label)
            struct_attrs = {"struct_label_id": int(label_id)}
            struct_attrs.update(struct_transition_attrs)
            struct_tok = EventToken(
                value_id=val_id,
                category_id=int(TokenCategory.STRUCTURAL),
                t_from_start_hours=_t_from_start_hours(t) if isinstance(t, datetime) else 0.0,
                dt_from_prev_hours=dt_hours if not emitted_for_event else 0.0,
                cat_attrs=struct_attrs,
                num_attrs={},
                raw_time=t if isinstance(t, datetime) else None,
                window_hook=window_hook_label if is_boundary else None,
            )
            emitted_for_event.append(struct_tok)

        # Skip original token if structural-only, or if the routed category is already
        # STRUCTURAL and the codebook emitted the canonical structural marker.
        if struct_hit and (
            category == TokenCategory.STRUCTURAL
            or (code_str not in struct_keep_orig and code_str in struct_only)
        ):
            # nothing else; record timestamp advance
            tokens.extend(emitted_for_event)
            if emitted_for_event and isinstance(t, datetime):
                last_emitted_time = t
            continue

        if process_transition is not None and drop_original_process_marker_tokens:
            tokens.extend(emitted_for_event)
            if emitted_for_event and isinstance(t, datetime):
                last_emitted_time = t
            continue

        if encoder is None:
            # unsupported category -> only structural tokens (if any)
            tokens.extend(emitted_for_event)
            if emitted_for_event and isinstance(t, datetime):
                last_emitted_time = t
            continue

        # Encoders may return multiple tokens for a single event (e.g., MEDTOK)
        event_tokens = encoder.encode_event(ev_view, dt_hours=dt_hours)
        if not event_tokens and category == TokenCategory.MEASUREMENT:
            event_tokens = _emit_qual_obs_tokens(
                code_value=code_str,
                t_value=t if isinstance(t, datetime) else None,
                dt_value=dt_hours if not emitted_for_event else 0.0,
            )
        should_hook = False
        if bool(window_hook_label) and code_str is not None and not (structural_codebook is not None and struct_hit):
            if routed_transition_action in ACTIVE_TRANSITION_ACTIONS:
                should_hook = True
            elif routed_transition_action == "suppress":
                should_hook = False
            else:
                code_prefix = code_str.split("//", 1)[0].upper()
                legacy_struct_boundary = (
                    category == TokenCategory.STRUCTURAL
                    and code_prefix in LEGACY_STRUCTURAL_BOUNDARY_PREFIXES
                )
                should_hook = (
                    (structural_event_map is None and legacy_struct_boundary)
                    or (structural_event_map is not None and code_str in structural_codes)
                )

        # Guarantee boundary presence if map says so (fallback to UNK)
        if not event_tokens and should_hook:
            fallback_id = 0
            base_vocab = getattr(encoder, "base_vocab", None)
            if base_vocab is not None:
                fallback_id = base_vocab.offset + base_vocab.unk_id
            event_tokens = [
                EventToken(
                    value_id=int(fallback_id),
                    category_id=int(category),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=float(dt_hours),
                    cat_attrs={},
                    num_attrs={},
                )
            ]
        if not event_tokens:
            # still may have structural tokens
            tokens.extend(emitted_for_event)
            if emitted_for_event and isinstance(t, datetime):
                last_emitted_time = t
            continue

        t_from_start = _t_from_start_hours(t) if isinstance(t, datetime) else 0.0

        # First token in this event gets dt_hours; subsequent get 0
        dt_budget = dt_hours
        if emitted_for_event:
            dt_budget = 0.0  # structural token already consumed dt

        for idx, tok in enumerate(event_tokens):
            cat_attrs = dict(tok.cat_attrs)
            if idx == 0 and routed_transition_attrs:
                cat_attrs.update(routed_transition_attrs)
            emitted = EventToken(
                value_id=int(tok.value_id),
                category_id=int(category),
                t_from_start_hours=t_from_start,
                dt_from_prev_hours=dt_budget if idx == 0 else 0.0,
                cat_attrs=cat_attrs,
                num_attrs=dict(tok.num_attrs),
                raw_time=t if isinstance(t, datetime) else tok.raw_time,
                window_hook=window_hook_label if should_hook else tok.window_hook,
            )
            if category == TokenCategory.MEDICATION and attach_med_numeric:
                med_val = _extract_med_numeric(ev)
                # preserve encoder-provided value unless missing; attach None explicitly
                if "numeric_value" not in emitted.num_attrs or emitted.num_attrs.get("numeric_value") is None:
                    emitted.num_attrs["numeric_value"] = med_val

            emitted_for_event.append(emitted)

        tokens.extend(emitted_for_event)

        if emitted_for_event and isinstance(t, datetime):
            last_emitted_time = t

    return tokens
