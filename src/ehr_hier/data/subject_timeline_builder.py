from typing import Dict, List, Optional, Set
from datetime import datetime
import math

import meds_reader as mr   # pip install meds_reader

from .token_types import EventToken, TokenCategory
from .event_router import classify_code_to_category
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.data.structural_codes import StructuralCodebook
from src.ehr_hier.data.demographics import (
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


def build_subject_timeline(
    db: mr.SubjectDatabase,
    subject_id: int,
    encoders: Dict[TokenCategory, EventTokenEncoder],
    add_summary_tokens: Optional[List[EventToken]] = None,
    structural_event_map: Optional[Dict[str, float]] = None,
    window_hook_label: str = "window_boundary",
    attach_med_numeric: bool = True,
    structural_codebook: Optional[StructuralCodebook] = None,
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
        if transition_action_id is not None:
            attrs["transition_action_id"] = int(transition_action_id)
        if transition_window_type_id is not None:
            attrs["transition_window_type_id"] = int(transition_window_type_id)
            if transition_action in {"open_next", "close_open"}:
                attrs["window_type_id"] = int(transition_window_type_id)
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
                dt_from_prev_hours=dt_hours,
                cat_attrs=struct_attrs,
                num_attrs={},
                raw_time=t if isinstance(t, datetime) else None,
                window_hook=window_hook_label if is_boundary else None,
            )
            emitted_for_event.append(struct_tok)

        # Skip original token if structural-only, or if the routed category is already
        # STRUCTURAL and the codebook emitted the canonical structural marker.
        if struct_hit and code_str not in struct_keep_orig and (
            code_str in struct_only or category == TokenCategory.STRUCTURAL
        ):
            # nothing else; record timestamp advance
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
        should_hook = False
        if bool(window_hook_label) and code_str is not None and not (structural_codebook is not None and struct_hit):
            if routed_transition_action in ACTIVE_TRANSITION_ACTIONS:
                should_hook = True
            elif routed_transition_action == "suppress":
                should_hook = False
            else:
                should_hook = (
                    (structural_event_map is None and category == TokenCategory.STRUCTURAL)
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
