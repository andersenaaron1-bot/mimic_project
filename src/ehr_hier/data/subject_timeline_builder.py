from typing import Dict, List, Optional
from datetime import datetime

import meds_reader as mr   # pip install meds_reader

from .token_types import TokenTriplet, TokenCategory, SpecialToken
from .event_router import classify_code_to_category
from src.ehr_hier.tokenizers.base_encoder import EventTokenEncoder


def build_subject_timeline(
    db: mr.SubjectDatabase,
    subject_id: int,
    encoders: Dict[TokenCategory, EventTokenEncoder],
    add_summary_tokens: Optional[List[TokenTriplet]] = None,
) -> List[TokenTriplet]:
    """
    Build a flat token timeline for a single subject.

    Steps:
      1. Reset encoder state (per subject).
      2. Add [PT_CLS].
      3. Optionally add summary tokens + [SEP].
      4. Walk through MEDS events in time order:
           - classify by code → category
           - get corresponding encoder
           - compute dt_hours since previous emitted token
           - extend the token list with encoder outputs
      5. Append a final [SEP] as timeline terminator.

    Returns
    -------
    tokens : List[TokenTriplet]
        Ready to be batched and fed into a transformer.
    """
    subj = db[int(subject_id)]
    events = list(subj.events)  # already time-sorted per MEDS spec

    # 1) Reset per-subject state in encoders (dt_prev etc.)
    for enc in encoders.values():
        if hasattr(enc, "reset_state"):
            enc.reset_state()

    tokens: List[TokenTriplet] = []

    # 2) [PT_CLS] at the very beginning
    tokens.append(
        TokenTriplet(
            value_id=int(SpecialToken.PT_CLS),
            category_id=int(TokenCategory.SPECIAL),
            dt_hours=0.0,
        )
    )

    # 3) Optional global summary tokens (already prepared as TokenTriplet list)
    if add_summary_tokens:
        # assume caller set dt_hours appropriately; we set dt for 1st summary token
        # as 0 relative to [PT_CLS] for simplicity
        if len(add_summary_tokens) > 0:
            add_summary_tokens[0].dt_hours = 0.0
        tokens.extend(add_summary_tokens)

        # Separator between summary and timeline
        tokens.append(
            TokenTriplet(
                value_id=int(SpecialToken.SEP),
                category_id=int(TokenCategory.SPECIAL),
                dt_hours=0.0,
            )
        )

    # 4) Timeline tokens from events
    last_time: Optional[datetime] = None
    have_emitted_any = len(tokens) > 0

    for ev in events:
        code = getattr(ev, "code", None)
        category = classify_code_to_category(code)
        encoder = encoders.get(category)
        if encoder is None:
            # unsupported category → skip
            continue

        t = getattr(ev, "time", None)
        if isinstance(t, datetime):
            if have_emitted_any and last_time is not None:
                dt_hours = max(0.0, (t - last_time).total_seconds() / 3600.0)
            else:
                dt_hours = 0.0
            last_time = t
        else:
            dt_hours = 0.0

        # Encoders may return multiple tokens for a single event (e.g., MEDTOK)
        event_tokens = encoder.encode_event(ev, dt_hours=dt_hours)
        if not event_tokens:
            continue

        # Convention: encoder uses dt_hours only on FIRST token for that event;
        # if it returns multiple tokens, it should set dt_hours=0.0 on the rest.
        tokens.extend(event_tokens)
        have_emitted_any = True

    # 5) Final [SEP] as end-of-timeline marker
    tokens.append(
        TokenTriplet(
            value_id=int(SpecialToken.SEP),
            category_id=int(TokenCategory.SPECIAL),
            dt_hours=0.0,
        )
    )

    return tokens
