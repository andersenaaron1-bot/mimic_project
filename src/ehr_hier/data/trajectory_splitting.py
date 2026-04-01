from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from src.ehr_hier.data.demographics import build_global_demographic_special_tokens
from src.ehr_hier.data.subject_timeline_builder import GLOBAL_DEMOGRAPHIC_TOKEN_IDS
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.data.window_segmentation import (
    SegmentedWindow,
    WindowSegmentationConfig,
    segment_event_tokens,
)


@dataclass(frozen=True)
class TrajectorySplitConfig:
    mode: str = "full_subject"
    post_discharge_cutoff_hours: float = 31.0 * 24.0


def _clone_token(
    tok: EventToken,
    *,
    force_window_type_id: Optional[int] = None,
    force_window_site_id: Optional[int] = None,
    time_offset_hours: float = 0.0,
    force_dt_from_prev_hours: Optional[float] = None,
) -> EventToken:
    cat_attrs = dict(tok.cat_attrs or {})
    if force_window_type_id is not None:
        cat_attrs["window_type_id"] = int(force_window_type_id)
    if force_window_site_id is not None and int(force_window_site_id) > 0:
        cat_attrs["window_site_id"] = int(force_window_site_id)
    return EventToken(
        value_id=int(tok.value_id),
        category_id=int(tok.category_id),
        t_from_start_hours=max(0.0, float(tok.t_from_start_hours) - float(time_offset_hours)),
        dt_from_prev_hours=(
            float(force_dt_from_prev_hours)
            if force_dt_from_prev_hours is not None
            else float(tok.dt_from_prev_hours)
        ),
        cat_attrs=cat_attrs,
        num_attrs=dict(tok.num_attrs or {}),
        raw_time=tok.raw_time,
        window_hook=tok.window_hook,
    )


def split_special_tokens(timeline: Iterable[EventToken]) -> tuple[list[EventToken], list[EventToken]]:
    specials: list[EventToken] = []
    events: list[EventToken] = []
    for tok in timeline:
        cloned = _clone_token(tok)
        if int(cloned.category_id) == int(TokenCategory.SPECIAL):
            specials.append(cloned)
        else:
            events.append(cloned)
    return specials, events


def _window_fragment(
    window: SegmentedWindow,
    tokens: list[EventToken],
    *,
    start_idx_offset: int,
    end_idx_offset: int,
    force_first_window_type: bool = False,
) -> SegmentedWindow:
    if not tokens:
        return SegmentedWindow(
            tokens=[],
            window_type_id=int(window.window_type_id),
            start_time_hours=float(window.start_time_hours),
            window_site_id=int(window.window_site_id),
        )
    cloned_tokens = [
        _clone_token(
            tok,
            force_window_type_id=(int(window.window_type_id) if force_first_window_type and idx == 0 else None),
            force_window_site_id=(int(window.window_site_id) if force_first_window_type and idx == 0 else None),
        )
        for idx, tok in enumerate(tokens)
    ]
    local_breaks = [
        int(idx) - int(start_idx_offset)
        for idx in (window.chunk_break_token_indices or [])
        if int(start_idx_offset) < int(idx) < int(end_idx_offset)
    ]
    return SegmentedWindow(
        tokens=cloned_tokens,
        window_type_id=int(window.window_type_id),
        start_time_hours=float(cloned_tokens[0].t_from_start_hours),
        window_site_id=int(window.window_site_id),
        opening_action=window.opening_action if int(start_idx_offset) == 0 else None,
        closing_action=window.closing_action if int(end_idx_offset) >= len(window.tokens) else None,
        opening_time_hours=(
            window.opening_time_hours if int(start_idx_offset) == 0 else float(cloned_tokens[0].t_from_start_hours)
        ),
        closing_time_hours=window.closing_time_hours if int(end_idx_offset) >= len(window.tokens) else None,
        closing_discharge_like=bool(window.closing_discharge_like and int(end_idx_offset) >= len(window.tokens)),
        chunk_break_token_indices=local_breaks,
    )


def _split_window_at_time(window: SegmentedWindow, *, cutoff_time_hours: float) -> tuple[Optional[SegmentedWindow], Optional[SegmentedWindow]]:
    if not window.tokens:
        return window, None
    split_idx = None
    for idx, tok in enumerate(window.tokens):
        if float(tok.t_from_start_hours) > float(cutoff_time_hours):
            split_idx = idx
            break
    if split_idx is None or split_idx <= 0:
        return window, None
    left = _window_fragment(window, list(window.tokens[:split_idx]), start_idx_offset=0, end_idx_offset=split_idx)
    right = _window_fragment(
        window,
        list(window.tokens[split_idx:]),
        start_idx_offset=split_idx,
        end_idx_offset=len(window.tokens),
        force_first_window_type=True,
    )
    return left, right


def split_segmented_windows_into_trajectories(
    windows: List[SegmentedWindow],
    *,
    config: TrajectorySplitConfig,
    post_discharge_window_type_id: Optional[int],
) -> list[list[SegmentedWindow]]:
    if not windows:
        return []
    mode = str(config.mode or "full_subject").strip().lower()
    if mode == "full_subject":
        return [[window for window in windows]]
    if mode != "admission_chain":
        raise ValueError(f"Unsupported trajectory split mode {config.mode!r}")

    post_type = int(post_discharge_window_type_id) if post_discharge_window_type_id is not None else None
    cutoff_h = float(config.post_discharge_cutoff_hours)

    trajectories: list[list[SegmentedWindow]] = []
    current: list[SegmentedWindow] = []
    active_cutoff: Optional[float] = None

    for window in windows:
        if active_cutoff is not None and current:
            if (
                post_type is not None
                and int(window.window_type_id) == int(post_type)
                and window.tokens
                and float(window.tokens[0].t_from_start_hours) <= float(active_cutoff) < float(window.tokens[-1].t_from_start_hours)
            ):
                left, right = _split_window_at_time(window, cutoff_time_hours=float(active_cutoff))
                if left is not None and left.tokens:
                    current.append(left)
                if current:
                    trajectories.append(list(current))
                current = []
                active_cutoff = None
                if right is not None and right.tokens:
                    current.append(right)
                continue
            if float(window.start_time_hours) > float(active_cutoff):
                trajectories.append(list(current))
                current = []
                active_cutoff = None

        current.append(window)

        if active_cutoff is not None and (post_type is None or int(window.window_type_id) != int(post_type)):
            active_cutoff = None

        if window.closing_discharge_like and window.closing_time_hours is not None:
            active_cutoff = float(window.closing_time_hours) + float(cutoff_h)

    if current:
        trajectories.append(list(current))
    return trajectories


def build_trajectory_timelines(
    *,
    timeline: Iterable[EventToken],
    segmentation_config: WindowSegmentationConfig,
    split_config: TrajectorySplitConfig,
    subject_metadata: Optional[dict[str, Any]] = None,
    special_token_offset: int = 0,
) -> list[list[EventToken]]:
    specials, events = split_special_tokens(timeline)
    if not events:
        return [list(specials)] if specials else []
    windows = segment_event_tokens(events, config=segmentation_config)
    if not windows:
        return [list(specials) + list(events)]
    trajectory_windows = split_segmented_windows_into_trajectories(
        windows,
        config=split_config,
        post_discharge_window_type_id=segmentation_config.post_discharge_window_type_id,
    )
    out: list[list[EventToken]] = []
    for windows_in_traj in trajectory_windows:
        traj_events: list[EventToken] = []
        for window_idx, window in enumerate(windows_in_traj):
            force_first = bool(window_idx == 0 and window.tokens)
            for tok_idx, tok in enumerate(window.tokens):
                traj_events.append(
                    _clone_token(
                        tok,
                        force_window_type_id=(int(window.window_type_id) if force_first and tok_idx == 0 else None),
                        force_window_site_id=(int(window.window_site_id) if force_first and tok_idx == 0 else None),
                    )
                )
        if not traj_events:
            continue
        time_offset_hours = float(traj_events[0].t_from_start_hours)
        rebased_events = [
            _clone_token(
                tok,
                time_offset_hours=time_offset_hours,
                force_dt_from_prev_hours=(0.0 if idx == 0 else None),
            )
            for idx, tok in enumerate(traj_events)
        ]

        if subject_metadata is not None:
            first_tok = traj_events[0]
            anchor_epoch_s = (
                float(first_tok.raw_time.timestamp())
                if first_tok.raw_time is not None and hasattr(first_tok.raw_time, "timestamp")
                else None
            )
            traj_specials = build_global_demographic_special_tokens(
                metadata=dict(subject_metadata),
                anchor_time_hours=float(first_tok.t_from_start_hours),
                anchor_time_epoch_s=anchor_epoch_s,
                token_ids=dict(GLOBAL_DEMOGRAPHIC_TOKEN_IDS),
                special_token_offset=int(special_token_offset),
            )
        else:
            traj_specials = [_clone_token(tok) for tok in specials]
        out.append(list(traj_specials) + rebased_events)
    return out
