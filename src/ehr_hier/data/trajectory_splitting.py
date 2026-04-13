from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from src.ehr_hier.data.demographics import build_global_demographic_special_tokens
from src.ehr_hier.data.event_frames import (
    EventFrame,
    EventPayloadKind,
    clone_event_frame,
    ensure_event_frames,
    flatten_event_frames,
)
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


@dataclass
class SegmentedFrameWindow:
    frames: list[EventFrame]
    window_type_id: int
    start_time_hours: float
    window_site_id: int = 0
    opening_action: Optional[str] = None
    closing_action: Optional[str] = None
    opening_time_hours: Optional[float] = None
    closing_time_hours: Optional[float] = None
    closing_discharge_like: bool = False

    @property
    def tokens(self) -> list[EventToken]:
        return flatten_event_frames(self.frames, clone=False)


def split_special_tokens(
    timeline: Iterable[EventFrame | EventToken],
) -> tuple[list[EventFrame], list[EventFrame]]:
    specials: list[EventFrame] = []
    events: list[EventFrame] = []
    for frame in ensure_event_frames(timeline):
        if int(frame.category_id) == int(TokenCategory.SPECIAL):
            specials.append(frame)
        else:
            events.append(frame)
    return specials, events


def _frame_start_time(frame: EventFrame) -> float:
    return float(frame.t_from_start_hours)


def _segment_event_frames(
    events: list[EventFrame],
    *,
    config: WindowSegmentationConfig,
) -> list[SegmentedFrameWindow]:
    flat_tokens = flatten_event_frames(events, clone=False)
    if not flat_tokens:
        return []
    token_windows = segment_event_tokens(flat_tokens, config=config)
    token_to_frame_idx: dict[int, int] = {}
    for frame_idx, frame in enumerate(events):
        for tok in frame.token_bundle:
            token_to_frame_idx[id(tok)] = int(frame_idx)

    out: list[SegmentedFrameWindow] = []
    for window in token_windows:
        ordered_frame_ids: list[int] = []
        seen: set[int] = set()
        for tok in window.tokens:
            frame_idx = token_to_frame_idx.get(id(tok))
            if frame_idx is None or frame_idx in seen:
                continue
            seen.add(frame_idx)
            ordered_frame_ids.append(int(frame_idx))
        if not ordered_frame_ids:
            continue
        frame_start = int(min(ordered_frame_ids))
        frame_stop = int(max(ordered_frame_ids)) + 1
        out.append(
            SegmentedFrameWindow(
                frames=[clone_event_frame(frame) for frame in events[frame_start:frame_stop]],
                window_type_id=int(window.window_type_id),
                start_time_hours=float(window.start_time_hours),
                window_site_id=int(window.window_site_id),
                opening_action=window.opening_action,
                closing_action=window.closing_action,
                opening_time_hours=window.opening_time_hours,
                closing_time_hours=window.closing_time_hours,
                closing_discharge_like=bool(window.closing_discharge_like),
            )
        )
    return out


def _ensure_frame_window(window: SegmentedFrameWindow | SegmentedWindow) -> SegmentedFrameWindow:
    if isinstance(window, SegmentedFrameWindow):
        return window
    return SegmentedFrameWindow(
        frames=ensure_event_frames(window.tokens),
        window_type_id=int(window.window_type_id),
        start_time_hours=float(window.start_time_hours),
        window_site_id=int(window.window_site_id),
        opening_action=window.opening_action,
        closing_action=window.closing_action,
        opening_time_hours=window.opening_time_hours,
        closing_time_hours=window.closing_time_hours,
        closing_discharge_like=bool(window.closing_discharge_like),
    )


def _window_fragment(
    window: SegmentedFrameWindow,
    frames: list[EventFrame],
    *,
    force_first_window_type: bool = False,
) -> SegmentedFrameWindow:
    if not frames:
        return SegmentedFrameWindow(
            frames=[],
            window_type_id=int(window.window_type_id),
            start_time_hours=float(window.start_time_hours),
            window_site_id=int(window.window_site_id),
        )
    cloned_frames = [
        clone_event_frame(
            frame,
            force_window_type_id=(int(window.window_type_id) if force_first_window_type and idx == 0 else None),
            force_window_site_id=(int(window.window_site_id) if force_first_window_type and idx == 0 else None),
        )
        for idx, frame in enumerate(frames)
    ]
    return SegmentedFrameWindow(
        frames=cloned_frames,
        window_type_id=int(window.window_type_id),
        start_time_hours=_frame_start_time(cloned_frames[0]),
        window_site_id=int(window.window_site_id),
        opening_action=window.opening_action,
        closing_action=window.closing_action,
        opening_time_hours=window.opening_time_hours,
        closing_time_hours=window.closing_time_hours,
        closing_discharge_like=bool(window.closing_discharge_like),
    )


def _split_window_at_time(
    window: SegmentedFrameWindow,
    *,
    cutoff_time_hours: float,
) -> tuple[Optional[SegmentedFrameWindow], Optional[SegmentedFrameWindow]]:
    if not window.frames:
        return window, None
    split_idx = None
    for idx, frame in enumerate(window.frames):
        if float(frame.t_from_start_hours) > float(cutoff_time_hours):
            split_idx = idx
            break
    if split_idx is None or split_idx <= 0:
        return window, None
    left = _window_fragment(window, list(window.frames[:split_idx]))
    right = _window_fragment(
        window,
        list(window.frames[split_idx:]),
        force_first_window_type=True,
    )
    return left, right


def split_segmented_windows_into_trajectories(
    windows: list[SegmentedFrameWindow | SegmentedWindow],
    *,
    config: TrajectorySplitConfig,
    post_discharge_window_type_id: Optional[int],
) -> list[list[SegmentedFrameWindow]]:
    normalized_windows = [_ensure_frame_window(window) for window in windows]
    if not normalized_windows:
        return []
    mode = str(config.mode or "full_subject").strip().lower()
    if mode == "full_subject":
        return [[window for window in normalized_windows]]
    if mode != "admission_chain":
        raise ValueError(f"Unsupported trajectory split mode {config.mode!r}")

    post_type = int(post_discharge_window_type_id) if post_discharge_window_type_id is not None else None
    cutoff_h = float(config.post_discharge_cutoff_hours)

    trajectories: list[list[SegmentedFrameWindow]] = []
    current: list[SegmentedFrameWindow] = []
    active_cutoff: Optional[float] = None

    for window in normalized_windows:
        if active_cutoff is not None and current:
            if (
                post_type is not None
                and int(window.window_type_id) == int(post_type)
                and window.frames
                and float(window.frames[0].t_from_start_hours) <= float(active_cutoff) < float(window.frames[-1].t_from_start_hours)
            ):
                left, right = _split_window_at_time(window, cutoff_time_hours=float(active_cutoff))
                if left is not None and left.frames:
                    current.append(left)
                if current:
                    trajectories.append(list(current))
                current = []
                active_cutoff = None
                if right is not None and right.frames:
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
    timeline: Iterable[EventFrame | EventToken],
    segmentation_config: WindowSegmentationConfig,
    split_config: TrajectorySplitConfig,
    subject_metadata: Optional[dict[str, Any]] = None,
    special_token_offset: int = 0,
) -> list[list[EventFrame]]:
    specials, events = split_special_tokens(timeline)
    if not events:
        return [list(specials)] if specials else []

    windows = _segment_event_frames(events, config=segmentation_config)
    if not windows:
        return [list(specials) + [clone_event_frame(frame) for frame in events]]

    trajectory_windows = split_segmented_windows_into_trajectories(
        windows,
        config=split_config,
        post_discharge_window_type_id=segmentation_config.post_discharge_window_type_id,
    )

    out: list[list[EventFrame]] = []
    for windows_in_traj in trajectory_windows:
        traj_events: list[EventFrame] = []
        for window_idx, window in enumerate(windows_in_traj):
            force_first = bool(window_idx == 0 and window.frames)
            for frame_idx, frame in enumerate(window.frames):
                traj_events.append(
                    clone_event_frame(
                        frame,
                        force_window_type_id=(int(window.window_type_id) if force_first and frame_idx == 0 else None),
                        force_window_site_id=(int(window.window_site_id) if force_first and frame_idx == 0 else None),
                    )
                )
        if not traj_events:
            continue

        time_offset_hours = float(traj_events[0].t_from_start_hours)
        rebased_events = [
            clone_event_frame(
                frame,
                time_offset_hours=time_offset_hours,
                force_dt_from_prev_hours=(0.0 if idx == 0 else None),
            )
            for idx, frame in enumerate(traj_events)
        ]

        if subject_metadata is not None:
            first_tok = traj_events[0].token_bundle[0]
            anchor_epoch_s = (
                float(first_tok.raw_time.timestamp())
                if first_tok.raw_time is not None and hasattr(first_tok.raw_time, "timestamp")
                else None
            )
            traj_special_tokens = build_global_demographic_special_tokens(
                metadata=dict(subject_metadata),
                anchor_time_hours=float(traj_events[0].t_from_start_hours),
                anchor_time_epoch_s=anchor_epoch_s,
                token_ids=dict(GLOBAL_DEMOGRAPHIC_TOKEN_IDS),
                special_token_offset=int(special_token_offset),
            )
            traj_specials = ensure_event_frames(
                traj_special_tokens,
                payload_kind=EventPayloadKind.DEMOGRAPHIC,
            )
        else:
            traj_specials = [clone_event_frame(frame) for frame in specials]
        out.append(list(traj_specials) + rebased_events)
    return out
