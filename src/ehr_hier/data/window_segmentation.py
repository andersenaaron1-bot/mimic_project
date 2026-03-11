from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.ehr_hier.data.structural_codes import TRANSITION_ACTION_FROM_ID
from src.ehr_hier.data.token_types import EventToken, TokenCategory


ACTIVE_TRANSITION_ACTIONS = {"open_next", "close_current", "close_open"}


@dataclass(frozen=True)
class WindowSegmentationConfig:
    """
    Runtime segmentation policy used by the collator and any future generation code.

    The policy intentionally mirrors the bundle-based audit path:
      1. detect transition candidates from explicit transition metadata or legacy hooks
      2. group nearby candidates into local bundles
      3. merge sparse administrative transition chains
      4. split the linear sequence with directional actions
    """

    bundle_gap_hours: float = 0.5
    bundle_max_index_gap: int = 2
    merge_transition_chains: bool = True
    chain_gap_hours: float = 6.0
    chain_max_intervening_tokens: int = 16
    rebalance_dense_windows: bool = True
    rebalance_target_frac: float = 0.8
    rebalance_min_tokens: int = 32
    rebalance_tail_tokens: int = 16
    unk_window_type_id: int = 0
    # Optional fallback type for leading windows before the first typed transition.
    default_first_window_type_id: int | None = None
    # Carry forward the most recent known type for untyped windows.
    propagate_prev_type_for_unknown_windows: bool = True
    # Optionally synthesize a short discharge-to-readmission gap window.
    enable_inter_admission_windows: bool = False
    inter_admission_window_type_id: int | None = None
    inter_admission_max_gap_hours: float = 24.0
    inter_admission_token_id: int | None = None
    inter_admission_struct_label_id: int | None = None


@dataclass
class SegmentedWindow:
    tokens: List[EventToken]
    window_type_id: int
    start_time_hours: float
    opening_action: Optional[str] = None
    closing_action: Optional[str] = None
    chunks: List["SegmentedChunk"] = field(default_factory=list)
    truncated_chunks: int = 0
    truncated_tokens: int = 0
    truncated_structural_tokens: int = 0


@dataclass
class SegmentedChunk:
    tokens: List[EventToken]
    start_time_hours: float
    chunk_index: int
    is_first_chunk: bool
    is_last_chunk: bool


def _same_time_group(left: EventToken, right: EventToken) -> bool:
    if left.raw_time is not None and right.raw_time is not None:
        return left.raw_time == right.raw_time
    return float(left.t_from_start_hours) == float(right.t_from_start_hours)


def _group_tokens_by_time(tokens: List[EventToken]) -> List[List[EventToken]]:
    if not tokens:
        return []
    groups: List[List[EventToken]] = [[tokens[0]]]
    for tok in tokens[1:]:
        if _same_time_group(groups[-1][-1], tok):
            groups[-1].append(tok)
        else:
            groups.append([tok])
    return groups


def _split_oversized_group(group: List[EventToken], *, max_content_tokens: int) -> List[List[EventToken]]:
    if len(group) <= max_content_tokens:
        return [group]
    return [
        list(group[start : start + max_content_tokens])
        for start in range(0, len(group), max_content_tokens)
    ]


def _cat_attr_int(tok: EventToken, key: str) -> Optional[int]:
    if tok.cat_attrs is None or key not in tok.cat_attrs:
        return None
    try:
        return int(tok.cat_attrs[key])
    except Exception:
        return None


def _token_transition_action(tok: EventToken) -> Optional[str]:
    action_id = _cat_attr_int(tok, "transition_action_id")
    if action_id is not None and action_id in TRANSITION_ACTION_FROM_ID:
        return TRANSITION_ACTION_FROM_ID[action_id]
    if tok.window_hook is not None:
        # Backward-compatible fallback for older timelines/tests.
        return "open_next"
    return None


def _token_transition_type_id(tok: EventToken) -> Optional[int]:
    for key in ("transition_window_type_id", "window_type_id"):
        type_id = _cat_attr_int(tok, key)
        if type_id is not None and type_id >= 0:
            return type_id
    return None


def _infer_window_type_from_tokens(window_tokens: List[EventToken], *, unk_type_id: int) -> int:
    if not window_tokens:
        return int(unk_type_id)

    for key in ("window_type_id", "transition_window_type_id"):
        val = _cat_attr_int(window_tokens[0], key)
        if val is not None:
            return int(val)
    return int(unk_type_id)


def _resolve_opening_window_type(
    opening_tokens: List[EventToken],
    *,
    previous_type_id: int,
    config: WindowSegmentationConfig,
) -> int:
    explicit_ids = [
        int(type_id)
        for tok in opening_tokens
        for type_id in [_token_transition_type_id(tok)]
        if type_id is not None
    ]
    if explicit_ids:
        # Later items in a transition chain tend to be more specific than the opener.
        return int(explicit_ids[-1])

    inferred = _infer_window_type_from_tokens(opening_tokens, unk_type_id=config.unk_window_type_id)
    if inferred != int(config.unk_window_type_id):
        return int(inferred)
    if previous_type_id != int(config.unk_window_type_id):
        return int(previous_type_id)
    return int(config.unk_window_type_id)


def _apply_window_type_fallbacks(
    windows: List[SegmentedWindow],
    *,
    config: WindowSegmentationConfig,
) -> List[SegmentedWindow]:
    if not windows:
        return []

    unk = int(config.unk_window_type_id)
    first_default = (
        int(config.default_first_window_type_id)
        if config.default_first_window_type_id is not None
        else None
    )
    carry_prev = bool(config.propagate_prev_type_for_unknown_windows)

    out: List[SegmentedWindow] = []
    prev_known: int | None = None
    for idx, window in enumerate(windows):
        w_type = int(window.window_type_id)
        if w_type == unk:
            if idx == 0 and first_default is not None:
                w_type = int(first_default)
            elif carry_prev and prev_known is not None:
                w_type = int(prev_known)
        if w_type != unk:
            prev_known = int(w_type)

        out.append(
            SegmentedWindow(
                tokens=list(window.tokens),
                window_type_id=int(w_type),
                start_time_hours=float(window.start_time_hours),
                opening_action=window.opening_action,
                closing_action=window.closing_action,
                chunks=list(window.chunks),
            )
        )
    return out


def _has_explicit_transition(tokens: List[EventToken]) -> bool:
    return any(_cat_attr_int(tok, "transition_action_id") is not None for tok in tokens)


def _token_has_flag(tok: EventToken, key: str) -> bool:
    val = _cat_attr_int(tok, key)
    return val is not None and int(val) != 0


def _window_has_flag(window: SegmentedWindow, key: str) -> bool:
    return any(_token_has_flag(tok, key) for tok in window.tokens)


def _window_end_time_hours(window: SegmentedWindow) -> float:
    if window.tokens:
        return float(window.tokens[-1].t_from_start_hours)
    return float(window.start_time_hours)


def _build_inter_admission_window(
    *,
    left_window: SegmentedWindow,
    gap_hours: float,
    config: WindowSegmentationConfig,
) -> SegmentedWindow | None:
    inter_type_id = config.inter_admission_window_type_id
    token_id = config.inter_admission_token_id
    if inter_type_id is None or token_id is None:
        return None

    left_end = _window_end_time_hours(left_window)
    gap_midpoint = float(left_end) + (0.5 * float(gap_hours))
    token_attrs = {
        "window_type_id": int(inter_type_id),
        "inter_admission_window": 1,
    }
    if config.inter_admission_struct_label_id is not None:
        token_attrs["struct_label_id"] = int(config.inter_admission_struct_label_id)

    gap_token = EventToken(
        value_id=int(token_id),
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=float(gap_midpoint),
        dt_from_prev_hours=max(0.0, 0.5 * float(gap_hours)),
        cat_attrs=token_attrs,
        num_attrs={"numeric_value": float(gap_hours)},
        raw_time=None,
        window_hook=None,
    )
    return SegmentedWindow(
        tokens=[gap_token],
        window_type_id=int(inter_type_id),
        start_time_hours=float(gap_midpoint),
        opening_action=None,
        closing_action=None,
    )


def _inject_inter_admission_windows(
    windows: List[SegmentedWindow],
    *,
    config: WindowSegmentationConfig,
) -> List[SegmentedWindow]:
    if (
        not config.enable_inter_admission_windows
        or config.inter_admission_window_type_id is None
        or config.inter_admission_max_gap_hours <= 0.0
        or len(windows) < 2
    ):
        return list(windows)

    inter_type_id = int(config.inter_admission_window_type_id)
    out: List[SegmentedWindow] = []
    for idx, window in enumerate(windows[:-1]):
        out.append(window)
        next_window = windows[idx + 1]

        if int(window.window_type_id) == inter_type_id or int(next_window.window_type_id) == inter_type_id:
            continue
        if not _window_has_flag(window, "transition_discharge_like"):
            continue
        if _window_has_flag(window, "transition_death_like"):
            continue
        if not _window_has_flag(next_window, "transition_admission_like"):
            continue

        gap_hours = max(0.0, float(next_window.start_time_hours) - _window_end_time_hours(window))
        if gap_hours <= 0.0 or gap_hours > float(config.inter_admission_max_gap_hours):
            continue

        inter_window = _build_inter_admission_window(
            left_window=window,
            gap_hours=float(gap_hours),
            config=config,
        )
        if inter_window is not None:
            out.append(inter_window)

    out.append(windows[-1])
    return out


def _should_merge_transition_chain(
    left_bundle: Dict[str, object],
    right_bundle: Dict[str, object],
    events: List[EventToken],
    *,
    config: WindowSegmentationConfig,
) -> bool:
    if not config.merge_transition_chains:
        return False

    left_tokens = events[int(left_bundle["start_idx"]) : int(left_bundle["end_idx"]) + 1]
    right_tokens = events[int(right_bundle["start_idx"]) : int(right_bundle["end_idx"]) + 1]
    if not (_has_explicit_transition(left_tokens) and _has_explicit_transition(right_tokens)):
        return False
    if (
        any(_token_has_flag(tok, "transition_discharge_like") for tok in left_tokens)
        and not any(_token_has_flag(tok, "transition_death_like") for tok in left_tokens)
        and any(_token_has_flag(tok, "transition_admission_like") for tok in right_tokens)
    ):
        return False

    start = int(left_bundle["end_idx"]) + 1
    stop = int(right_bundle["start_idx"])
    if stop < start:
        return True

    intervening = events[start:stop]
    if len(intervening) > int(config.chain_max_intervening_tokens):
        return False
    if any(
        tok.category_id in {
            int(TokenCategory.MEASUREMENT),
            int(TokenCategory.DIAGNOSIS),
            int(TokenCategory.PROCEDURE),
            int(TokenCategory.MEDICATION),
        }
        for tok in intervening
    ):
        return False

    left_t = float(events[int(left_bundle["end_idx"])].t_from_start_hours)
    right_t = float(events[int(right_bundle["start_idx"])].t_from_start_hours)
    return max(0.0, right_t - left_t) <= float(config.chain_gap_hours)


def _build_boundary_bundles(events: List[EventToken], *, config: WindowSegmentationConfig) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for idx, tok in enumerate(events):
        action = _token_transition_action(tok)
        if action in ACTIVE_TRANSITION_ACTIONS:
            candidates.append({"idx": int(idx), "action": str(action)})

    if not candidates:
        return []

    bundles: List[Dict[str, object]] = []
    current: Dict[str, object] = {
        "start_idx": int(candidates[0]["idx"]),
        "end_idx": int(candidates[0]["idx"]),
        "candidate_indices": [int(candidates[0]["idx"])],
        "candidate_actions": [str(candidates[0]["action"])],
    }

    for cand in candidates[1:]:
        idx = int(cand["idx"])
        prev_idx = int(current["candidate_indices"][-1])  # type: ignore[index]
        time_gap = abs(float(events[idx].t_from_start_hours) - float(events[prev_idx].t_from_start_hours))
        idx_gap = idx - prev_idx
        if idx_gap <= int(config.bundle_max_index_gap) and time_gap <= float(config.bundle_gap_hours):
            current["end_idx"] = idx
            current["candidate_indices"].append(idx)  # type: ignore[union-attr]
            current["candidate_actions"].append(str(cand["action"]))  # type: ignore[union-attr]
        else:
            bundles.append(dict(current))
            current = {
                "start_idx": idx,
                "end_idx": idx,
                "candidate_indices": [idx],
                "candidate_actions": [str(cand["action"])],
            }
    bundles.append(dict(current))

    if not config.merge_transition_chains or len(bundles) < 2:
        return bundles

    merged: List[Dict[str, object]] = [dict(bundles[0])]
    for bundle in bundles[1:]:
        prev = merged[-1]
        if _should_merge_transition_chain(prev, bundle, events, config=config):
            prev["end_idx"] = int(bundle["end_idx"])
            prev["candidate_indices"] = list(prev["candidate_indices"]) + list(bundle["candidate_indices"])  # type: ignore[index]
            prev["candidate_actions"] = list(prev["candidate_actions"]) + list(bundle["candidate_actions"])  # type: ignore[index]
        else:
            merged.append(dict(bundle))
    return merged


def _resolve_bundle_action(
    bundle_tokens: List[EventToken],
    bundle_candidate_indices: List[int],
    *,
    bundle_start_idx: int,
) -> tuple[str, List[EventToken], List[EventToken]]:
    candidate_positions = [int(idx) - int(bundle_start_idx) for idx in bundle_candidate_indices]
    candidate_actions = [
        _token_transition_action(bundle_tokens[pos])
        for pos in candidate_positions
        if 0 <= pos < len(bundle_tokens)
    ]
    open_like_positions = [
        pos
        for pos, action in zip(candidate_positions, candidate_actions)
        if action in {"open_next", "close_open"}
    ]
    close_like_positions = [
        pos
        for pos, action in zip(candidate_positions, candidate_actions)
        if action in {"close_current", "close_open"}
    ]

    if "close_open" in candidate_actions or (open_like_positions and close_like_positions):
        bundle_action = "close_open"
    elif open_like_positions:
        bundle_action = "open_next"
    elif close_like_positions:
        bundle_action = "close_current"
    else:
        bundle_action = "open_next"

    if bundle_action == "open_next":
        return bundle_action, [], list(bundle_tokens)
    if bundle_action == "close_current":
        return bundle_action, list(bundle_tokens), []

    pivot = min(open_like_positions) if open_like_positions else max(0, len(bundle_tokens) - 1)
    closing = list(bundle_tokens[:pivot])
    opening = list(bundle_tokens[pivot:])
    if not opening and closing:
        opening.append(closing.pop())
    return bundle_action, closing, opening


def segment_event_tokens(
    events: List[EventToken],
    *,
    config: WindowSegmentationConfig | None = None,
) -> List[SegmentedWindow]:
    config = config or WindowSegmentationConfig()
    if not events:
        return []

    bundles = _build_boundary_bundles(events, config=config)
    if not bundles:
        return _apply_window_type_fallbacks(
            [
                SegmentedWindow(
                    tokens=list(events),
                    window_type_id=_infer_window_type_from_tokens(list(events), unk_type_id=config.unk_window_type_id),
                    start_time_hours=float(events[0].t_from_start_hours),
                    opening_action=None,
                    closing_action=None,
                )
            ],
            config=config,
        )

    windows: List[SegmentedWindow] = []
    current_tokens: List[EventToken] = []
    current_type_id = int(config.unk_window_type_id)
    current_opening_action: Optional[str] = None

    cursor = 0
    for bundle in bundles:
        start_idx = int(bundle["start_idx"])
        end_idx = int(bundle["end_idx"])
        if cursor < start_idx:
            current_tokens.extend(events[cursor:start_idx])

        bundle_tokens = events[start_idx : end_idx + 1]
        bundle_action, closing_items, opening_items = _resolve_bundle_action(
            bundle_tokens,
            list(bundle["candidate_indices"]),  # type: ignore[arg-type]
            bundle_start_idx=start_idx,
        )

        if bundle_action == "close_current":
            current_tokens.extend(closing_items)
            if current_tokens:
                windows.append(
                    SegmentedWindow(
                        tokens=list(current_tokens),
                        window_type_id=(
                            int(current_type_id)
                            if int(current_type_id) != int(config.unk_window_type_id)
                            else _infer_window_type_from_tokens(current_tokens, unk_type_id=config.unk_window_type_id)
                        ),
                        start_time_hours=float(current_tokens[0].t_from_start_hours),
                        opening_action=current_opening_action,
                        closing_action=bundle_action,
                    )
                )
            current_tokens = []
            current_type_id = int(config.unk_window_type_id)
            current_opening_action = None
        elif bundle_action == "open_next":
            if current_tokens:
                windows.append(
                    SegmentedWindow(
                        tokens=list(current_tokens),
                        window_type_id=(
                            int(current_type_id)
                            if int(current_type_id) != int(config.unk_window_type_id)
                            else _infer_window_type_from_tokens(current_tokens, unk_type_id=config.unk_window_type_id)
                        ),
                        start_time_hours=float(current_tokens[0].t_from_start_hours),
                        opening_action=current_opening_action,
                        closing_action=None,
                    )
                )
            current_tokens = list(opening_items)
            current_type_id = _resolve_opening_window_type(
                opening_items,
                previous_type_id=current_type_id,
                config=config,
            )
            current_opening_action = bundle_action
        elif bundle_action == "close_open":
            current_tokens.extend(closing_items)
            if current_tokens:
                windows.append(
                    SegmentedWindow(
                        tokens=list(current_tokens),
                        window_type_id=(
                            int(current_type_id)
                            if int(current_type_id) != int(config.unk_window_type_id)
                            else _infer_window_type_from_tokens(current_tokens, unk_type_id=config.unk_window_type_id)
                        ),
                        start_time_hours=float(current_tokens[0].t_from_start_hours),
                        opening_action=current_opening_action,
                        closing_action=bundle_action,
                    )
                )
            current_tokens = list(opening_items)
            current_type_id = _resolve_opening_window_type(
                opening_items,
                previous_type_id=current_type_id,
                config=config,
            )
            current_opening_action = bundle_action
        else:
            raise ValueError(f"Unsupported bundle action: {bundle_action}")

        cursor = end_idx + 1

    if cursor < len(events):
        current_tokens.extend(events[cursor:])

    if current_tokens:
        windows.append(
            SegmentedWindow(
                tokens=list(current_tokens),
                window_type_id=(
                    int(current_type_id)
                    if int(current_type_id) != int(config.unk_window_type_id)
                    else _infer_window_type_from_tokens(current_tokens, unk_type_id=config.unk_window_type_id)
                ),
                start_time_hours=float(current_tokens[0].t_from_start_hours),
                opening_action=current_opening_action,
                closing_action=None,
            )
        )

    windows = _apply_window_type_fallbacks(windows, config=config)
    return _inject_inter_admission_windows(windows, config=config)


def rebalance_segmented_windows(
    windows: List[SegmentedWindow],
    *,
    max_content_tokens: int,
    config: WindowSegmentationConfig | None = None,
) -> List[SegmentedWindow]:
    config = config or WindowSegmentationConfig()
    if not config.rebalance_dense_windows or max_content_tokens <= 0:
        return list(windows)

    target_tokens = int(round(float(max_content_tokens) * float(config.rebalance_target_frac)))
    target_tokens = max(1, min(int(max_content_tokens), target_tokens))
    min_tokens = max(1, int(config.rebalance_min_tokens))
    tail_tokens = max(1, int(config.rebalance_tail_tokens))

    out: List[SegmentedWindow] = []
    for window in windows:
        if len(window.tokens) <= int(max_content_tokens):
            out.append(window)
            continue

        groups: List[List[EventToken]] = []
        for group in _group_tokens_by_time(window.tokens):
            groups.extend(_split_oversized_group(group, max_content_tokens=int(max_content_tokens)))

        chunks: List[List[EventToken]] = []
        current: List[EventToken] = []
        current_len = 0
        for group in groups:
            g_len = len(group)
            if not current:
                current = list(group)
                current_len = g_len
                continue

            if current_len < min_tokens and current_len + g_len <= int(max_content_tokens):
                current.extend(group)
                current_len += g_len
                continue

            if current_len + g_len <= int(max_content_tokens) and current_len < target_tokens:
                current.extend(group)
                current_len += g_len
                continue

            chunks.append(list(current))
            current = list(group)
            current_len = g_len

        if current:
            chunks.append(list(current))

        if len(chunks) >= 2 and len(chunks[-1]) < tail_tokens:
            if len(chunks[-2]) + len(chunks[-1]) <= int(max_content_tokens):
                chunks[-2].extend(chunks[-1])
                chunks.pop()

        for idx, chunk in enumerate(chunks):
            out.append(
                SegmentedWindow(
                    tokens=list(chunk),
                    window_type_id=int(window.window_type_id),
                    start_time_hours=float(chunk[0].t_from_start_hours),
                    opening_action=window.opening_action if idx == 0 else None,
                    closing_action=window.closing_action if idx == len(chunks) - 1 else None,
                )
            )

    return out


def chunk_segmented_windows(
    windows: List[SegmentedWindow],
    *,
    max_content_tokens: int,
    max_chunks_per_window: int,
    config: WindowSegmentationConfig | None = None,
) -> List[SegmentedWindow]:
    """
    Derive bounded local chunks *within* each semantic window.

    Unlike `rebalance_segmented_windows`, this preserves the semantic global chain:
    one `SegmentedWindow` remains one global regime step, while `chunks` provides the
    local attention units used inside that regime.
    """
    config = config or WindowSegmentationConfig()
    if max_content_tokens <= 0:
        max_content_tokens = 1
    if max_chunks_per_window <= 0:
        max_chunks_per_window = 1

    target_tokens = int(round(float(max_content_tokens) * float(config.rebalance_target_frac)))
    target_tokens = max(1, min(int(max_content_tokens), target_tokens))
    min_tokens = max(1, int(config.rebalance_min_tokens))
    tail_tokens = max(1, int(config.rebalance_tail_tokens))

    out: List[SegmentedWindow] = []
    for window in windows:
        if not window.tokens:
            out.append(
                SegmentedWindow(
                    tokens=[],
                    window_type_id=int(window.window_type_id),
                    start_time_hours=float(window.start_time_hours),
                    opening_action=window.opening_action,
                    closing_action=window.closing_action,
                    chunks=[],
                    truncated_chunks=0,
                    truncated_tokens=0,
                    truncated_structural_tokens=0,
                )
            )
            continue

        groups: List[List[EventToken]] = []
        for group in _group_tokens_by_time(window.tokens):
            groups.extend(_split_oversized_group(group, max_content_tokens=int(max_content_tokens)))

        raw_chunks: List[List[EventToken]] = []
        current: List[EventToken] = []
        current_len = 0
        for group in groups:
            g_len = len(group)
            if not current:
                current = list(group)
                current_len = g_len
                continue

            if current_len < min_tokens and current_len + g_len <= int(max_content_tokens):
                current.extend(group)
                current_len += g_len
                continue

            if current_len + g_len <= int(max_content_tokens) and current_len < target_tokens:
                current.extend(group)
                current_len += g_len
                continue

            raw_chunks.append(list(current))
            current = list(group)
            current_len = g_len

        if current:
            raw_chunks.append(list(current))

        if len(raw_chunks) >= 2 and len(raw_chunks[-1]) < tail_tokens:
            if len(raw_chunks[-2]) + len(raw_chunks[-1]) <= int(max_content_tokens):
                raw_chunks[-2].extend(raw_chunks[-1])
                raw_chunks.pop()

        raw_chunk_count_before_cap = len(raw_chunks)
        raw_chunks_kept = raw_chunks[: int(max_chunks_per_window)]
        dropped_chunk_groups = raw_chunks[int(max_chunks_per_window) :]
        dropped_chunks = max(0, int(raw_chunk_count_before_cap) - len(raw_chunks_kept))
        dropped_tokens = int(sum(len(chunk) for chunk in dropped_chunk_groups))
        dropped_structural_tokens = int(
            sum(
                1
                for chunk in dropped_chunk_groups
                for tok in chunk
                if int(tok.category_id) == int(TokenCategory.STRUCTURAL)
            )
        )

        chunks = [
            SegmentedChunk(
                tokens=list(chunk),
                start_time_hours=float(chunk[0].t_from_start_hours),
                chunk_index=idx,
                is_first_chunk=(idx == 0),
                is_last_chunk=(idx == len(raw_chunks_kept) - 1),
            )
            for idx, chunk in enumerate(raw_chunks_kept)
            if chunk
        ]

        out.append(
            SegmentedWindow(
                tokens=list(window.tokens),
                window_type_id=int(window.window_type_id),
                start_time_hours=float(window.start_time_hours),
                opening_action=window.opening_action,
                closing_action=window.closing_action,
                chunks=chunks,
                truncated_chunks=int(dropped_chunks),
                truncated_tokens=int(dropped_tokens),
                truncated_structural_tokens=int(dropped_structural_tokens),
            )
        )

    return out
