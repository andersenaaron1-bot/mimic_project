from ehr_hier.data.structural_codes import TRANSITION_ACTION_TO_ID
from ehr_hier.data.token_types import EventToken, TokenCategory
from ehr_hier.data.window_segmentation import WindowSegmentationConfig, segment_event_tokens


def test_segment_default_first_window_type_id_applies_to_leading_unknown_window() -> None:
    events = [
        EventToken(
            value_id=101,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={},
            num_attrs={},
        ),
        EventToken(
            value_id=201,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=1.0,
            dt_from_prev_hours=1.0,
            cat_attrs={"window_type_id": 3},
            num_attrs={},
            window_hook="episode",
        ),
    ]
    cfg = WindowSegmentationConfig(
        unk_window_type_id=0,
        default_first_window_type_id=1,
        propagate_prev_type_for_unknown_windows=True,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 2
    assert windows[0].window_type_id == 1
    assert windows[1].window_type_id == 3


def test_segment_leaves_unknown_trailing_window_untyped() -> None:
    events = [
        EventToken(
            value_id=301,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=101,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=2.0,
            dt_from_prev_hours=2.0,
            cat_attrs={},
            num_attrs={},
        ),
        EventToken(
            value_id=302,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=3.0,
            dt_from_prev_hours=1.0,
            cat_attrs={"transition_action_id": TRANSITION_ACTION_TO_ID["close_current"]},
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=401,
            category_id=int(TokenCategory.DIAGNOSIS),
            t_from_start_hours=4.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
        ),
    ]
    cfg = WindowSegmentationConfig(
        unk_window_type_id=0,
        default_first_window_type_id=None,
        propagate_prev_type_for_unknown_windows=False,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 2
    assert windows[0].window_type_id == 2
    assert windows[1].window_type_id == 0


def test_segment_assigns_post_discharge_window_between_discharge_and_next_opener() -> None:
    events = [
        EventToken(
            value_id=301,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 3,
                "window_type_id": 3,
                "transition_admission_like": 1,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=101,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=1.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
        ),
        EventToken(
            value_id=302,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=2.0,
            dt_from_prev_hours=1.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_current"],
                "transition_discharge_like": 1,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=401,
            category_id=int(TokenCategory.DIAGNOSIS),
            t_from_start_hours=4.0,
            dt_from_prev_hours=2.0,
            cat_attrs={},
            num_attrs={},
        ),
        EventToken(
            value_id=303,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=6.0,
            dt_from_prev_hours=4.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_admission_like": 1,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=402,
            category_id=int(TokenCategory.DIAGNOSIS),
            t_from_start_hours=7.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
        ),
    ]
    cfg = WindowSegmentationConfig(
        unk_window_type_id=0,
        default_first_window_type_id=1,
        post_discharge_window_type_id=6,
        propagate_prev_type_for_unknown_windows=False,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 3
    assert [window.window_type_id for window in windows] == [3, 6, 2]
    assert [tok.value_id for tok in windows[1].tokens] == [401]


def test_segment_does_not_create_post_discharge_window_after_death() -> None:
    events = [
        EventToken(
            value_id=301,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 4,
                "window_type_id": 4,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=101,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=1.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
        ),
        EventToken(
            value_id=302,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=2.0,
            dt_from_prev_hours=1.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_current"],
                "transition_death_like": 1,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=401,
            category_id=int(TokenCategory.DIAGNOSIS),
            t_from_start_hours=4.0,
            dt_from_prev_hours=2.0,
            cat_attrs={},
            num_attrs={},
        ),
    ]
    cfg = WindowSegmentationConfig(
        unk_window_type_id=0,
        default_first_window_type_id=1,
        post_discharge_window_type_id=6,
        propagate_prev_type_for_unknown_windows=False,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 2
    assert [window.window_type_id for window in windows] == [4, 0]


def test_segment_prefers_transfer_to_for_bundle_action_and_opening_type() -> None:
    events = [
        EventToken(
            value_id=101,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={},
            num_attrs={},
        ),
        EventToken(
            value_id=201,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=1.0,
            dt_from_prev_hours=1.0,
            cat_attrs={"transition_action_id": TRANSITION_ACTION_TO_ID["close_current"]},
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=202,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=1.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
                "transition_window_type_id": 4,
                "window_type_id": 4,
                "transition_transfer_like": 1,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=203,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=1.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 3,
                "window_type_id": 3,
            },
            num_attrs={},
            window_hook="episode",
        ),
        EventToken(
            value_id=301,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=2.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
        ),
    ]
    cfg = WindowSegmentationConfig(
        unk_window_type_id=0,
        default_first_window_type_id=1,
        propagate_prev_type_for_unknown_windows=False,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 2
    assert windows[0].window_type_id == 1
    assert windows[1].window_type_id == 4
    assert [tok.value_id for tok in windows[1].tokens[:2]] == [202, 203]
