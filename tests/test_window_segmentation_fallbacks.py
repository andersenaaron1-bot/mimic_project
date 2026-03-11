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


def test_segment_propagates_previous_type_for_unknown_trailing_window() -> None:
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
        propagate_prev_type_for_unknown_windows=True,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 2
    assert windows[0].window_type_id == 2
    # trailing diagnosis-only segment after close_current inherits the last known type.
    assert windows[1].window_type_id == 2


def test_segment_injects_inter_admission_window_for_short_discharge_gap() -> None:
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
            value_id=401,
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
        propagate_prev_type_for_unknown_windows=True,
        enable_inter_admission_windows=True,
        inter_admission_window_type_id=6,
        inter_admission_max_gap_hours=24.0,
        inter_admission_token_id=2200999,
        inter_admission_struct_label_id=9,
    )

    windows = segment_event_tokens(events, config=cfg)
    assert len(windows) == 3
    assert [window.window_type_id for window in windows] == [3, 6, 2]
    inter_window = windows[1]
    assert inter_window.start_time_hours == 4.0
    assert len(inter_window.tokens) == 1
    gap_token = inter_window.tokens[0]
    assert gap_token.value_id == 2200999
    assert gap_token.category_id == int(TokenCategory.STRUCTURAL)
    assert gap_token.cat_attrs["inter_admission_window"] == 1
    assert gap_token.cat_attrs["window_type_id"] == 6
    assert gap_token.cat_attrs["struct_label_id"] == 9
    assert gap_token.num_attrs["numeric_value"] == 4.0
