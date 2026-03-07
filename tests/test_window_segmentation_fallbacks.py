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

