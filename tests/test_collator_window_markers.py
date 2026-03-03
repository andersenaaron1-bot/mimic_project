import pytest


def test_collator_inserts_window_markers_and_emits_window_metadata() -> None:
    from ehr_hier.data.token_types import EventToken, TokenCategory
    from ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig

    summary = EventToken(
        value_id=1,
        category_id=int(TokenCategory.SPECIAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={},
        num_attrs={},
    )

    # Two windows, split by token2.window_hook.
    token1 = EventToken(
        value_id=100,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=5.0,
        dt_from_prev_hours=5.0,
        cat_attrs={"window_type_id": 2},
        num_attrs={"numeric_value": 1.23},
    )
    token2 = EventToken(
        value_id=200,
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=7.0,
        dt_from_prev_hours=2.0,
        cat_attrs={"struct_label_id": 0},  # -> window_type_id = 1
        num_attrs={},
        window_hook="episode",
    )
    token3 = EventToken(
        value_id=201,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=8.0,
        dt_from_prev_hours=1.0,
        cat_attrs={},
        num_attrs={"numeric_value": 4.56},
    )

    collator = AETHierarchicalCollator(
        max_windows=8,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4, end_token_id=None),
    )

    batch = collator([[summary, token1, token2, token3]])

    # (B=1,W=2,L=8)
    assert batch["input_ids"].shape[:3] == (1, 2, 8)

    # Window 0: [summary] [WIN_TYPE=10+2] [token1] [WIN_END=10+4]
    w0 = batch["input_ids"][0, 0, :4].tolist()
    assert w0 == [1, 12, 100, 14]

    # Window 1: [summary] [WIN_TYPE=10+1] [token2] [token3] [WIN_END]
    w1 = batch["input_ids"][0, 1, :5].tolist()
    assert w1 == [1, 11, 200, 201, 14]

    # Window metadata is per-window and uses absolute time for starts.
    assert batch["window_type_ids"][0, :2].tolist() == [2, 1]
    assert batch["window_start_times"][0, :2].tolist() == pytest.approx([5.0, 7.0], rel=1e-6)


def test_collator_merges_sparse_transition_chain_and_uses_last_opening_type() -> None:
    from ehr_hier.data.structural_codes import TRANSITION_ACTION_TO_ID
    from ehr_hier.data.token_types import EventToken, TokenCategory
    from ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig

    summary = EventToken(
        value_id=1,
        category_id=int(TokenCategory.SPECIAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={},
        num_attrs={},
    )
    ed_entry = EventToken(
        value_id=300,
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
    )
    admin_bridge = EventToken(
        value_id=400,
        category_id=int(TokenCategory.OTHER),
        t_from_start_hours=0.2,
        dt_from_prev_hours=0.2,
        cat_attrs={},
        num_attrs={},
    )
    hospital_admission = EventToken(
        value_id=301,
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=1.0,
        dt_from_prev_hours=0.8,
        cat_attrs={
            "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
            "transition_window_type_id": 3,
            "window_type_id": 3,
        },
        num_attrs={},
        window_hook="episode",
    )
    meas = EventToken(
        value_id=100,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=1.5,
        dt_from_prev_hours=0.5,
        cat_attrs={},
        num_attrs={"numeric_value": 1.23},
    )
    discharge = EventToken(
        value_id=302,
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=5.0,
        dt_from_prev_hours=3.5,
        cat_attrs={"transition_action_id": TRANSITION_ACTION_TO_ID["close_current"]},
        num_attrs={},
        window_hook="episode",
    )

    collator = AETHierarchicalCollator(
        max_windows=8,
        max_len_per_window=12,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=8, end_token_id=None),
    )

    batch = collator([[summary, ed_entry, admin_bridge, hospital_admission, meas, discharge]])

    # One typed regime window rather than ED -> inpatient fragmentation.
    assert batch["window_mask"].shape[1] == 1
    assert batch["window_mask"][0].tolist() == [1]
    assert batch["window_type_ids"][0, 0].item() == 3
    assert batch["window_start_times"][0, 0].item() == pytest.approx(0.0, rel=1e-6)
    w0 = batch["input_ids"][0, 0, :8].tolist()
    assert w0 == [1, 13, 300, 400, 301, 100, 302, 18]
