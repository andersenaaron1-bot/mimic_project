import pytest


def test_collator_inserts_window_markers_and_emits_window_metadata() -> None:
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

    # Two windows, split by a transfer-driven opener.
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
        cat_attrs={
            "struct_label_id": 0,
            "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
            "transition_window_type_id": 1,
            "window_type_id": 1,
            "transition_transfer_like": 1,
        },
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

    # (B=1,W=2,C=1,L=8)
    assert batch["input_ids"].shape == (1, 2, 1, 8)

    # Window 0: [summary] [WIN_TYPE=10+2] [token1] [WIN_END=10+4]
    w0 = batch["input_ids"][0, 0, 0, :4].tolist()
    assert w0 == [1, 12, 100, 14]

    # Window 1: [summary] [WIN_TYPE=10+1] [token2] [token3] [WIN_END]
    w1 = batch["input_ids"][0, 1, 0, :5].tolist()
    assert w1 == [1, 11, 200, 201, 14]

    # Window metadata is per-window and uses absolute time for starts.
    assert batch["window_type_ids"][0, :2].tolist() == [2, 1]
    assert batch["window_start_times"][0, :2].tolist() == pytest.approx([5.0, 7.0], rel=1e-6)
    assert batch["chunk_mask"][0, :2, 0].tolist() == [1, 1]


def test_collator_keeps_nontransfer_admin_markers_inside_leading_history_window() -> None:
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

    # No transfer opener: everything stays inside one leading unresolved window.
    assert batch["window_mask"].shape[1] == 1
    assert batch["window_mask"][0].tolist() == [1]
    assert batch["window_type_ids"][0, 0].item() == 0
    assert batch["window_start_times"][0, 0].item() == pytest.approx(0.0, rel=1e-6)
    w0 = batch["input_ids"][0, 0, 0, :8].tolist()
    assert w0 == [1, 10, 300, 400, 301, 100, 302, 18]
    assert batch["chunk_mask"][0, 0, 0].item() == 1


def test_collator_chunks_dense_semantic_window_without_creating_new_global_windows() -> None:
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
    tokens = [
        EventToken(
            value_id=101,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={"window_type_id": 2},
            num_attrs={"numeric_value": 1.0},
        ),
        EventToken(
            value_id=102,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={},
            num_attrs={"numeric_value": 2.0},
        ),
        EventToken(
            value_id=103,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={},
            num_attrs={"numeric_value": 3.0},
        ),
        EventToken(
            value_id=104,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=1.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={"numeric_value": 4.0},
        ),
        EventToken(
            value_id=105,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=1.0,
            dt_from_prev_hours=0.0,
            cat_attrs={},
            num_attrs={"numeric_value": 5.0},
        ),
        EventToken(
            value_id=106,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=1.0,
            dt_from_prev_hours=0.0,
            cat_attrs={},
            num_attrs={"numeric_value": 6.0},
        ),
    ]

    collator = AETHierarchicalCollator(
        max_windows=8,
        max_chunks_per_window=4,
        max_len_per_window=6,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, end_mode="next_type", type_token_offset=10, num_types=4),
    )

    batch = collator([[summary, *tokens]])

    # One semantic window with two local chunks.
    assert batch["window_mask"][0, :1].tolist() == [1]
    assert batch["window_type_ids"][0, :1].tolist() == [2]
    assert batch["window_start_times"][0, :1].tolist() == pytest.approx([0.0], rel=1e-6)
    assert batch["chunk_mask"][0, 0, :2].tolist() == [1, 1]
    assert batch["chunk_start_offsets"][0, 0, :2].tolist() == pytest.approx([0.0, 1.0], rel=1e-6)
    assert batch["chunk_is_last"][0, 0, :2].tolist() == [0, 1]
    assert batch["semantic_token_counts"][0, 0].item() == pytest.approx(6.0, rel=1e-6)

    # First local chunk ends with WIN_CONTINUE rather than a semantic transition marker.
    w0 = batch["input_ids"][0, 0, 0, :6].tolist()
    assert w0 == [1, 12, 101, 102, 103, 15]

    # Final chunk closes the semantic window.
    w1 = batch["input_ids"][0, 0, 1, :6].tolist()
    assert w1 == [1, 12, 104, 105, 106, 14]


def test_collator_clamps_out_of_range_window_type_to_unk() -> None:
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
    # No transition metadata: segmentation falls back to first-token window_type_id=99.
    token = EventToken(
        value_id=101,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=1.0,
        dt_from_prev_hours=1.0,
        cat_attrs={"window_type_id": 99},
        num_attrs={"numeric_value": 1.0},
    )

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=6,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4, unk_type_id=0),
    )

    batch = collator([[summary, token]])
    # clamped to UNK=0
    assert batch["window_type_ids"][0, 0].item() == 0
    # WIN_TYPE marker should be offset + 0
    assert batch["input_ids"][0, 0, 0, 1].item() == 10


def test_collator_emits_post_discharge_window_chunk_when_tokens_follow_discharge() -> None:
    from ehr_hier.data.structural_codes import TRANSITION_ACTION_TO_ID
    from ehr_hier.data.token_types import EventToken, TokenCategory
    from ehr_hier.data.window_segmentation import WindowSegmentationConfig
    from ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig

    summary = EventToken(
        value_id=1,
        category_id=int(TokenCategory.SPECIAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={},
        num_attrs={},
    )
    admit = EventToken(
        value_id=300,
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={
            "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
            "transition_window_type_id": 3,
            "window_type_id": 3,
            "transition_transfer_like": 1,
        },
        num_attrs={},
        window_hook="episode",
    )
    discharge = EventToken(
        value_id=301,
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=2.0,
        dt_from_prev_hours=2.0,
        cat_attrs={
            "transition_action_id": TRANSITION_ACTION_TO_ID["close_current"],
            "transition_discharge_like": 1,
        },
        num_attrs={},
        window_hook="episode",
    )
    inpatient_meas = EventToken(
        value_id=325,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=1.0,
        dt_from_prev_hours=1.0,
        cat_attrs={},
        num_attrs={"numeric_value": 2.5},
    )
    post_dx = EventToken(
        value_id=350,
        category_id=int(TokenCategory.DIAGNOSIS),
        t_from_start_hours=4.0,
        dt_from_prev_hours=2.0,
        cat_attrs={},
        num_attrs={},
    )
    readmit = EventToken(
        value_id=302,
        category_id=int(TokenCategory.STRUCTURAL),
        t_from_start_hours=6.0,
        dt_from_prev_hours=4.0,
        cat_attrs={
            "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
            "transition_window_type_id": 2,
            "window_type_id": 2,
            "transition_transfer_like": 1,
        },
        num_attrs={},
        window_hook="episode",
    )
    meas = EventToken(
        value_id=100,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=6.5,
        dt_from_prev_hours=0.5,
        cat_attrs={},
        num_attrs={"numeric_value": 1.23},
    )

    collator = AETHierarchicalCollator(
        max_windows=8,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=8, unk_type_id=0),
        segmentation=WindowSegmentationConfig(
            unk_window_type_id=0,
            default_first_window_type_id=1,
            post_discharge_window_type_id=6,
            propagate_prev_type_for_unknown_windows=False,
        ),
    )

    batch = collator([[summary, admit, inpatient_meas, discharge, post_dx, readmit, meas]])

    assert batch["window_mask"][0, :3].tolist() == [1, 1, 1]
    assert batch["window_type_ids"][0, :3].tolist() == [3, 6, 2]
    assert batch["chunk_mask"][0, :3, 0].tolist() == [1, 1, 1]
    assert batch["window_start_times"][0, :3].tolist() == pytest.approx([0.0, 4.0, 6.0], rel=1e-6)
