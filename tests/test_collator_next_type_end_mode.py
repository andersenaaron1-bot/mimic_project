import pytest


def test_collator_can_end_windows_with_next_type_marker() -> None:
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
        window_markers=WindowMarkerConfig(enabled=True, end_mode="next_type", type_token_offset=10, num_types=4, end_token_id=None),
    )

    batch = collator([[summary, token1, token2, token3]])

    # (B=1,W=2,C=1,L=8)
    assert batch["input_ids"].shape == (1, 2, 1, 8)

    # Window 0 ends with WIN_<NEXT_TYPE=1> rather than WIN_END.
    w0 = batch["input_ids"][0, 0, 0, :4].tolist()
    assert w0 == [1, 12, 100, 11]

    # Last window still ends with WIN_END.
    w1 = batch["input_ids"][0, 1, 0, :5].tolist()
    assert w1 == [1, 11, 200, 201, 14]

    # Window metadata is per-window and uses absolute time for starts.
    assert batch["window_type_ids"][0, :2].tolist() == [2, 1]
    assert batch["window_start_times"][0, :2].tolist() == pytest.approx([5.0, 7.0], rel=1e-6)
    assert batch["chunk_mask"][0, :2, 0].tolist() == [1, 1]
