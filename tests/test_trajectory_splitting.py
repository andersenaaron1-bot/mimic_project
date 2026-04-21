from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import src.ehr_hier.data.compile_dataset as compile_mod
from src.ehr_hier.data.dataset import PrecompiledMEDSDataset
from src.ehr_hier.data.precompiled_format import save_packed_shard, serialize_timeline_compact
from src.ehr_hier.data.structural_codes import TRANSITION_ACTION_TO_ID
from src.ehr_hier.data.subject_timeline_builder import GLOBAL_DEMOGRAPHIC_TOKEN_IDS
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.data.trajectory_splitting import (
    TrajectorySplitConfig,
    build_trajectory_timelines,
    split_segmented_windows_into_trajectories,
)
from src.ehr_hier.data.window_segmentation import (
    SegmentedWindow,
    WindowSegmentationConfig,
    chunk_segmented_windows,
    segment_event_tokens,
)


_BASE_TIME = datetime(2026, 1, 1, 8, 0, 0)


def _tok(
    value_id: int,
    *,
    hour: float,
    category: TokenCategory = TokenCategory.STRUCTURAL,
    dt_hours: float = 0.0,
    cat_attrs: dict[str, int] | None = None,
    num_attrs: dict[str, float] | None = None,
) -> EventToken:
    return EventToken(
        value_id=int(value_id),
        category_id=int(category),
        t_from_start_hours=float(hour),
        dt_from_prev_hours=float(dt_hours),
        cat_attrs=dict(cat_attrs or {}),
        num_attrs=dict(num_attrs or {}),
        raw_time=_BASE_TIME + timedelta(hours=float(hour)),
        window_hook="window_boundary" if "transition_action_id" in (cat_attrs or {}) else None,
    )


def test_consecutive_transfer_burst_without_semantic_content_coalesces() -> None:
    events = [
        _tok(
            1,
            hour=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 101,
                "window_site_id": 101,
                "transition_transfer_like": 1,
            },
        ),
        _tok(
            3,
            hour=0.2,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 101,
                "window_site_id": 101,
                "transition_transfer_like": 1,
            },
        ),
        _tok(4, hour=2.5, category=TokenCategory.MEASUREMENT),
    ]
    cfg = WindowSegmentationConfig(unk_window_type_id=0)

    windows = segment_event_tokens(events, config=cfg)

    assert len(windows) == 1
    assert windows[0].window_type_id == 2
    assert windows[0].window_site_id == 101
    assert windows[0].chunk_break_token_indices == []
    assert [tok.value_id for tok in windows[0].tokens] == [1, 3, 4]

    chunked = chunk_segmented_windows(
        windows,
        max_content_tokens=16,
        max_chunks_per_window=8,
        config=cfg,
    )
    assert len(chunked[0].chunks) == 1
    assert [tok.value_id for tok in chunked[0].chunks[0].tokens] == [1, 3, 4]


def test_transfer_opener_after_semantic_content_starts_new_window() -> None:
    events = [
        _tok(
            1,
            hour=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 101,
                "window_site_id": 101,
                "transition_transfer_like": 1,
            },
        ),
        _tok(2, hour=1.0, category=TokenCategory.MEASUREMENT),
        _tok(
            3,
            hour=2.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_open"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 202,
                "window_site_id": 202,
                "transition_transfer_like": 1,
            },
        ),
        _tok(4, hour=2.5, category=TokenCategory.MEASUREMENT),
    ]
    cfg = WindowSegmentationConfig(unk_window_type_id=0)

    windows = segment_event_tokens(events, config=cfg)

    assert len(windows) == 2
    assert windows[0].window_site_id == 101
    assert [tok.value_id for tok in windows[0].tokens] == [1, 2]
    assert windows[1].window_site_id == 202
    assert [tok.value_id for tok in windows[1].tokens] == [3, 4]


def test_admission_chain_split_cuts_after_31_days_post_discharge() -> None:
    windows = [
        SegmentedWindow(
            tokens=[_tok(1, hour=0.0)],
            window_type_id=2,
            start_time_hours=0.0,
            closing_time_hours=10.0,
            closing_discharge_like=True,
        ),
        SegmentedWindow(
            tokens=[_tok(2, hour=40.0, category=TokenCategory.MEASUREMENT)],
            window_type_id=5,
            start_time_hours=40.0,
        ),
        SegmentedWindow(
            tokens=[_tok(3, hour=900.0)],
            window_type_id=2,
            start_time_hours=900.0,
        ),
    ]

    trajectories = split_segmented_windows_into_trajectories(
        windows,
        config=TrajectorySplitConfig(mode="admission_chain", post_discharge_cutoff_hours=31.0 * 24.0),
        post_discharge_window_type_id=5,
    )

    assert len(trajectories) == 2
    assert [tok.value_id for tok in trajectories[0][0].tokens] == [1]
    assert [tok.value_id for tok in trajectories[0][1].tokens] == [2]
    assert [tok.value_id for tok in trajectories[1][0].tokens] == [3]


def test_admission_chain_can_split_inside_long_post_discharge_window() -> None:
    long_post = SegmentedWindow(
        tokens=[
            _tok(20, hour=100.0, category=TokenCategory.MEASUREMENT),
            _tok(21, hour=700.0, category=TokenCategory.MEASUREMENT),
            _tok(22, hour=800.0, category=TokenCategory.MEASUREMENT),
        ],
        window_type_id=5,
        start_time_hours=100.0,
    )
    windows = [
        SegmentedWindow(
            tokens=[_tok(10, hour=0.0)],
            window_type_id=2,
            start_time_hours=0.0,
            closing_time_hours=10.0,
            closing_discharge_like=True,
        ),
        long_post,
    ]

    trajectories = split_segmented_windows_into_trajectories(
        windows,
        config=TrajectorySplitConfig(mode="admission_chain", post_discharge_cutoff_hours=31.0 * 24.0),
        post_discharge_window_type_id=5,
    )

    assert len(trajectories) == 2
    assert [tok.value_id for tok in trajectories[0][1].tokens] == [20, 21]
    assert [tok.value_id for tok in trajectories[1][0].tokens] == [22]


def test_precompiled_trajectory_dataset_rebuilds_slice_demographics(tmp_path: Path) -> None:
    segmentation_cfg = WindowSegmentationConfig(
        unk_window_type_id=0,
        post_discharge_window_type_id=5,
    )
    split_cfg = TrajectorySplitConfig(
        mode="admission_chain",
        post_discharge_cutoff_hours=31.0 * 24.0,
    )
    full_timeline = [
        _tok(
            GLOBAL_DEMOGRAPHIC_TOKEN_IDS["WEIGHT_AT_ADMISSION"],
            hour=0.0,
            category=TokenCategory.SPECIAL,
            num_attrs={"numeric_value": 50.0},
            cat_attrs={"global_demographic": 1, "demographic_feature_id": 5, "demographic_numeric": 1},
        ),
        _tok(
            1,
            hour=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 101,
                "window_site_id": 101,
            },
        ),
        _tok(2, hour=1.0, category=TokenCategory.MEASUREMENT, dt_hours=1.0),
        _tok(
            3,
            hour=10.0,
            dt_hours=9.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_current"],
                "transition_discharge_like": 1,
            },
        ),
        _tok(4, hour=20.0, category=TokenCategory.MEASUREMENT, dt_hours=10.0),
        _tok(
            5,
            hour=900.0,
            dt_hours=880.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 202,
                "window_site_id": 202,
            },
        ),
        _tok(6, hour=901.0, category=TokenCategory.MEASUREMENT, dt_hours=1.0),
    ]
    metadata = {
        "subject_demographics": {
            "sex_value": 0.0,
            "birth_ts": float(datetime(1980, 1, 1).timestamp()),
            "timeline_start_ts": float(_BASE_TIME.timestamp()),
            "observations": {
                "BMI": [],
                "HEIGHT_CM": [{"t_from_start_hours": 0.0, "value": 170.0}],
                "WEIGHT_KG": [
                    {"t_from_start_hours": 0.0, "value": 70.0},
                    {"t_from_start_hours": 850.0, "value": 90.0},
                ],
            },
        }
    }
    shard_path = tmp_path / "shards" / "000000.ptz"
    save_packed_shard(
        shard_path,
        subject_ids=[101],
        serialized_timelines=[serialize_timeline_compact(full_timeline, metadata=metadata)],
    )
    rows = compile_mod._build_trajectory_index_records(
        output_dir=str(tmp_path),
        records=[{"subject_id": 101, "rel_path": "shards/000000.ptz", "subject_idx": 0}],
        segmentation_config=segmentation_cfg,
        trajectory_split_config=split_cfg,
    )
    rows_df = pd.DataFrame(rows)
    rows_df["split"] = "train"
    rows_df.to_csv(tmp_path / "trajectory_index.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        '{"version": 2, "storage_format": "packed_shard_v2"}',
        encoding="utf-8",
    )

    ds = PrecompiledMEDSDataset(
        str(tmp_path),
        split="train",
        index_filename="trajectory_index.csv",
    )

    assert len(ds) == 2
    second_traj = ds[1]
    weight_tokens = [
        tok
        for tok in second_traj
        if int(tok.category_id) == int(TokenCategory.SPECIAL)
        and int(tok.value_id) == int(GLOBAL_DEMOGRAPHIC_TOKEN_IDS["WEIGHT_AT_ADMISSION"])
    ]
    assert len(weight_tokens) == 1
    assert weight_tokens[0].num_attrs["numeric_value"] == 90.0

    first_event = next(tok for tok in second_traj if int(tok.category_id) != int(TokenCategory.SPECIAL))
    assert first_event.t_from_start_hours == 0.0
    assert first_event.dt_from_prev_hours == 0.0


def test_build_trajectory_timelines_rebases_sample_clock() -> None:
    timeline = [
        _tok(
            1,
            hour=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 101,
                "window_site_id": 101,
            },
        ),
        _tok(2, hour=1.0, category=TokenCategory.MEASUREMENT, dt_hours=1.0),
        _tok(
            3,
            hour=10.0,
            dt_hours=9.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_current"],
                "transition_discharge_like": 1,
            },
        ),
        _tok(4, hour=900.0, category=TokenCategory.MEASUREMENT, dt_hours=890.0),
    ]
    trajectories = build_trajectory_timelines(
        timeline=timeline,
        segmentation_config=WindowSegmentationConfig(
            unk_window_type_id=0,
            post_discharge_window_type_id=5,
        ),
        split_config=TrajectorySplitConfig(
            mode="admission_chain",
            post_discharge_cutoff_hours=31.0 * 24.0,
        ),
    )

    assert len(trajectories) == 2
    tail_event = trajectories[1][0]
    assert tail_event.t_from_start_hours == 0.0
    assert tail_event.dt_from_prev_hours == 0.0
