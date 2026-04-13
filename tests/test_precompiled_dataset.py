from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch

import src.ehr_hier.data.compile_dataset as compile_mod
import src.ehr_hier.data.dataset as dataset_mod
from src.ehr_hier.data.event_frames import ensure_event_frames, flatten_event_frames
from src.ehr_hier.data.structural_codes import TRANSITION_ACTION_TO_ID
from src.ehr_hier.data.precompiled_format import (
    deserialize_timeline_compact,
    save_packed_shard,
    serialize_timeline_compact,
)
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.data.trajectory_splitting import TrajectorySplitConfig
from src.ehr_hier.data.window_segmentation import WindowSegmentationConfig


def _write_timeline(root: Path, rel_path: str, payload: object) -> None:
    fp = root / rel_path
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, fp)


def _make_timeline(subject_id: int, *, length: int = 6) -> list[EventToken]:
    start = datetime(2026, 3, 31, 8, 0, 0)
    timeline: list[EventToken] = []
    for idx in range(length):
        timeline.append(
            EventToken(
                value_id=10_000 + subject_id + idx,
                category_id=(idx % 6),
                t_from_start_hours=float(idx) * 0.25,
                dt_from_prev_hours=0.25 if idx else 0.0,
                cat_attrs={
                    "window_type_id": (idx % 5) + 1,
                    "transition_action_id": (idx % 3),
                    "struct_label_id": 40 + idx,
                },
                num_attrs={
                    "numeric_value": float(idx) * 1.5,
                    "dose": None if idx % 2 else float(idx) / 10.0,
                },
                raw_time=start + timedelta(minutes=15 * idx),
                window_hook="window_boundary" if idx in {0, length - 1} else None,
            )
        )
    return timeline


def _make_split_timeline() -> list[EventToken]:
    start = datetime(2026, 3, 31, 8, 0, 0)
    return [
        EventToken(
            value_id=1,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=0.0,
            dt_from_prev_hours=0.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 101,
                "window_site_id": 101,
            },
            num_attrs={},
            raw_time=start,
            window_hook="window_boundary",
        ),
        EventToken(
            value_id=2,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=1.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
            raw_time=start + timedelta(hours=1),
            window_hook=None,
        ),
        EventToken(
            value_id=3,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=10.0,
            dt_from_prev_hours=9.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["close_current"],
                "transition_discharge_like": 1,
            },
            num_attrs={},
            raw_time=start + timedelta(hours=10),
            window_hook="window_boundary",
        ),
        EventToken(
            value_id=4,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=20.0,
            dt_from_prev_hours=10.0,
            cat_attrs={},
            num_attrs={},
            raw_time=start + timedelta(hours=20),
            window_hook=None,
        ),
        EventToken(
            value_id=5,
            category_id=int(TokenCategory.STRUCTURAL),
            t_from_start_hours=900.0,
            dt_from_prev_hours=880.0,
            cat_attrs={
                "transition_action_id": TRANSITION_ACTION_TO_ID["open_next"],
                "transition_window_type_id": 2,
                "window_type_id": 2,
                "transition_site_id": 202,
                "window_site_id": 202,
            },
            num_attrs={},
            raw_time=start + timedelta(hours=900),
            window_hook="window_boundary",
        ),
        EventToken(
            value_id=6,
            category_id=int(TokenCategory.MEASUREMENT),
            t_from_start_hours=901.0,
            dt_from_prev_hours=1.0,
            cat_attrs={},
            num_attrs={},
            raw_time=start + timedelta(hours=901),
            window_hook=None,
        ),
    ]


def test_precompiled_dataset_filters_from_index_split_column(tmp_path: Path) -> None:
    _write_timeline(tmp_path, "00/101.pt", serialize_timeline_compact(_make_timeline(101, length=2)))
    _write_timeline(tmp_path, "01/202.pt", serialize_timeline_compact(_make_timeline(202, length=2)))
    pd.DataFrame(
        [
            {"subject_id": 101, "rel_path": "00/101.pt", "split": "train"},
            {"subject_id": 202, "rel_path": "01/202.pt", "split": "tuning"},
        ]
    ).to_csv(tmp_path / "index.csv", index=False)

    train_ds = dataset_mod.PrecompiledMEDSDataset(str(tmp_path), split="train")
    val_ds = dataset_mod.PrecompiledMEDSDataset(str(tmp_path), split="val")

    assert len(train_ds) == 1
    assert flatten_event_frames(train_ds[0])[0].value_id == _make_timeline(101, length=2)[0].value_id
    assert len(val_ds) == 1
    assert flatten_event_frames(val_ds[0])[0].value_id == _make_timeline(202, length=2)[0].value_id


def test_precompiled_dataset_prefers_canonical_splits_parquet_over_index_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_timeline(tmp_path, "00/101.pt", serialize_timeline_compact(_make_timeline(101, length=2)))
    _write_timeline(tmp_path, "01/202.pt", serialize_timeline_compact(_make_timeline(202, length=2)))
    pd.DataFrame(
        [
            {"subject_id": 101, "rel_path": "00/101.pt", "split": "train"},
            {"subject_id": 202, "rel_path": "01/202.pt", "split": "train"},
        ]
    ).to_csv(tmp_path / "index.csv", index=False)

    def _fake_read_parquet(_: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"subject_id": 101, "split": "train"},
                {"subject_id": 202, "split": "tuning"},
            ]
        )

    monkeypatch.setattr(dataset_mod.pd, "read_parquet", _fake_read_parquet)

    tuning_ds = dataset_mod.PrecompiledMEDSDataset(
        str(tmp_path),
        split="val",
        splits_parquet=str(tmp_path / "splits.parquet"),
    )

    assert len(tuning_ds) == 1
    assert Path(tuning_ds.file_paths[0]).name == "202.pt"
    assert flatten_event_frames(tuning_ds[0])[0].value_id == _make_timeline(202, length=2)[0].value_id


def test_write_precompiled_index_writes_manifest_and_split_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_timeline(tmp_path, "00/101.pt", [{"subject_id": 101}])
    _write_timeline(tmp_path, "01/202.pt", [{"subject_id": 202}])

    def _fake_read_parquet(_: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"subject_id": 101, "split": "train"},
                {"subject_id": 202, "split": "tuning"},
            ]
        )

    monkeypatch.setattr(compile_mod.pd, "read_parquet", _fake_read_parquet)
    manifest = compile_mod.write_precompiled_index(
        output_dir=str(tmp_path),
        splits_parquet=str(tmp_path / "splits.parquet"),
    )

    index_df = pd.read_csv(tmp_path / "index.csv")
    assert index_df["rel_path"].tolist() == ["00/101.pt", "01/202.pt"]
    assert manifest["total_timelines"] == 2
    assert manifest["split_counts"] == {"train": 1, "tuning": 1}


def test_write_precompiled_index_supports_duplicate_subject_rows_for_trajectories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_read_parquet(_: str) -> pd.DataFrame:
        return pd.DataFrame([{"subject_id": 101, "split": "train"}])

    monkeypatch.setattr(compile_mod.pd, "read_parquet", _fake_read_parquet)
    manifest = compile_mod.write_precompiled_index(
        output_dir=str(tmp_path),
        splits_parquet=str(tmp_path / "splits.parquet"),
        index_filename="trajectory_index.csv",
        manifest_filename="trajectory_manifest.json",
        records=[
            {
                "subject_id": 101,
                "rel_path": "shards/000000.ptz",
                "subject_idx": 0,
                "trajectory_ord": 0,
            },
            {
                "subject_id": 101,
                "rel_path": "shards/000000.ptz",
                "subject_idx": 0,
                "trajectory_ord": 1,
            },
        ],
        storage_format="packed_shard_v2",
    )

    index_df = pd.read_csv(tmp_path / "trajectory_index.csv")
    assert index_df["trajectory_ord"].tolist() == [0, 1]
    assert manifest["total_timelines"] == 2
    assert manifest["split_counts"] == {"train": 2}


def test_compact_timeline_roundtrip_preserves_event_fields() -> None:
    timeline = _make_timeline(101, length=8)

    compact = serialize_timeline_compact(timeline)
    restored = deserialize_timeline_compact(compact)

    assert restored == ensure_event_frames(timeline)


def test_precompiled_dataset_loads_packed_shard_from_index(tmp_path: Path) -> None:
    shard_path = tmp_path / "shards" / "000000.ptz"
    train_timeline = _make_timeline(101, length=5)
    tuning_timeline = _make_timeline(202, length=7)
    save_packed_shard(
        shard_path,
        subject_ids=[101, 202],
        serialized_timelines=[
            serialize_timeline_compact(train_timeline),
            serialize_timeline_compact(tuning_timeline),
        ],
    )
    pd.DataFrame(
        [
            {
                "subject_id": 101,
                "rel_path": "shards/000000.ptz",
                "subject_idx": 0,
                "split": "train",
            },
            {
                "subject_id": 202,
                "rel_path": "shards/000000.ptz",
                "subject_idx": 1,
                "split": "tuning",
            },
        ]
    ).to_csv(tmp_path / "index.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        '{"version": 2, "storage_format": "packed_shard_v2"}',
        encoding="utf-8",
    )

    train_ds = dataset_mod.PrecompiledMEDSDataset(str(tmp_path), split="train")
    val_ds = dataset_mod.PrecompiledMEDSDataset(str(tmp_path), split="val")

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert train_ds[0] == ensure_event_frames(train_timeline)
    assert val_ds[0] == ensure_event_frames(tuning_timeline)


def test_precompiled_dataset_can_return_metadata(tmp_path: Path) -> None:
    shard_path = tmp_path / "shards" / "000000.ptz"
    timeline = _make_timeline(101, length=5)
    save_packed_shard(
        shard_path,
        subject_ids=[101],
        serialized_timelines=[serialize_timeline_compact(timeline)],
    )
    pd.DataFrame(
        [
            {
                "subject_id": 101,
                "rel_path": "shards/000000.ptz",
                "subject_idx": 0,
                "trajectory_ord": 0,
                "materialized_trajectory": 1,
                "split": "train",
            }
        ]
    ).to_csv(tmp_path / "trajectory_index.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        '{"version": 2, "storage_format": "packed_shard_v2"}',
        encoding="utf-8",
    )

    ds = dataset_mod.PrecompiledMEDSDataset(
        str(tmp_path),
        split="train",
        index_filename="trajectory_index.csv",
        return_metadata=True,
    )
    sample = ds[0]

    assert int(sample["subject_id"]) == 101
    assert int(sample["trajectory_ord"]) == 0
    assert str(sample["rel_path"]) == "shards/000000.ptz"
    assert int(sample["subject_idx"]) == 0
    assert bool(sample["materialized_trajectory"]) is True
    assert sample["timeline"] == ensure_event_frames(timeline)


def test_packed_shard_is_smaller_than_legacy_subject_pickle(tmp_path: Path) -> None:
    timeline = _make_timeline(303, length=128)
    legacy_path = tmp_path / "303.pt"
    shard_path = tmp_path / "shards" / "000000.ptz"

    torch.save(timeline, legacy_path)
    save_packed_shard(
        shard_path,
        subject_ids=[303],
        serialized_timelines=[serialize_timeline_compact(timeline)],
    )

    assert shard_path.stat().st_size < legacy_path.stat().st_size


def test_precompiled_dataset_loads_materialized_trajectory_rows_directly(tmp_path: Path) -> None:
    trajectory = _make_timeline(404, length=4)
    shard_path = tmp_path / "trajectory_shards" / "000000.ptz"
    save_packed_shard(
        shard_path,
        subject_ids=[404],
        serialized_timelines=[serialize_timeline_compact(trajectory)],
    )
    pd.DataFrame(
        [
            {
                "subject_id": 404,
                "rel_path": "trajectory_shards/000000.ptz",
                "subject_idx": 0,
                "trajectory_id": 0,
                "trajectory_ord": 0,
                "materialized_trajectory": 1,
                "split": "train",
            }
        ]
    ).to_csv(tmp_path / "trajectory_index.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        '{"version": 2, "storage_format": "packed_shard_v2"}',
        encoding="utf-8",
    )

    ds = dataset_mod.PrecompiledMEDSDataset(
        str(tmp_path),
        split="train",
        index_filename="trajectory_index.csv",
    )

    assert len(ds) == 1
    assert ds[0] == ensure_event_frames(trajectory)


def test_build_trajectory_index_records_materializes_direct_trajectory_shards(tmp_path: Path) -> None:
    full_timeline = _make_split_timeline()
    save_packed_shard(
        tmp_path / "shards" / "000000.ptz",
        subject_ids=[101],
        serialized_timelines=[serialize_timeline_compact(full_timeline)],
    )

    rows = compile_mod._build_trajectory_index_records(
        output_dir=str(tmp_path),
        records=[{"subject_id": 101, "rel_path": "shards/000000.ptz", "subject_idx": 0}],
        segmentation_config=WindowSegmentationConfig(
            unk_window_type_id=0,
            post_discharge_window_type_id=5,
        ),
        trajectory_split_config=TrajectorySplitConfig(
            mode="admission_chain",
            post_discharge_cutoff_hours=31.0 * 24.0,
        ),
    )

    assert len(rows) == 2
    assert all(int(row["materialized_trajectory"]) == 1 for row in rows)
    assert all(str(row["rel_path"]).startswith("trajectory_shards/") for row in rows)
    assert (tmp_path / str(rows[0]["rel_path"])).exists()
