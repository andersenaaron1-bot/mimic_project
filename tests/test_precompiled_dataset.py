from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch

import src.ehr_hier.data.compile_dataset as compile_mod
import src.ehr_hier.data.dataset as dataset_mod
from src.ehr_hier.data.precompiled_format import (
    deserialize_timeline_compact,
    save_packed_shard,
    serialize_timeline_compact,
)
from src.ehr_hier.data.token_types import EventToken


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


def test_precompiled_dataset_filters_from_index_split_column(tmp_path: Path) -> None:
    _write_timeline(tmp_path, "00/101.pt", [{"subject_id": 101}])
    _write_timeline(tmp_path, "01/202.pt", [{"subject_id": 202}])
    pd.DataFrame(
        [
            {"subject_id": 101, "rel_path": "00/101.pt", "split": "train"},
            {"subject_id": 202, "rel_path": "01/202.pt", "split": "tuning"},
        ]
    ).to_csv(tmp_path / "index.csv", index=False)

    train_ds = dataset_mod.PrecompiledMEDSDataset(str(tmp_path), split="train")
    val_ds = dataset_mod.PrecompiledMEDSDataset(str(tmp_path), split="val")

    assert len(train_ds) == 1
    assert train_ds[0][0]["subject_id"] == 101
    assert len(val_ds) == 1
    assert val_ds[0][0]["subject_id"] == 202


def test_precompiled_dataset_prefers_canonical_splits_parquet_over_index_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_timeline(tmp_path, "00/101.pt", [{"subject_id": 101}])
    _write_timeline(tmp_path, "01/202.pt", [{"subject_id": 202}])
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
    assert tuning_ds[0][0]["subject_id"] == 202


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


def test_compact_timeline_roundtrip_preserves_event_fields() -> None:
    timeline = _make_timeline(101, length=8)

    compact = serialize_timeline_compact(timeline)
    restored = deserialize_timeline_compact(compact)

    assert restored == timeline


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
    assert train_ds[0] == train_timeline
    assert val_ds[0] == tuning_timeline


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
