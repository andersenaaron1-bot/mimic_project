from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

import src.ehr_hier.data.compile_dataset as compile_mod
import src.ehr_hier.data.dataset as dataset_mod


def _write_timeline(root: Path, rel_path: str, payload: object) -> None:
    fp = root / rel_path
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, fp)


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
