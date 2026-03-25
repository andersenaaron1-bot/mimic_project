from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import List
import warnings

import pandas as pd

import torch
from torch.utils.data import Dataset

from src.ehr_hier.data.token_types import EventToken


class PrecompiledMEDSDataset(Dataset):
    """
    Simple dataset over precompiled timelines saved by compile_dataset.
    """

    def __init__(
        self,
        data_root: str,
        *,
        split: str = "train",
        split_ratio: float = 0.8,
        splits_parquet: str | None = None,
    ) -> None:
        self.data_root = str(data_root)
        root = Path(self.data_root)
        index_csv = root / "index.csv"

        index_df: pd.DataFrame | None = None
        if index_csv.exists():
            index_df = pd.read_csv(index_csv)
            required = {"subject_id", "rel_path"}
            if not required.issubset(index_df.columns):
                raise ValueError(
                    f"Precompiled dataset index at {index_csv} must contain columns {sorted(required)}"
                )
            index_df = index_df.copy()
            index_df["subject_id"] = index_df["subject_id"].astype("int64")
            index_df["file_path"] = index_df["rel_path"].map(lambda rel: str(root / str(rel)))

        if index_df is None:
            files: List[str] = sorted(glob(os.path.join(self.data_root, "**", "*.pt"), recursive=True))
            if not files:
                raise ValueError(f"No .pt files found under {data_root}; run compile_dataset first.")
            records = []
            skipped_non_integer = 0
            for fp in files:
                stem = Path(fp).stem
                if not stem.isdigit():
                    skipped_non_integer += 1
                    continue
                records.append({"file_path": str(fp), "subject_id": int(stem)})
            index_df = pd.DataFrame.from_records(records)
            if skipped_non_integer:
                raise ValueError(
                    "Found non-integer precompiled timeline file stems without an index.csv. "
                    "Regenerate the precompiled index or rename files to <subject_id>.pt."
                )

        # Preferred path: enforce a canonical subject split file.
        if splits_parquet is not None:
            split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]]

            # MEDS conventions typically use train/tuning/held_out.
            aliases = {"val": "tuning", "test": "held_out"}
            target_split = aliases.get(split, split)

            keep = set(split_df.loc[split_df["split"] == target_split, "subject_id"].astype("int64").tolist())
            if not keep:
                raise ValueError(f"No subjects for split='{target_split}' in {splits_parquet}")

            selected = index_df.loc[index_df["subject_id"].isin(keep), "file_path"].astype(str).tolist()
            skipped = 0

            if not selected:
                raise ValueError(
                    f"No precompiled timeline files matched split='{target_split}' under {data_root}. "
                    "Make sure file names are <subject_id>.pt and were compiled for this cohort."
                )
            if skipped > 0:
                warnings.warn(
                    f"Skipped {skipped} timeline files with non-integer stems while applying split filter.",
                    stacklevel=2,
                )
            self.file_paths = selected
            return

        # Backward-compatible fallback: deterministic but not cohort-aware.
        if "split" in index_df.columns:
            aliases = {"val": "tuning", "test": "held_out"}
            target_split = aliases.get(split, split)
            selected = (
                index_df.loc[index_df["split"].astype(str) == str(target_split), "file_path"]
                .astype(str)
                .tolist()
            )
            if not selected:
                raise ValueError(
                    f"No precompiled timeline files matched split='{target_split}' under {data_root} index."
                )
            self.file_paths = selected
            return

        files = index_df["file_path"].astype(str).tolist()
        cut = int(len(files) * float(split_ratio))
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val' when splits_parquet is not provided")
        warnings.warn(
            "Using split_ratio fallback; pass splits_parquet or compile an index.csv for canonical train/val/test consistency.",
            stacklevel=2,
        )
        self.file_paths = files[:cut] if split == "train" else files[cut:]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> List[EventToken]:
        fp = self.file_paths[idx]
        timeline = torch.load(fp)
        return timeline
