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

        files: List[str] = sorted(glob(os.path.join(self.data_root, "**", "*.pt"), recursive=True))
        if not files:
            raise ValueError(f"No .pt files found under {data_root}; run compile_dataset first.")

        # Preferred path: enforce a canonical subject split file.
        if splits_parquet is not None:
            split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]]

            # MEDS conventions typically use train/tuning/held_out.
            aliases = {"val": "tuning", "test": "held_out"}
            target_split = aliases.get(split, split)

            keep = set(split_df.loc[split_df["split"] == target_split, "subject_id"].astype("int64").tolist())
            if not keep:
                raise ValueError(f"No subjects for split='{target_split}' in {splits_parquet}")

            selected: List[str] = []
            skipped = 0
            for fp in files:
                stem = Path(fp).stem
                try:
                    sid = int(stem)
                except ValueError:
                    skipped += 1
                    continue
                if sid in keep:
                    selected.append(fp)

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
        cut = int(len(files) * float(split_ratio))
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val' when splits_parquet is not provided")
        warnings.warn(
            "Using split_ratio fallback; pass splits_parquet for canonical train/val/test consistency.",
            stacklevel=2,
        )
        self.file_paths = files[:cut] if split == "train" else files[cut:]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> List[EventToken]:
        fp = self.file_paths[idx]
        timeline = torch.load(fp)
        return timeline
