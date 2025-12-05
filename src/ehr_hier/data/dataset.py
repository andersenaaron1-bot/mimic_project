from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from src.ehr_hier.data.token_types import EventToken


class PrecompiledMEDSDataset(Dataset):
    """
    Simple dataset over precompiled timelines saved by compile_dataset.
    """

    def __init__(self, data_root: str, *, split: str = "train", split_ratio: float = 0.8) -> None:
        self.data_root = str(data_root)

        files: List[str] = sorted(glob(os.path.join(self.data_root, "**", "*.pt"), recursive=True))
        if not files:
            raise ValueError(f"No .pt files found under {data_root}; run compile_dataset first.")

        cut = int(len(files) * float(split_ratio))
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.file_paths = files[:cut] if split == "train" else files[cut:]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> List[EventToken]:
        fp = self.file_paths[idx]
        timeline = torch.load(fp)
        return timeline
