"""
Utility to materialize subject timelines to disk for fast training.

This is intentionally minimal: callers must construct encoders/structural codebook
upstream and pass a meds_reader DB path. We shard outputs as <root>/<prefix>/<sid>.pt.
"""
from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional

import torch
import meds_reader as mr
from tqdm import tqdm

from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.data.structural_codes import StructuralCodebook


def _process_subject(
    subject_id: int,
    *,
    db_path: str,
    encoders: Dict[TokenCategory, EventTokenEncoder],
    codebook: Optional[StructuralCodebook],
    output_dir: str,
    window_hook_label: str = "window_boundary",
    attach_med_numeric: bool = True,
) -> bool:
    """
    Worker-safe timeline build + save for a single subject.
    """
    db = mr.SubjectDatabase(db_path)
    timeline = build_subject_timeline(
        db=db,
        subject_id=subject_id,
        encoders=encoders,
        structural_codebook=codebook,
        window_hook_label=window_hook_label,
        attach_med_numeric=attach_med_numeric,
    )

    shard_folder = f"{subject_id:02d}"[:2]
    save_path = Path(output_dir) / shard_folder / f"{subject_id}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(timeline, save_path)
    return True


def compile_dataset(
    *,
    db_path: str,
    encoders: Dict[TokenCategory, EventTokenEncoder],
    output_dir: str,
    structural_codebook: Optional[StructuralCodebook] = None,
    num_workers: Optional[int] = None,
) -> None:
    """
    Build timelines for all subjects and persist them to disk.
    """
    if not encoders:
        raise ValueError("encoders must be provided (TokenCategory -> EventTokenEncoder)")

    db = mr.SubjectDatabase(db_path)
    subject_ids = list(db)

    workers = num_workers if num_workers is not None else max(1, cpu_count() - 2)
    worker_fn = partial(
        _process_subject,
        db_path=db_path,
        encoders=encoders,
        codebook=structural_codebook,
        output_dir=output_dir,
    )

    os.makedirs(output_dir, exist_ok=True)
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_fn, subject_ids),
                total=len(subject_ids),
                desc="compiling timelines",
            )
        )

    ok = sum(results)
    print(f"compiled {ok}/{len(subject_ids)} subjects -> {output_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "compile_dataset is a library entrypoint; construct encoders externally "
        "and call compile_dataset(db_path=..., encoders=..., output_dir=...)."
    )
