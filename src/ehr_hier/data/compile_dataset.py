"""
Utility to materialize subject timelines to disk for fast training.

This is intentionally minimal: callers must construct encoders/structural codebook
upstream and pass a meds_reader DB path. We shard outputs as <root>/<shard>/<sid>.pt.
"""
from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional, Iterable, List

import torch
import meds_reader as mr
from tqdm import tqdm

from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.data.structural_codes import StructuralCodebook


def _process_subject(
    subject_id: int,
    *,
    db_path: str,
    encoders: Dict[TokenCategory, EventTokenEncoder],
    codebook: Optional[StructuralCodebook],
    output_dir: str,
    num_output_shards: int = 100,
    window_hook_label: str = "window_boundary",
    attach_med_numeric: bool = True,
    qual_obs_code_vocab: Optional[CategoryVocab] = None,
    qual_obs_value_vocab: Optional[CategoryVocab] = None,
    qual_obs_tail_policy: str = "drop",
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
        qual_obs_code_vocab=qual_obs_code_vocab,
        qual_obs_value_vocab=qual_obs_value_vocab,
        qual_obs_tail_policy=qual_obs_tail_policy,
    )

    shard_idx = int(subject_id) % int(num_output_shards)
    shard_folder = f"{shard_idx:02d}"
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
    qual_obs_code_vocab: Optional[CategoryVocab] = None,
    qual_obs_value_vocab: Optional[CategoryVocab] = None,
    qual_obs_tail_policy: str = "drop",
    num_workers: Optional[int] = None,
    subject_ids: Optional[Iterable[int]] = None,
    num_output_shards: int = 100,
) -> None:
    """
    Build timelines for selected subjects and persist them to disk.
    """
    if not encoders:
        raise ValueError("encoders must be provided (TokenCategory -> EventTokenEncoder)")
    if num_output_shards <= 0:
        raise ValueError("num_output_shards must be > 0")

    db = mr.SubjectDatabase(db_path)
    if subject_ids is None:
        subject_id_list: List[int] = [int(sid) for sid in db]
    else:
        subject_id_list = [int(sid) for sid in subject_ids]

    workers = num_workers if num_workers is not None else max(1, cpu_count() - 2)
    worker_fn = partial(
        _process_subject,
        db_path=db_path,
        encoders=encoders,
        codebook=structural_codebook,
        output_dir=output_dir,
        num_output_shards=num_output_shards,
        qual_obs_code_vocab=qual_obs_code_vocab,
        qual_obs_value_vocab=qual_obs_value_vocab,
        qual_obs_tail_policy=qual_obs_tail_policy,
    )

    os.makedirs(output_dir, exist_ok=True)
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_fn, subject_id_list),
                total=len(subject_id_list),
                desc="compiling timelines",
            )
        )

    ok = sum(results)
    print(f"compiled {ok}/{len(subject_id_list)} subjects -> {output_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "compile_dataset is a library entrypoint; construct encoders externally "
        "and call compile_dataset(db_path=..., encoders=..., output_dir=..., "
        "subject_ids=..., num_output_shards=...)."
    )
