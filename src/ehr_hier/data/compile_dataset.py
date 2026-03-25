"""
Utility to materialize subject timelines to disk for fast training.

This is intentionally minimal: callers must construct encoders/structural codebook
upstream and pass a meds_reader DB path. We shard outputs as <root>/<shard>/<sid>.pt.
"""
from __future__ import annotations

import os
import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional, Iterable, List

import pandas as pd
import torch
import meds_reader as mr
from tqdm import tqdm

from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.interfaces import EventTokenEncoder
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.data.structural_codes import StructuralCodebook

_WORKER_STATE: Dict[str, object] = {}


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_compile_progress(
    *,
    done: int,
    total: int,
    compiled: int,
    skipped_existing: int,
    started_at: float,
) -> None:
    elapsed = max(1e-9, time.perf_counter() - started_at)
    rate = float(done) / elapsed
    remaining = max(0, int(total) - int(done))
    eta = (float(remaining) / rate) if rate > 0.0 else float("inf")
    print(
        (
            "[compile_dataset] "
            f"done={int(done)}/{int(total)} "
            f"compiled={int(compiled)} "
            f"skipped_existing={int(skipped_existing)} "
            f"elapsed={_format_elapsed(elapsed)} "
            f"rate={rate:.2f} subjects/s "
            f"eta={_format_elapsed(eta if eta != float('inf') else 0.0)}"
        ),
        flush=True,
    )


def write_precompiled_index(
    *,
    output_dir: str,
    splits_parquet: str | None = None,
    index_filename: str = "index.csv",
    manifest_filename: str = "manifest.json",
) -> Dict[str, object]:
    root = Path(output_dir)
    files = sorted(root.glob("**/*.pt"))
    rows: List[Dict[str, object]] = []
    skipped_non_integer = 0
    for fp in files:
        stem = fp.stem
        if not stem.isdigit():
            skipped_non_integer += 1
            continue
        rows.append(
            {
                "subject_id": int(stem),
                "rel_path": str(fp.relative_to(root)).replace("\\", "/"),
            }
        )
    if not rows:
        raise ValueError(f"No precompiled .pt timelines found under {output_dir}")

    index_df = pd.DataFrame.from_records(rows).sort_values("subject_id").reset_index(drop=True)
    split_counts: Dict[str, int] = {}
    if splits_parquet is not None:
        split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]].copy()
        split_df["subject_id"] = split_df["subject_id"].astype("int64")
        index_df = index_df.merge(split_df, on="subject_id", how="left", validate="one_to_one")
        split_counts = {
            str(k): int(v)
            for k, v in index_df["split"].fillna("<missing>").value_counts().sort_index().items()
        }

    index_path = root / str(index_filename)
    index_df.to_csv(index_path, index=False)

    manifest = {
        "version": 1,
        "data_root": str(root),
        "index_filename": str(index_filename),
        "total_timelines": int(len(index_df)),
        "skipped_non_integer_files": int(skipped_non_integer),
        "subject_id_min": int(index_df["subject_id"].min()),
        "subject_id_max": int(index_df["subject_id"].max()),
        "split_counts": split_counts,
    }
    manifest_path = root / str(manifest_filename)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _init_compile_worker(
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
    skip_existing: bool = True,
) -> None:
    """
    Initialize per-worker resources once to avoid reopening the DB for every subject.
    """
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "db": mr.SubjectDatabase(db_path),
            "encoders": encoders,
            "codebook": codebook,
            "output_dir": str(output_dir),
            "num_output_shards": int(num_output_shards),
            "window_hook_label": str(window_hook_label),
            "attach_med_numeric": bool(attach_med_numeric),
            "qual_obs_code_vocab": qual_obs_code_vocab,
            "qual_obs_value_vocab": qual_obs_value_vocab,
            "qual_obs_tail_policy": str(qual_obs_tail_policy),
            "skip_existing": bool(skip_existing),
        }
    )


def _process_subject(subject_id: int) -> str:
    """
    Worker-safe timeline build + save for a single subject.
    """
    if not _WORKER_STATE:
        raise RuntimeError("compile_dataset worker state was not initialized")

    shard_idx = int(subject_id) % int(_WORKER_STATE["num_output_shards"])
    shard_folder = f"{shard_idx:02d}"
    save_path = Path(str(_WORKER_STATE["output_dir"])) / shard_folder / f"{subject_id}.pt"
    if bool(_WORKER_STATE["skip_existing"]) and save_path.exists():
        return "skipped_existing"

    try:
        timeline = build_subject_timeline(
            db=_WORKER_STATE["db"],
            subject_id=int(subject_id),
            encoders=_WORKER_STATE["encoders"],
            structural_codebook=_WORKER_STATE["codebook"],
            window_hook_label=str(_WORKER_STATE["window_hook_label"]),
            attach_med_numeric=bool(_WORKER_STATE["attach_med_numeric"]),
            qual_obs_code_vocab=_WORKER_STATE["qual_obs_code_vocab"],
            qual_obs_value_vocab=_WORKER_STATE["qual_obs_value_vocab"],
            qual_obs_tail_policy=str(_WORKER_STATE["qual_obs_tail_policy"]),
        )
    except Exception as exc:  # pragma: no cover - exercised in integration contexts
        raise RuntimeError(f"Failed to compile subject_id={int(subject_id)}") from exc

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(timeline, save_path)
    return "compiled"


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
    splits_parquet: str | None = None,
    write_index: bool = True,
    skip_existing: bool = True,
    progress_every: int = 100,
) -> Dict[str, object]:
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
    workers = max(1, int(workers))
    os.makedirs(output_dir, exist_ok=True)
    chunksize = max(1, min(64, len(subject_id_list) // max(1, workers * 8)))
    started_at = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "compile_dataset_start",
                "output_dir": str(output_dir),
                "subjects": int(len(subject_id_list)),
                "workers": int(workers),
                "chunksize": int(chunksize),
                "num_output_shards": int(num_output_shards),
                "skip_existing": bool(skip_existing),
                "write_index": bool(write_index),
            },
            indent=2,
        ),
        flush=True,
    )
    with Pool(
        processes=workers,
        initializer=_init_compile_worker,
        initargs=(
            db_path,
            encoders,
            structural_codebook,
            output_dir,
            num_output_shards,
            "window_boundary",
            True,
            qual_obs_code_vocab,
            qual_obs_value_vocab,
            qual_obs_tail_policy,
            bool(skip_existing),
        ),
    ) as pool:
        compiled = 0
        skipped_existing_count = 0
        done = 0
        with tqdm(
            total=len(subject_id_list),
            desc="compiling timelines",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as pbar:
            for status in pool.imap_unordered(_process_subject, subject_id_list, chunksize=chunksize):
                done += 1
                if status == "compiled":
                    compiled += 1
                elif status == "skipped_existing":
                    skipped_existing_count += 1
                pbar.update(1)
                if int(progress_every) > 0 and (
                    done % int(progress_every) == 0 or done == len(subject_id_list)
                ):
                    _print_compile_progress(
                        done=done,
                        total=len(subject_id_list),
                        compiled=compiled,
                        skipped_existing=skipped_existing_count,
                        started_at=started_at,
                    )

    elapsed = time.perf_counter() - started_at
    print(
        (
            f"compiled {compiled}/{len(subject_id_list)} subjects "
            f"(skipped_existing={skipped_existing_count}) -> {output_dir} "
            f"in {_format_elapsed(elapsed)} "
            f"({(float(done) / max(elapsed, 1e-9)):.2f} subjects/s)"
        ),
        flush=True,
    )
    if bool(write_index):
        manifest = write_precompiled_index(
            output_dir=output_dir,
            splits_parquet=splits_parquet,
        )
        manifest["compile_workers"] = int(workers)
        manifest["compile_chunksize"] = int(chunksize)
        manifest["compile_elapsed_seconds"] = float(elapsed)
        manifest["compiled_subjects"] = int(compiled)
        manifest["skipped_existing_subjects"] = int(skipped_existing_count)
        manifest_path = Path(output_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"wrote precompiled index -> {Path(output_dir) / 'index.csv'}")
        return manifest
    return {
        "version": 1,
        "data_root": str(output_dir),
        "total_timelines_requested": int(len(subject_id_list)),
        "total_timelines_compiled": int(compiled),
        "total_timelines_skipped_existing": int(skipped_existing_count),
        "compile_workers": int(workers),
        "compile_chunksize": int(chunksize),
        "compile_elapsed_seconds": float(elapsed),
    }


if __name__ == "__main__":
    raise SystemExit(
        "compile_dataset is a library entrypoint; construct encoders externally "
        "and call compile_dataset(db_path=..., encoders=..., output_dir=..., "
        "subject_ids=..., num_output_shards=...)."
    )
