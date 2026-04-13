"""
Utility to materialize subject frame timelines to disk for fast training.

This is intentionally minimal: callers must construct encoders/structural codebook
upstream and pass a meds_reader DB path. Packed event-frame shard outputs are written under
<root>/shards/<shard>.ptz with an index.csv mapping subject_id -> shard row.
"""
from __future__ import annotations

import json
import math
import os
import time
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import meds_reader as mr
import torch
from tqdm import tqdm

from src.ehr_hier.data.precompiled_format import (
    PRECOMPILED_PAYLOAD_VERSION,
    PRECOMPILED_SHARD_SUFFIX,
    PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
    deserialize_timeline_compact,
    load_packed_shard,
    save_packed_shard,
    serialize_timeline_compact,
)
from src.ehr_hier.data.demographics import collect_subject_demographic_metadata
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.data.trajectory_splitting import (
    TrajectorySplitConfig,
    build_trajectory_timelines,
)
from src.ehr_hier.data.window_segmentation import WindowSegmentationConfig
from src.ehr_hier.tokenizers.interfaces import EventFrameEncoder
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
    records: Iterable[Dict[str, object]] | None = None,
    storage_format: str | None = None,
) -> Dict[str, object]:
    root = Path(output_dir)
    rows: List[Dict[str, object]]
    skipped_non_integer = 0
    if records is not None:
        rows = []
        for rec in records:
            row = {
                "subject_id": int(rec["subject_id"]),
                "rel_path": str(rec["rel_path"]),
            }
            if "subject_idx" in rec and rec["subject_idx"] is not None and not pd.isna(rec["subject_idx"]):
                row["subject_idx"] = int(rec["subject_idx"])
            for key, value in dict(rec).items():
                if key in row or key in {"subject_id", "rel_path", "subject_idx", "split", "file_path"}:
                    continue
                row[str(key)] = value
            rows.append(row)
    else:
        files = sorted(root.glob("**/*.pt"))
        rows = []
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
        raise ValueError(f"No precompiled timelines found under {output_dir}")

    sort_cols = ["subject_id"] + [col for col in ("trajectory_ord", "trajectory_id") if col in rows[0]]
    index_df = pd.DataFrame.from_records(rows).sort_values(sort_cols).reset_index(drop=True)
    split_counts: Dict[str, int] = {}
    if splits_parquet is not None:
        split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]].copy()
        split_df["subject_id"] = split_df["subject_id"].astype("int64")
        merge_validate = "one_to_one" if index_df["subject_id"].is_unique else "many_to_one"
        index_df = index_df.merge(split_df, on="subject_id", how="left", validate=merge_validate)
        split_counts = {
            str(k): int(v)
            for k, v in index_df["split"].fillna("<missing>").value_counts().sort_index().items()
        }

    index_path = root / str(index_filename)
    index_df.to_csv(index_path, index=False)

    manifest = {
        "version": PRECOMPILED_PAYLOAD_VERSION if storage_format else 1,
        "data_root": str(root),
        "index_filename": str(index_filename),
        "total_timelines": int(len(index_df)),
        "skipped_non_integer_files": int(skipped_non_integer),
        "subject_id_min": int(index_df["subject_id"].min()),
        "subject_id_max": int(index_df["subject_id"].max()),
        "split_counts": split_counts,
        "storage_format": str(storage_format or "legacy_subject_pt"),
    }
    manifest_path = root / str(manifest_filename)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _init_compile_worker(
    db_path: str,
    encoders: Dict[TokenCategory, EventFrameEncoder],
    codebook: Optional[StructuralCodebook],
    output_dir: str,
    window_hook_label: str = "window_boundary",
    attach_med_numeric: bool = True,
    qual_obs_code_vocab: Optional[CategoryVocab] = None,
    qual_obs_value_vocab: Optional[CategoryVocab] = None,
    qual_obs_tail_policy: str = "drop",
) -> None:
    """
    Initialize per-worker resources once to avoid reopening the DB for every subject.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    proc = current_process()
    worker_tag = (
        f"w{int(proc._identity[0]):03d}"  # type: ignore[attr-defined]
        if getattr(proc, "_identity", None)
        else f"pid{os.getpid()}"
    )
    shards_dir = Path(output_dir) / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "db": mr.SubjectDatabase(db_path),
            "encoders": encoders,
            "codebook": codebook,
            "output_dir": str(output_dir),
            "window_hook_label": str(window_hook_label),
            "attach_med_numeric": bool(attach_med_numeric),
            "qual_obs_code_vocab": qual_obs_code_vocab,
            "qual_obs_value_vocab": qual_obs_value_vocab,
            "qual_obs_tail_policy": str(qual_obs_tail_policy),
            "worker_tag": worker_tag,
            "worker_local_shard_idx": 0,
        }
    )


def _flush_worker_rows(rows: Sequence[tuple[int, Dict[str, Any]]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    worker_tag = str(_WORKER_STATE["worker_tag"])
    shard_idx = int(_WORKER_STATE["worker_local_shard_idx"])
    shard_rel_path = f"shards/{worker_tag}_{shard_idx:06d}{PRECOMPILED_SHARD_SUFFIX}"
    ordered = sorted(rows, key=lambda item: int(item[0]))
    subject_ids = [int(subject_id) for subject_id, _ in ordered]
    timelines = [payload for _, payload in ordered]
    save_packed_shard(
        Path(str(_WORKER_STATE["output_dir"])) / shard_rel_path,
        subject_ids=subject_ids,
        serialized_timelines=timelines,
    )
    _WORKER_STATE["worker_local_shard_idx"] = shard_idx + 1
    return [
        {
            "subject_id": int(subject_id),
            "rel_path": shard_rel_path,
            "subject_idx": int(pos),
        }
        for pos, subject_id in enumerate(subject_ids)
    ]


def _process_subject_batch(subject_ids: Sequence[int]) -> List[Dict[str, object]]:
    """
    Worker-safe timeline build + compact serialization for a batch of subjects.
    """
    if not _WORKER_STATE:
        raise RuntimeError("compile_dataset worker state was not initialized")

    rows: List[tuple[int, Dict[str, Any]]] = []
    for subject_id in subject_ids:
        try:
            subject = _WORKER_STATE["db"][int(subject_id)]
            subject_events = list(subject.events)
            timeline_start_ts = None
            for ev in subject_events:
                t_ev = getattr(ev, "time", None)
                if t_ev is None or not hasattr(t_ev, "timestamp"):
                    continue
                try:
                    timeline_start_ts = float(t_ev.timestamp())
                except Exception:
                    timeline_start_ts = None
                break
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
        metadata = collect_subject_demographic_metadata(
            subject_events,
            timeline_start_ts=timeline_start_ts,
        )
        rows.append(
            (
                int(subject_id),
                serialize_timeline_compact(
                    timeline,
                    metadata={"subject_demographics": metadata},
                ),
            )
        )
    return _flush_worker_rows(rows)


def _load_existing_index_records(output_dir: str, *, index_filename: str = "index.csv") -> List[Dict[str, object]]:
    index_path = Path(output_dir) / str(index_filename)
    if not index_path.exists():
        return []
    index_df = pd.read_csv(index_path)
    required = {"subject_id", "rel_path"}
    if not required.issubset(index_df.columns):
        raise ValueError(
            f"Existing index at {index_path} must contain columns {sorted(required)}"
        )
    records = index_df.to_dict(orient="records")
    out: List[Dict[str, object]] = []
    for rec in records:
        row = {
            "subject_id": int(rec["subject_id"]),
            "rel_path": str(rec["rel_path"]),
        }
        if "subject_idx" in rec and not pd.isna(rec["subject_idx"]):
            row["subject_idx"] = int(rec["subject_idx"])
        for key, value in rec.items():
            if key in row or key in {"subject_id", "rel_path", "subject_idx", "split", "file_path"}:
                continue
            row[str(key)] = value
        out.append(row)
    return out


def _build_trajectory_index_records(
    *,
    output_dir: str,
    records: Sequence[Dict[str, object]],
    segmentation_config: WindowSegmentationConfig,
    trajectory_split_config: TrajectorySplitConfig,
) -> List[Dict[str, object]]:
    root = Path(output_dir)
    shard_cache: Dict[str, Dict[str, Any]] = {}
    materialized_subject_ids: List[int] = []
    materialized_payloads: List[Dict[str, Any]] = []
    materialized_rows: List[Dict[str, object]] = []
    out: List[Dict[str, object]] = []
    shard_idx = 0
    target_trajectories_per_shard = 512

    def _flush_materialized_shard() -> None:
        nonlocal shard_idx
        if not materialized_payloads:
            return
        shard_rel_path = f"trajectory_shards/{shard_idx:06d}{PRECOMPILED_SHARD_SUFFIX}"
        save_packed_shard(
            root / shard_rel_path,
            subject_ids=list(materialized_subject_ids),
            serialized_timelines=list(materialized_payloads),
        )
        for pos, row in enumerate(materialized_rows):
            out.append(
                {
                    "subject_id": int(row["subject_id"]),
                    "rel_path": shard_rel_path,
                    "subject_idx": int(pos),
                    "trajectory_id": int(row["trajectory_id"]),
                    "trajectory_ord": int(row["trajectory_ord"]),
                    "materialized_trajectory": 1,
                    "source_rel_path": str(row["source_rel_path"]),
                    **(
                        {"source_subject_idx": int(row["source_subject_idx"])}
                        if row.get("source_subject_idx") is not None
                        else {}
                    ),
                }
            )
        materialized_subject_ids.clear()
        materialized_payloads.clear()
        materialized_rows.clear()
        shard_idx += 1

    for rec in sorted(records, key=lambda row: (int(row["subject_id"]), int(row.get("subject_idx", 0) or 0))):
        rel_path = str(rec["rel_path"])
        subject_idx = rec.get("subject_idx", None)
        subject_id = int(rec["subject_id"])
        file_path = root / rel_path
        if subject_idx is None or pd.isna(subject_idx):
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "value_ids" in payload:
                serialized = payload
            else:
                continue
        else:
            cached = shard_cache.get(rel_path)
            if cached is None:
                cached = load_packed_shard(file_path)
                shard_cache[rel_path] = cached
            serialized = cached["timelines"][int(subject_idx)]
        timeline = deserialize_timeline_compact(serialized)
        metadata = dict((serialized or {}).get("metadata", {}) or {})
        subject_metadata = metadata.get("subject_demographics", None)
        trajectories = build_trajectory_timelines(
            timeline=timeline,
            segmentation_config=segmentation_config,
            split_config=trajectory_split_config,
            subject_metadata=subject_metadata if isinstance(subject_metadata, dict) else None,
        )
        if not trajectories:
            trajectories = [timeline]
        for trajectory_ord, trajectory in enumerate(trajectories):
            trajectory_id = len(out) + len(materialized_rows)
            materialized_subject_ids.append(int(subject_id))
            materialized_payloads.append(
                serialize_timeline_compact(
                    trajectory,
                    metadata={
                        "view": "trajectory",
                        "subject_id": int(subject_id),
                        "trajectory_ord": int(trajectory_ord),
                        "source_rel_path": rel_path,
                        **(
                            {"source_subject_idx": int(subject_idx)}
                            if subject_idx is not None and not pd.isna(subject_idx)
                            else {}
                        ),
                    },
                )
            )
            materialized_rows.append(
                {
                    "subject_id": int(subject_id),
                    "trajectory_id": int(trajectory_id),
                    "trajectory_ord": int(trajectory_ord),
                    "source_rel_path": rel_path,
                    "source_subject_idx": (
                        int(subject_idx) if subject_idx is not None and not pd.isna(subject_idx) else None
                    ),
                }
            )
            if len(materialized_payloads) >= int(target_trajectories_per_shard):
                _flush_materialized_shard()
    _flush_materialized_shard()
    return out


def _iter_subject_batches(subject_ids: Sequence[int], batch_size: int) -> List[List[int]]:
    size = max(1, int(batch_size))
    return [
        [int(sid) for sid in subject_ids[start : start + size]]
        for start in range(0, len(subject_ids), size)
    ]


def compile_dataset(
    *,
    db_path: str,
    encoders: Dict[TokenCategory, EventFrameEncoder],
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
    chunksize: int | None = None,
    segmentation_config: WindowSegmentationConfig | None = None,
    trajectory_split_config: TrajectorySplitConfig | None = None,
) -> Dict[str, object]:
    """
    Build timelines for selected subjects and persist them to disk.
    """
    if not encoders:
        raise ValueError("encoders must be provided (TokenCategory -> EventFrameEncoder)")
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
    existing_records = _load_existing_index_records(output_dir) if bool(skip_existing) else []
    existing_subject_ids = {int(rec["subject_id"]) for rec in existing_records}
    requested_subjects = list(subject_id_list)
    if existing_subject_ids:
        subject_id_list = [sid for sid in subject_id_list if int(sid) not in existing_subject_ids]
    skipped_existing_count = int(len(requested_subjects) - len(subject_id_list))
    resolved_chunksize = (
        max(1, int(chunksize))
        if chunksize is not None
        else max(1, min(8, len(subject_id_list) // max(1, workers * 32)))
    )
    target_subjects_per_shard = max(
        1,
        int(
            math.ceil(
                float(max(1, len(subject_id_list)))
                / float(max(1, int(num_output_shards)))
            )
        ),
    )
    resolved_subjects_per_shard = max(8, min(128, int(target_subjects_per_shard)))
    subject_batches = _iter_subject_batches(subject_id_list, resolved_subjects_per_shard)
    started_at = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "compile_dataset_start",
                "output_dir": str(output_dir),
                "subjects": int(len(requested_subjects)),
                "subjects_to_compile": int(len(subject_id_list)),
                "workers": int(workers),
                "chunksize": int(resolved_chunksize),
                "num_output_shards": int(num_output_shards),
                "target_subjects_per_shard": int(target_subjects_per_shard),
                "resolved_subjects_per_shard": int(resolved_subjects_per_shard),
                "task_batch_count": int(len(subject_batches)),
                "skip_existing": bool(skip_existing),
                "write_index": bool(write_index),
                "storage_format": PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
            },
            indent=2,
        ),
        flush=True,
    )
    if not subject_id_list:
        elapsed = time.perf_counter() - started_at
        manifest = write_precompiled_index(
            output_dir=output_dir,
            splits_parquet=splits_parquet,
            records=existing_records,
            storage_format=PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
        )
        trajectory_index_count = 0
        if segmentation_config is not None and trajectory_split_config is not None:
            trajectory_records = _build_trajectory_index_records(
                output_dir=output_dir,
                records=existing_records,
                segmentation_config=segmentation_config,
                trajectory_split_config=trajectory_split_config,
            )
            write_precompiled_index(
                output_dir=output_dir,
                splits_parquet=splits_parquet,
                index_filename="trajectory_index.csv",
                manifest_filename="trajectory_manifest.json",
                records=trajectory_records,
                storage_format=PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
            )
            trajectory_index_count = int(len(trajectory_records))
        manifest["compile_workers"] = int(workers)
        manifest["compile_chunksize"] = int(resolved_chunksize)
        manifest["compile_elapsed_seconds"] = float(elapsed)
        manifest["compiled_subjects"] = 0
        manifest["skipped_existing_subjects"] = int(skipped_existing_count)
        manifest["target_subjects_per_shard"] = int(target_subjects_per_shard)
        manifest["shard_count"] = len({str(rec["rel_path"]) for rec in existing_records})
        if trajectory_index_count > 0:
            manifest["trajectory_index_filename"] = "trajectory_index.csv"
            manifest["trajectory_count"] = int(trajectory_index_count)
        manifest_path = Path(output_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"wrote precompiled index -> {Path(output_dir) / 'index.csv'}")
        return manifest

    with Pool(
        processes=workers,
        initializer=_init_compile_worker,
        initargs=(
            db_path,
            encoders,
            structural_codebook,
            output_dir,
            "window_boundary",
            True,
            qual_obs_code_vocab,
            qual_obs_value_vocab,
            qual_obs_tail_policy,
        ),
    ) as pool:
        compiled = 0
        done = 0
        index_rows: List[Dict[str, object]] = list(existing_records)
        with tqdm(
            total=len(subject_id_list),
            desc="compiling timelines",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as pbar:
            for batch_rows in pool.imap_unordered(
                _process_subject_batch,
                subject_batches,
                chunksize=resolved_chunksize,
            ):
                batch_done = int(len(batch_rows))
                done += batch_done
                compiled += batch_done
                index_rows.extend(batch_rows)
                pbar.update(batch_done)
                if int(progress_every) > 0 and (
                    done % int(progress_every) == 0 or done >= len(subject_id_list)
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
            f"compiled {compiled}/{len(requested_subjects)} subjects "
            f"(skipped_existing={skipped_existing_count}) -> {output_dir} "
            f"in {_format_elapsed(elapsed)} "
            f"({(float(compiled) / max(elapsed, 1e-9)):.2f} subjects/s)"
        ),
        flush=True,
    )
    if bool(write_index):
        manifest = write_precompiled_index(
            output_dir=output_dir,
            splits_parquet=splits_parquet,
            records=index_rows,
            storage_format=PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
        )
        trajectory_index_count = 0
        if segmentation_config is not None and trajectory_split_config is not None:
            trajectory_records = _build_trajectory_index_records(
                output_dir=output_dir,
                records=index_rows,
                segmentation_config=segmentation_config,
                trajectory_split_config=trajectory_split_config,
            )
            write_precompiled_index(
                output_dir=output_dir,
                splits_parquet=splits_parquet,
                index_filename="trajectory_index.csv",
                manifest_filename="trajectory_manifest.json",
                records=trajectory_records,
                storage_format=PRECOMPILED_STORAGE_FORMAT_PACKED_V2,
            )
            trajectory_index_count = int(len(trajectory_records))
        manifest["compile_workers"] = int(workers)
        manifest["compile_chunksize"] = int(resolved_chunksize)
        manifest["compile_elapsed_seconds"] = float(elapsed)
        manifest["compiled_subjects"] = int(compiled)
        manifest["skipped_existing_subjects"] = int(skipped_existing_count)
        manifest["target_subjects_per_shard"] = int(target_subjects_per_shard)
        manifest["resolved_subjects_per_shard"] = int(resolved_subjects_per_shard)
        manifest["shard_count"] = len({str(rec["rel_path"]) for rec in index_rows})
        if trajectory_index_count > 0:
            manifest["trajectory_index_filename"] = "trajectory_index.csv"
            manifest["trajectory_count"] = int(trajectory_index_count)
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
        "compile_chunksize": int(resolved_chunksize),
        "compile_elapsed_seconds": float(elapsed),
    }


if __name__ == "__main__":
    raise SystemExit(
        "compile_dataset is a library entrypoint; construct encoders externally "
        "and call compile_dataset(db_path=..., encoders=..., output_dir=..., "
        "subject_ids=..., num_output_shards=...)."
    )
