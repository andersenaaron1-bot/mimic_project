#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as cf
import inspect
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meds_reader as mr

from scripts.audit_tokenization_flow import (
    _build_segmentation_config,
    _build_measurement_config,
    _build_static_artifacts,
    _build_struct_vocab,
    _load_subject_ids,
    _load_tokenization_contract,
    _resolve_residual_policy,
)
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders
from src.ehr_hier.transformer.vocab_runtime import (
    build_compact_runtime_vocab_and_remapper,
    build_runtime_vocab_and_remapper,
)


def _parse_csv_set(text: str | None) -> set[str]:
    if text is None:
        return set()
    out: set[str] = set()
    for part in str(text).split(","):
        p = part.strip()
        if p:
            out.add(p)
    return out


def _maybe_print_progress(*, idx: int, total: int, every: int, started_at: float) -> None:
    elapsed = max(0.0, time.time() - started_at)
    rate = float(idx) / elapsed if elapsed > 0 else 0.0
    remaining = (float(total - idx) / rate) if rate > 0 else float("inf")
    eta_text = f"{remaining / 60.0:.1f}m" if remaining == remaining and remaining != float("inf") else "unknown"
    print(
        f"[compact-vocab] {idx}/{total} subjects | elapsed={elapsed / 60.0:.1f}m | eta={eta_text}",
        flush=True,
    )


def _chunk_subjects(subject_ids: Sequence[int], chunk_size: int) -> List[List[int]]:
    sz = max(1, int(chunk_size))
    return [list(subject_ids[i : i + sz]) for i in range(0, len(subject_ids), sz)]


def _auto_worker_count(requested_workers: int) -> int:
    if int(requested_workers) > 0:
        return int(requested_workers)
    cpu_budget = 0
    raw_slurm = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if raw_slurm:
        try:
            cpu_budget = int(raw_slurm)
        except ValueError:
            cpu_budget = 0
    if cpu_budget <= 0:
        cpu_budget = int(os.cpu_count() or 1)
    if cpu_budget <= 2:
        return 1
    return max(1, min(8, cpu_budget // 2))


_WORKER_STATE: Dict[str, Any] = {}


def _build_worker_runtime(*, args: argparse.Namespace, block_ranges: Sequence[tuple[str, int, int]]) -> Dict[str, Any]:
    # Avoid loading massive full MedTok code2embeddings in every worker unless explicitly requested.
    worker_args = argparse.Namespace(**vars(args))
    if not bool(getattr(worker_args, "allow_full_medtok_in_workers", False)):
        worker_args.medtok_code2embeds = None

    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    artifacts = _build_static_artifacts(worker_args)
    residual_enabled, residual_buckets, residual_offsets = _resolve_residual_policy(
        worker_args,
        tokenization_contract=tokenization_contract,
    )

    struct_codes_union = set()
    if artifacts.structural_codebook is not None:
        struct_codes_union.update(artifacts.structural_codebook.code2label.keys())
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)

    meas_cfg = _build_measurement_config(worker_args, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError("Measurement artifacts missing; cannot build timelines for compact vocab.")

    encoders = build_base_encoders(
        meas_cfg,
        diag_vocab=artifacts.diag_vocab,
        proc_vocab=artifacts.proc_vocab,
        med_vocab=artifacts.med_vocab,
        struct_vocab=struct_vocab,
        med_attr_vocabs=artifacts.med_attr_vocabs,
        med_numeric_attrs=artifacts.med_numeric_attrs,
        medtok_parent_lookup=artifacts.medtok_parent_lookup,
        enable_residual_fallback=bool(residual_enabled),
        residual_fallback_buckets=int(residual_buckets),
        residual_fallback_offsets=dict(residual_offsets),
    )

    sig = inspect.signature(build_subject_timeline)
    timeline_kwargs = {
        "encoders": encoders,
        "structural_codebook": artifacts.structural_codebook,
        "window_hook_label": "window_boundary",
        "attach_med_numeric": True,
        "emit_process_struct_tokens": False,
        "drop_original_process_marker_tokens": False,
        "emit_global_demographic_tokens": True,
        "special_token_offset": 0,
    }
    timeline_kwargs = {k: v for k, v in timeline_kwargs.items() if k in sig.parameters}
    db = mr.SubjectDatabase(str(worker_args.meds_reader_db))
    return {
        "db": db,
        "encoders": encoders,
        "timeline_kwargs": timeline_kwargs,
        "block_ranges": [(str(n), int(lo), int(hi)) for n, lo, hi in block_ranges],
    }


def _init_worker(payload: Mapping[str, Any]) -> None:
    global _WORKER_STATE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch  # local import to avoid hard dependency during arg parsing

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    args = argparse.Namespace(**dict(payload.get("args", {})))
    block_ranges_raw = payload.get("block_ranges", [])
    block_ranges: List[tuple[str, int, int]] = [
        (str(t[0]), int(t[1]), int(t[2])) for t in block_ranges_raw
    ]
    _WORKER_STATE = _build_worker_runtime(args=args, block_ranges=block_ranges)


def _scan_subject_chunk(subject_chunk: Sequence[int]) -> Dict[str, Any]:
    db = _WORKER_STATE["db"]
    encoders = _WORKER_STATE["encoders"]
    timeline_kwargs = _WORKER_STATE["timeline_kwargs"]
    block_ranges = _WORKER_STATE["block_ranges"]

    observed_ids_by_block: Dict[str, set[int]] = {str(n): set() for n, _, _ in block_ranges}
    total_tokens = 0
    unmatched_tokens = 0
    subjects_built = 0

    for sid in subject_chunk:
        for enc in encoders.values():
            reset = getattr(enc, "reset_state", None)
            if callable(reset):
                reset()

        timeline = build_subject_timeline(
            db=db,
            subject_id=int(sid),
            **timeline_kwargs,
        )
        subjects_built += 1
        total_tokens += int(len(timeline))

        for tok in timeline:
            gid = int(tok.value_id)
            matched = False
            for name, lo, hi in block_ranges:
                if lo <= gid <= hi:
                    observed_ids_by_block[name].add(gid)
                    matched = True
                    break
            if not matched:
                unmatched_tokens += 1

    return {
        "subjects_built": int(subjects_built),
        "total_tokens": int(total_tokens),
        "unmatched_tokens": int(unmatched_tokens),
        "observed_ids_by_block": {k: sorted(v) for k, v in observed_ids_by_block.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build compact runtime vocab/remapper from actually observed token IDs in v1 timelines."
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=20000)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0, help="Process workers (0 = auto, 1 = serial).")
    ap.add_argument("--subject_chunk_size", type=int, default=128, help="Subjects per worker task.")
    ap.add_argument(
        "--allow_full_medtok_in_workers",
        action="store_true",
        help="If set, workers may load --medtok_code2embeds; default uses --medtok_vocab_dir only (lower RAM).",
    )

    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--vocab_manifest", default="artifacts/vocab_manifest.json")
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")

    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", required=True)
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--codes_parquet_parent_lookup", default=None)

    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--tokenizer_ckpt", required=True)

    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=39999)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)

    ap.add_argument(
        "--preserve_full_blocks",
        default="special,structural",
        help="Comma-separated block names to keep full (default: preserve special,structural).",
    )
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    random.seed(int(args.sample_seed))

    base_vocab_config, base_remapper = build_runtime_vocab_and_remapper(
        tokenization_contract=args.tokenization_yaml,
        vocab_manifest=args.vocab_manifest,
        structural_yaml=args.structural_yaml,
        medtok_vocab_dir=args.medtok_vocab_dir,
        code2id_pt=args.code2id_pt,
        tokenizer_ckpt=args.tokenizer_ckpt,
    )

    observed_ids_by_block: Dict[str, set[int]] = defaultdict(set)
    for block in base_remapper.blocks:
        observed_ids_by_block[str(block.name)] = set()

    # Mandatory special IDs for padding and marker grammar.
    marker_cfg = dict(base_vocab_config.get("window_markers", {}) or {})
    t_off = int(marker_cfg.get("type_token_offset", 10))
    n_types = int(marker_cfg.get("num_types", 0))
    end_id = int(marker_cfg.get("end_token_id", t_off + n_types))
    cont_id = int(marker_cfg.get("continue_token_id", end_id + 1))
    observed_ids_by_block["special"].update({0, 1, 2, 3, end_id, cont_id})
    observed_ids_by_block["special"].update(range(t_off, t_off + max(0, n_types)))

    subject_ids = _load_subject_ids(
        str(args.splits_parquet),
        str(args.split),
        int(args.max_subjects),
        sample_seed=int(args.sample_seed),
    )
    if not subject_ids:
        raise ValueError("No subjects loaded; cannot build compact runtime vocab.")

    block_ranges = [
        (str(b.name), int(b.global_offset), int(b.global_max))
        for b in base_remapper.blocks
    ]

    started_at = time.time()
    total = len(subject_ids)
    total_tokens = 0
    unmatched_tokens = 0
    progress_every = int(args.progress_every)
    next_progress = progress_every if progress_every > 0 else None
    workers = _auto_worker_count(int(args.workers))
    chunk_size = max(1, int(args.subject_chunk_size))
    subject_chunks = _chunk_subjects(subject_ids, chunk_size)
    print(f"[compact-vocab] using workers={workers} chunk_size={chunk_size}", flush=True)
    if workers > 1 and args.medtok_code2embeds and not bool(args.allow_full_medtok_in_workers):
        print(
            "[compact-vocab] workers>1: ignoring --medtok_code2embeds in workers and using --medtok_vocab_dir to avoid OOM",
            flush=True,
        )

    done_subjects = 0
    if workers <= 1:
        global _WORKER_STATE
        _WORKER_STATE = _build_worker_runtime(args=args, block_ranges=block_ranges)
        for chunk in subject_chunks:
            res = _scan_subject_chunk(chunk)
            done_subjects += int(res["subjects_built"])
            total_tokens += int(res["total_tokens"])
            unmatched_tokens += int(res["unmatched_tokens"])
            obs = dict(res.get("observed_ids_by_block", {}))
            for name, vals in obs.items():
                observed_ids_by_block[str(name)].update(int(v) for v in vals)
            while next_progress is not None and done_subjects >= next_progress and next_progress < total:
                _maybe_print_progress(
                    idx=next_progress,
                    total=total,
                    every=progress_every,
                    started_at=started_at,
                )
                next_progress += progress_every
    else:
        payload = {
            "args": vars(args),
            "block_ranges": block_ranges,
        }
        max_workers = max(1, workers)
        with cf.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(payload,),
        ) as ex:
            for res in ex.map(_scan_subject_chunk, subject_chunks, chunksize=1):
                done_subjects += int(res["subjects_built"])
                total_tokens += int(res["total_tokens"])
                unmatched_tokens += int(res["unmatched_tokens"])
                obs = dict(res.get("observed_ids_by_block", {}))
                for name, vals in obs.items():
                    observed_ids_by_block[str(name)].update(int(v) for v in vals)
                while next_progress is not None and done_subjects >= next_progress and next_progress < total:
                    _maybe_print_progress(
                        idx=next_progress,
                        total=total,
                        every=progress_every,
                        started_at=started_at,
                    )
                    next_progress += progress_every

    _maybe_print_progress(
        idx=done_subjects,
        total=total,
        every=progress_every,
        started_at=started_at,
    )

    preserve_full_blocks = _parse_csv_set(args.preserve_full_blocks)
    compact_vocab_config, compact_remapper = build_compact_runtime_vocab_and_remapper(
        base_vocab_config=base_vocab_config,
        base_remapper=base_remapper,
        observed_ids_by_block=observed_ids_by_block,
        preserve_full_blocks=preserve_full_blocks,
    )

    out_fp = Path(args.out_json)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab_config": compact_vocab_config,
        "id_remapper": compact_remapper.serialize(),
        "summary": {
            "subjects_scanned": int(len(subject_ids)),
            "tokens_scanned": int(total_tokens),
            "tokens_unmatched_to_base_blocks": int(unmatched_tokens),
            "preserve_full_blocks": sorted(preserve_full_blocks),
            "total_size_base": int(base_vocab_config.get("total_size", 0)),
            "total_size_compact": int(compact_vocab_config.get("total_size", 0)),
            "size_special_base": int(base_vocab_config.get("size_special", 0)),
            "size_special_compact": int(compact_vocab_config.get("size_special", 0)),
            "size_rvq_base": int(base_vocab_config.get("size_rvq", 0)),
            "size_rvq_compact": int(compact_vocab_config.get("size_rvq", 0)),
            "size_meas_base": int(base_vocab_config.get("size_meas_labels", 0)),
            "size_meas_compact": int(compact_vocab_config.get("size_meas_labels", 0)),
            "size_med_base": int(base_vocab_config.get("size_meds", 0)),
            "size_med_compact": int(compact_vocab_config.get("size_meds", 0)),
            "observed_ids_per_block": {
                str(k): int(len(v)) for k, v in sorted(observed_ids_by_block.items())
            },
        },
    }
    out_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            payload["summary"],
            indent=2,
        )
    )
    print(f"Wrote compact runtime vocab bundle to {out_fp}")


if __name__ == "__main__":
    main()
