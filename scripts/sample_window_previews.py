#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping

import meds_reader as mr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import (  # noqa: E402
    _build_measurement_config,
    _build_segmentation_config,
    _build_static_artifacts,
    _build_struct_vocab,
    _build_window_marker_config,
    _load_subject_ids,
    _load_tokenization_contract,
    _resolve_residual_policy,
)
from scripts.preview_chunked_trajectory import (  # noqa: E402
    _gather_structural_codes,
    _json_preview,
    _window_type_id2name,
)
from src.ehr_hier.data.structural_codes import structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402
from src.ehr_hier.tokenizers.decode_tokens import invert_code2id  # noqa: E402
from src.ehr_hier.transformer.collator import AETHierarchicalCollator  # noqa: E402


def _parse_subject_ids(arg: str | None) -> List[int]:
    if arg is None or not str(arg).strip():
        return []
    out: List[int] = []
    for part in str(arg).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _append_limited(store: List[int], value: int, *, limit: int) -> None:
    if value in store:
        return
    if len(store) < limit:
        store.append(int(value))


def _counter_dict(counter: Counter[Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))}


def _defaultdict_lists_to_dict(store: DefaultDict[str, List[int]]) -> Dict[str, List[int]]:
    return {str(k): [int(x) for x in v] for k, v in sorted(store.items(), key=lambda kv: str(kv[0]))}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Sample a representative set of subject trajectories and dump the built semantic windows "
            "plus per-chunk token traces as JSONL."
        )
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subject_ids", default=None, help="Optional comma-separated explicit subject ids.")
    ap.add_argument("--max_subjects", type=int, default=2000)
    ap.add_argument("--sample_seed", type=int, default=13)
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--medtok_crosswalk_json", default=None)
    ap.add_argument("--allow_smoke_medtok", action="store_true")
    ap.add_argument("--sparse_vocab_json", default=None)
    ap.add_argument(
        "--tokenization_yaml",
        default="configs/data/tokenization_v1.yaml",
        help="Optional token/window contract. Missing file falls back to built-in defaults.",
    )
    ap.add_argument(
        "--codes_parquet_parent_lookup",
        default=None,
        help="Optional metadata/codes.parquet for code->parent_codes lookup used by MedTok encoders.",
    )
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--tokenizer_ckpt", required=True)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--max_chunks_per_window", type=int, default=8)
    ap.add_argument("--max_len_per_window", type=int, default=128)
    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=40_000)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)
    ap.add_argument(
        "--include_decoded_timeline_preview",
        action="store_true",
        help="Include the full decoded timeline preview in each JSONL record. Disabled by default to keep output smaller.",
    )
    ap.add_argument("--examples_per_window_type", type=int, default=12)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--output_summary_json", default=None)
    args = ap.parse_args()

    subject_ids = _parse_subject_ids(args.subject_ids)
    if not subject_ids:
        subject_ids = _load_subject_ids(
            args.splits_parquet,
            args.split,
            max_subjects=int(args.max_subjects),
            sample_seed=int(args.sample_seed),
        )
    if not subject_ids:
        raise ValueError("No subject ids selected.")

    artifacts = _build_static_artifacts(args)
    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    window_markers_cfg = _build_window_marker_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
    )
    segmentation_cfg = _build_segmentation_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
        unk_type_id=int(window_markers_cfg.unk_type_id),
    )
    residual_enabled, residual_buckets, residual_offsets = _resolve_residual_policy(
        args,
        tokenization_contract=tokenization_contract,
    )
    meas_cfg = _build_measurement_config(args, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError("Measurement artifacts are required.")

    collator = AETHierarchicalCollator(
        max_windows=int(args.max_windows),
        max_chunks_per_window=int(args.max_chunks_per_window),
        max_len_per_window=int(args.max_len_per_window),
        pad_id=0,
        window_markers=window_markers_cfg,
        segmentation=segmentation_cfg,
    )
    window_type_id2name_map = _window_type_id2name(artifacts.structural_codebook, collator)
    db = mr.SubjectDatabase(args.meds_reader_db)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "split": str(args.split),
        "subject_ids_source": "explicit" if args.subject_ids else "split_sample",
        "requested_subject_count": int(len(subject_ids)),
        "sample_seed": None if args.subject_ids else int(args.sample_seed),
        "subjects_scanned": 0,
        "subjects_written": 0,
        "subjects_failed": 0,
        "failed_subject_samples": [],
        "semantic_window_total": 0,
        "chunked_window_total": 0,
        "special_token_total": 0,
        "window_type_counts": {},
        "opening_action_counts": {},
        "closing_action_counts": {},
        "chunk_count_distribution": {},
        "window_type_example_subject_ids": {},
        "special_case_subject_ids": {
            "history_prefix": [],
            "post_discharge": [],
            "unk": [],
        },
        "started_at_unix": float(time.time()),
        "elapsed_sec": 0.0,
    }
    failed_subject_samples: List[Dict[str, Any]] = []
    window_type_counts: Counter[str] = Counter()
    opening_action_counts: Counter[str] = Counter()
    closing_action_counts: Counter[str] = Counter()
    chunk_count_distribution: Counter[int] = Counter()
    window_type_example_subject_ids: DefaultDict[str, List[int]] = defaultdict(list)
    history_prefix_subject_ids: List[int] = []
    post_discharge_subject_ids: List[int] = []
    unk_subject_ids: List[int] = []

    started_at = time.time()
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for idx, subject_id in enumerate(subject_ids, start=1):
            summary["subjects_scanned"] = int(idx)
            try:
                subj = db[int(subject_id)]
                struct_codes_union = _gather_structural_codes(subj)
                if artifacts.structural_codebook is not None:
                    struct_codes_union.update(structural_surface_vocab_codes(artifacts.structural_codebook))
                struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)
                struct_id2code = invert_code2id(struct_vocab.code2id)
                encoders = build_base_encoders(
                    meas_cfg,
                    diag_vocab=artifacts.diag_vocab,
                    proc_vocab=artifacts.proc_vocab,
                    med_vocab=artifacts.med_vocab,
                    struct_vocab=struct_vocab,
                    med_attr_vocabs=artifacts.med_attr_vocabs,
                    med_numeric_attrs=artifacts.med_numeric_attrs,
                    medtok_parent_lookup=artifacts.medtok_parent_lookup,
                    medtok_crosswalks=artifacts.medtok_crosswalks,
                    residual_fallback_vocabs=artifacts.residual_fallback_vocabs,
                    enable_residual_fallback=bool(residual_enabled),
                    residual_fallback_buckets=int(residual_buckets),
                    residual_fallback_offsets=dict(residual_offsets),
                )
                timeline = build_subject_timeline(
                    db=db,
                    subject_id=int(subject_id),
                    encoders=encoders,
                    structural_codebook=artifacts.structural_codebook,
                    qual_obs_code_vocab=artifacts.obs_code_vocab,
                    qual_obs_value_vocab=artifacts.obs_value_vocab,
                    qual_obs_tail_policy=artifacts.obs_tail_policy,
                )
                special_tokens, events = collator._split_special(timeline)
                semantic_windows = collator._segment_windows(events)[: int(args.max_windows)]
                chunked_windows = collator._chunk_windows(semantic_windows, special_tokens=special_tokens)
                payload = _json_preview(
                    subject_id=int(subject_id),
                    timeline=timeline,
                    semantic_windows=semantic_windows,
                    chunked_windows=chunked_windows,
                    collator=collator,
                    special_tokens=special_tokens,
                    artifacts=artifacts,
                    struct_id2code=struct_id2code,
                    window_type_id2name_map=window_type_id2name_map,
                )
                if not args.include_decoded_timeline_preview:
                    payload.pop("decoded_timeline_preview", None)

                f_out.write(json.dumps(payload, ensure_ascii=True) + "\n")

                summary["subjects_written"] = int(summary["subjects_written"]) + 1
                summary["semantic_window_total"] = int(summary["semantic_window_total"]) + int(payload["semantic_window_count"])
                summary["chunked_window_total"] = int(summary["chunked_window_total"]) + int(payload["chunked_window_count"])
                summary["special_token_total"] = int(summary["special_token_total"]) + int(payload["special_token_count"])

                for window in payload["windows"]:
                    window_type = str(window["semantic_window_type"])
                    opening = str(window.get("opening_action"))
                    closing = str(window.get("closing_action"))
                    window_type_counts[window_type] += 1
                    opening_action_counts[opening] += 1
                    closing_action_counts[closing] += 1
                    chunk_count_distribution[int(window["chunk_count"])] += 1
                    _append_limited(
                        window_type_example_subject_ids[window_type],
                        int(subject_id),
                        limit=int(args.examples_per_window_type),
                    )
                    if window_type == "HISTORY_PREFIX":
                        _append_limited(history_prefix_subject_ids, int(subject_id), limit=int(args.examples_per_window_type))
                    elif window_type == "POST_DISCHARGE":
                        _append_limited(post_discharge_subject_ids, int(subject_id), limit=int(args.examples_per_window_type))
                    elif window_type == "UNK":
                        _append_limited(unk_subject_ids, int(subject_id), limit=int(args.examples_per_window_type))
            except Exception as exc:
                summary["subjects_failed"] = int(summary["subjects_failed"]) + 1
                if len(failed_subject_samples) < 24:
                    failed_subject_samples.append(
                        {
                            "subject_id": int(subject_id),
                            "error": repr(exc),
                        }
                    )

            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                rate = float(idx) / elapsed
                print(
                    f"[window-preview] {idx}/{len(subject_ids)} subjects | "
                    f"elapsed={elapsed/60.0:.1f}m | rate={rate:.2f} subj/s"
                )

    summary["failed_subject_samples"] = failed_subject_samples
    summary["window_type_counts"] = _counter_dict(window_type_counts)
    summary["opening_action_counts"] = _counter_dict(opening_action_counts)
    summary["closing_action_counts"] = _counter_dict(closing_action_counts)
    summary["chunk_count_distribution"] = _counter_dict(chunk_count_distribution)
    summary["window_type_example_subject_ids"] = _defaultdict_lists_to_dict(window_type_example_subject_ids)
    summary["special_case_subject_ids"] = {
        "history_prefix": [int(x) for x in history_prefix_subject_ids],
        "post_discharge": [int(x) for x in post_discharge_subject_ids],
        "unk": [int(x) for x in unk_subject_ids],
    }
    summary["elapsed_sec"] = float(time.time() - started_at)

    if args.output_summary_json:
        out_summary = Path(args.output_summary_json)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote preview JSONL to {out_jsonl}")
    if args.output_summary_json:
        print(f"Wrote summary JSON to {args.output_summary_json}")


if __name__ == "__main__":
    main()
