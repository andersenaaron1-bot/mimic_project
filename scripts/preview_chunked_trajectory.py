#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import meds_reader as mr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import (  # noqa: E402
    SPECIAL_ID2NAME,
    _build_segmentation_config,
    _build_window_marker_config,
    _build_measurement_config,
    _load_tokenization_contract,
    _build_static_artifacts,
    _build_struct_vocab,
    _load_subject_ids,
    _make_structural_id2label,
    _offset,
    _resolve_residual_policy,
)
from src.ehr_hier.data.event_router import classify_code_to_category  # noqa: E402
from src.ehr_hier.data.structural_codes import structural_surface_code, structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline  # noqa: E402
from src.ehr_hier.data.token_types import EventToken, TokenCategory  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402
from src.ehr_hier.tokenizers.decode_tokens import decode_timeline_tokens, invert_code2id  # noqa: E402
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


def _window_type_id2name(codebook, collator: AETHierarchicalCollator) -> Dict[int, str]:
    out = {int(collator.window_markers.unk_type_id): "UNK"}
    if codebook is None:
        return out
    for name, idx in codebook.window_type2id().items():
        out[int(idx)] = str(name)
    return out


def _continue_token_id(collator: AETHierarchicalCollator) -> int:
    if collator.window_markers.continue_token_id is not None:
        return int(collator.window_markers.continue_token_id)
    end_token_id = (
        int(collator.window_markers.end_token_id)
        if collator.window_markers.end_token_id is not None
        else int(collator.window_markers.type_token_offset) + int(collator.window_markers.num_types)
    )
    return int(end_token_id) + 1


def _special_names(collator: AETHierarchicalCollator) -> Dict[int, str]:
    names = dict(SPECIAL_ID2NAME)
    type_offset = int(collator.window_markers.type_token_offset)
    num_types = int(collator.window_markers.num_types)
    for i in range(num_types):
        names[type_offset + i] = f"WIN_TYPE_OR_NEXT::{i}"
    end_token_id = (
        int(collator.window_markers.end_token_id)
        if collator.window_markers.end_token_id is not None
        else type_offset + num_types
    )
    names[end_token_id] = "WIN_END"
    names[_continue_token_id(collator)] = "WIN_CONTINUE"
    return names


def _gather_structural_codes(subj) -> set[str]:
    codes: set[str] = set()
    for ev in subj.events:
        code = getattr(ev, "code", None)
        if classify_code_to_category(code) == TokenCategory.STRUCTURAL and code is not None:
            surface = structural_surface_code(code, routed_category=TokenCategory.STRUCTURAL)
            if surface:
                codes.add(str(surface))
    return codes


def _decode_chunk_tokens(
    chunk_seq_ids: Sequence[int],
    chunk_seq_times: Sequence[float],
    chunk_seq_types: Sequence[int],
    *,
    collator: AETHierarchicalCollator,
    artifacts,
    struct_id2code: Mapping[int, str],
) -> List[Dict[str, Any]]:
    toks: List[EventToken] = []
    for vid, rel_t, cat in zip(chunk_seq_ids, chunk_seq_times, chunk_seq_types):
        toks.append(
            EventToken(
                value_id=int(vid),
                category_id=int(cat),
                t_from_start_hours=float(rel_t),
                dt_from_prev_hours=0.0,
                cat_attrs={},
                num_attrs={},
            )
        )

    return decode_timeline_tokens(
        toks,
        code_token_offset=_offset(artifacts.manifest, "measurement_code", 2_000_000),
        rvq_token_offset=_offset(artifacts.manifest, "measurement_value", 2_100_000),
        rvq_codebook_stride=artifacts.measurement_stride or 256,
        measurement_num_codebooks=artifacts.measurement_num_codebooks,
        measurement_code2name=invert_code2id(artifacts.code2id or {}),
        diagnosis_offset=artifacts.diag_vocab.offset,
        diagnosis_id2code=invert_code2id(artifacts.diag_vocab.code2id),
        procedure_offset=artifacts.proc_vocab.offset,
        procedure_id2code=invert_code2id(artifacts.proc_vocab.code2id),
        medication_offset=artifacts.med_vocab.offset,
        medication_id2code=invert_code2id(artifacts.med_vocab.code2id),
        observation_code_offset=_offset(artifacts.manifest, "observation_code", 2_300_000),
        observation_value_offset=_offset(artifacts.manifest, "observation_value", 2_320_000),
        structural_offset=_offset(artifacts.manifest, "structural", 2_200_000),
        structural_id2label=_make_structural_id2label(artifacts.structural_codebook),
        structural_id2code=struct_id2code,
        special_id2name=_special_names(collator),
    )


def _json_preview(
    *,
    subject_id: int,
    timeline: List[EventToken],
    semantic_windows,
    chunked_windows,
    collator: AETHierarchicalCollator,
    special_tokens: List[EventToken],
    artifacts,
    struct_id2code: Mapping[int, str],
    window_type_id2name_map: Mapping[int, str],
) -> Dict[str, Any]:
    decoded_timeline = decode_timeline_tokens(
        timeline,
        code_token_offset=_offset(artifacts.manifest, "measurement_code", 2_000_000),
        rvq_token_offset=_offset(artifacts.manifest, "measurement_value", 2_100_000),
        rvq_codebook_stride=artifacts.measurement_stride or 256,
        measurement_num_codebooks=artifacts.measurement_num_codebooks,
        measurement_code2name=invert_code2id(artifacts.code2id or {}),
        diagnosis_offset=artifacts.diag_vocab.offset,
        diagnosis_id2code=invert_code2id(artifacts.diag_vocab.code2id),
        procedure_offset=artifacts.proc_vocab.offset,
        procedure_id2code=invert_code2id(artifacts.proc_vocab.code2id),
        medication_offset=artifacts.med_vocab.offset,
        medication_id2code=invert_code2id(artifacts.med_vocab.code2id),
        observation_code_offset=_offset(artifacts.manifest, "observation_code", 2_300_000),
        observation_value_offset=_offset(artifacts.manifest, "observation_value", 2_320_000),
        structural_offset=_offset(artifacts.manifest, "structural", 2_200_000),
        structural_id2label=_make_structural_id2label(artifacts.structural_codebook),
        structural_id2code=struct_id2code,
        special_id2name=_special_names(collator),
    )

    windows_payload: List[Dict[str, Any]] = []
    for wi, window in enumerate(chunked_windows):
        next_type_id = chunked_windows[wi + 1].window_type_id if wi + 1 < len(chunked_windows) else None
        next_start = chunked_windows[wi + 1].start_time_hours if wi + 1 < len(chunked_windows) else None
        chunk_payloads: List[Dict[str, Any]] = []
        for chunk in window.chunks:
            ids, times, vals, valmask, types, chunk_start_abs, chunk_start_offset = collator._process_chunk(
                chunk,
                special_tokens,
                w_type_id=int(window.window_type_id),
                w_start_abs=float(window.start_time_hours),
                next_type_id=next_type_id,
                next_start_abs=next_start,
            )
            seq_len = len(ids)
            chunk_payloads.append(
                {
                    "chunk_index": int(chunk.chunk_index),
                    "is_first_chunk": bool(chunk.is_first_chunk),
                    "is_last_chunk": bool(chunk.is_last_chunk),
                    "chunk_start_abs_hours": float(chunk_start_abs),
                    "chunk_start_offset_hours": float(chunk_start_offset),
                    "raw_token_count": int(len(chunk.tokens)),
                    "processed_seq_len": int(seq_len),
                    "raw_token_ids": [int(tok.value_id) for tok in chunk.tokens],
                    "processed_ids": [int(x) for x in ids],
                    "processed_times": [float(x) for x in times],
                    "processed_type_ids": [int(x) for x in types],
                    "numeric_values": [float(x) for x in vals],
                    "numeric_mask": [int(x) for x in valmask],
                    "decoded_sequence": _decode_chunk_tokens(
                        ids,
                        times,
                        types,
                        collator=collator,
                        artifacts=artifacts,
                        struct_id2code=struct_id2code,
                    ),
                }
            )

        windows_payload.append(
            {
                "window_index": int(wi),
                "semantic_window_type_id": int(window.window_type_id),
                "semantic_window_type": window_type_id2name_map.get(int(window.window_type_id), f"TYPE::{int(window.window_type_id)}"),
                "window_start_abs_hours": float(window.start_time_hours),
                "opening_action": window.opening_action,
                "closing_action": window.closing_action,
                "semantic_raw_token_count": int(len(window.tokens)),
                "chunk_count": int(len(window.chunks)),
                "chunks": chunk_payloads,
            }
        )

    return {
        "subject_id": int(subject_id),
        "special_token_count": int(len(special_tokens)),
        "semantic_window_count": int(len(semantic_windows)),
        "chunked_window_count": int(len(chunked_windows)),
        "decoded_timeline_preview": decoded_timeline,
        "windows": windows_payload,
    }


def _format_decoded_item(item: Mapping[str, Any]) -> str:
    if item.get("kind") == "measurement_bundle":
        name = item.get("var_name") or f"VAR::{item.get('var_id')}"
        rvq = item.get("rvq_indices", [])
        return f"{name} rvq={rvq}"
    label = item.get("label")
    if label is not None:
        return str(label)
    category = item.get("category", "UNK")
    return f"{category}::{item.get('value_id')}"


def _print_text_preview(payload: Mapping[str, Any], *, max_items_per_chunk: int) -> None:
    print(f"subject_id: {payload['subject_id']}")
    print(f"semantic_windows: {payload['semantic_window_count']} | chunked_windows: {payload['chunked_window_count']}")
    for window in payload["windows"]:
        print(
            f"\n[Window {window['window_index']}] "
            f"type={window['semantic_window_type']}({window['semantic_window_type_id']}) "
            f"start={window['window_start_abs_hours']:.2f}h "
            f"raw_tokens={window['semantic_raw_token_count']} "
            f"chunks={window['chunk_count']} "
            f"open={window['opening_action']} close={window['closing_action']}"
        )
        for chunk in window["chunks"]:
            print(
                f"  [Chunk {chunk['chunk_index']}] "
                f"offset={chunk['chunk_start_offset_hours']:.2f}h "
                f"abs={chunk['chunk_start_abs_hours']:.2f}h "
                f"raw={chunk['raw_token_count']} "
                f"seq={chunk['processed_seq_len']} "
                f"first={chunk['is_first_chunk']} last={chunk['is_last_chunk']}"
            )
            for i, item in enumerate(chunk["decoded_sequence"][:max_items_per_chunk]):
                time_val = chunk["processed_times"][i] if i < len(chunk["processed_times"]) else None
                print(f"    - t={time_val:.2f} | {_format_decoded_item(item)}")
            if len(chunk["decoded_sequence"]) > max_items_per_chunk:
                print(f"    ... {len(chunk['decoded_sequence']) - max_items_per_chunk} more items")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview one subject trajectory as semantic windows and local chunks.")
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subject_id", type=int, default=None)
    ap.add_argument("--subject_index", type=int, default=0, help="Index within the split if --subject_id is omitted.")
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
    ap.add_argument("--max_items_per_chunk", type=int, default=30)
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

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
    db = mr.SubjectDatabase(args.meds_reader_db)

    subject_id = args.subject_id
    if subject_id is None:
        split_subjects = _load_subject_ids(args.splits_parquet, args.split, max_subjects=max(args.subject_index + 1, 1))
        if not split_subjects or args.subject_index >= len(split_subjects):
            raise ValueError(f"subject_index={args.subject_index} is out of range for split={args.split}")
        subject_id = int(split_subjects[args.subject_index])

    subj = db[int(subject_id)]
    struct_codes_union = _gather_structural_codes(subj)
    if artifacts.structural_codebook is not None:
        struct_codes_union.update(structural_surface_vocab_codes(artifacts.structural_codebook))
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)
    struct_id2code = invert_code2id(struct_vocab.code2id)

    meas_cfg = _build_measurement_config(args, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError("Measurement artifacts are required.")

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

    collator = AETHierarchicalCollator(
        max_windows=int(args.max_windows),
        max_chunks_per_window=int(args.max_chunks_per_window),
        max_len_per_window=int(args.max_len_per_window),
        pad_id=0,
        window_markers=window_markers_cfg,
        segmentation=segmentation_cfg,
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
        window_type_id2name_map=_window_type_id2name(artifacts.structural_codebook, collator),
    )

    if args.format == "json":
        text = json.dumps(payload, indent=2)
        print(text)
    else:
        _print_text_preview(payload, max_items_per_chunk=int(args.max_items_per_chunk))

    if args.output_json:
        out_fp = Path(args.output_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote preview JSON to {out_fp}")


if __name__ == "__main__":
    main()
