#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import (  # noqa: E402
    _build_segmentation_config,
    _build_static_artifacts,
    _build_trajectory_split_config,
    _build_window_marker_config,
    _load_tokenization_contract,
)
from scripts.train_transformer_v1 import (  # noqa: E402
    TimelineCollateAdapter,
    TrainModelConfig,
    _build_runtime_bundle,
    _move_batch_to_device,
    _resolve_carry_inputs,
    _resolve_device,
    _update_carry_cache,
)
from src.ehr_hier.data.dataset import PrecompiledMEDSDataset  # noqa: E402
from src.ehr_hier.transformer.collator import AETHierarchicalCollator  # noqa: E402
from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer  # noqa: E402
from src.ehr_hier.transformer.precedent_memory import (  # noqa: E402
    build_anchor_mask_flags,
    build_future_summary,
    build_window_support_flags,
    materialize_precedent_index_store,
    save_precedent_index_store,
    select_future_window_indices,
)
from src.ehr_hier.transformer.world_model_contract import (  # noqa: E402
    FutureSummary,
    FutureSnippetRef,
    PrecedentIndexItem,
    compose_precedent_key_state,
)


def _build_collator(args: argparse.Namespace) -> tuple[AETHierarchicalCollator, dict[str, Any]]:
    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    vocab_config, remapper = _build_runtime_bundle(args)
    artifact_args = argparse.Namespace(
        structural_yaml=args.structural_yaml,
        sparse_vocab_json=args.sparse_vocab_json,
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=args.medtok_vocab_dir,
        medtok_attr_dir=args.medtok_attr_dir,
        medtok_crosswalk_json=args.medtok_crosswalk_json,
        code2id_pt=args.code2id_pt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        codes_parquet_parent_lookup=args.codes_parquet_parent_lookup,
    )
    artifacts = _build_static_artifacts(artifact_args)
    window_markers_cfg = _build_window_marker_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
    )
    segmentation_cfg = _build_segmentation_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
        unk_type_id=int(window_markers_cfg.unk_type_id),
    )
    collator = AETHierarchicalCollator(
        max_windows=int(args.max_windows),
        max_chunks_per_window=int(args.max_chunks_per_window),
        max_len_per_window=int(args.max_len_per_window),
        window_markers=window_markers_cfg,
        segmentation=segmentation_cfg,
        id_remapper=remapper,
    )
    return collator, vocab_config


def _ordered_indices(ds: PrecompiledMEDSDataset) -> list[int]:
    def _key(i: int) -> tuple[int, int, str, int]:
        traj_ord = ds.trajectory_orders[i] if i < len(ds.trajectory_orders) else None
        subject_pos = ds.subject_positions[i] if i < len(ds.subject_positions) else None
        return (
            int(ds.subject_ids[i]) if i < len(ds.subject_ids) else -1,
            int(traj_ord) if traj_ord is not None else 0,
            str(ds.file_paths[i]),
            int(subject_pos) if subject_pos is not None else -1,
        )

    return sorted(range(len(ds)), key=_key)


def _support_flags_for_window(tensor_batch: dict[str, torch.Tensor], window_idx: int) -> torch.Tensor:
    return build_window_support_flags(
        event_type_ids=tensor_batch["event_type_ids"][0, window_idx],
        event_attention_mask=tensor_batch["event_attention_mask"][0, window_idx],
        event_memory_chronic_flags=tensor_batch.get("event_memory_chronic_flags", None)[0, window_idx]
        if tensor_batch.get("event_memory_chronic_flags", None) is not None
        else None,
        event_numeric_values=tensor_batch.get("event_numeric_values", None)[0, window_idx]
        if tensor_batch.get("event_numeric_values", None) is not None
        else None,
        event_numeric_mask=tensor_batch.get("event_numeric_mask", None)[0, window_idx]
        if tensor_batch.get("event_numeric_mask", None) is not None
        else None,
    )


def _cpu_future_summary(summary: FutureSummary) -> FutureSummary:
    return FutureSummary(
        next_window_type_id=summary.next_window_type_id.detach().cpu(),
        next_window_gap_h=summary.next_window_gap_h.detach().cpu(),
        next_window_duration_h=summary.next_window_duration_h.detach().cpu(),
        event_family_hist=summary.event_family_hist.detach().cpu(),
        payload_hist=summary.payload_hist.detach().cpu(),
        support_flags=summary.support_flags.detach().cpu(),
        transition_flags=summary.transition_flags.detach().cpu(),
        event_count=summary.event_count.detach().cpu(),
        measurement_count=summary.measurement_count.detach().cpu(),
        extreme_measurement_count=summary.extreme_measurement_count.detach().cpu(),
        numeric_severity=summary.numeric_severity.detach().cpu(),
        terminal_window_type_id=summary.terminal_window_type_id.detach().cpu(),
        future_window_count=summary.future_window_count.detach().cpu(),
    )


def _emit_items_from_batch(
    *,
    tensor_batch: dict[str, torch.Tensor],
    sample_meta: dict[str, Any],
    aux_state: dict[str, Any],
    rel_path_to_id: dict[str, int],
    next_item_id: int,
) -> tuple[list[PrecedentIndexItem], int]:
    items: list[PrecedentIndexItem] = []
    window_mask = tensor_batch["window_mask"][0].to(dtype=torch.bool)
    n_real = int(window_mask.to(dtype=torch.long).sum().item())
    if n_real <= 1:
        return items, next_item_id

    window_type_ids = tensor_batch["window_type_ids"][0]
    window_start_times = tensor_batch["window_start_times"][0]
    semantic_duration_hours = tensor_batch["semantic_duration_hours"][0]
    packet = aux_state["window_state_packet"]
    global_states = aux_state["window_global_states"][0]
    memory_digests = aux_state.get("patient_memory_state_digests_by_bank", None) or {}
    persistent_digest = memory_digests.get("persistent", None)
    if persistent_digest is None:
        persistent_digest = torch.zeros_like(global_states)
    else:
        persistent_digest = persistent_digest[0]

    rel_path = str(sample_meta.get("rel_path", ""))
    rel_path_id = rel_path_to_id.setdefault(rel_path, len(rel_path_to_id))
    subject_id = int(sample_meta.get("subject_id", -1))
    subject_idx = int(sample_meta.get("subject_idx", -1))
    trajectory_ord = int(sample_meta.get("trajectory_ord", 0))

    for w in range(n_real - 1):
        current_end_h = float(window_start_times[w].item() + semantic_duration_hours[w].item())
        future_h1_idx, future_h1_truncated = select_future_window_indices(
            window_mask=window_mask,
            window_start_times=window_start_times,
            anchor_end_h=current_end_h,
            start_idx=w + 1,
            max_windows=1,
            max_hours=None,
        )
        future_h2_idx, future_h2_truncated = select_future_window_indices(
            window_mask=window_mask,
            window_start_times=window_start_times,
            anchor_end_h=current_end_h,
            start_idx=w + 1,
            max_windows=2,
            max_hours=24.0,
        )
        future_h3_idx, future_h3_truncated = select_future_window_indices(
            window_mask=window_mask,
            window_start_times=window_start_times,
            anchor_end_h=current_end_h,
            start_idx=w + 1,
            max_windows=4,
            max_hours=24.0 * 7.0,
        )
        if not future_h1_idx:
            continue

        support_flags = _support_flags_for_window(tensor_batch, w)
        anchor_mask_flags = build_anchor_mask_flags(
            boundary_ord=w,
            gap_prev_h=float(packet.gap_prev_hours[0, w].item()) if packet.gap_prev_hours is not None else 0.0,
            future_truncated=bool(future_h3_truncated),
            future_missing=False,
        )
        key_packet = packet.query_token[0, w].detach().cpu().to(dtype=torch.float32)
        key_memory = persistent_digest[w].detach().cpu().to(dtype=torch.float32)
        key_state = compose_precedent_key_state(
            packet_query=packet.query_token[0, w : w + 1].to(dtype=torch.float32),
            latent_state=global_states[w : w + 1].to(dtype=torch.float32),
            memory_digest=persistent_digest[w : w + 1].to(dtype=torch.float32),
            window_type_ids=packet.window_type_ids[0, w : w + 1] if packet.window_type_ids is not None else None,
            gap_prev_hours=packet.gap_prev_hours[0, w : w + 1] if packet.gap_prev_hours is not None else None,
            duration_hours=packet.duration_hours[0, w : w + 1] if packet.duration_hours is not None else None,
        ).squeeze(0).detach().cpu()

        future_h1 = build_future_summary(
            current_window_type_id=int(window_type_ids[w].item()),
            current_window_end_h=current_end_h,
            future_window_indices=future_h1_idx,
            future_truncated=bool(future_h1_truncated),
            window_type_ids=window_type_ids,
            window_start_times=window_start_times,
            semantic_duration_hours=semantic_duration_hours,
            event_type_ids=tensor_batch["event_type_ids"][0],
            event_payload_ids=tensor_batch["event_payload_ids"][0],
            event_attention_mask=tensor_batch["event_attention_mask"][0],
            event_memory_chronic_flags=tensor_batch.get("event_memory_chronic_flags", None)[0]
            if tensor_batch.get("event_memory_chronic_flags", None) is not None
            else None,
            event_numeric_values=tensor_batch.get("event_numeric_values", None)[0]
            if tensor_batch.get("event_numeric_values", None) is not None
            else None,
            event_numeric_mask=tensor_batch.get("event_numeric_mask", None)[0]
            if tensor_batch.get("event_numeric_mask", None) is not None
            else None,
        )
        future_h2 = build_future_summary(
            current_window_type_id=int(window_type_ids[w].item()),
            current_window_end_h=current_end_h,
            future_window_indices=future_h2_idx,
            future_truncated=bool(future_h2_truncated),
            window_type_ids=window_type_ids,
            window_start_times=window_start_times,
            semantic_duration_hours=semantic_duration_hours,
            event_type_ids=tensor_batch["event_type_ids"][0],
            event_payload_ids=tensor_batch["event_payload_ids"][0],
            event_attention_mask=tensor_batch["event_attention_mask"][0],
            event_memory_chronic_flags=tensor_batch.get("event_memory_chronic_flags", None)[0]
            if tensor_batch.get("event_memory_chronic_flags", None) is not None
            else None,
            event_numeric_values=tensor_batch.get("event_numeric_values", None)[0]
            if tensor_batch.get("event_numeric_values", None) is not None
            else None,
            event_numeric_mask=tensor_batch.get("event_numeric_mask", None)[0]
            if tensor_batch.get("event_numeric_mask", None) is not None
            else None,
        )
        future_h3 = build_future_summary(
            current_window_type_id=int(window_type_ids[w].item()),
            current_window_end_h=current_end_h,
            future_window_indices=future_h3_idx,
            future_truncated=bool(future_h3_truncated),
            window_type_ids=window_type_ids,
            window_start_times=window_start_times,
            semantic_duration_hours=semantic_duration_hours,
            event_type_ids=tensor_batch["event_type_ids"][0],
            event_payload_ids=tensor_batch["event_payload_ids"][0],
            event_attention_mask=tensor_batch["event_attention_mask"][0],
            event_memory_chronic_flags=tensor_batch.get("event_memory_chronic_flags", None)[0]
            if tensor_batch.get("event_memory_chronic_flags", None) is not None
            else None,
            event_numeric_values=tensor_batch.get("event_numeric_values", None)[0]
            if tensor_batch.get("event_numeric_values", None) is not None
            else None,
            event_numeric_mask=tensor_batch.get("event_numeric_mask", None)[0]
            if tensor_batch.get("event_numeric_mask", None) is not None
            else None,
        )
        future_h1 = _cpu_future_summary(future_h1)
        future_h2 = _cpu_future_summary(future_h2)
        future_h3 = _cpu_future_summary(future_h3)
        next_prompt_idx = int(future_h1_idx[0])
        future_prefix_prompt = packet.slot_tokens[0, next_prompt_idx].detach().cpu().to(dtype=torch.float32)

        items.append(
            PrecedentIndexItem(
                item_id=torch.tensor(next_item_id, dtype=torch.long),
                subject_id=torch.tensor(subject_id, dtype=torch.long),
                trajectory_ord=torch.tensor(trajectory_ord, dtype=torch.long),
                boundary_ord=torch.tensor(w, dtype=torch.long),
                anchor_window_ord=torch.tensor(w, dtype=torch.long),
                current_window_type_id=torch.tensor(int(window_type_ids[w].item()), dtype=torch.long),
                current_window_start_h=torch.tensor(float(window_start_times[w].item()), dtype=torch.float32),
                current_window_duration_h=torch.tensor(float(semantic_duration_hours[w].item()), dtype=torch.float32),
                gap_prev_h=torch.tensor(
                    float(packet.gap_prev_hours[0, w].item()) if packet.gap_prev_hours is not None else 0.0,
                    dtype=torch.float32,
                ),
                support_flags=support_flags.detach().cpu().to(dtype=torch.float32),
                anchor_mask_flags=anchor_mask_flags.detach().cpu().to(dtype=torch.float32),
                key_state=key_state.to(dtype=torch.float32),
                key_packet=key_packet,
                key_memory=key_memory,
                future_summary_h1=future_h1,
                future_summary_h2=future_h2,
                future_summary_h3=future_h3,
                future_prefix_prompt=future_prefix_prompt,
                future_snippet_ref=FutureSnippetRef(
                    rel_path_id=torch.tensor(rel_path_id, dtype=torch.long),
                    subject_idx=torch.tensor(subject_idx, dtype=torch.long),
                    trajectory_ord=torch.tensor(trajectory_ord, dtype=torch.long),
                    start_boundary_ord=torch.tensor(w + 1, dtype=torch.long),
                    stop_boundary_ord=torch.tensor(
                        (future_h3_idx[-1] + 1) if future_h3_idx else (w + 1),
                        dtype=torch.long,
                    ),
                ),
            )
        )
        next_item_id += 1
    return items, next_item_id


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompiled_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--splits_parquet", default=None)
    ap.add_argument("--trajectory_mode", default="admission_chain")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--resume_from", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runtime_vocab_json", default=None)
    ap.add_argument("--sparse_vocab_json", default=None)
    ap.add_argument("--tokenization_yaml", required=True)
    ap.add_argument("--vocab_manifest", default=None)
    ap.add_argument("--structural_yaml", required=True)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--medtok_attr_dir", default=None)
    ap.add_argument("--medtok_crosswalk_json", default=None)
    ap.add_argument("--codes_parquet_parent_lookup", default=None)
    ap.add_argument("--code2id_pt", default=None)
    ap.add_argument("--tokenizer_ckpt", default=None)
    ap.add_argument("--allow_smoke_medtok", action="store_true")
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--max_chunks_per_window", type=int, default=8)
    ap.add_argument("--max_len_per_window", type=int, default=128)
    ap.add_argument("--max_subjects", type=int, default=0)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    device = _resolve_device(str(args.device))
    collator, vocab_config = _build_collator(args)
    trajectory_split_cfg = _build_trajectory_split_config(
        mode=str(args.trajectory_mode),
        post_discharge_cutoff_days=30.0,
    )
    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    artifact_args = argparse.Namespace(
        structural_yaml=args.structural_yaml,
        sparse_vocab_json=args.sparse_vocab_json,
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=args.medtok_vocab_dir,
        medtok_attr_dir=args.medtok_attr_dir,
        medtok_crosswalk_json=args.medtok_crosswalk_json,
        code2id_pt=args.code2id_pt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        codes_parquet_parent_lookup=args.codes_parquet_parent_lookup,
    )
    artifacts = _build_static_artifacts(artifact_args)
    segmentation_cfg = _build_segmentation_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
        unk_type_id=int(
            _build_window_marker_config(
                tokenization_contract=tokenization_contract,
                structural_codebook=artifacts.structural_codebook,
            ).unk_type_id
        ),
    )

    ds = PrecompiledMEDSDataset(
        args.precompiled_root,
        split=str(args.split),
        splits_parquet=str(args.splits_parquet) if args.splits_parquet else None,
        index_filename=("trajectory_index.csv" if str(args.trajectory_mode) == "admission_chain" else "index.csv"),
        segmentation_config=segmentation_cfg,
        trajectory_split_config=trajectory_split_cfg,
        return_metadata=True,
    )

    ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
    model_cfg = TrainModelConfig(**dict(ckpt["model_config"]))
    model = AdaptiveEpisodicTransformer(model_cfg, vocab_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ordered = _ordered_indices(ds)
    if int(args.max_subjects) > 0:
        ordered = ordered[: int(args.max_subjects)]

    rel_path_to_id: dict[str, int] = {}
    items: list[PrecedentIndexItem] = []
    carry_cache: dict[int, dict[str, Any]] = {}
    next_item_id = 0
    collate = TimelineCollateAdapter(collator)

    with torch.no_grad():
        for ds_idx in ordered:
            sample = ds[ds_idx]
            if not isinstance(sample, dict):
                continue
            batch = collate([sample])
            tensor_batch = _move_batch_to_device(batch, device)
            prev_global_state, prev_memory_state = _resolve_carry_inputs(
                tensor_batch=tensor_batch,
                model=model,
                carry_cache=carry_cache,
            )
            _, aux_state = model(
                input_ids=tensor_batch["input_ids"],
                time_ids=tensor_batch["time_ids"],
                numeric_values=tensor_batch["numeric_values"],
                token_type_ids=tensor_batch["token_type_ids"],
                attention_mask=tensor_batch["attention_mask"],
                numeric_mask=tensor_batch.get("numeric_mask", None),
                prev_global_state=prev_global_state,
                prev_memory_state=prev_memory_state,
                window_start_times=tensor_batch.get("window_start_times", None),
                window_mask=tensor_batch.get("window_mask", None),
                window_type_ids=tensor_batch.get("window_type_ids", None),
                chunk_mask=tensor_batch.get("chunk_mask", None),
                chunk_start_offsets=tensor_batch.get("chunk_start_offsets", None),
                chunk_is_last=tensor_batch.get("chunk_is_last", None),
                semantic_token_counts=tensor_batch.get("semantic_token_counts", None),
                semantic_duration_hours=tensor_batch.get("semantic_duration_hours", None),
                chunk_token_counts=tensor_batch.get("chunk_token_counts", None),
                chunk_duration_hours=tensor_batch.get("chunk_duration_hours", None),
                token_event_index=tensor_batch.get("token_event_index", None),
                token_event_slot_ids=tensor_batch.get("token_event_slot_ids", None),
                event_input_ids=tensor_batch.get("event_input_ids", None),
                event_time_ids=tensor_batch.get("event_time_ids", None),
                event_numeric_values=tensor_batch.get("event_numeric_values", None),
                event_numeric_mask=tensor_batch.get("event_numeric_mask", None),
                event_type_ids=tensor_batch.get("event_type_ids", None),
                event_payload_ids=tensor_batch.get("event_payload_ids", None),
                event_demographic_feature_ids=tensor_batch.get("event_demographic_feature_ids", None),
                event_attention_mask=tensor_batch.get("event_attention_mask", None),
                event_memory_rule_scores=tensor_batch.get("event_memory_rule_scores", None),
                event_memory_group_ids=tensor_batch.get("event_memory_group_ids", None),
                event_memory_first_flags=tensor_batch.get("event_memory_first_flags", None),
                event_memory_chronic_flags=tensor_batch.get("event_memory_chronic_flags", None),
                return_aux_state=True,
            )
            _update_carry_cache(
                tensor_batch=tensor_batch,
                aux_state=aux_state,
                carry_cache=carry_cache,
            )
            new_items, next_item_id = _emit_items_from_batch(
                tensor_batch=tensor_batch,
                sample_meta=sample,
                aux_state=aux_state,
                rel_path_to_id=rel_path_to_id,
                next_item_id=next_item_id,
            )
            items.extend(new_items)

    if not items:
        raise ValueError("No precedent items were emitted; check the split and segmentation settings.")

    rel_path_vocab = [None] * len(rel_path_to_id)
    for rel_path, rel_id in rel_path_to_id.items():
        rel_path_vocab[int(rel_id)] = str(rel_path)
    store = materialize_precedent_index_store(
        items=items,
        rel_path_vocab=[path or "" for path in rel_path_vocab],
        num_window_types=int(model.num_window_types),
    )
    save_precedent_index_store(args.output_path, store)

    summary = {
        "items": int(store.item_ids.shape[0]),
        "subjects": int(store.subject_ids.unique().numel()),
        "num_window_types": int(store.num_window_types),
        "output_path": str(args.output_path),
        "checkpoint": str(args.resume_from),
        "split": str(args.split),
    }
    summary_path = Path(args.output_path).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
