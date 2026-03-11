#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import inspect
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meds_reader as mr

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
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline  # noqa: E402
from src.ehr_hier.data.structural_codes import structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.data.token_types import TokenCategory  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402
from src.ehr_hier.transformer.collator import AETHierarchicalCollator  # noqa: E402
from src.ehr_hier.transformer.loss import AETLossModule  # noqa: E402
from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer  # noqa: E402
from src.ehr_hier.transformer.vocab_runtime import (  # noqa: E402
    build_runtime_vocab_and_remapper,
    load_runtime_vocab_bundle,
)


@dataclass
class SmokeModelConfig:
    d_model: int = 128
    num_heads: int = 8
    d_ff: int = 256
    num_local_layers: int = 2
    num_global_layers: int = 2
    num_chunk_layers: int = 1
    rope_max_period: float = 10_000.0
    dropout: float = 0.1
    special_type_id: int = 0
    enable_transition_bias: bool = True
    enable_time_embedding: bool = False
    global_fusion_mode: str = "add"
    exclude_special_from_global_fusion: bool = True


def _reset_encoders(encoders: Dict[TokenCategory, Any]) -> None:
    for enc in encoders.values():
        reset = getattr(enc, "reset_state", None)
        if callable(reset):
            reset()


def _build_runtime_bundle(
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], Any]:
    if args.runtime_vocab_json:
        fp = Path(args.runtime_vocab_json)
        if fp.exists():
            return load_runtime_vocab_bundle(fp)
    vocab_config, remapper = build_runtime_vocab_and_remapper(
        sparse_vocab_contract=args.sparse_vocab_json,
        tokenization_contract=args.tokenization_yaml,
        vocab_manifest=args.vocab_manifest,
        structural_yaml=args.structural_yaml,
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=args.medtok_vocab_dir,
        medtok_attr_dir=args.medtok_attr_dir,
        code2id_pt=args.code2id_pt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        allow_smoke_medtok=bool(args.allow_smoke_medtok),
    )
    if args.runtime_vocab_json:
        out_fp = Path(args.runtime_vocab_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(
            json.dumps(
                {
                    "sparse_vocab_contract": dict(vocab_config.get("sparse_vocab_contract", {}) or {}),
                    "vocab_config": vocab_config,
                    "id_remapper": remapper.serialize(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return vocab_config, remapper


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bare-bones smoke test: timeline build -> collation -> model forward/backward -> optimizer step."
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--vocab_manifest", default="artifacts/vocab_manifest.json")
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--sparse_vocab_json", default=None)

    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--medtok_crosswalk_json", default=None)
    ap.add_argument("--allow_smoke_medtok", action="store_true")
    ap.add_argument("--codes_parquet_parent_lookup", default=None)

    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--tokenizer_ckpt", required=True)

    ap.add_argument("--runtime_vocab_json", default=None)
    ap.add_argument("--structural_entity_dense_size", type=int, default=65536)
    ap.add_argument("--structural_entity_source_size", type=int, default=900000)

    ap.add_argument("--max_windows", type=int, default=32)
    ap.add_argument("--max_chunks_per_window", type=int, default=8)
    ap.add_argument("--max_len_per_window", type=int, default=128)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--num_local_layers", type=int, default=2)
    ap.add_argument("--num_global_layers", type=int, default=2)
    ap.add_argument("--num_chunk_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=39999)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)

    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

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

    struct_codes_union = set(structural_surface_vocab_codes(artifacts.structural_codebook))
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)

    meas_cfg = _build_measurement_config(args, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError("Missing measurement artifacts; need code2id/stats/cvae/tokenizer checkpoints.")

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
        enable_residual_fallback=bool(residual_enabled),
        residual_fallback_buckets=int(residual_buckets),
        residual_fallback_offsets=dict(residual_offsets),
    )

    db = mr.SubjectDatabase(str(args.meds_reader_db))
    subject_ids = _load_subject_ids(
        str(args.splits_parquet),
        str(args.split),
        int(args.max_subjects),
        sample_seed=int(args.seed),
    )
    if not subject_ids:
        raise ValueError("No subjects loaded for smoke run.")

    timelines: List[List[Any]] = []
    token_counts: List[int] = []
    for sid in subject_ids:
        _reset_encoders(encoders)
        timeline_kwargs = {
            "db": db,
            "subject_id": int(sid),
            "encoders": encoders,
            "structural_codebook": artifacts.structural_codebook,
            "window_hook_label": "window_boundary",
            "attach_med_numeric": True,
            "emit_process_struct_tokens": False,
            "drop_original_process_marker_tokens": False,
            "emit_global_demographic_tokens": True,
            "special_token_offset": 0,
        }
        sig = inspect.signature(build_subject_timeline)
        tl = build_subject_timeline(
            **{k: v for k, v in timeline_kwargs.items() if k in sig.parameters}
        )
        if tl:
            timelines.append(tl)
            token_counts.append(len(tl))
    if not timelines:
        raise ValueError("No non-empty timelines built.")

    vocab_config, remapper = _build_runtime_bundle(args)

    collator = AETHierarchicalCollator(
        max_windows=int(args.max_windows),
        max_chunks_per_window=int(args.max_chunks_per_window),
        max_len_per_window=int(args.max_len_per_window),
        window_markers=window_markers_cfg,
        segmentation=segmentation_cfg,
        id_remapper=remapper,
    )
    batch = collator(timelines)
    tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    cfg = SmokeModelConfig(
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        d_ff=int(args.d_ff),
        num_local_layers=int(args.num_local_layers),
        num_global_layers=int(args.num_global_layers),
        num_chunk_layers=int(args.num_chunk_layers),
        dropout=float(args.dropout),
    )
    model = AdaptiveEpisodicTransformer(cfg, vocab_config).to(device)
    criterion = AETLossModule(vocab_config=vocab_config, strict_routing=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    model.train()
    optim.zero_grad(set_to_none=True)
    head_outputs, _ = model(
        input_ids=tensor_batch["input_ids"],
        time_ids=tensor_batch["time_ids"],
        numeric_values=tensor_batch["numeric_values"],
        token_type_ids=tensor_batch["token_type_ids"],
        attention_mask=tensor_batch["attention_mask"],
        window_start_times=tensor_batch.get("window_start_times", None),
        window_mask=tensor_batch.get("window_mask", None),
        window_type_ids=tensor_batch.get("window_type_ids", None),
        chunk_mask=tensor_batch.get("chunk_mask", None),
        chunk_start_offsets=tensor_batch.get("chunk_start_offsets", None),
        chunk_is_last=tensor_batch.get("chunk_is_last", None),
    )
    loss, logs = criterion(head_outputs, tensor_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite smoke loss: {float(loss.detach().cpu().item())}")
    for k, v in logs.items():
        if not math.isfinite(float(v)):
            raise FloatingPointError(f"Non-finite smoke metric: {k}={v}")

    loss.backward()
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu().item()
    )
    if not math.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite grad_norm: {grad_norm}")
    optim.step()

    payload = {
        "subjects_requested": int(args.max_subjects),
        "subjects_built": int(len(timelines)),
        "avg_timeline_tokens": float(sum(token_counts) / max(1, len(token_counts))),
        "batch_shapes": {
            "input_ids": list(tensor_batch["input_ids"].shape),
            "time_ids": list(tensor_batch["time_ids"].shape),
            "numeric_values": list(tensor_batch["numeric_values"].shape),
            "attention_mask": list(tensor_batch["attention_mask"].shape),
            "window_type_ids": list(tensor_batch["window_type_ids"].shape),
        },
        "overflow_stats": batch.get("overflow_stats", {}),
        "id_remap_stats": batch.get("id_remap_stats", {}),
        "vocab_config": {
            "total_size": int(vocab_config["total_size"]),
            "size_special": int(vocab_config["size_special"]),
            "size_rvq": int(vocab_config["size_rvq"]),
            "size_meas_labels": int(vocab_config["size_meas_labels"]),
            "size_meds": int(vocab_config["size_meds"]),
        },
        "loss": float(loss.detach().cpu().item()),
        "loss_logs": {k: float(v) for k, v in logs.items()},
        "grad_norm": float(grad_norm),
        "status": "ok",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
