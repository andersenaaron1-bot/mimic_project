#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict

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
from scripts.train_transformer_v1 import (  # noqa: E402
    TrainModelConfig,
    _build_runtime_bundle,
    _resolve_device,
    _reset_encoders,
)
from src.ehr_hier.data.structural_codes import structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline  # noqa: E402
from src.ehr_hier.data.token_types import TokenCategory  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402
from src.ehr_hier.transformer.collator import AETHierarchicalCollator  # noqa: E402
from src.ehr_hier.transformer.generation import (  # noqa: E402
    RolloutConfig,
    RolloutSubjectState,
    WindowGenerationGrammar,
    build_dense_token_metadata,
    rollout_subject_with_model,
)
from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer  # noqa: E402


def _resolve_arg(
    name: str,
    *,
    cli_value: Any,
    checkpoint_args: Dict[str, Any],
    default: Any = None,
    required: bool = False,
) -> Any:
    value = cli_value if cli_value is not None else checkpoint_args.get(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required argument {name!r} in CLI and checkpoint metadata.")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Roll out one subject from a trained v1 checkpoint using the current window-marker grammar."
        )
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--device", default="auto")

    ap.add_argument("--split", default=None)
    ap.add_argument("--subject_id", type=int, default=None)
    ap.add_argument("--subject_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--meds_reader_db", default=None)
    ap.add_argument("--splits_parquet", default=None)
    ap.add_argument("--tokenization_yaml", default=None)
    ap.add_argument("--vocab_manifest", default=None)
    ap.add_argument("--structural_yaml", default=None)
    ap.add_argument("--sparse_vocab_json", default=None)
    ap.add_argument("--runtime_vocab_json", default=None)

    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--medtok_attr_dir", default=None)
    ap.add_argument("--medtok_crosswalk_json", default=None)
    ap.add_argument("--codes_parquet_parent_lookup", default=None)
    ap.add_argument("--allow_smoke_medtok", action="store_true")

    ap.add_argument("--code2id_pt", default=None)
    ap.add_argument("--stats_pt", default=None)
    ap.add_argument("--cvae_ckpt", default=None)
    ap.add_argument("--tokenizer_ckpt", default=None)

    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--max_chunks_per_window", type=int, default=None)
    ap.add_argument("--max_len_per_window", type=int, default=None)

    ap.add_argument("--prefix_windows", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_new_windows", type=int, default=1)
    ap.add_argument("--rollout_max_chunks_per_window", type=int, default=None)
    ap.add_argument("--rollout_max_content_tokens_per_chunk", type=int, default=None)
    ap.add_argument("--min_content_tokens_per_chunk", type=int, default=1)
    ap.add_argument("--default_content_dt_hours", type=float, default=1.0)
    ap.add_argument("--boundary_logit_margin", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--stop_after_end_token", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = dict(ckpt.get("args", {}) or {})
    model_config_raw = dict(ckpt.get("model_config", {}) or {})
    if not model_config_raw:
        raise ValueError("Checkpoint is missing model_config; cannot reconstruct transformer.")

    effective = argparse.Namespace(
        meds_reader_db=_resolve_arg(
            "meds_reader_db",
            cli_value=args.meds_reader_db,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        splits_parquet=_resolve_arg(
            "splits_parquet",
            cli_value=args.splits_parquet,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        tokenization_yaml=_resolve_arg(
            "tokenization_yaml",
            cli_value=args.tokenization_yaml,
            checkpoint_args=checkpoint_args,
            default="configs/data/tokenization_v1.yaml",
        ),
        vocab_manifest=_resolve_arg(
            "vocab_manifest",
            cli_value=args.vocab_manifest,
            checkpoint_args=checkpoint_args,
            default="artifacts/vocab_manifest.json",
        ),
        structural_yaml=_resolve_arg(
            "structural_yaml",
            cli_value=args.structural_yaml,
            checkpoint_args=checkpoint_args,
            default="configs/data/structural_codes.yaml",
        ),
        sparse_vocab_json=_resolve_arg(
            "sparse_vocab_json",
            cli_value=args.sparse_vocab_json,
            checkpoint_args=checkpoint_args,
            default=None,
        ),
        runtime_vocab_json=_resolve_arg(
            "runtime_vocab_json",
            cli_value=args.runtime_vocab_json,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        medtok_code2embeds=_resolve_arg(
            "medtok_code2embeds",
            cli_value=args.medtok_code2embeds,
            checkpoint_args=checkpoint_args,
            default=None,
        ),
        medtok_vocab_dir=_resolve_arg(
            "medtok_vocab_dir",
            cli_value=args.medtok_vocab_dir,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        medtok_attr_dir=_resolve_arg(
            "medtok_attr_dir",
            cli_value=args.medtok_attr_dir,
            checkpoint_args=checkpoint_args,
            default="artifacts/medtok_attrs",
        ),
        medtok_crosswalk_json=_resolve_arg(
            "medtok_crosswalk_json",
            cli_value=args.medtok_crosswalk_json,
            checkpoint_args=checkpoint_args,
            default=None,
        ),
        codes_parquet_parent_lookup=_resolve_arg(
            "codes_parquet_parent_lookup",
            cli_value=args.codes_parquet_parent_lookup,
            checkpoint_args=checkpoint_args,
            default=None,
        ),
        allow_smoke_medtok=bool(args.allow_smoke_medtok)
        or bool(checkpoint_args.get("allow_smoke_medtok", False)),
        code2id_pt=_resolve_arg(
            "code2id_pt",
            cli_value=args.code2id_pt,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        stats_pt=_resolve_arg(
            "stats_pt",
            cli_value=args.stats_pt,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        cvae_ckpt=_resolve_arg(
            "cvae_ckpt",
            cli_value=args.cvae_ckpt,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        tokenizer_ckpt=_resolve_arg(
            "tokenizer_ckpt",
            cli_value=args.tokenizer_ckpt,
            checkpoint_args=checkpoint_args,
            required=True,
        ),
        max_windows=int(
            _resolve_arg(
                "max_windows",
                cli_value=args.max_windows,
                checkpoint_args=checkpoint_args,
                default=32,
            )
        ),
        max_chunks_per_window=int(
            _resolve_arg(
                "max_chunks_per_window",
                cli_value=args.max_chunks_per_window,
                checkpoint_args=checkpoint_args,
                default=4,
            )
        ),
        max_len_per_window=int(
            _resolve_arg(
                "max_len_per_window",
                cli_value=args.max_len_per_window,
                checkpoint_args=checkpoint_args,
                default=128,
            )
        ),
        disable_residual_fallback=bool(checkpoint_args.get("disable_residual_fallback", False)),
        residual_fallback_buckets=int(checkpoint_args.get("residual_fallback_buckets", 39999)),
        diag_residual_offset=checkpoint_args.get("diag_residual_offset", None),
        proc_residual_offset=checkpoint_args.get("proc_residual_offset", None),
        med_residual_offset=checkpoint_args.get("med_residual_offset", None),
    )

    split = str(
        args.split
        if args.split is not None
        else checkpoint_args.get("eval_split", checkpoint_args.get("train_split", "tuning"))
    )
    sample_seed = int(args.seed if args.seed is not None else checkpoint_args.get("seed", 1337))

    tokenization_contract = _load_tokenization_contract(effective.tokenization_yaml)
    vocab_config, remapper = _build_runtime_bundle(effective)

    artifact_args = argparse.Namespace(
        structural_yaml=effective.structural_yaml,
        sparse_vocab_json=effective.sparse_vocab_json,
        medtok_code2embeds=effective.medtok_code2embeds,
        medtok_vocab_dir=effective.medtok_vocab_dir,
        medtok_attr_dir=effective.medtok_attr_dir,
        medtok_crosswalk_json=effective.medtok_crosswalk_json,
        code2id_pt=effective.code2id_pt,
        tokenizer_ckpt=effective.tokenizer_ckpt,
        codes_parquet_parent_lookup=effective.codes_parquet_parent_lookup,
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
        effective,
        tokenization_contract=tokenization_contract,
    )

    struct_codes_union = set(structural_surface_vocab_codes(artifacts.structural_codebook))
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)
    meas_cfg = _build_measurement_config(effective, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError("Missing measurement artifacts; cannot rebuild rollout timeline.")

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

    if args.subject_id is not None:
        subject_id = int(args.subject_id)
    else:
        subject_cap = max(1, int(args.subject_index) + 1)
        subject_ids = _load_subject_ids(
            str(effective.splits_parquet),
            split,
            subject_cap,
            sample_seed=sample_seed,
        )
        if not subject_ids:
            raise ValueError(f"No subject ids available for split {split!r}.")
        if int(args.subject_index) < 0 or int(args.subject_index) >= len(subject_ids):
            raise IndexError(
                f"subject_index {args.subject_index} out of range for split {split!r} with {len(subject_ids)} subjects."
            )
        subject_id = int(subject_ids[int(args.subject_index)])

    _reset_encoders(encoders)
    db = mr.SubjectDatabase(str(effective.meds_reader_db))
    timeline_kwargs = {
        "db": db,
        "subject_id": int(subject_id),
        "encoders": encoders,
        "structural_codebook": artifacts.structural_codebook,
        "window_hook_label": "window_boundary",
        "attach_med_numeric": True,
        "emit_global_demographic_tokens": True,
        "special_token_offset": 0,
        "qual_obs_code_vocab": artifacts.obs_code_vocab,
        "qual_obs_value_vocab": artifacts.obs_value_vocab,
        "qual_obs_tail_policy": artifacts.obs_tail_policy,
    }
    timeline_sig = inspect.signature(build_subject_timeline)
    timeline = build_subject_timeline(
        **{k: v for k, v in timeline_kwargs.items() if k in timeline_sig.parameters}
    )
    if not timeline:
        raise ValueError(f"Subject {subject_id} produced an empty timeline.")

    collator = AETHierarchicalCollator(
        max_windows=int(effective.max_windows),
        max_chunks_per_window=int(effective.max_chunks_per_window),
        max_len_per_window=int(effective.max_len_per_window),
        window_markers=window_markers_cfg,
        segmentation=segmentation_cfg,
        id_remapper=remapper,
    )
    batch = collator([timeline])
    grammar = WindowGenerationGrammar.from_vocab_config(vocab_config)
    dense_token_meta = build_dense_token_metadata(vocab_config)
    subject_state = RolloutSubjectState.from_collated_batch(
        batch,
        sample_idx=0,
        grammar=grammar,
        trim_trailing_marker=True,
        pad_id=int(getattr(collator, "pad_id", 0)),
    )
    original_window_count = len(subject_state.input_ids)
    prefix_windows = (
        int(args.prefix_windows)
        if args.prefix_windows is not None
        else max(1, int(original_window_count) - 1)
    )
    subject_state.truncate_to_window_prefix(
        int(prefix_windows),
        grammar=grammar,
        trim_trailing_marker=True,
    )

    device = _resolve_device(str(args.device))
    model = AdaptiveEpisodicTransformer(
        TrainModelConfig(**model_config_raw),
        vocab_config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    rollout_cfg = RolloutConfig(
        max_new_tokens=int(args.max_new_tokens),
        max_new_windows=int(args.max_new_windows),
        max_chunks_per_window=int(
            args.rollout_max_chunks_per_window
            if args.rollout_max_chunks_per_window is not None
            else effective.max_chunks_per_window
        ),
        max_content_tokens_per_chunk=int(
            args.rollout_max_content_tokens_per_chunk
            if args.rollout_max_content_tokens_per_chunk is not None
            else max(1, effective.max_len_per_window - 2)
        ),
        min_content_tokens_per_chunk=int(args.min_content_tokens_per_chunk),
        default_content_dt_hours=float(args.default_content_dt_hours),
        boundary_logit_margin=float(args.boundary_logit_margin),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if args.top_k is not None else None,
        sample=bool(args.sample),
        stop_after_end_token=bool(args.stop_after_end_token),
        trim_trailing_marker=False,
    )
    prefix_state = subject_state.export_sequences(dense_token_meta=dense_token_meta)
    rollout = rollout_subject_with_model(
        model=model,
        vocab_config=vocab_config,
        subject_state=subject_state,
        config=rollout_cfg,
        device=device,
    )

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_meta": {
            "epoch": int(ckpt.get("epoch", 0)),
            "global_step": int(ckpt.get("global_step", 0)),
            "best_val_loss": ckpt.get("best_val_loss", None),
            "latest_train_metrics": ckpt.get("latest_train_metrics", {}),
            "latest_val_metrics": ckpt.get("latest_val_metrics", {}),
        },
        "subject": {
            "split": str(split),
            "subject_id": int(subject_id),
            "timeline_token_count": int(len(timeline)),
            "original_window_count": int(original_window_count),
            "prefix_window_count": int(len(subject_state.input_ids)),
        },
        "rollout_config": {
            "max_new_tokens": int(rollout_cfg.max_new_tokens),
            "max_new_windows": int(rollout_cfg.max_new_windows),
            "max_chunks_per_window": int(rollout_cfg.max_chunks_per_window),
            "max_content_tokens_per_chunk": int(rollout_cfg.max_content_tokens_per_chunk),
            "min_content_tokens_per_chunk": int(rollout_cfg.min_content_tokens_per_chunk),
            "default_content_dt_hours": float(rollout_cfg.default_content_dt_hours),
            "boundary_logit_margin": float(rollout_cfg.boundary_logit_margin),
            "temperature": float(rollout_cfg.temperature),
            "top_k": int(rollout_cfg.top_k) if rollout_cfg.top_k is not None else None,
            "sample": bool(rollout_cfg.sample),
            "stop_after_end_token": bool(rollout_cfg.stop_after_end_token),
        },
        "window_markers": dict(vocab_config.get("window_markers", {}) or {}),
        "prefix_state": prefix_state,
        "rollout": rollout,
    }

    text = json.dumps(payload, indent=2)
    if args.output_json:
        out_fp = Path(args.output_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
