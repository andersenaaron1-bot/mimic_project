#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meds_reader as mr

from scripts.audit_tokenization_flow import (  # noqa: E402
    _build_measurement_config,
    _build_segmentation_config,
    _build_static_artifacts,
    _build_struct_vocab,
    _build_trajectory_split_config,
    _build_window_marker_config,
    _load_subject_ids,
    _load_tokenization_contract,
    _resolve_residual_policy,
    _resolve_residual_tail_policies,
)
from src.ehr_hier.data.dataset import PrecompiledMEDSDataset  # noqa: E402
from src.ehr_hier.data.event_frames import EventFrame  # noqa: E402
from src.ehr_hier.data.structural_codes import structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline  # noqa: E402
from src.ehr_hier.data.token_types import TokenCategory  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402
from src.ehr_hier.transformer.collator import AETHierarchicalCollator  # noqa: E402
from src.ehr_hier.transformer.episodic_memory import (  # noqa: E402
    EpisodicMemoryState,
    PatientMemoryState,
)
from src.ehr_hier.transformer.loss import AETLossModule  # noqa: E402
from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer  # noqa: E402
from src.ehr_hier.transformer.vocab_runtime import (  # noqa: E402
    build_runtime_vocab_and_remapper,
    load_runtime_vocab_bundle,
)

TOKEN_FAMILY_WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "none": {},
    "semantic_boost_v1": {
        "diagnosis": 3.0,
        "diagnosis_residual": 2.5,
        "procedure": 5.0,
        "procedure_residual": 3.0,
        "medication": 2.0,
        "medication_residual": 1.5,
        "measurement_value": 0.5,
        "observation_code": 0.75,
        "observation_value": 0.75,
        "structural": 0.75,
        "special_marker": 0.5,
        "unk": 0.1,
    },
    "semantic_boost_v2": {
        "diagnosis": 2.5,
        "diagnosis_residual": 2.0,
        "procedure": 4.0,
        "procedure_residual": 2.5,
        "medication": 1.75,
        "medication_residual": 1.35,
        "measurement_value": 0.75,
        "observation_code": 0.9,
        "observation_value": 0.9,
        "structural": 0.85,
        "special_marker": 0.75,
        "unk": 0.35,
    },
}

OBJECTIVE_PRESETS: Dict[str, Dict[str, Any]] = {
    "legacy_v1": {
        "prefer_unified_token_loss": True,
        "token_family_weight_preset": "none",
        "precedent_loss_start_step": 0,
        "precedent_loss_ramp_steps": 0,
        "loss_weights": {
            "token": 1.0,
            "event_token": 1.0,
            "event_family": 1.0,
            "event_payload": 1.0,
            "event_concept": 1.0,
            "event_dt": 1.0,
            "event_value": 1.0,
            "val": 0.0,
            "transition": 1.0,
            "win_boundary": 1.0,
            "win": 0.5,
            "len": 0.0,
            "chunk": 0.0,
            "time": 0.0,
            "dt": 0.0,
            "next_window_gap": 1.0,
            "next_window_duration": 1.0,
            "next_window_support": 0.5,
            "precedent_future": 1.0,
            "precedent_contrast": 1.0,
            "precedent_anchor": 0.25,
        },
    },
    "world_model_mttee": {
        "prefer_unified_token_loss": False,
        "token_family_weight_preset": "none",
        "precedent_loss_start_step": 2000,
        "precedent_loss_ramp_steps": 2000,
        "loss_weights": {
            "token": 0.05,
            "event_token": 0.0,
            "event_family": 1.0,
            "event_payload": 1.0,
            "event_concept": 1.0,
            "event_dt": 1.0,
            "event_value": 1.0,
            "val": 0.0,
            "transition": 1.0,
            "win_boundary": 1.0,
            "win": 1.0,
            "len": 0.0,
            "chunk": 0.0,
            "time": 0.0,
            "dt": 0.0,
            "next_window_gap": 1.0,
            "next_window_duration": 1.0,
            "next_window_support": 0.5,
            "precedent_future": 1.0,
            "precedent_contrast": 1.0,
            "precedent_anchor": 0.25,
        },
    },
}

LOSS_WEIGHT_ARG_TO_KEY: Dict[str, str] = {
    "token_loss_weight": "token",
    "event_token_loss_weight": "event_token",
    "event_family_loss_weight": "event_family",
    "event_payload_loss_weight": "event_payload",
    "event_concept_loss_weight": "event_concept",
    "event_dt_loss_weight": "event_dt",
    "event_value_loss_weight": "event_value",
    "value_loss_weight": "val",
    "transition_loss_weight": "transition",
    "win_boundary_loss_weight": "win_boundary",
    "win_loss_weight": "win",
    "len_loss_weight": "len",
    "chunk_loss_weight": "chunk",
    "time_loss_weight": "time",
    "dt_loss_weight": "dt",
    "next_window_gap_loss_weight": "next_window_gap",
    "next_window_duration_loss_weight": "next_window_duration",
    "next_window_support_loss_weight": "next_window_support",
    "precedent_future_loss_weight": "precedent_future",
    "precedent_contrast_loss_weight": "precedent_contrast",
    "precedent_anchor_loss_weight": "precedent_anchor",
}


@dataclass
class TrainModelConfig:
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 512
    num_local_layers: int = 2
    num_global_layers: int = 2
    num_chunk_layers: int = 1
    rope_max_period: float = 10_000.0
    dropout: float = 0.1
    special_type_id: int = 0
    enable_transition_bias: bool = True
    enable_time_embedding: bool = True
    time_embedding_max_hours: float = 31.0 * 24.0
    local_time_embedding_max_hours: float = 72.0
    global_time_embedding_max_hours: float = 365.25 * 24.0 * 10.0
    enable_chunk_meta_sidechannel: bool = True
    enable_window_sequence_meta: bool = True
    num_token_types: int = 8
    condition_numeric_on_token_type: bool = True
    numeric_value_transform: str = "signed_log1p"
    global_fusion_mode: str = "add"
    exclude_special_from_global_fusion: bool = True
    global_context_mode: str = "latent_state"
    use_unified_token_head: bool = True
    emit_switched_heads: bool = False
    use_event_composer: bool = True
    enable_exact_memory: bool = True
    enable_precedent_memory: bool = False
    exact_memory_slots: int = 16
    exact_memory_static_slots: int = 2
    exact_memory_persistent_slots: int = 16
    exact_memory_episodic_slots: int = 16
    exact_memory_write_per_window: int = 2
    exact_memory_retrieve_k: int = 4
    exact_memory_static_retrieve_k: int = 2
    exact_memory_persistent_retrieve_k: int = 4
    exact_memory_episodic_retrieve_k: int = 4
    exact_memory_static_feature_ids: str = "1,2"
    exact_memory_max_same_group: int = 2
    exact_memory_age_decay: float = 0.05
    exact_memory_rule_write_scale: float = 1.0
    exact_memory_rule_retrieval_scale: float = 0.25
    exact_memory_learned_write_scale: float = 1.0
    exact_memory_first_occurrence_bonus: float = 0.75
    exact_memory_chronic_bonus: float = 1.5
    precedent_retrieve_k: int = 4
    precedent_strict_window_type_match: bool = True
    precedent_support_overlap_bias: float = 0.25
    precedent_score_temperature: float = 1.0
    carry_state_across_segments: bool = True
    event_bundle_slots: int = 32
    enable_event_time_nll_head: bool = True
    enable_next_window_gap_nll_head: bool = True
    enable_next_window_duration_nll_head: bool = True
    enable_next_window_support_head: bool = True


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


def build_optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> List[dict]:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim < 2 or lname.endswith("bias") or "norm" in lname:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_lr_lambda(*, total_steps: int, warmup_steps: int, min_lr_scale: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_scale = float(min_lr_scale)

    def _fn(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(total_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_scale) + (1.0 - float(min_lr_scale)) * cosine

    return _fn


def resolve_epoch_range(*, start_epoch: int, epochs_to_run: int) -> range:
    start_epoch = max(1, int(start_epoch))
    epochs_to_run = max(1, int(epochs_to_run))
    end_epoch = start_epoch + epochs_to_run - 1
    return range(start_epoch, end_epoch + 1)


def resolve_token_family_weights(
    *,
    preset: str = "none",
    overrides: Iterable[str] | None = None,
) -> Dict[str, float]:
    preset_key = str(preset or "none").strip() or "none"
    if preset_key not in TOKEN_FAMILY_WEIGHT_PRESETS:
        raise KeyError(
            f"Unknown token family weight preset {preset_key!r}; expected one of {sorted(TOKEN_FAMILY_WEIGHT_PRESETS)}"
        )
    resolved = dict(TOKEN_FAMILY_WEIGHT_PRESETS[preset_key])
    allowed = set(AETLossModule.TOKEN_FAMILY_GROUPS)
    for raw in overrides or []:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --token_family_weight entry {item!r}; expected format family=value"
            )
        name, value = item.split("=", 1)
        family = str(name).strip()
        if family not in allowed:
            raise KeyError(
                f"Unknown token family {family!r}; expected one of {sorted(allowed)}"
            )
        weight = float(str(value).strip())
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"Token family weight for {family!r} must be finite and > 0, got {value!r}"
            )
        resolved[family] = weight
    return resolved


def _cli_flag_present(raw_argv: Sequence[str], flag: str) -> bool:
    target = str(flag)
    return any(
        str(token) == target or str(token).startswith(target + "=")
        for token in raw_argv
    )


def resolve_objective_configuration(
    *,
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> Dict[str, Any]:
    preset_key = str(getattr(args, "objective_preset", "legacy_v1"))
    if preset_key not in OBJECTIVE_PRESETS:
        raise KeyError(
            f"Unknown objective preset {preset_key!r}; expected one of {sorted(OBJECTIVE_PRESETS)}"
        )
    preset = OBJECTIVE_PRESETS[preset_key]

    prefer_unified = getattr(args, "prefer_unified_token_loss", None)
    if prefer_unified is None:
        prefer_unified = bool(preset["prefer_unified_token_loss"])

    token_family_weight_preset = str(getattr(args, "token_family_weight_preset", "none"))
    if not _cli_flag_present(raw_argv, "--token_family_weight_preset"):
        token_family_weight_preset = str(preset["token_family_weight_preset"])

    resolved_weights = dict(preset["loss_weights"])
    for arg_name, weight_key in LOSS_WEIGHT_ARG_TO_KEY.items():
        flag = f"--{arg_name}"
        if _cli_flag_present(raw_argv, flag):
            resolved_weights[weight_key] = float(getattr(args, arg_name))

    precedent_loss_start_step = getattr(args, "precedent_loss_start_step", None)
    if precedent_loss_start_step is None:
        precedent_loss_start_step = int(preset["precedent_loss_start_step"])
    precedent_loss_ramp_steps = getattr(args, "precedent_loss_ramp_steps", None)
    if precedent_loss_ramp_steps is None:
        precedent_loss_ramp_steps = int(preset["precedent_loss_ramp_steps"])

    return {
        "preset": preset_key,
        "prefer_unified_token_loss": bool(prefer_unified),
        "token_family_weight_preset": str(token_family_weight_preset),
        "loss_weights": {str(k): float(v) for k, v in resolved_weights.items()},
        "precedent_loss_start_step": int(precedent_loss_start_step),
        "precedent_loss_ramp_steps": int(precedent_loss_ramp_steps),
    }


def precedent_loss_scale_for_step(
    *,
    global_step: int,
    start_step: int,
    ramp_steps: int,
) -> float:
    step = int(global_step)
    start = max(0, int(start_step))
    ramp = max(0, int(ramp_steps))
    if start <= 0:
        return 1.0 if ramp == 0 else min(1.0, float(max(0, step + 1)) / float(max(1, ramp)))
    if step < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, float(step - start + 1) / float(ramp))


def resolve_scheduled_loss_weights(
    *,
    base_weights: Dict[str, float],
    global_step: int,
    precedent_loss_start_step: int,
    precedent_loss_ramp_steps: int,
) -> tuple[Dict[str, float], Dict[str, float]]:
    weights = {str(k): float(v) for k, v in dict(base_weights).items()}
    precedent_scale = precedent_loss_scale_for_step(
        global_step=int(global_step),
        start_step=int(precedent_loss_start_step),
        ramp_steps=int(precedent_loss_ramp_steps),
    )
    for key in ("precedent_future", "precedent_contrast", "precedent_anchor"):
        weights[key] = float(weights.get(key, 0.0)) * float(precedent_scale)
    return weights, {"precedent_loss_scale": float(precedent_scale)}


def resolve_precompiled_num_workers(requested_num_workers: int) -> int:
    requested = int(requested_num_workers)
    if requested > 0:
        return requested
    cpu_total = os.cpu_count() or 2
    return max(1, min(8, cpu_total - 1))


def build_dataloader_kwargs(
    *,
    device: torch.device,
    num_workers: int,
    precompiled: bool,
    prefetch_factor: int,
) -> Dict[str, Any]:
    resolved_workers = (
        resolve_precompiled_num_workers(int(num_workers))
        if precompiled
        else int(num_workers)
    )
    kwargs: Dict[str, Any] = {
        "num_workers": int(resolved_workers),
        "pin_memory": (device.type == "cuda"),
    }
    if int(resolved_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


class OnTheFlyTimelineDataset(Dataset):
    def __init__(
        self,
        *,
        db_path: str,
        subject_ids: Iterable[int],
        encoders: Dict[TokenCategory, Any],
        structural_codebook: Any,
        qual_obs_code_vocab: Any = None,
        qual_obs_value_vocab: Any = None,
        qual_obs_tail_policy: str = "drop",
        emit_global_demographic_tokens: bool = True,
    ) -> None:
        self.db_path = str(db_path)
        self.subject_ids = [int(sid) for sid in subject_ids]
        self.encoders = encoders
        self.structural_codebook = structural_codebook
        self.qual_obs_code_vocab = qual_obs_code_vocab
        self.qual_obs_value_vocab = qual_obs_value_vocab
        self.qual_obs_tail_policy = str(qual_obs_tail_policy)
        self.emit_global_demographic_tokens = bool(emit_global_demographic_tokens)
        self._db: Any | None = None
        self._timeline_sig = inspect.signature(build_subject_timeline)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _ensure_db(self) -> Any:
        if self._db is None:
            self._db = mr.SubjectDatabase(self.db_path)
        return self._db

    def __getitem__(self, idx: int) -> Dict[str, Any] | None:
        sid = int(self.subject_ids[idx])
        db = self._ensure_db()
        _reset_encoders(self.encoders)
        timeline_kwargs = {
            "db": db,
            "subject_id": sid,
            "encoders": self.encoders,
            "structural_codebook": self.structural_codebook,
            "window_hook_label": "window_boundary",
            "attach_med_numeric": True,
            "emit_global_demographic_tokens": self.emit_global_demographic_tokens,
            "special_token_offset": 0,
            "qual_obs_code_vocab": self.qual_obs_code_vocab,
            "qual_obs_value_vocab": self.qual_obs_value_vocab,
            "qual_obs_tail_policy": self.qual_obs_tail_policy,
        }
        timeline = build_subject_timeline(
            **{k: v for k, v in timeline_kwargs.items() if k in self._timeline_sig.parameters}
        )
        if not timeline:
            return None
        return {
            "timeline": timeline,
            "subject_id": int(sid),
            "trajectory_ord": 0,
        }


class TimelineCollateAdapter:
    def __init__(self, collator: AETHierarchicalCollator) -> None:
        self.collator = collator

    def __call__(self, batch: List[Dict[str, Any] | None]) -> Dict[str, Any]:
        items = [item for item in batch if item]
        if not items:
            raise ValueError("All timelines in batch were empty after on-the-fly building.")
        return self.collator(items)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _resolve_carry_inputs(
    *,
    tensor_batch: Dict[str, torch.Tensor],
    model: AdaptiveEpisodicTransformer,
    carry_cache: Dict[int, Dict[str, Any]] | None,
) -> tuple[
    torch.Tensor | None,
    PatientMemoryState | EpisodicMemoryState | None,
]:
    if carry_cache is None:
        return None, None
    subject_ids = tensor_batch.get("subject_ids", None)
    trajectory_ords = tensor_batch.get("trajectory_ords", None)
    if subject_ids is None or trajectory_ords is None:
        return None, None

    B = int(subject_ids.shape[0])
    device = tensor_batch["input_ids"].device
    dtype = model.embeddings.token_embedding.weight.dtype
    d_model = int(model.config.d_model)
    static_slots = int(getattr(model.config, "exact_memory_static_slots", 0))
    persistent_slots = int(getattr(model.config, "exact_memory_persistent_slots", 0))
    episodic_slots = int(getattr(model.config, "exact_memory_episodic_slots", 0))

    prev_global_rows: List[torch.Tensor | None] = []
    prev_memory_rows: List[PatientMemoryState | EpisodicMemoryState | None] = []
    any_global = False
    any_memory = False

    for idx in range(B):
        sid = int(subject_ids[idx].detach().cpu().item())
        trajectory_ord = int(trajectory_ords[idx].detach().cpu().item())
        cached = carry_cache.get(sid) if sid >= 0 else None
        if cached is not None and int(cached.get("next_trajectory_ord", -1)) == trajectory_ord:
            global_state = cached.get("global_state", None)
            memory_state = cached.get("memory_state", None)
            prev_global_rows.append(global_state)
            prev_memory_rows.append(memory_state)
            any_global = any_global or global_state is not None
            any_memory = any_memory or memory_state is not None
        else:
            prev_global_rows.append(None)
            prev_memory_rows.append(None)

    prev_global_state = None
    if any_global:
        prev_global_state = torch.zeros((B, d_model), device=device, dtype=dtype)
        for idx, row in enumerate(prev_global_rows):
            if row is None:
                continue
            prev_global_state[idx] = row.to(device=device, dtype=dtype)

    prev_memory_state = None
    if any_memory and (static_slots > 0 or persistent_slots > 0 or episodic_slots > 0):
        prev_memory_state = PatientMemoryState.stack(
            prev_memory_rows,
            static_slots=static_slots,
            persistent_slots=persistent_slots,
            episodic_slots=episodic_slots,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    return prev_global_state, prev_memory_state


def _update_carry_cache(
    *,
    tensor_batch: Dict[str, torch.Tensor],
    aux_state: Dict[str, Any] | None,
    carry_cache: Dict[int, Dict[str, Any]] | None,
) -> None:
    if carry_cache is None or not aux_state:
        return
    subject_ids = tensor_batch.get("subject_ids", None)
    trajectory_ords = tensor_batch.get("trajectory_ords", None)
    if subject_ids is None or trajectory_ords is None:
        return

    global_state = aux_state.get("global_state", None)
    memory_state = aux_state.get("memory_state", None)

    B = int(subject_ids.shape[0])
    for idx in range(B):
        sid = int(subject_ids[idx].detach().cpu().item())
        trajectory_ord = int(trajectory_ords[idx].detach().cpu().item())
        if sid < 0 or trajectory_ord < 0:
            continue
        entry: Dict[str, Any] = {"next_trajectory_ord": int(trajectory_ord + 1)}
        if global_state is not None:
            entry["global_state"] = global_state[idx].detach().cpu()
        if memory_state is not None:
            entry["memory_state"] = memory_state.select(idx).detach().to("cpu")
        carry_cache[sid] = entry


def _accumulate_logs(acc: Dict[str, float], logs: Dict[str, float]) -> None:
    for k, v in logs.items():
        try:
            fv = float(v)
        except Exception:
            continue
        acc[k] = acc.get(k, 0.0) + fv


def _mean_logs(acc: Dict[str, float], denom: int) -> Dict[str, float]:
    denom = max(1, int(denom))
    return {k: float(v) / float(denom) for k, v in acc.items()}


def _run_model_and_loss(
    *,
    model: AdaptiveEpisodicTransformer,
    criterion: AETLossModule,
    tensor_batch: Dict[str, torch.Tensor],
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    prev_global_state: torch.Tensor | None = None,
    prev_memory_state: PatientMemoryState | EpisodicMemoryState | None = None,
    return_aux_state: bool = False,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any], Dict[str, Any] | None]:
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=bool(autocast_enabled),
    ):
        head_outputs, final_state = model(
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
            subject_ids=tensor_batch.get("subject_ids", None),
            trajectory_ords=tensor_batch.get("trajectory_ords", None),
            return_aux_state=bool(return_aux_state),
        )
        loss, logs = criterion(head_outputs, tensor_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu().item())}")
    aux_state = final_state if isinstance(final_state, dict) else None
    return loss, {k: float(v) for k, v in logs.items()}, head_outputs, aux_state


def evaluate(
    *,
    model: AdaptiveEpisodicTransformer,
    criterion: AETLossModule,
    base_loss_weights: Dict[str, float],
    dataloader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    max_batches: int | None = None,
    carry_across_segments: bool = False,
    global_step: int = 0,
    precedent_loss_start_step: int = 0,
    precedent_loss_ramp_steps: int = 0,
) -> Dict[str, float]:
    model.eval()
    scheduled_weights, schedule_logs = resolve_scheduled_loss_weights(
        base_weights=base_loss_weights,
        global_step=int(global_step),
        precedent_loss_start_step=int(precedent_loss_start_step),
        precedent_loss_ramp_steps=int(precedent_loss_ramp_steps),
    )
    criterion.set_weights(scheduled_weights)
    total_loss = 0.0
    n_batches = 0
    log_acc: Dict[str, float] = {}
    carry_cache: Dict[int, Dict[str, Any]] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            tensor_batch = _move_batch_to_device(batch, device)
            prev_global_state, prev_memory_state = _resolve_carry_inputs(
                tensor_batch=tensor_batch,
                model=model,
                carry_cache=carry_cache if carry_across_segments else None,
            )
            loss, logs, _, aux_state = _run_model_and_loss(
                model=model,
                criterion=criterion,
                tensor_batch=tensor_batch,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                prev_global_state=prev_global_state,
                prev_memory_state=prev_memory_state,
                return_aux_state=bool(carry_across_segments),
            )
            _update_carry_cache(
                tensor_batch=tensor_batch,
                aux_state=aux_state,
                carry_cache=carry_cache if carry_across_segments else None,
            )
            total_loss += float(loss.detach().cpu().item())
            _accumulate_logs(log_acc, logs)
            n_batches += 1
    if n_batches == 0:
        return {"loss": float("nan")}
    out = {"loss": total_loss / float(n_batches)}
    out.update(_mean_logs(log_acc, n_batches))
    out.update(schedule_logs)
    return out


def _save_checkpoint(
    *,
    path: Path,
    model: AdaptiveEpisodicTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: Any | None,
    args: argparse.Namespace,
    model_config: TrainModelConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float | None,
    latest_train_metrics: Dict[str, float] | None,
    latest_val_metrics: Dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "args": vars(args),
        "model_config": asdict(model_config),
        "best_val_loss": best_val_loss,
        "latest_train_metrics": latest_train_metrics or {},
        "latest_val_metrics": latest_val_metrics or {},
    }
    torch.save(payload, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "First representative v1 trainer for the hierarchical transformer. "
            "Uses the current tokenization/windowing contract with an explicit "
            "objective preset for either legacy unified-token training or the "
            "world-model marked event objective."
        )
    )
    ap.add_argument("--meds_reader_db", default=None)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--eval_split", default="tuning")
    ap.add_argument("--max_train_subjects", type=int, default=None)
    ap.add_argument("--max_eval_subjects", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--precompiled_root", default=None)
    ap.add_argument("--precompiled_train_root", default=None)
    ap.add_argument("--precompiled_eval_root", default=None)
    ap.add_argument(
        "--trajectory_mode",
        default="full_subject",
        choices=["full_subject", "admission_chain"],
    )
    ap.add_argument("--post_discharge_cutoff_days", type=float, default=31.0)

    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--vocab_manifest", default="artifacts/vocab_manifest.json")
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--sparse_vocab_json", default=None)

    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", required=True)
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--medtok_crosswalk_json", default=None)
    ap.add_argument("--allow_smoke_medtok", action="store_true")
    ap.add_argument("--codes_parquet_parent_lookup", default=None)

    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", default=None)
    ap.add_argument("--cvae_ckpt", default=None)
    ap.add_argument("--tokenizer_ckpt", required=True)

    ap.add_argument("--runtime_vocab_json", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--eval_every_steps", type=int, default=500)
    ap.add_argument("--max_eval_batches", type=int, default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--eval_only", action="store_true")

    ap.add_argument("--max_windows", type=int, default=32)
    ap.add_argument("--max_chunks_per_window", type=int, default=4)
    ap.add_argument("--max_len_per_window", type=int, default=128)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--min_lr_scale", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--disable_amp", action="store_true")
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--num_local_layers", type=int, default=2)
    ap.add_argument("--num_global_layers", type=int, default=2)
    ap.add_argument("--num_chunk_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--enable_time_embedding", dest="enable_time_embedding", action="store_true")
    ap.add_argument("--disable_time_embedding", dest="enable_time_embedding", action="store_false")
    ap.add_argument("--enable_chunk_meta_sidechannel", dest="enable_chunk_meta_sidechannel", action="store_true")
    ap.add_argument("--disable_chunk_meta_sidechannel", dest="enable_chunk_meta_sidechannel", action="store_false")
    ap.add_argument("--enable_window_sequence_meta", dest="enable_window_sequence_meta", action="store_true")
    ap.add_argument("--disable_window_sequence_meta", dest="enable_window_sequence_meta", action="store_false")
    ap.add_argument("--condition_numeric_on_token_type", dest="condition_numeric_on_token_type", action="store_true")
    ap.add_argument("--disable_numeric_type_conditioning", dest="condition_numeric_on_token_type", action="store_false")
    ap.add_argument("--numeric_value_transform", choices=["identity", "signed_log1p"], default="signed_log1p")
    ap.add_argument("--time_embedding_max_hours", type=float, default=31.0 * 24.0)
    ap.add_argument("--local_time_embedding_max_hours", type=float, default=72.0)
    ap.add_argument("--global_time_embedding_max_hours", type=float, default=365.25 * 24.0 * 10.0)
    ap.add_argument("--disable_transition_bias", action="store_true")
    ap.add_argument("--emit_switched_heads", action="store_true")
    ap.add_argument("--global_context_mode", choices=["transformer", "latent_state"], default="latent_state")
    ap.add_argument(
        "--objective_preset",
        choices=sorted(OBJECTIVE_PRESETS.keys()),
        default="world_model_mttee",
    )
    ap.add_argument("--prefer_unified_token_loss", dest="prefer_unified_token_loss", action="store_true")
    ap.add_argument("--disable_prefer_unified_token_loss", dest="prefer_unified_token_loss", action="store_false")
    ap.add_argument("--carry_state_across_segments", dest="carry_state_across_segments", action="store_true")
    ap.add_argument("--disable_carry_state_across_segments", dest="carry_state_across_segments", action="store_false")
    ap.add_argument("--enable_exact_memory", dest="enable_exact_memory", action="store_true")
    ap.add_argument("--disable_exact_memory", dest="enable_exact_memory", action="store_false")
    ap.add_argument("--enable_precedent_memory", dest="enable_precedent_memory", action="store_true")
    ap.add_argument("--disable_precedent_memory", dest="enable_precedent_memory", action="store_false")
    ap.add_argument("--precedent_index_path", default=None)
    ap.add_argument("--precedent_retrieve_k", type=int, default=4)
    ap.add_argument(
        "--precedent_strict_window_type_match",
        dest="precedent_strict_window_type_match",
        action="store_true",
    )
    ap.add_argument(
        "--disable_precedent_strict_window_type_match",
        dest="precedent_strict_window_type_match",
        action="store_false",
    )
    ap.add_argument("--precedent_support_overlap_bias", type=float, default=0.25)
    ap.add_argument("--precedent_score_temperature", type=float, default=1.0)
    ap.add_argument("--exact_memory_slots", type=int, default=16)
    ap.add_argument("--exact_memory_static_slots", type=int, default=2)
    ap.add_argument("--exact_memory_persistent_slots", type=int, default=16)
    ap.add_argument("--exact_memory_episodic_slots", type=int, default=16)
    ap.add_argument("--exact_memory_write_per_window", type=int, default=2)
    ap.add_argument("--exact_memory_retrieve_k", type=int, default=4)
    ap.add_argument("--exact_memory_static_retrieve_k", type=int, default=2)
    ap.add_argument("--exact_memory_persistent_retrieve_k", type=int, default=4)
    ap.add_argument("--exact_memory_episodic_retrieve_k", type=int, default=4)
    ap.add_argument("--exact_memory_static_feature_ids", type=str, default="1,2")
    ap.add_argument("--exact_memory_max_same_group", type=int, default=2)
    ap.add_argument("--exact_memory_age_decay", type=float, default=0.05)
    ap.add_argument("--exact_memory_rule_write_scale", type=float, default=1.0)
    ap.add_argument("--exact_memory_rule_retrieval_scale", type=float, default=0.25)
    ap.add_argument("--exact_memory_learned_write_scale", type=float, default=1.0)
    ap.add_argument("--exact_memory_first_occurrence_bonus", type=float, default=0.75)
    ap.add_argument("--exact_memory_chronic_bonus", type=float, default=1.5)
    ap.add_argument("--enable_event_time_nll_head", dest="enable_event_time_nll_head", action="store_true")
    ap.add_argument("--disable_event_time_nll_head", dest="enable_event_time_nll_head", action="store_false")
    ap.add_argument("--enable_next_window_gap_nll_head", dest="enable_next_window_gap_nll_head", action="store_true")
    ap.add_argument("--disable_next_window_gap_nll_head", dest="enable_next_window_gap_nll_head", action="store_false")
    ap.add_argument(
        "--enable_next_window_duration_nll_head",
        dest="enable_next_window_duration_nll_head",
        action="store_true",
    )
    ap.add_argument(
        "--disable_next_window_duration_nll_head",
        dest="enable_next_window_duration_nll_head",
        action="store_false",
    )
    ap.add_argument(
        "--enable_next_window_support_head",
        dest="enable_next_window_support_head",
        action="store_true",
    )
    ap.add_argument(
        "--disable_next_window_support_head",
        dest="enable_next_window_support_head",
        action="store_false",
    )
    ap.set_defaults(
        enable_time_embedding=True,
        enable_chunk_meta_sidechannel=True,
        enable_window_sequence_meta=True,
        condition_numeric_on_token_type=True,
        carry_state_across_segments=True,
        enable_exact_memory=True,
        enable_precedent_memory=False,
        precedent_strict_window_type_match=True,
        enable_event_time_nll_head=True,
        enable_next_window_gap_nll_head=True,
        enable_next_window_duration_nll_head=True,
        enable_next_window_support_head=True,
        prefer_unified_token_loss=None,
    )

    ap.add_argument("--token_loss_weight", type=float, default=1.0)
    ap.add_argument("--event_token_loss_weight", type=float, default=1.0)
    ap.add_argument("--event_family_loss_weight", type=float, default=1.0)
    ap.add_argument("--event_payload_loss_weight", type=float, default=1.0)
    ap.add_argument("--event_concept_loss_weight", type=float, default=1.0)
    ap.add_argument("--event_dt_loss_weight", type=float, default=1.0)
    ap.add_argument("--event_value_loss_weight", type=float, default=1.0)
    ap.add_argument(
        "--token_family_weight_preset",
        choices=sorted(TOKEN_FAMILY_WEIGHT_PRESETS.keys()),
        default="none",
    )
    ap.add_argument(
        "--token_family_weight",
        action="append",
        default=[],
        help="Optional per-family unified token loss reweighting, e.g. diagnosis=3.0",
    )
    ap.add_argument("--value_loss_weight", type=float, default=0.0)
    ap.add_argument("--transition_loss_weight", type=float, default=1.0)
    ap.add_argument("--win_boundary_loss_weight", type=float, default=1.0)
    ap.add_argument("--win_loss_weight", type=float, default=0.5)
    ap.add_argument("--len_loss_weight", type=float, default=0.0)
    ap.add_argument("--chunk_loss_weight", type=float, default=0.0)
    ap.add_argument("--time_loss_weight", type=float, default=0.0)
    ap.add_argument("--dt_loss_weight", type=float, default=0.0)
    ap.add_argument("--next_window_gap_loss_weight", type=float, default=1.0)
    ap.add_argument("--next_window_duration_loss_weight", type=float, default=1.0)
    ap.add_argument("--next_window_support_loss_weight", type=float, default=0.5)
    ap.add_argument("--precedent_future_loss_weight", type=float, default=1.0)
    ap.add_argument("--precedent_contrast_loss_weight", type=float, default=1.0)
    ap.add_argument("--precedent_anchor_loss_weight", type=float, default=0.25)
    ap.add_argument("--precedent_loss_start_step", type=int, default=None)
    ap.add_argument("--precedent_loss_ramp_steps", type=int, default=None)

    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=39999)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)

    raw_argv = tuple(sys.argv[1:])
    args = ap.parse_args()
    objective_cfg = resolve_objective_configuration(args=args, raw_argv=raw_argv)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _resolve_device(str(args.device))
    autocast_enabled = device.type == "cuda" and not bool(args.disable_amp)
    autocast_dtype = torch.bfloat16 if str(args.amp_dtype) == "bf16" else torch.float16

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_jsonl = output_dir / "train_metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    trajectory_split_cfg = _build_trajectory_split_config(
        mode=str(args.trajectory_mode),
        post_discharge_cutoff_days=float(args.post_discharge_cutoff_days),
    )

    collator = AETHierarchicalCollator(
        max_windows=int(args.max_windows),
        max_chunks_per_window=int(args.max_chunks_per_window),
        max_len_per_window=int(args.max_len_per_window),
        window_markers=window_markers_cfg,
        segmentation=segmentation_cfg,
        id_remapper=remapper,
    )
    collate_fn = TimelineCollateAdapter(collator)

    train_loader: DataLoader
    eval_loader: DataLoader | None = None

    precompiled_train_root = args.precompiled_train_root or args.precompiled_root
    precompiled_eval_root = args.precompiled_eval_root or args.precompiled_root
    pipeline_summary: Dict[str, Any] = {}
    if precompiled_train_root:
        train_loader_kwargs = build_dataloader_kwargs(
            device=device,
            num_workers=int(args.num_workers),
            precompiled=True,
            prefetch_factor=int(args.prefetch_factor),
        )
        train_ds = PrecompiledMEDSDataset(
            precompiled_train_root,
            split=str(args.train_split),
            splits_parquet=str(args.splits_parquet),
            index_filename=("trajectory_index.csv" if str(args.trajectory_mode) == "admission_chain" else "index.csv"),
            segmentation_config=segmentation_cfg,
            trajectory_split_config=trajectory_split_cfg,
            return_metadata=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=collate_fn,
            **train_loader_kwargs,
        )
        pipeline_summary = {
            "mode": "precompiled",
            "train_root": str(precompiled_train_root),
            "eval_root": str(precompiled_eval_root or precompiled_train_root),
            "trajectory_mode": str(args.trajectory_mode),
            "carry_state_across_segments": bool(args.carry_state_across_segments),
            "resolved_num_workers": int(train_loader_kwargs["num_workers"]),
            "prefetch_factor": int(train_loader_kwargs.get("prefetch_factor", 0)),
            "persistent_workers": bool(train_loader_kwargs.get("persistent_workers", False)),
            "train_subject_count": int(len(train_ds)),
        }
        if args.eval_split:
            eval_root = precompiled_eval_root or precompiled_train_root
            eval_loader_kwargs = build_dataloader_kwargs(
                device=device,
                num_workers=int(args.num_workers),
                precompiled=True,
                prefetch_factor=int(args.prefetch_factor),
            )
            eval_ds = PrecompiledMEDSDataset(
                eval_root,
                split=str(args.eval_split),
                splits_parquet=str(args.splits_parquet),
                index_filename=("trajectory_index.csv" if str(args.trajectory_mode) == "admission_chain" else "index.csv"),
                segmentation_config=segmentation_cfg,
                trajectory_split_config=trajectory_split_cfg,
                return_metadata=True,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=int(args.eval_batch_size),
                shuffle=False,
                collate_fn=collate_fn,
                **eval_loader_kwargs,
            )
            pipeline_summary["eval_subject_count"] = int(len(eval_ds))
    else:
        if not args.meds_reader_db:
            raise ValueError("--meds_reader_db is required unless --precompiled_root is provided.")
        if int(args.num_workers) != 0:
            raise ValueError("On-the-fly timeline building currently requires --num_workers 0.")
        if str(args.trajectory_mode) != "full_subject":
            raise ValueError(
                "On-the-fly timeline building does not support --trajectory_mode admission_chain; use precompiled trajectories."
            )
        residual_enabled, residual_buckets, residual_offsets = _resolve_residual_policy(
            args,
            tokenization_contract=tokenization_contract,
        )
        residual_tail_policies = _resolve_residual_tail_policies(
            tokenization_contract=tokenization_contract,
        )
        struct_codes_union = set(structural_surface_vocab_codes(artifacts.structural_codebook))
        struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)
        meas_cfg = _build_measurement_config(args, artifacts=artifacts)
        if meas_cfg is None:
            raise ValueError(
                "Missing measurement artifacts; need code2id/stats/cvae/tokenizer checkpoints."
            )
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
            residual_tail_policies=residual_tail_policies,
        )
        train_subject_ids = _load_subject_ids(
            str(args.splits_parquet),
            str(args.train_split),
            int(args.max_train_subjects) if args.max_train_subjects is not None else None,
            sample_seed=int(args.seed),
        )
        train_ds = OnTheFlyTimelineDataset(
            db_path=str(args.meds_reader_db),
            subject_ids=train_subject_ids,
            encoders=encoders,
            structural_codebook=artifacts.structural_codebook,
            qual_obs_code_vocab=artifacts.obs_code_vocab,
            qual_obs_value_vocab=artifacts.obs_value_vocab,
            qual_obs_tail_policy=artifacts.obs_tail_policy,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        pipeline_summary = {
            "mode": "on_the_fly",
            "trajectory_mode": "full_subject",
            "carry_state_across_segments": bool(args.carry_state_across_segments),
            "resolved_num_workers": 0,
            "train_subject_count": int(len(train_ds)),
        }
        if args.eval_split:
            eval_subject_ids = _load_subject_ids(
                str(args.splits_parquet),
                str(args.eval_split),
                int(args.max_eval_subjects) if args.max_eval_subjects is not None else None,
                sample_seed=int(args.seed),
            )
            eval_ds = OnTheFlyTimelineDataset(
                db_path=str(args.meds_reader_db),
                subject_ids=eval_subject_ids,
                encoders=encoders,
                structural_codebook=artifacts.structural_codebook,
                qual_obs_code_vocab=artifacts.obs_code_vocab,
                qual_obs_value_vocab=artifacts.obs_value_vocab,
                qual_obs_tail_policy=artifacts.obs_tail_policy,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=int(args.eval_batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=(device.type == "cuda"),
            )
            pipeline_summary["eval_subject_count"] = int(len(eval_ds))

    model_cfg = TrainModelConfig(
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        d_ff=int(args.d_ff),
        num_local_layers=int(args.num_local_layers),
        num_global_layers=int(args.num_global_layers),
        num_chunk_layers=int(args.num_chunk_layers),
        dropout=float(args.dropout),
        enable_time_embedding=bool(args.enable_time_embedding),
        time_embedding_max_hours=float(args.time_embedding_max_hours),
        local_time_embedding_max_hours=float(args.local_time_embedding_max_hours),
        global_time_embedding_max_hours=float(args.global_time_embedding_max_hours),
        enable_chunk_meta_sidechannel=bool(args.enable_chunk_meta_sidechannel),
        enable_window_sequence_meta=bool(args.enable_window_sequence_meta),
        condition_numeric_on_token_type=bool(args.condition_numeric_on_token_type),
        numeric_value_transform=str(args.numeric_value_transform),
        global_context_mode=str(args.global_context_mode),
        enable_transition_bias=not bool(args.disable_transition_bias),
        emit_switched_heads=bool(args.emit_switched_heads),
        carry_state_across_segments=bool(args.carry_state_across_segments),
        enable_exact_memory=bool(args.enable_exact_memory),
        enable_precedent_memory=bool(args.enable_precedent_memory or args.precedent_index_path),
        exact_memory_slots=int(args.exact_memory_slots),
        exact_memory_static_slots=int(args.exact_memory_static_slots),
        exact_memory_persistent_slots=int(args.exact_memory_persistent_slots),
        exact_memory_episodic_slots=int(args.exact_memory_episodic_slots),
        exact_memory_write_per_window=int(args.exact_memory_write_per_window),
        exact_memory_retrieve_k=int(args.exact_memory_retrieve_k),
        exact_memory_static_retrieve_k=int(args.exact_memory_static_retrieve_k),
        exact_memory_persistent_retrieve_k=int(args.exact_memory_persistent_retrieve_k),
        exact_memory_episodic_retrieve_k=int(args.exact_memory_episodic_retrieve_k),
        exact_memory_static_feature_ids=str(args.exact_memory_static_feature_ids),
        exact_memory_max_same_group=int(args.exact_memory_max_same_group),
        exact_memory_age_decay=float(args.exact_memory_age_decay),
        exact_memory_rule_write_scale=float(args.exact_memory_rule_write_scale),
        exact_memory_rule_retrieval_scale=float(args.exact_memory_rule_retrieval_scale),
        exact_memory_learned_write_scale=float(args.exact_memory_learned_write_scale),
        exact_memory_first_occurrence_bonus=float(args.exact_memory_first_occurrence_bonus),
        exact_memory_chronic_bonus=float(args.exact_memory_chronic_bonus),
        precedent_retrieve_k=int(args.precedent_retrieve_k),
        precedent_strict_window_type_match=bool(args.precedent_strict_window_type_match),
        precedent_support_overlap_bias=float(args.precedent_support_overlap_bias),
        precedent_score_temperature=float(args.precedent_score_temperature),
        enable_event_time_nll_head=bool(args.enable_event_time_nll_head),
        enable_next_window_gap_nll_head=bool(args.enable_next_window_gap_nll_head),
        enable_next_window_duration_nll_head=bool(args.enable_next_window_duration_nll_head),
        enable_next_window_support_head=bool(args.enable_next_window_support_head),
    )
    token_family_weights = resolve_token_family_weights(
        preset=str(objective_cfg["token_family_weight_preset"]),
        overrides=args.token_family_weight,
    )
    model = AdaptiveEpisodicTransformer(model_cfg, vocab_config).to(device)
    criterion = AETLossModule(
        vocab_config=vocab_config,
        token_family_weights=token_family_weights,
        strict_routing=True,
        prefer_unified_token_loss=bool(objective_cfg["prefer_unified_token_loss"]),
        weights=dict(objective_cfg["loss_weights"]),
    ).to(device)
    base_loss_weights = dict(objective_cfg["loss_weights"])
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, float(args.weight_decay)),
        lr=float(args.lr),
    )
    total_steps_est = int(args.max_steps) if args.max_steps is not None else int(len(train_loader) * max(1, int(args.epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(
            total_steps=max(1, total_steps_est),
            warmup_steps=int(args.warmup_steps),
            min_lr_scale=float(args.min_lr_scale),
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not bool(args.disable_amp)))

    start_epoch = 1
    global_step = 0
    best_val_loss: float | None = None
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = ckpt.get("best_val_loss", None)
    if args.precedent_index_path:
        model.load_precedent_index(args.precedent_index_path, map_location="cpu")

    run_meta = {
        "args": vars(args),
        "objective": objective_cfg,
        "token_family_weights": token_family_weights,
        "model_config": asdict(model_cfg),
        "input_pipeline": pipeline_summary,
        "vocab_summary": {
            "total_size": int(vocab_config["total_size"]),
            "size_special": int(vocab_config["size_special"]),
            "size_rvq": int(vocab_config["size_rvq"]),
            "size_meas_labels": int(vocab_config["size_meas_labels"]),
            "size_meds": int(vocab_config["size_meds"]),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if args.eval_only:
        if not args.resume_from:
            raise ValueError("--eval_only requires --resume_from to load a trained checkpoint.")
        if eval_loader is None:
            raise ValueError("--eval_only requires a non-empty eval loader.")
        latest_val_metrics = evaluate(
            model=model,
            criterion=criterion,
            base_loss_weights=base_loss_weights,
            dataloader=eval_loader,
            device=device,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            max_batches=args.max_eval_batches,
            carry_across_segments=bool(args.carry_state_across_segments),
            global_step=global_step,
            precedent_loss_start_step=int(objective_cfg["precedent_loss_start_step"]),
            precedent_loss_ramp_steps=int(objective_cfg["precedent_loss_ramp_steps"]),
        )
        eval_payload = {
            "event": "eval_only",
            "checkpoint": str(args.resume_from),
            "epoch": int(start_epoch - 1),
            "global_step": int(global_step),
            "best_val_loss": best_val_loss,
            "val": latest_val_metrics,
        }
        print(json.dumps(eval_payload, indent=2))
        (output_dir / "eval_only_metrics.json").write_text(
            json.dumps(eval_payload, indent=2),
            encoding="utf-8",
        )
        with log_jsonl.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(eval_payload) + "\n")
        return

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    max_steps = max(1, int(args.max_steps))
    latest_train_metrics: Dict[str, float] | None = None
    latest_val_metrics: Dict[str, float] | None = None

    for epoch in resolve_epoch_range(start_epoch=start_epoch, epochs_to_run=int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_log_acc: Dict[str, float] = {}
        epoch_batches = 0
        stopped_on_max_steps = False
        carry_cache: Dict[int, Dict[str, Any]] = {}
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch_idx, batch in enumerate(pbar, start=1):
            tensor_batch = _move_batch_to_device(batch, device)
            scheduled_weights, schedule_logs = resolve_scheduled_loss_weights(
                base_weights=base_loss_weights,
                global_step=global_step,
                precedent_loss_start_step=int(objective_cfg["precedent_loss_start_step"]),
                precedent_loss_ramp_steps=int(objective_cfg["precedent_loss_ramp_steps"]),
            )
            criterion.set_weights(scheduled_weights)
            prev_global_state, prev_memory_state = _resolve_carry_inputs(
                tensor_batch=tensor_batch,
                model=model,
                carry_cache=carry_cache if bool(args.carry_state_across_segments) else None,
            )
            loss, logs, _, aux_state = _run_model_and_loss(
                model=model,
                criterion=criterion,
                tensor_batch=tensor_batch,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                prev_global_state=prev_global_state,
                prev_memory_state=prev_memory_state,
                return_aux_state=bool(args.carry_state_across_segments),
            )
            _update_carry_cache(
                tensor_batch=tensor_batch,
                aux_state=aux_state,
                carry_cache=carry_cache if bool(args.carry_state_across_segments) else None,
            )
            scaler.scale(loss / float(grad_accum_steps)).backward()

            epoch_loss_sum += float(loss.detach().cpu().item())
            _accumulate_logs(epoch_log_acc, logs)
            epoch_batches += 1
            pbar.set_postfix(
                loss=f"{float(loss.detach().cpu().item()):.4f}",
                token=f"{logs.get('loss_token', 0.0):.4f}",
                acc=f"{logs.get('acc_token', 0.0):.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == len(train_loader))
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                .detach()
                .cpu()
                .item()
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            step_metrics = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            step_metrics.update(schedule_logs)
            step_metrics.update(logs)
            with log_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(step_metrics) + "\n")

            if eval_loader is not None and int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
                latest_val_metrics = evaluate(
                    model=model,
                    criterion=criterion,
                    base_loss_weights=base_loss_weights,
                    dataloader=eval_loader,
                    device=device,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    max_batches=args.max_eval_batches,
                    carry_across_segments=bool(args.carry_state_across_segments),
                    global_step=global_step,
                    precedent_loss_start_step=int(objective_cfg["precedent_loss_start_step"]),
                    precedent_loss_ramp_steps=int(objective_cfg["precedent_loss_ramp_steps"]),
                )
                val_loss = float(latest_val_metrics.get("loss", float("inf")))
                if best_val_loss is None or val_loss < float(best_val_loss):
                    best_val_loss = val_loss
                    _save_checkpoint(
                        path=ckpt_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        args=args,
                        model_config=model_cfg,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        latest_train_metrics=step_metrics,
                        latest_val_metrics=latest_val_metrics,
                    )

            if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                _save_checkpoint(
                    path=ckpt_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    model_config=model_cfg,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    latest_train_metrics=step_metrics,
                    latest_val_metrics=latest_val_metrics,
                )

            if global_step >= max_steps:
                stopped_on_max_steps = True
                break

        latest_train_metrics = {
            "loss": float(epoch_loss_sum) / float(max(1, epoch_batches)),
            **_mean_logs(epoch_log_acc, epoch_batches),
        }
        should_run_final_eval = (
            eval_loader is not None
            and not stopped_on_max_steps
            and (int(args.eval_every_steps) <= 0 or global_step % int(args.eval_every_steps) != 0)
        )
        if should_run_final_eval:
            latest_val_metrics = evaluate(
                model=model,
                criterion=criterion,
                base_loss_weights=base_loss_weights,
                dataloader=eval_loader,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                max_batches=args.max_eval_batches,
                carry_across_segments=bool(args.carry_state_across_segments),
                global_step=global_step,
                precedent_loss_start_step=int(objective_cfg["precedent_loss_start_step"]),
                precedent_loss_ramp_steps=int(objective_cfg["precedent_loss_ramp_steps"]),
            )
            val_loss = float(latest_val_metrics.get("loss", float("inf")))
            if best_val_loss is None or val_loss < float(best_val_loss):
                best_val_loss = val_loss
                _save_checkpoint(
                    path=ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    model_config=model_cfg,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    latest_train_metrics=latest_train_metrics,
                    latest_val_metrics=latest_val_metrics,
                )

        _save_checkpoint(
            path=ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            model_config=model_cfg,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            latest_train_metrics=latest_train_metrics,
            latest_val_metrics=latest_val_metrics,
        )

        epoch_summary = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train": latest_train_metrics,
            "val": latest_val_metrics,
            "best_val_loss": best_val_loss,
            "stopped_on_max_steps": bool(stopped_on_max_steps),
        }
        print(json.dumps(epoch_summary, indent=2))
        with log_jsonl.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"event": "epoch_end", **epoch_summary}) + "\n")

        if global_step >= max_steps:
            break


if __name__ == "__main__":
    main()
