#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    _build_window_marker_config,
    _load_subject_ids,
    _load_tokenization_contract,
    _resolve_residual_policy,
)
from src.ehr_hier.data.dataset import PrecompiledMEDSDataset  # noqa: E402
from src.ehr_hier.data.structural_codes import structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline  # noqa: E402
from src.ehr_hier.data.token_types import EventToken, TokenCategory  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402
from src.ehr_hier.transformer.collator import AETHierarchicalCollator  # noqa: E402
from src.ehr_hier.transformer.loss import AETLossModule  # noqa: E402
from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer  # noqa: E402
from src.ehr_hier.transformer.vocab_runtime import (  # noqa: E402
    build_runtime_vocab_and_remapper,
    load_runtime_vocab_bundle,
)


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
    enable_time_embedding: bool = False
    global_fusion_mode: str = "add"
    exclude_special_from_global_fusion: bool = True
    use_unified_token_head: bool = True
    emit_switched_heads: bool = False


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

    def __getitem__(self, idx: int) -> List[EventToken] | None:
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
        return timeline if timeline else None


class TimelineCollateAdapter:
    def __init__(self, collator: AETHierarchicalCollator) -> None:
        self.collator = collator

    def __call__(self, batch: List[List[EventToken] | None]) -> Dict[str, Any]:
        timelines = [timeline for timeline in batch if timeline]
        if not timelines:
            raise ValueError("All timelines in batch were empty after on-the-fly building.")
        return self.collator(timelines)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


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
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=bool(autocast_enabled),
    ):
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
        raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu().item())}")
    return loss, {k: float(v) for k, v in logs.items()}, head_outputs


def evaluate(
    *,
    model: AdaptiveEpisodicTransformer,
    criterion: AETLossModule,
    dataloader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    log_acc: Dict[str, float] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            tensor_batch = _move_batch_to_device(batch, device)
            loss, logs, _ = _run_model_and_loss(
                model=model,
                criterion=criterion,
                tensor_batch=tensor_batch,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            total_loss += float(loss.detach().cpu().item())
            _accumulate_logs(log_acc, logs)
            n_batches += 1
    if n_batches == 0:
        return {"loss": float("nan")}
    out = {"loss": total_loss / float(n_batches)}
    out.update(_mean_logs(log_acc, n_batches))
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
            "Uses the current tokenization/windowing contract and a unified autoregressive token loss."
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

    ap.add_argument("--max_windows", type=int, default=32)
    ap.add_argument("--max_chunks_per_window", type=int, default=4)
    ap.add_argument("--max_len_per_window", type=int, default=128)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
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
    ap.add_argument("--enable_time_embedding", action="store_true")
    ap.add_argument("--disable_transition_bias", action="store_true")
    ap.add_argument("--emit_switched_heads", action="store_true")

    ap.add_argument("--token_loss_weight", type=float, default=1.0)
    ap.add_argument("--value_loss_weight", type=float, default=0.0)
    ap.add_argument("--transition_loss_weight", type=float, default=1.0)
    ap.add_argument("--win_boundary_loss_weight", type=float, default=1.0)
    ap.add_argument("--win_loss_weight", type=float, default=0.5)
    ap.add_argument("--len_loss_weight", type=float, default=0.0)
    ap.add_argument("--chunk_loss_weight", type=float, default=0.0)
    ap.add_argument("--time_loss_weight", type=float, default=0.0)
    ap.add_argument("--dt_loss_weight", type=float, default=0.0)

    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=39999)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)

    args = ap.parse_args()

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
    if precompiled_train_root:
        train_ds = PrecompiledMEDSDataset(
            precompiled_train_root,
            split=str(args.train_split),
            splits_parquet=str(args.splits_parquet),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        if args.eval_split:
            eval_root = precompiled_eval_root or precompiled_train_root
            eval_ds = PrecompiledMEDSDataset(
                eval_root,
                split=str(args.eval_split),
                splits_parquet=str(args.splits_parquet),
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=int(args.eval_batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                collate_fn=collate_fn,
                pin_memory=(device.type == "cuda"),
            )
    else:
        if not args.meds_reader_db:
            raise ValueError("--meds_reader_db is required unless --precompiled_root is provided.")
        if int(args.num_workers) != 0:
            raise ValueError("On-the-fly timeline building currently requires --num_workers 0.")
        residual_enabled, residual_buckets, residual_offsets = _resolve_residual_policy(
            args,
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

    model_cfg = TrainModelConfig(
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        d_ff=int(args.d_ff),
        num_local_layers=int(args.num_local_layers),
        num_global_layers=int(args.num_global_layers),
        num_chunk_layers=int(args.num_chunk_layers),
        dropout=float(args.dropout),
        enable_time_embedding=bool(args.enable_time_embedding),
        enable_transition_bias=not bool(args.disable_transition_bias),
        emit_switched_heads=bool(args.emit_switched_heads),
    )
    model = AdaptiveEpisodicTransformer(model_cfg, vocab_config).to(device)
    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={
            "token": float(args.token_loss_weight),
            "val": float(args.value_loss_weight),
            "transition": float(args.transition_loss_weight),
            "win_boundary": float(args.win_boundary_loss_weight),
            "win": float(args.win_loss_weight),
            "len": float(args.len_loss_weight),
            "chunk": float(args.chunk_loss_weight),
            "time": float(args.time_loss_weight),
            "dt": float(args.dt_loss_weight),
        },
    ).to(device)
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

    run_meta = {
        "args": vars(args),
        "model_config": asdict(model_cfg),
        "vocab_summary": {
            "total_size": int(vocab_config["total_size"]),
            "size_special": int(vocab_config["size_special"]),
            "size_rvq": int(vocab_config["size_rvq"]),
            "size_meas_labels": int(vocab_config["size_meas_labels"]),
            "size_meds": int(vocab_config["size_meds"]),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    max_steps = max(1, int(args.max_steps))
    latest_train_metrics: Dict[str, float] | None = None
    latest_val_metrics: Dict[str, float] | None = None

    for epoch in range(start_epoch, max(1, int(args.epochs)) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_log_acc: Dict[str, float] = {}
        epoch_batches = 0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch_idx, batch in enumerate(pbar, start=1):
            tensor_batch = _move_batch_to_device(batch, device)
            loss, logs, _ = _run_model_and_loss(
                model=model,
                criterion=criterion,
                tensor_batch=tensor_batch,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
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
            step_metrics.update(logs)
            with log_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(step_metrics) + "\n")

            if eval_loader is not None and int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
                latest_val_metrics = evaluate(
                    model=model,
                    criterion=criterion,
                    dataloader=eval_loader,
                    device=device,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    max_batches=args.max_eval_batches,
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
                break

        latest_train_metrics = {
            "loss": float(epoch_loss_sum) / float(max(1, epoch_batches)),
            **_mean_logs(epoch_log_acc, epoch_batches),
        }
        if eval_loader is not None and (int(args.eval_every_steps) <= 0 or global_step % int(args.eval_every_steps) != 0):
            latest_val_metrics = evaluate(
                model=model,
                criterion=criterion,
                dataloader=eval_loader,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                max_batches=args.max_eval_batches,
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
        }
        print(json.dumps(epoch_summary, indent=2))
        with log_jsonl.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"event": "epoch_end", **epoch_summary}) + "\n")

        if global_step >= max_steps:
            break


if __name__ == "__main__":
    main()
