#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meds_reader as mr

from src.ehr_hier.data.demographics import (
    infer_birth_timestamp,
    infer_event_age_years,
    infer_subject_sex,
)
from src.ehr_hier.data.event_frames import EventFrame, flatten_event_frames
from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.structural_codes import (
    StructuralCodebook,
    load_structural_codebook_yaml,
    structural_surface_code,
    structural_surface_vocab_codes,
)
from src.ehr_hier.data.subject_timeline_builder import (
    GLOBAL_DEMOGRAPHIC_TOKEN_IDS,
    build_subject_timeline,
)
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders
from src.ehr_hier.tokenizers.decode_tokens import decode_timeline_tokens, invert_code2id
from src.ehr_hier.tokenizers.measurement_encoder import MeasurementEncoderConfig
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_medication_code,
    canonicalize_procedure_code,
    diagnosis_filter,
    medication_filter,
    procedure_filter,
)
from src.ehr_hier.tokenizers.medtok_loader import (
    build_vocab_from_code2embeddings,
    CategoryVocab,
    load_attr_vocab,
    load_medtok_vocab,
    load_observation_vocab,
    load_residual_fallback_vocab,
)
from src.ehr_hier.data.observation_vocab import observation_surfaces
from src.ehr_hier.tokenizers.medtok_attr_encoder import (
    EXPLICIT_MEDTOK_RESOLUTION_STAGES,
    MedTokenWithAttrsEncoder,
    load_parent_lookup_from_codes_parquet,
)
from src.ehr_hier.tokenizers.medtok_crosswalk import (
    load_crosswalk_candidate_map,
    load_resolved_crosswalk_lookup,
)
from src.ehr_hier.tokenizers.vocab_contract import (
    DEFAULT_SPARSE_VOCAB_JSON,
    build_legacy_manifest_from_sparse_contract,
    load_sparse_vocab_contract,
    validate_medtok_inputs,
)
from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig
from src.ehr_hier.data.window_segmentation import WindowSegmentationConfig
from src.ehr_hier.data.trajectory_splitting import TrajectorySplitConfig


SPECIAL_ID2NAME = {
    0: "PAD",
    1: "PT_CLS",
    2: "SEP",
    3: "MASK",
    **{int(v): str(k) for k, v in GLOBAL_DEMOGRAPHIC_TOKEN_IDS.items()},
}


@dataclass
class AuditArtifacts:
    manifest: Dict[str, Any]
    structural_codebook: Optional[StructuralCodebook]
    code2id: Dict[str, int] | None
    measurement_num_codebooks: int | None
    measurement_codebook_size: int | None
    measurement_stride: int | None
    diag_vocab: CategoryVocab
    proc_vocab: CategoryVocab
    med_vocab: CategoryVocab
    med_attr_vocabs: Dict[str, CategoryVocab]
    med_numeric_attrs: Dict[str, NumericBinConfig]
    medtok_parent_lookup: Dict[str, List[str]]
    medtok_crosswalks: Dict[str, Dict[str, str]]
    residual_fallback_vocabs: Dict[str, CategoryVocab]
    obs_code_vocab: Optional[CategoryVocab]
    obs_value_vocab: Optional[CategoryVocab]
    obs_tail_policy: str


class _EventWithDemographics:
    __slots__ = ("_ev", "age_years", "sex")

    def __init__(self, ev: object, *, age_years: float, sex: float) -> None:
        self._ev = ev
        self.age_years = float(age_years)
        self.sex = float(sex)

    def __getattr__(self, name: str):
        return getattr(self._ev, name)


def _offset(manifest: Mapping[str, Any], key: str, default: int) -> int:
    return int(manifest.get(key, {}).get("offset", default))


def _top_counter(counter: Counter[str] | Counter[int], top_k: int) -> List[Dict[str, Any]]:
    return [{"key": str(key), "count": int(count)} for key, count in counter.most_common(top_k)]


def _as_plain_counter(counter: Counter[Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def _load_manifest(sparse_vocab_json: Optional[str] = None) -> Dict[str, Any]:
    sparse_candidates: List[Path] = []
    if sparse_vocab_json:
        fp = Path(str(sparse_vocab_json))
        sparse_candidates.append(fp if fp.is_absolute() else PROJECT_ROOT / fp)
    sparse_candidates.append(PROJECT_ROOT / DEFAULT_SPARSE_VOCAB_JSON)
    for fp in sparse_candidates:
        if fp.exists():
            return build_legacy_manifest_from_sparse_contract(load_sparse_vocab_contract(fp))
    fp = PROJECT_ROOT / "artifacts" / "vocab_manifest.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def _load_tokenization_contract(tokenization_yaml: Optional[str]) -> Dict[str, Any]:
    if tokenization_yaml is None or not str(tokenization_yaml).strip():
        return {}
    fp = Path(str(tokenization_yaml))
    if not fp.is_absolute():
        fp = PROJECT_ROOT / fp
    if not fp.exists():
        return {}
    payload = yaml.safe_load(fp.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_observation_tail_policy(
    *,
    tokenization_contract: Mapping[str, Any],
) -> str:
    cfg = tokenization_contract.get("qualitative_observation", {})
    if not isinstance(cfg, dict):
        cfg = tokenization_contract.get("observation", {})
    if not isinstance(cfg, dict):
        cfg = {}
    tail_policy = str(cfg.get("tail_policy", "drop")).strip().lower() or "drop"
    if tail_policy not in {"drop", "hash"}:
        raise ValueError(
            f"Unsupported qualitative observation tail policy {tail_policy!r}; expected drop|hash"
        )
    return tail_policy


def _build_window_marker_config(
    *,
    tokenization_contract: Mapping[str, Any],
    structural_codebook: Optional[StructuralCodebook],
) -> WindowMarkerConfig:
    cfg = tokenization_contract.get("window_markers", {})
    if not isinstance(cfg, dict):
        cfg = {}

    codebook_num_types: Optional[int] = None
    if structural_codebook is not None:
        type_map = structural_codebook.window_type2id()
        if type_map:
            codebook_num_types = max(int(v) for v in type_map.values()) + 1

    num_types = int(cfg.get("num_types", codebook_num_types if codebook_num_types is not None else 16))
    num_types = max(1, num_types)
    end_token_id = cfg.get("end_token_id", None)
    continue_token_id = cfg.get("continue_token_id", None)

    return WindowMarkerConfig(
        enabled=bool(cfg.get("enabled", True)),
        end_mode=str(cfg.get("end_mode", "end_token")),
        type_token_offset=int(cfg.get("type_token_offset", 10)),
        num_types=int(num_types),
        end_token_id=int(end_token_id) if end_token_id is not None else None,
        continue_token_id=int(continue_token_id) if continue_token_id is not None else None,
        unk_type_id=int(cfg.get("unk_type_id", 0)),
    )


def _build_segmentation_config(
    *,
    tokenization_contract: Mapping[str, Any],
    structural_codebook: Optional[StructuralCodebook],
    unk_type_id: int,
) -> WindowSegmentationConfig:
    cfg = tokenization_contract.get("window_segmentation", {})
    if not isinstance(cfg, dict):
        cfg = {}

    first_type_id = cfg.get("default_first_window_type_id", None)
    if first_type_id is None:
        first_type_name = cfg.get("default_first_window_type", None)
        if first_type_name is not None and structural_codebook is not None:
            first_type_id = structural_codebook.window_type2id().get(str(first_type_name))

    post_discharge_type_id = cfg.get("post_discharge_window_type_id", None)
    if post_discharge_type_id is None:
        post_discharge_type_name = cfg.get("post_discharge_window_type", None)
        if post_discharge_type_name is not None and structural_codebook is not None:
            post_discharge_type_id = structural_codebook.window_type2id().get(str(post_discharge_type_name))

    return WindowSegmentationConfig(
        bundle_gap_hours=float(cfg.get("bundle_gap_hours", 0.5)),
        bundle_max_index_gap=int(cfg.get("bundle_max_index_gap", 2)),
        merge_transition_chains=bool(cfg.get("merge_transition_chains", True)),
        chain_gap_hours=float(cfg.get("chain_gap_hours", 6.0)),
        chain_max_intervening_tokens=int(cfg.get("chain_max_intervening_tokens", 16)),
        rebalance_dense_windows=bool(cfg.get("rebalance_dense_windows", True)),
        rebalance_target_frac=float(cfg.get("rebalance_target_frac", 0.8)),
        rebalance_min_tokens=int(cfg.get("rebalance_min_tokens", 32)),
        rebalance_tail_tokens=int(cfg.get("rebalance_tail_tokens", 16)),
        unk_window_type_id=int(unk_type_id),
        default_first_window_type_id=(
            int(first_type_id) if first_type_id is not None else None
        ),
        post_discharge_window_type_id=(
            int(post_discharge_type_id) if post_discharge_type_id is not None else None
        ),
        propagate_prev_type_for_unknown_windows=bool(
            cfg.get("propagate_prev_type_for_unknown_windows", False)
        ),
        preserve_same_site_within_window=bool(
            cfg.get("preserve_same_site_within_window", True)
        ),
        site_change_starts_new_window=bool(
            cfg.get("site_change_starts_new_window", True)
        ),
    )


def _build_trajectory_split_config(
    *,
    mode: str,
    post_discharge_cutoff_days: float,
) -> TrajectorySplitConfig | None:
    mode_norm = str(mode or "full_subject").strip().lower()
    if mode_norm in {"", "none"}:
        return None
    if mode_norm not in {"full_subject", "admission_chain"}:
        raise ValueError(
            f"Unsupported trajectory split mode {mode!r}; expected one of ['full_subject', 'admission_chain', 'none']"
        )
    return TrajectorySplitConfig(
        mode=mode_norm,
        post_discharge_cutoff_hours=float(post_discharge_cutoff_days) * 24.0,
    )


def _resolve_residual_policy(
    args: argparse.Namespace,
    *,
    tokenization_contract: Mapping[str, Any],
) -> tuple[bool, int, Dict[str, int]]:
    cfg = tokenization_contract.get("residual_fallback", {})
    if not isinstance(cfg, dict):
        cfg = {}
    offsets_cfg = cfg.get("offsets", {})
    if not isinstance(offsets_cfg, dict):
        offsets_cfg = {}

    enabled_default = bool(cfg.get("enabled", True))
    residual_enabled = False if bool(args.disable_residual_fallback) else enabled_default
    residual_buckets = int(cfg.get("buckets", int(args.residual_fallback_buckets)))

    offsets: Dict[str, int] = {}
    for key, cli_val in (
        ("diagnosis", args.diag_residual_offset),
        ("procedure", args.proc_residual_offset),
        ("medication", args.med_residual_offset),
    ):
        if cli_val is not None:
            offsets[key] = int(cli_val)
        elif key in offsets_cfg and offsets_cfg.get(key) is not None:
            offsets[key] = int(offsets_cfg[key])
    return residual_enabled, residual_buckets, offsets


def _resolve_residual_tail_policies(
    *,
    tokenization_contract: Mapping[str, Any],
) -> Dict[str, str]:
    cfg = tokenization_contract.get("residual_fallback", {})
    if not isinstance(cfg, dict):
        cfg = {}
    families_cfg = cfg.get("families", {})
    if not isinstance(families_cfg, dict):
        families_cfg = {}
    default_tail_policy = str(cfg.get("tail_policy", "drop")).strip().lower() or "drop"

    out: Dict[str, str] = {}
    for family in ("diagnosis", "procedure", "medication"):
        family_cfg = families_cfg.get(family, {})
        if not isinstance(family_cfg, dict):
            family_cfg = {}
        tail_policy = str(family_cfg.get("tail_policy", default_tail_policy)).strip().lower() or "drop"
        if tail_policy not in {"drop", "hash"}:
            raise ValueError(
                f"Unsupported residual tail policy {tail_policy!r} for family {family!r}; expected drop|hash"
            )
        out[family] = tail_policy
    return out


def _load_subject_ids(
    splits_parquet: str,
    split: str,
    max_subjects: Optional[int],
    *,
    sample_seed: Optional[int] = None,
) -> List[int]:
    split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]]
    aliases = {"val": "tuning", "test": "held_out"}
    target_split = aliases.get(split, split)
    ids = split_df.loc[split_df["split"] == target_split, "subject_id"].astype("int64").tolist()
    if sample_seed is not None:
        rng = random.Random(int(sample_seed))
        rng.shuffle(ids)
    if max_subjects is not None and max_subjects > 0:
        ids = ids[:max_subjects]
    return [int(x) for x in ids]


def _parse_subject_ids(arg: str | None) -> List[int]:
    if arg is None or not str(arg).strip():
        return []
    out: List[int] = []
    for part in str(arg).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _maybe_print_progress(
    pass_name: str,
    *,
    idx: int,
    total: int,
    every: int,
    started_at: float,
) -> None:
    if every <= 0:
        return
    if idx != total and idx % every != 0:
        return
    elapsed = max(0.0, time.time() - started_at)
    rate = float(idx) / elapsed if elapsed > 0 else 0.0
    remaining = (float(total - idx) / rate) if rate > 0 else float("inf")
    eta_text = f"{remaining / 60.0:.1f}m" if math.isfinite(remaining) else "unknown"
    print(
        f"[{pass_name}] {idx}/{total} subjects | "
        f"elapsed={elapsed / 60.0:.1f}m | eta={eta_text}",
        flush=True,
    )


def _prefix_of(code: object) -> str:
    if code is None:
        return "<NONE>"
    return str(code).split("//", 1)[0].upper()


def _is_finite_numeric(value: object) -> bool:
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fv)


def _has_nonempty_attr(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _is_nonzero_numeric(value: object) -> bool:
    if not _is_finite_numeric(value):
        return False
    return abs(float(value)) > 0.0


def _count_medication_raw_attrs(
    ev: object,
    *,
    categorical_attrs: Iterable[str],
    numeric_attrs: Iterable[str],
    categorical_counter: Counter[str],
    numeric_finite_counter: Counter[str],
    numeric_nonzero_counter: Counter[str],
) -> None:
    for attr_name in categorical_attrs:
        if _has_nonempty_attr(getattr(ev, attr_name, None)):
            categorical_counter[str(attr_name)] += 1
    for attr_name in numeric_attrs:
        raw = getattr(ev, attr_name, None)
        if _is_finite_numeric(raw):
            numeric_finite_counter[str(attr_name)] += 1
        if _is_nonzero_numeric(raw):
            numeric_nonzero_counter[str(attr_name)] += 1


def _window_end_time_hours(window) -> float:
    if getattr(window, "tokens", None):
        return float(window.tokens[-1].t_from_start_hours)
    return float(getattr(window, "start_time_hours", 0.0))


def _window_position_name(index: int, total: int) -> str:
    if total <= 1:
        return "singleton"
    if index == 0:
        return "leading"
    if index == total - 1:
        return "trailing"
    return "interior"


def _gap_bucket_name(hours: Optional[float]) -> str:
    if hours is None:
        return "<none>"
    h = max(0.0, float(hours))
    if h <= 1.0:
        return "<=1h"
    if h <= 6.0:
        return "1-6h"
    if h <= 24.0:
        return "6-24h"
    if h <= 24.0 * 7.0:
        return "1-7d"
    if h <= 24.0 * 31.0:
        return "7-31d"
    return ">31d"


def _window_category_counter(window) -> Counter[str]:
    out: Counter[str] = Counter()
    for tok in getattr(window, "tokens", []) or []:
        try:
            name = TokenCategory(int(tok.category_id)).name
        except Exception:
            name = f"CATEGORY::{int(tok.category_id)}"
        out[name] += 1
    return out


def _build_semantic_resolution_encoders(
    *,
    artifacts: AuditArtifacts,
    residual_enabled: bool,
    residual_buckets: int,
    residual_offsets: Mapping[str, int],
    residual_tail_policies: Mapping[str, str],
) -> Dict[str, MedTokenWithAttrsEncoder]:
    diag_residual = int(residual_offsets["diagnosis"]) if residual_enabled and "diagnosis" in residual_offsets else None
    proc_residual = int(residual_offsets["procedure"]) if residual_enabled and "procedure" in residual_offsets else None
    med_residual = int(residual_offsets["medication"]) if residual_enabled and "medication" in residual_offsets else None
    residual_vocabs = artifacts.residual_fallback_vocabs if residual_enabled else {}
    return {
        "diagnosis": MedTokenWithAttrsEncoder(
            TokenCategory.DIAGNOSIS,
            artifacts.diag_vocab,
            canonicalize_fn=canonicalize_diagnosis_code,
            parent_lookup=artifacts.medtok_parent_lookup,
            crosswalk_lookup=artifacts.medtok_crosswalks.get("diagnosis"),
            residual_exact_vocab=residual_vocabs.get("diagnosis"),
            residual_fallback_offset=diag_residual,
            residual_fallback_buckets=int(residual_buckets),
            residual_tail_policy=str(residual_tail_policies.get("diagnosis", "drop")),
        ),
        "procedure": MedTokenWithAttrsEncoder(
            TokenCategory.PROCEDURE,
            artifacts.proc_vocab,
            canonicalize_fn=canonicalize_procedure_code,
            parent_lookup=artifacts.medtok_parent_lookup,
            crosswalk_lookup=artifacts.medtok_crosswalks.get("procedure"),
            residual_exact_vocab=residual_vocabs.get("procedure"),
            residual_fallback_offset=proc_residual,
            residual_fallback_buckets=int(residual_buckets),
            residual_tail_policy=str(residual_tail_policies.get("procedure", "drop")),
        ),
        "medication": MedTokenWithAttrsEncoder(
            TokenCategory.MEDICATION,
            artifacts.med_vocab,
            canonicalize_fn=canonicalize_medication_code,
            parent_lookup=artifacts.medtok_parent_lookup,
            crosswalk_lookup=artifacts.medtok_crosswalks.get("medication"),
            residual_exact_vocab=residual_vocabs.get("medication"),
            residual_fallback_offset=med_residual,
            residual_fallback_buckets=int(residual_buckets),
            residual_tail_policy=str(residual_tail_policies.get("medication", "drop")),
        ),
    }


def _is_expected_process_reroute(raw_code: object, category: TokenCategory) -> bool:
    s = str(raw_code).upper() if raw_code is not None else ""
    if category == TokenCategory.MEDICATION:
        return (
            s.startswith("INFUSION_START//")
            or s.startswith("INFUSION_END//")
            or s.startswith("MEDICATION//START//")
            or s.startswith("MEDICATION//END//")
            or s.startswith("MEDICATION//STOP//")
        )
    if category == TokenCategory.PROCEDURE:
        return (
            s.startswith("PROCEDURE//START//")
            or s.startswith("PROCEDURE//END//")
            or s.startswith("PROCEDURE//STOP//")
        )
    return False


def _maybe_attr_vocab(path: Path, offset: int, name: str) -> Optional[CategoryVocab]:
    return load_attr_vocab(str(path), offset=offset, name=name) if path.exists() else None


def _build_med_numeric_cfg(manifest: Mapping[str, Any]) -> Dict[str, NumericBinConfig]:
    return {
        "dosage": NumericBinConfig(
            offset=_offset(manifest, "med_dosage", 1_680_000),
            bins=8,
            min_val=0.1,
            max_val=2000.0,
            log=True,
        ),
        "rate": NumericBinConfig(
            offset=_offset(manifest, "med_rate", 1_700_000),
            bins=6,
            min_val=0.1,
            max_val=2000.0,
            log=True,
        ),
        "duration_hours": NumericBinConfig(
            offset=_offset(manifest, "med_duration", 1_720_000),
            bins=6,
            min_val=0.1,
            max_val=240.0,
            log=True,
        ),
    }


def _build_static_artifacts(args: argparse.Namespace) -> AuditArtifacts:
    tokenization_contract = _load_tokenization_contract(
        getattr(args, "tokenization_yaml", None)
    )
    manifest = _load_manifest(getattr(args, "sparse_vocab_json", None))
    structural_codebook = (
        load_structural_codebook_yaml(
            args.structural_yaml,
            default_offset=_offset(manifest, "structural", 2_200_000),
        )
        if args.structural_yaml
        else None
    )

    medtok_inputs = validate_medtok_inputs(
        medtok_code2embeds=getattr(args, "medtok_code2embeds", None),
        medtok_vocab_dir=getattr(args, "medtok_vocab_dir", None),
        allow_smoke_medtok=bool(getattr(args, "allow_smoke_medtok", False)),
    )
    medtok_code2embeds = (
        Path(medtok_inputs["medtok_code2embeds"])
        if medtok_inputs["medtok_code2embeds"] is not None
        else None
    )
    medtok_vocab_dir = (
        Path(medtok_inputs["medtok_vocab_dir"])
        if medtok_inputs["medtok_vocab_dir"] is not None
        else None
    )
    medtok_attr_dir = Path(args.medtok_attr_dir)
    medtok_crosswalk_json = getattr(args, "medtok_crosswalk_json", None)

    if medtok_code2embeds is not None:
        diag_vocab = build_vocab_from_code2embeddings(
            str(medtok_code2embeds),
            offset=_offset(manifest, "diagnosis", 1_000_000),
            name="diagnosis",
            filter_fn=diagnosis_filter,
        )
        proc_vocab = build_vocab_from_code2embeddings(
            str(medtok_code2embeds),
            offset=_offset(manifest, "procedure", 1_200_000),
            name="procedure",
            filter_fn=procedure_filter,
        )
        med_vocab = build_vocab_from_code2embeddings(
            str(medtok_code2embeds),
            offset=_offset(manifest, "medication", 1_400_000),
            name="medication",
            filter_fn=medication_filter,
        )
    else:
        diag_vocab = load_medtok_vocab(
            str(medtok_vocab_dir / "diag_vocab.json"),
            offset=_offset(manifest, "diagnosis", 1_000_000),
            name="diagnosis",
        )
        proc_vocab = load_medtok_vocab(
            str(medtok_vocab_dir / "proc_vocab.json"),
            offset=_offset(manifest, "procedure", 1_200_000),
            name="procedure",
        )
        med_vocab = load_medtok_vocab(
            str(medtok_vocab_dir / "med_vocab.json"),
            offset=_offset(manifest, "medication", 1_400_000),
            name="medication",
        )

    medtok_crosswalk_candidate_maps = {
        "diagnosis": load_crosswalk_candidate_map(medtok_crosswalk_json, "diagnosis"),
        "procedure": load_crosswalk_candidate_map(medtok_crosswalk_json, "procedure"),
        "medication": load_crosswalk_candidate_map(medtok_crosswalk_json, "medication"),
    }
    if medtok_code2embeds is not None:
        for vocab, family in (
            (diag_vocab, "diagnosis"),
            (proc_vocab, "procedure"),
            (med_vocab, "medication"),
        ):
            next_id = max(vocab.code2id.values()) + 1 if vocab.code2id else 0
            for target_candidates in medtok_crosswalk_candidate_maps.get(family, {}).values():
                for target_code in target_candidates:
                    if str(target_code) in vocab.code2id:
                        break
                    vocab.code2id[str(target_code)] = int(next_id)
                    next_id += 1
                    break
    medtok_crosswalks = {
        "diagnosis": load_resolved_crosswalk_lookup(
            medtok_crosswalk_json,
            "diagnosis",
            available_codes=diag_vocab.code2id.keys(),
        ),
        "procedure": load_resolved_crosswalk_lookup(
            medtok_crosswalk_json,
            "procedure",
            available_codes=proc_vocab.code2id.keys(),
        ),
        "medication": load_resolved_crosswalk_lookup(
            medtok_crosswalk_json,
            "medication",
            available_codes=med_vocab.code2id.keys(),
        ),
    }
    residual_fallback_vocabs = {
        family: vocab
        for family in ("diagnosis", "procedure", "medication")
        for vocab in [
            load_residual_fallback_vocab(
                medtok_vocab_dir,
                family=family,
                offset=_offset(
                    manifest,
                    f"{family}_residual",
                    {
                        "diagnosis": 1_160_000,
                        "procedure": 1_360_000,
                        "medication": 1_800_000,
                    }[family],
                ),
            )
        ]
        if vocab is not None
    }
    obs_code_vocab = load_observation_vocab(
        medtok_vocab_dir,
        kind="code",
        offset=_offset(manifest, "observation_code", 2_300_000),
    )
    obs_value_vocab = load_observation_vocab(
        medtok_vocab_dir,
        kind="value",
        offset=_offset(manifest, "observation_value", 2_320_000),
    )

    med_attr_vocabs: Dict[str, CategoryVocab] = {}
    for name, filename, default_offset in (
        ("route", "route_vocab.json", 1_600_000),
        ("form", "form_vocab.json", 1_620_000),
        ("freq", "freq_vocab.json", 1_640_000),
        ("unit", "unit_vocab.json", 1_660_000),
    ):
        vocab = _maybe_attr_vocab(
            medtok_attr_dir / filename,
            _offset(manifest, f"med_{name}", default_offset),
            name,
        )
        if vocab is not None:
            med_attr_vocabs[name] = vocab

    code2id: Dict[str, int] | None = None
    measurement_num_codebooks: int | None = None
    measurement_codebook_size: int | None = None
    measurement_stride: int | None = None
    if args.code2id_pt:
        code2id = torch.load(args.code2id_pt, map_location="cpu")
    if args.tokenizer_ckpt:
        tok_ckpt = torch.load(args.tokenizer_ckpt, map_location="cpu")
        measurement_num_codebooks = int(tok_ckpt["cfg"]["num_codebooks"])
        measurement_codebook_size = int(tok_ckpt["cfg"]["codebook_size"])
        measurement_stride = measurement_codebook_size

    medtok_parent_lookup: Dict[str, List[str]] = {}
    codes_parquet_parent_lookup = getattr(args, "codes_parquet_parent_lookup", None)
    if codes_parquet_parent_lookup:
        medtok_parent_lookup = load_parent_lookup_from_codes_parquet(codes_parquet_parent_lookup)

    obs_tail_policy = _resolve_observation_tail_policy(
        tokenization_contract=tokenization_contract,
    )

    return AuditArtifacts(
        manifest=manifest,
        structural_codebook=structural_codebook,
        code2id=code2id,
        measurement_num_codebooks=measurement_num_codebooks,
        measurement_codebook_size=measurement_codebook_size,
        measurement_stride=measurement_stride,
        diag_vocab=diag_vocab,
        proc_vocab=proc_vocab,
        med_vocab=med_vocab,
        med_attr_vocabs=med_attr_vocabs,
        med_numeric_attrs=_build_med_numeric_cfg(manifest),
        medtok_parent_lookup=medtok_parent_lookup,
        medtok_crosswalks=medtok_crosswalks,
        residual_fallback_vocabs=residual_fallback_vocabs,
        obs_code_vocab=obs_code_vocab,
        obs_value_vocab=obs_value_vocab,
        obs_tail_policy=str(obs_tail_policy),
    )


def _summarize_raw_subjects(
    db: mr.SubjectDatabase,
    subject_ids: List[int],
    *,
    artifacts: AuditArtifacts,
    semantic_resolvers: Mapping[str, MedTokenWithAttrsEncoder],
    top_k: int,
    progress_every: int = 0,
) -> tuple[Dict[str, Any], set[str]]:
    category_counts = Counter()
    prefix_counts = Counter()
    numeric_by_category = Counter()
    code_counts_by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    marker_counts = Counter()
    measurement_status = Counter()
    medtok_hits: Dict[str, Counter[str]] = {
        "diagnosis": Counter(),
        "procedure": Counter(),
        "medication": Counter(),
    }
    medtok_resolution_stages: Dict[str, Counter[str]] = {
        "diagnosis": Counter(),
        "procedure": Counter(),
        "medication": Counter(),
    }
    medtok_miss_samples: Dict[str, Counter[str]] = {
        "diagnosis": Counter(),
        "procedure": Counter(),
        "medication": Counter(),
    }
    med_raw_categorical_attrs = Counter()
    med_raw_numeric_attrs_finite = Counter()
    med_raw_numeric_attrs_nonzero = Counter()
    structural_labels = Counter()
    structural_boundaries = Counter()
    structural_overlays = Counter()
    other_codes = Counter()
    structural_raw_codes: set[str] = set()

    started_at = time.time()
    total = len(subject_ids)
    for idx, sid in enumerate(subject_ids, start=1):
        subj = db[int(sid)]
        for ev in subj.events:
            code = getattr(ev, "code", None)
            code_str = str(code) if code is not None else "<NONE>"
            prefix = _prefix_of(code)
            category = classify_code_to_category(code)
            cat_name = category.name

            category_counts[cat_name] += 1
            prefix_counts[prefix] += 1
            code_counts_by_category[cat_name][code_str] += 1

            raw_upper = code_str.upper()
            if "START" in raw_upper:
                marker_counts["START_like"] += 1
            if "END" in raw_upper:
                marker_counts["END_like"] += 1
            if "STOP" in raw_upper:
                marker_counts["STOP_like"] += 1

            if _is_finite_numeric(getattr(ev, "numeric_value", None)):
                numeric_by_category[cat_name] += 1

            if category == TokenCategory.MEASUREMENT:
                if code is None:
                    measurement_status["missing_code"] += 1
                elif artifacts.code2id is None:
                    measurement_status["mapping_missing"] += 1
                elif code_str not in artifacts.code2id:
                    measurement_status["code_unmapped"] += 1
                elif getattr(ev, "numeric_value", None) is None:
                    measurement_status["no_numeric_value"] += 1
                elif not _is_finite_numeric(getattr(ev, "numeric_value", None)):
                    measurement_status["nonfinite_numeric_value"] += 1
                else:
                    measurement_status["measurement_ok"] += 1
            elif category == TokenCategory.DIAGNOSIS:
                resolution = semantic_resolvers["diagnosis"].resolve_event(ev)
                medtok_resolution_stages["diagnosis"][resolution.stage] += 1
                if resolution.stage in EXPLICIT_MEDTOK_RESOLUTION_STAGES:
                    medtok_hits["diagnosis"]["hit"] += 1
                else:
                    medtok_hits["diagnosis"]["miss"] += 1
                    medtok_miss_samples["diagnosis"][code_str] += 1
            elif category == TokenCategory.PROCEDURE:
                resolution = semantic_resolvers["procedure"].resolve_event(ev)
                medtok_resolution_stages["procedure"][resolution.stage] += 1
                if resolution.stage in EXPLICIT_MEDTOK_RESOLUTION_STAGES:
                    medtok_hits["procedure"]["hit"] += 1
                else:
                    medtok_hits["procedure"]["miss"] += 1
                    medtok_miss_samples["procedure"][code_str] += 1
            elif category == TokenCategory.MEDICATION:
                _count_medication_raw_attrs(
                    ev,
                    categorical_attrs=artifacts.med_attr_vocabs.keys(),
                    numeric_attrs=artifacts.med_numeric_attrs.keys(),
                    categorical_counter=med_raw_categorical_attrs,
                    numeric_finite_counter=med_raw_numeric_attrs_finite,
                    numeric_nonzero_counter=med_raw_numeric_attrs_nonzero,
                )
                resolution = semantic_resolvers["medication"].resolve_event(ev)
                medtok_resolution_stages["medication"][resolution.stage] += 1
                if resolution.stage in EXPLICIT_MEDTOK_RESOLUTION_STAGES:
                    medtok_hits["medication"]["hit"] += 1
                else:
                    medtok_hits["medication"]["miss"] += 1
                    medtok_miss_samples["medication"][code_str] += 1

            if artifacts.structural_codebook is not None and code_str in artifacts.structural_codebook:
                structural_raw_codes.add(code_str)
                label = artifacts.structural_codebook.code2label[code_str]
                structural_labels[label] += 1
                if artifacts.structural_codebook.is_window_boundary(code=code_str, label=label):
                    structural_boundaries[label] += 1
                else:
                    structural_overlays[label] += 1

            if category == TokenCategory.STRUCTURAL:
                structural_raw_codes.add(code_str)
            if category == TokenCategory.OTHER:
                other_codes[code_str] += 1

        _maybe_print_progress(
            "raw",
            idx=idx,
            total=total,
            every=progress_every,
            started_at=started_at,
        )

    raw_summary = {
        "subjects_scanned": len(subject_ids),
        "total_events": int(sum(category_counts.values())),
        "events_by_category": _as_plain_counter(category_counts),
        "numeric_events_by_category": _as_plain_counter(numeric_by_category),
        "measurement_status": _as_plain_counter(measurement_status),
        "medtok_hits": {k: _as_plain_counter(v) for k, v in medtok_hits.items()},
        "medtok_resolution_stages": {
            k: _as_plain_counter(v) for k, v in medtok_resolution_stages.items()
        },
        "medication_raw_attr_presence": {
            "categorical_nonempty": _as_plain_counter(med_raw_categorical_attrs),
            "numeric_finite": _as_plain_counter(med_raw_numeric_attrs_finite),
            "numeric_nonzero": _as_plain_counter(med_raw_numeric_attrs_nonzero),
        },
        "top_prefixes": _top_counter(prefix_counts, top_k),
        "top_other_codes": _top_counter(other_codes, top_k),
        "top_codes_by_category": {
            k: _top_counter(v, top_k) for k, v in code_counts_by_category.items()
        },
        "top_medtok_misses": {
            k: _top_counter(v, top_k) for k, v in medtok_miss_samples.items()
        },
        "marker_like_counts": _as_plain_counter(marker_counts),
        "structural_labels": _as_plain_counter(structural_labels),
        "structural_boundary_labels": _as_plain_counter(structural_boundaries),
        "structural_overlay_labels": _as_plain_counter(structural_overlays),
    }
    return raw_summary, structural_raw_codes


def _build_struct_vocab(
    structural_codes: Iterable[str],
    *,
    manifest: Mapping[str, Any],
) -> CategoryVocab:
    code2id = {"<UNK>": 0}
    for i, code in enumerate(sorted({str(c) for c in structural_codes if c is not None}), start=1):
        code2id[code] = i
    return CategoryVocab(
        name="structural",
        offset=_offset(manifest, "structural", 2_200_000),
        code2id=code2id,
    )


def _build_measurement_config(
    args: argparse.Namespace,
    *,
    artifacts: AuditArtifacts,
) -> Optional[MeasurementEncoderConfig]:
    if not (args.code2id_pt and args.stats_pt and args.cvae_ckpt and args.tokenizer_ckpt):
        return None
    stats = torch.load(args.stats_pt, map_location="cpu")
    return MeasurementEncoderConfig(
        cvae_ckpt=args.cvae_ckpt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        mean_by_var=stats["mean_by_var"],
        std_by_var=stats["std_by_var"],
        code2id=torch.load(args.code2id_pt, map_location="cpu"),
        code_token_offset=_offset(artifacts.manifest, "measurement_code", 2_000_000),
        rvq_token_offset=_offset(artifacts.manifest, "measurement_value", 2_100_000),
        rvq_codebook_stride=artifacts.measurement_stride,
    )


def _make_structural_id2label(codebook: Optional[StructuralCodebook]) -> Dict[int, str]:
    if codebook is None:
        return {}
    return {int(v): str(k) for k, v in codebook.label2id().items()}


def _family_name_for_token(tok: EventToken, *, artifacts: AuditArtifacts) -> str:
    value_id = int(tok.value_id)
    manifest = artifacts.manifest
    if tok.category_id == int(TokenCategory.SPECIAL):
        return "special_or_window"

    diag_offset = _offset(manifest, "diagnosis", 1_000_000)
    proc_offset = _offset(manifest, "procedure", 1_200_000)
    med_offset = _offset(manifest, "medication", 1_400_000)
    meas_code_offset = _offset(manifest, "measurement_code", 2_000_000)
    meas_value_offset = _offset(manifest, "measurement_value", 2_100_000)
    struct_offset = _offset(manifest, "structural", 2_200_000)
    obs_code_offset = _offset(manifest, "observation_code", 2_300_000)
    obs_value_offset = _offset(manifest, "observation_value", 2_320_000)

    if meas_value_offset <= value_id < struct_offset:
        return "measurement_value"
    if meas_code_offset <= value_id < meas_value_offset:
        return "measurement_code"
    if med_offset <= value_id < meas_code_offset:
        return "medication_or_med_attr"
    if proc_offset <= value_id < med_offset:
        return "procedure"
    if diag_offset <= value_id < proc_offset:
        return "diagnosis"
    if value_id >= obs_value_offset:
        return "observation_value"
    if obs_code_offset <= value_id < obs_value_offset:
        return "observation_code"
    if struct_offset <= value_id < obs_code_offset:
        return "structural"
    return "special_or_window"


def _record_window_marker_usage(
    *,
    ids: List[int],
    specials_count: int,
    collator: AETHierarchicalCollator,
    window_type_id: int,
    next_type_id: int | None,
    chunk_is_last: bool,
    totals: Counter[str],
    prefix_type_counts: Counter[int],
    prefix_raw_id_counts: Counter[int],
    suffix_mode_counts: Counter[str],
    suffix_next_type_counts: Counter[int],
    suffix_raw_id_counts: Counter[int],
) -> None:
    if not collator.window_markers.enabled or not ids:
        return
    prefix_index = int(specials_count)
    if prefix_index >= len(ids):
        return

    prefix_id = int(ids[prefix_index])
    suffix_id = int(ids[-1])
    totals["chunks_with_markers"] += 1
    totals["marker_tokens_total"] += 2
    prefix_type_counts[int(window_type_id)] += 1
    prefix_raw_id_counts[prefix_id] += 1
    suffix_raw_id_counts[suffix_id] += 1

    if not chunk_is_last:
        suffix_mode_counts["continue"] += 1
        return

    end_mode = str(getattr(collator.window_markers, "end_mode", "end_token"))
    if end_mode == "next_type" and next_type_id is not None:
        suffix_mode_counts["next_type"] += 1
        suffix_next_type_counts[int(next_type_id)] += 1
    else:
        suffix_mode_counts["end"] += 1


def _audit_subject_tokenization(
    db: mr.SubjectDatabase,
    subject_id: int,
    *,
    encoders: Dict[TokenCategory, Any],
    codebook: Optional[StructuralCodebook],
    artifacts: AuditArtifacts,
    struct_id2code: Mapping[int, str],
) -> Dict[str, Any]:
    subj = db[int(subject_id)]
    events = list(subj.events)
    sex_val = infer_subject_sex(events, default=0.0)
    birth_ts = infer_birth_timestamp(events)

    for enc in encoders.values():
        if hasattr(enc, "reset_state"):
            enc.reset_state()

    raw_events_by_category = Counter()
    dropped_events_by_category = Counter()
    emitted_events_by_category = Counter()
    emitted_tokens_by_category = Counter()
    unknown_base_tokens_by_category = Counter()
    process_reroute_events_by_category = Counter()
    bundle_sizes_by_category: Dict[str, Counter[int]] = defaultdict(Counter)
    semantic_base_outcomes_by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    observation_base_outcomes = Counter()
    family_counts = Counter()
    family_unique_ids: Dict[str, set[int]] = defaultdict(set)
    family_min_id: Dict[str, int] = {}
    family_max_id: Dict[str, int] = {}
    frame_payload_kind_counts = Counter()
    frame_bundle_sizes_by_payload_kind: Dict[str, Counter[int]] = defaultdict(Counter)
    observation_frame_integrity = Counter()
    medication_frame_categorical_attrs = Counter()
    medication_frame_numeric_attrs_nonzero = Counter()
    medication_frame_numeric_attrs_finite = Counter()
    medication_frame_marker_tokens = Counter()
    decoded_preview_kind_counts = Counter()
    decoded_observation_integrity = Counter()

    last_emitted_time = None
    timeline = build_subject_timeline(
        db=db,
        subject_id=int(subject_id),
        encoders=encoders,
        structural_codebook=codebook,
        qual_obs_code_vocab=artifacts.obs_code_vocab,
        qual_obs_value_vocab=artifacts.obs_value_vocab,
        qual_obs_tail_policy=artifacts.obs_tail_policy,
    )

    flattened_timeline = flatten_event_frames(timeline, clone=False)
    decoded_timeline = decode_timeline_tokens(
        flattened_timeline,
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
        structural_id2label=_make_structural_id2label(codebook),
        structural_id2code=struct_id2code,
        special_id2name=SPECIAL_ID2NAME,
    )

    for item in decoded_timeline:
        decoded_preview_kind_counts[str(item.get("kind", "<unk>"))] += 1
        label = item.get("label")
        label_str = label if isinstance(label, str) else ""
        if item.get("kind") == "observation_bundle":
            decoded_observation_integrity["bundle"] += 1
        elif label_str.startswith("OBS_CODE::"):
            decoded_observation_integrity["stray_code_token"] += 1
        elif label_str.startswith("OBS_VAL::"):
            decoded_observation_integrity["stray_value_token"] += 1

    for frame in timeline:
        if not isinstance(frame, EventFrame):
            continue
        payload_kind = str(frame.payload_kind)
        frame_payload_kind_counts[payload_kind] += 1
        frame_bundle_sizes_by_payload_kind[payload_kind][int(len(frame.token_bundle))] += 1

        if payload_kind == "qualitative_observation":
            obs_positions = [int((tok.cat_attrs or {}).get("obs_bundle_pos", 0)) for tok in frame.token_bundle]
            observation_frame_integrity["frames_total"] += 1
            if obs_positions == [1, 2]:
                observation_frame_integrity["complete_two_token"] += 1
            else:
                observation_frame_integrity["malformed"] += 1
            if 1 in obs_positions:
                observation_frame_integrity["has_code_token"] += 1
            if 2 in obs_positions:
                observation_frame_integrity["has_value_token"] += 1

        if int(frame.category_id) == int(TokenCategory.MEDICATION):
            medication_frame_marker_tokens[
                "has_marker_token" if any((tok.cat_attrs or {}).get("event_marker") is not None for tok in frame.token_bundle) else "base_only"
            ] += 1
            for attr_name in artifacts.med_attr_vocabs.keys():
                raw = (frame.cat_attrs or {}).get(attr_name, 0)
                if int(raw or 0) != 0:
                    medication_frame_categorical_attrs[str(attr_name)] += 1
            for attr_name in artifacts.med_numeric_attrs.keys():
                raw = (frame.num_attrs or {}).get(attr_name, None)
                if _is_finite_numeric(raw):
                    medication_frame_numeric_attrs_finite[str(attr_name)] += 1
                if _is_nonzero_numeric(raw):
                    medication_frame_numeric_attrs_nonzero[str(attr_name)] += 1

    struct_only = codebook.structural_only if codebook is not None else set()
    struct_keep_orig = codebook.keep_original if codebook is not None else set()

    for ev in events:
        age_years = infer_event_age_years(ev, birth_ts=birth_ts)
        ev_view = _EventWithDemographics(ev, age_years=age_years, sex=sex_val)
        code = getattr(ev_view, "code", None)
        code_str = str(code) if code is not None else None
        category = classify_code_to_category(code)
        is_process_reroute = _is_expected_process_reroute(code_str, category)
        raw_events_by_category[category.name] += 1
        if is_process_reroute:
            process_reroute_events_by_category[category.name] += 1
        encoder = encoders.get(category)
        resolution_stage: Optional[str] = None

        t = getattr(ev_view, "time", None)
        dt_hours = 0.0
        if t is not None and last_emitted_time is not None:
            dt_hours = max(0.0, (t - last_emitted_time).total_seconds() / 3600.0)

        emitted_for_event = 0
        if codebook is not None and code_str is not None and code_str in codebook.code2label:
            emitted_for_event += 1

        if codebook is not None and code_str is not None and code_str in struct_only and code_str not in struct_keep_orig:
            emitted_events_by_category[category.name] += 1
            emitted_tokens_by_category[category.name] += emitted_for_event
            bundle_sizes_by_category[category.name][emitted_for_event] += 1
            if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                if is_process_reroute:
                    semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
                else:
                    semantic_base_outcomes_by_category[category.name]["structural_only"] += 1
            last_emitted_time = t if emitted_for_event > 0 and t is not None else last_emitted_time
            continue

        if encoder is None:
            if emitted_for_event == 0:
                dropped_events_by_category[category.name] += 1
            else:
                emitted_events_by_category[category.name] += 1
                emitted_tokens_by_category[category.name] += emitted_for_event
                bundle_sizes_by_category[category.name][emitted_for_event] += 1
            if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                if is_process_reroute:
                    semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
                else:
                    semantic_base_outcomes_by_category[category.name]["dropped_or_no_encoder"] += 1
            last_emitted_time = t if emitted_for_event > 0 and t is not None else last_emitted_time
            continue

        if (
            category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}
            and hasattr(encoder, "resolve_event")
        ):
            resolution_stage = str(encoder.resolve_event(ev_view).stage)

        toks = encoder.encode_event(ev_view, dt_hours=dt_hours)
        if toks:
            emitted_for_event += len(toks)
            emitted_events_by_category[category.name] += 1
            emitted_tokens_by_category[category.name] += emitted_for_event
            bundle_sizes_by_category[category.name][emitted_for_event] += 1
            base_tok = toks[0]
            unk_gid = getattr(encoder, "unk_gid", None)
            if resolution_stage == "unk" or (
                resolution_stage is None
                and unk_gid is not None
                and int(base_tok.value_id) == int(unk_gid)
            ):
                unknown_base_tokens_by_category[category.name] += 1
            if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                if is_process_reroute:
                    semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
                elif resolution_stage in EXPLICIT_MEDTOK_RESOLUTION_STAGES:
                    semantic_base_outcomes_by_category[category.name][str(resolution_stage)] += 1
                elif resolution_stage in {"residual_exact", "residual_hash"}:
                    semantic_base_outcomes_by_category[category.name][str(resolution_stage)] += 1
                elif resolution_stage == "unk":
                    semantic_base_outcomes_by_category[category.name]["unk"] += 1
                elif resolution_stage == "drop":
                    semantic_base_outcomes_by_category[category.name]["drop"] += 1
                elif int(base_tok.cat_attrs.get("residual_fallback_hash", 0) or 0) == 1:
                    semantic_base_outcomes_by_category[category.name]["residual_hash"] += 1
                elif int(base_tok.cat_attrs.get("residual_fallback_exact", 0) or 0) == 1:
                    semantic_base_outcomes_by_category[category.name]["residual_exact"] += 1
                elif int(base_tok.cat_attrs.get("residual_fallback", 0) or 0) == 1:
                    semantic_base_outcomes_by_category[category.name]["residual_exact"] += 1
                elif unk_gid is not None and int(base_tok.value_id) == int(unk_gid):
                    semantic_base_outcomes_by_category[category.name]["unk"] += 1
                else:
                    semantic_base_outcomes_by_category[category.name]["exact"] += 1
            if t is not None:
                last_emitted_time = t
        else:
            if emitted_for_event > 0:
                emitted_events_by_category[category.name] += 1
                emitted_tokens_by_category[category.name] += emitted_for_event
                bundle_sizes_by_category[category.name][emitted_for_event] += 1
                if t is not None:
                    last_emitted_time = t
                if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                    if is_process_reroute:
                        semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
                    else:
                        semantic_base_outcomes_by_category[category.name]["structural_only"] += 1
            else:
                if category == TokenCategory.MEASUREMENT:
                    obs = observation_surfaces(ev_view, code_value=code_str)
                    if obs is not None:
                        code_exact = (
                            artifacts.obs_code_vocab is not None
                            and artifacts.obs_code_vocab.maybe_encode(obs.code_surface) is not None
                        )
                        value_exact = (
                            artifacts.obs_value_vocab is not None
                            and artifacts.obs_value_vocab.maybe_encode(obs.value_surface) is not None
                        )
                        if (
                            artifacts.obs_code_vocab is not None
                            and not code_exact
                            and artifacts.obs_tail_policy == "drop"
                        ) or (
                            artifacts.obs_value_vocab is not None
                            and not value_exact
                            and artifacts.obs_tail_policy == "drop"
                        ):
                            dropped_events_by_category[category.name] += 1
                            observation_base_outcomes["drop"] += 1
                        else:
                            emitted_for_event = 2
                            emitted_events_by_category[category.name] += 1
                            emitted_tokens_by_category[category.name] += emitted_for_event
                            bundle_sizes_by_category[category.name][emitted_for_event] += 1
                            if code_exact and value_exact:
                                observation_base_outcomes["exact"] += 1
                            elif code_exact or value_exact:
                                observation_base_outcomes["mixed"] += 1
                            else:
                                observation_base_outcomes["hash"] += 1
                            if t is not None:
                                last_emitted_time = t
                        continue

                dropped_events_by_category[category.name] += 1
                if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                    if is_process_reroute:
                        semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
                    elif resolution_stage == "drop":
                        semantic_base_outcomes_by_category[category.name]["drop"] += 1
                    else:
                        semantic_base_outcomes_by_category[category.name]["dropped_or_no_encoder"] += 1

    for tok in timeline:
        family = _family_name_for_token(tok, artifacts=artifacts)
        family_counts[family] += 1
        family_unique_ids[family].add(int(tok.value_id))
        family_min_id[family] = min(int(tok.value_id), family_min_id.get(family, int(tok.value_id)))
        family_max_id[family] = max(int(tok.value_id), family_max_id.get(family, int(tok.value_id)))

    return {
        "timeline": timeline,
        "raw_events_by_category": raw_events_by_category,
        "dropped_events_by_category": dropped_events_by_category,
        "emitted_events_by_category": emitted_events_by_category,
        "emitted_tokens_by_category": emitted_tokens_by_category,
        "unknown_base_tokens_by_category": unknown_base_tokens_by_category,
        "process_reroute_events_by_category": process_reroute_events_by_category,
        "semantic_base_outcomes_by_category": semantic_base_outcomes_by_category,
        "observation_base_outcomes": observation_base_outcomes,
        "bundle_sizes_by_category": bundle_sizes_by_category,
        "family_counts": family_counts,
        "family_unique_ids": family_unique_ids,
        "family_min_id": family_min_id,
        "family_max_id": family_max_id,
        "frame_payload_kind_counts": frame_payload_kind_counts,
        "frame_bundle_sizes_by_payload_kind": frame_bundle_sizes_by_payload_kind,
        "observation_frame_integrity": observation_frame_integrity,
        "decoded_preview_kind_counts": decoded_preview_kind_counts,
        "decoded_observation_integrity": decoded_observation_integrity,
        "medication_frame_attr_presence": {
            "categorical_nonzero": medication_frame_categorical_attrs,
            "numeric_finite": medication_frame_numeric_attrs_finite,
            "numeric_nonzero": medication_frame_numeric_attrs_nonzero,
            "marker_tokens": medication_frame_marker_tokens,
        },
    }


def _summarize_tokenization_and_collation(
    db: mr.SubjectDatabase,
    subject_ids: List[int],
    *,
    encoders: Dict[TokenCategory, Any],
    artifacts: AuditArtifacts,
    struct_id2code: Mapping[int, str],
    max_windows: int,
    max_chunks_per_window: int,
    max_len_per_window: int,
    collate_batch_size: int,
    example_subjects: int,
    example_tokens: int,
    window_markers: WindowMarkerConfig | None = None,
    segmentation_config: WindowSegmentationConfig | None = None,
    progress_every: int = 0,
) -> Dict[str, Any]:
    struct_id2label = _make_structural_id2label(artifacts.structural_codebook)
    type_id2name: Dict[int, str] = {}
    if artifacts.structural_codebook is not None:
        type_id2name.update(
            {int(v): str(k) for k, v in artifacts.structural_codebook.window_type2id().items()}
        )
    history_prefix_type_id = next(
        (k for k, v in type_id2name.items() if str(v) == "HISTORY_PREFIX"),
        None,
    )
    post_discharge_type_id = next(
        (k for k, v in type_id2name.items() if str(v) == "POST_DISCHARGE"),
        segmentation_config.post_discharge_window_type_id if segmentation_config is not None else None,
    )
    collator = AETHierarchicalCollator(
        max_windows=max_windows,
        max_chunks_per_window=max_chunks_per_window,
        max_len_per_window=max_len_per_window,
        pad_id=0,
        window_markers=window_markers or WindowMarkerConfig(),
        segmentation=segmentation_config,
    )

    total_tokens = 0
    emitted_tokens_by_category = Counter()
    raw_events_by_category = Counter()
    dropped_events_by_category = Counter()
    emitted_events_by_category = Counter()
    unknown_base_tokens_by_category = Counter()
    process_reroute_events_by_category = Counter()
    semantic_base_outcomes_by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    observation_base_outcomes = Counter()
    bundle_sizes_by_category: Dict[str, Counter[int]] = defaultdict(Counter)
    family_counts = Counter()
    family_unique_ids: Dict[str, set[int]] = defaultdict(set)
    family_min_id: Dict[str, int] = {}
    family_max_id: Dict[str, int] = {}
    frame_payload_kind_counts = Counter()
    frame_bundle_sizes_by_payload_kind: Dict[str, Counter[int]] = defaultdict(Counter)
    observation_frame_integrity = Counter()
    decoded_preview_kind_counts = Counter()
    decoded_observation_integrity = Counter()
    medication_frame_attr_presence: Dict[str, Counter[str]] = {
        "categorical_nonzero": Counter(),
        "numeric_finite": Counter(),
        "numeric_nonzero": Counter(),
        "marker_tokens": Counter(),
    }

    window_counts = Counter()
    chunk_counts = Counter()
    truncation_counts = Counter()
    numeric_mask_ones = 0
    attention_mask_ones = 0
    event_numeric_mask_ones = 0
    event_attention_mask_ones = 0
    window_mask_ones = 0
    chunk_mask_ones = 0
    window_type_zero = 0
    window_type_total = 0
    window_marker_totals = Counter()
    window_marker_prefix_types = Counter()
    window_marker_prefix_raw_ids = Counter()
    window_marker_suffix_modes = Counter()
    window_marker_suffix_next_types = Counter()
    window_marker_suffix_raw_ids = Counter()
    window_type_raw_counts = Counter()
    window_type_clamped_counts = Counter()
    unknown_window_opening_actions = Counter()
    unknown_window_closing_actions = Counter()
    unknown_window_first_categories = Counter()
    unknown_window_first_families = Counter()
    unknown_window_first_transition_action = Counter()
    unknown_window_first_transition_type = Counter()
    unknown_window_first_window_type_attr = Counter()
    unknown_window_first_struct_label = Counter()
    window_type_fallback_sources = Counter()
    unknown_window_samples: List[Dict[str, Any]] = []
    unknown_window_total = 0
    trajectory_shape = {
        "history_prefix_total": 0,
        "history_prefix_position_counts": Counter(),
        "history_prefix_next_type_counts": Counter(),
        "history_prefix_prev_gap_buckets": Counter(),
        "history_prefix_next_gap_buckets": Counter(),
        "history_prefix_duration_buckets": Counter(),
        "history_prefix_token_categories": Counter(),
        "history_prefix_samples": [],
        "post_discharge_total": 0,
        "post_discharge_position_counts": Counter(),
        "post_discharge_next_type_counts": Counter(),
        "post_discharge_prev_gap_buckets": Counter(),
        "post_discharge_next_gap_buckets": Counter(),
        "post_discharge_duration_buckets": Counter(),
        "post_discharge_token_categories": Counter(),
        "post_discharge_same_day_carry": Counter(),
        "post_discharge_samples": [],
        "residual_unknown_total": 0,
        "residual_unknown_position_counts": Counter(),
        "residual_unknown_prev_type_counts": Counter(),
        "residual_unknown_next_type_counts": Counter(),
        "residual_unknown_samples": [],
    }
    example_rows: List[Dict[str, Any]] = []
    batch_timelines: List[List[EventToken]] = []

    started_at = time.time()
    total = len(subject_ids)
    for idx, sid in enumerate(subject_ids, start=1):
        audited = _audit_subject_tokenization(
            db,
            sid,
            encoders=encoders,
            codebook=artifacts.structural_codebook,
            artifacts=artifacts,
            struct_id2code=struct_id2code,
        )
        timeline = audited["timeline"]
        batch_timelines.append(timeline)

        total_tokens += len(timeline)
        raw_events_by_category.update(audited["raw_events_by_category"])
        dropped_events_by_category.update(audited["dropped_events_by_category"])
        emitted_events_by_category.update(audited["emitted_events_by_category"])
        emitted_tokens_by_category.update(audited["emitted_tokens_by_category"])
        unknown_base_tokens_by_category.update(audited["unknown_base_tokens_by_category"])
        process_reroute_events_by_category.update(audited["process_reroute_events_by_category"])
        for cat_name, outcome_counter in audited["semantic_base_outcomes_by_category"].items():
            semantic_base_outcomes_by_category[cat_name].update(outcome_counter)
        observation_base_outcomes.update(audited.get("observation_base_outcomes", {}))
        family_counts.update(audited["family_counts"])
        for family, ids in audited["family_unique_ids"].items():
            family_unique_ids[family].update(ids)
        for family, min_id in audited["family_min_id"].items():
            family_min_id[family] = min(min_id, family_min_id.get(family, min_id))
        for family, max_id in audited["family_max_id"].items():
            family_max_id[family] = max(max_id, family_max_id.get(family, max_id))
        for cat_name, bundle_counter in audited["bundle_sizes_by_category"].items():
            bundle_sizes_by_category[cat_name].update(bundle_counter)
        frame_payload_kind_counts.update(audited.get("frame_payload_kind_counts", {}))
        for payload_kind, size_counter in audited.get("frame_bundle_sizes_by_payload_kind", {}).items():
            frame_bundle_sizes_by_payload_kind[payload_kind].update(size_counter)
        observation_frame_integrity.update(audited.get("observation_frame_integrity", {}))
        decoded_preview_kind_counts.update(audited.get("decoded_preview_kind_counts", {}))
        decoded_observation_integrity.update(audited.get("decoded_observation_integrity", {}))
        for section, counter in audited.get("medication_frame_attr_presence", {}).items():
            medication_frame_attr_presence.setdefault(str(section), Counter()).update(counter)

        specials, events = collator._split_special(timeline)
        all_windows = collator._segment_windows(events)
        kept_windows = all_windows[: max_windows]
        if len(all_windows) > max_windows:
            truncation_counts["subjects_truncated_by_max_windows"] += 1

        window_counts["subjects"] += 1
        window_counts["windows_total_pre_cap"] += len(all_windows)
        window_counts["windows_total_post_cap"] += len(kept_windows)
        chunked_windows = collator._chunk_windows(kept_windows, special_tokens=specials)
        chunk_counts["chunks_total_post_cap"] += sum(len(window.chunks) for window in chunked_windows)
        chunk_counts["semantic_windows_split_into_chunks"] += sum(1 for window in chunked_windows if len(window.chunks) > 1)
        for window in chunked_windows:
            if getattr(window, "fallback_window_type_source", None):
                window_type_fallback_sources[str(window.fallback_window_type_source)] += 1

        raw_win_types = [int(window.window_type_id) for window in chunked_windows]
        win_types = [collator._clamp_window_type_id(w) for w in raw_win_types]
        win_starts = [float(window.start_time_hours) for window in chunked_windows]
        for raw_w, clamped_w in zip(raw_win_types, win_types):
            window_type_raw_counts[int(raw_w)] += 1
            window_type_clamped_counts[int(clamped_w)] += 1
            if int(raw_w) != int(clamped_w):
                truncation_counts["windows_window_type_out_of_range"] += 1
        for wi, window in enumerate(chunked_windows):
            prefix_len = len(specials) + (1 if collator.window_markers.enabled else 0)
            suffix_len = 1 if collator.window_markers.enabled else 0
            budget = max(0, collator.max_len - prefix_len - suffix_len)
            if sum(len(chunk.tokens) for chunk in window.chunks) < len(window.tokens):
                truncation_counts["semantic_windows_truncated_by_max_chunks"] += 1
            if budget == 0 and len(window.tokens) > 0:
                truncation_counts["marker_only_windows"] += 1

            prev_type_name = (
                type_id2name.get(int(win_types[wi - 1]), str(int(win_types[wi - 1])))
                if wi > 0
                else "<none>"
            )
            next_type_name = (
                type_id2name.get(int(win_types[wi + 1]), str(int(win_types[wi + 1])))
                if wi + 1 < len(win_types)
                else "<none>"
            )
            prev_end = _window_end_time_hours(chunked_windows[wi - 1]) if wi > 0 else None
            cur_start = float(window.start_time_hours)
            cur_end = _window_end_time_hours(window)
            next_start_for_shape = float(chunked_windows[wi + 1].start_time_hours) if wi + 1 < len(chunked_windows) else None
            prev_gap_h = (cur_start - prev_end) if prev_end is not None else None
            next_gap_h = (next_start_for_shape - cur_end) if next_start_for_shape is not None else None
            duration_h = max(0.0, cur_end - cur_start)
            position_name = _window_position_name(wi, len(chunked_windows))
            token_categories = _window_category_counter(window)

            if int(win_types[wi]) == int(collator.window_markers.unk_type_id):
                unknown_window_total += 1
                unknown_window_opening_actions[str(window.opening_action)] += 1
                unknown_window_closing_actions[str(window.closing_action)] += 1
                if window.tokens:
                    first_tok = window.tokens[0]
                    try:
                        first_cat_name = TokenCategory(int(first_tok.category_id)).name
                    except Exception:
                        first_cat_name = f"CATEGORY::{int(first_tok.category_id)}"
                    unknown_window_first_categories[first_cat_name] += 1
                    unknown_window_first_families[
                        _family_name_for_token(first_tok, artifacts=artifacts)
                    ] += 1
                    if first_tok.cat_attrs is not None:
                        unknown_window_first_transition_action[
                            str(first_tok.cat_attrs.get("transition_action_id", "<none>"))
                        ] += 1
                        unknown_window_first_transition_type[
                            str(first_tok.cat_attrs.get("transition_window_type_id", "<none>"))
                        ] += 1
                        unknown_window_first_window_type_attr[
                            str(first_tok.cat_attrs.get("window_type_id", "<none>"))
                        ] += 1
                        unknown_window_first_struct_label[
                            str(first_tok.cat_attrs.get("struct_label_id", "<none>"))
                        ] += 1
                    if len(unknown_window_samples) < 32:
                        first_label = None
                        try:
                            dec = decode_timeline_tokens(
                                [first_tok],
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
                                structural_id2label=struct_id2label,
                                structural_id2code=struct_id2code,
                                special_id2name=SPECIAL_ID2NAME,
                            )
                            if dec:
                                first_label = (
                                    dec[0].get("label")
                                    or dec[0].get("raw_code")
                                    or dec[0].get("kind")
                                )
                        except Exception:
                            first_label = None

                        unknown_window_samples.append(
                            {
                                "subject_id": int(sid),
                                "window_index": int(wi),
                                "raw_window_type_id": int(raw_win_types[wi]),
                                "clamped_window_type_id": int(win_types[wi]),
                                "opening_action": window.opening_action,
                                "closing_action": window.closing_action,
                                "start_time_hours": float(window.start_time_hours),
                                "token_count": int(len(window.tokens)),
                                "chunk_count": int(len(window.chunks)),
                                "first_token_value_id": int(first_tok.value_id),
                                "first_token_category": first_cat_name,
                                "first_token_family": _family_name_for_token(
                                    first_tok,
                                    artifacts=artifacts,
                                ),
                                "first_token_label": first_label,
                                "first_transition_action_id": (
                                    int(first_tok.cat_attrs["transition_action_id"])
                                    if first_tok.cat_attrs is not None
                                    and "transition_action_id" in first_tok.cat_attrs
                                    else None
                                ),
                                "first_transition_window_type_id": (
                                    int(first_tok.cat_attrs["transition_window_type_id"])
                                    if first_tok.cat_attrs is not None
                                    and "transition_window_type_id" in first_tok.cat_attrs
                                    else None
                                ),
                                "first_window_type_id_attr": (
                                    int(first_tok.cat_attrs["window_type_id"])
                                    if first_tok.cat_attrs is not None
                                    and "window_type_id" in first_tok.cat_attrs
                                    else None
                                ),
                                "first_struct_label_id": (
                                    int(first_tok.cat_attrs["struct_label_id"])
                                    if first_tok.cat_attrs is not None
                                    and "struct_label_id" in first_tok.cat_attrs
                                    else None
                                ),
                            }
                        )

                trajectory_shape["residual_unknown_total"] += 1
                trajectory_shape["residual_unknown_position_counts"][position_name] += 1
                trajectory_shape["residual_unknown_prev_type_counts"][str(prev_type_name)] += 1
                trajectory_shape["residual_unknown_next_type_counts"][str(next_type_name)] += 1
                if len(trajectory_shape["residual_unknown_samples"]) < 24:
                    trajectory_shape["residual_unknown_samples"].append(
                        {
                            "subject_id": int(sid),
                            "window_index": int(wi),
                            "position": position_name,
                            "opening_action": window.opening_action,
                            "closing_action": window.closing_action,
                            "prev_type": str(prev_type_name),
                            "next_type": str(next_type_name),
                            "prev_gap_bucket": _gap_bucket_name(prev_gap_h),
                            "next_gap_bucket": _gap_bucket_name(next_gap_h),
                            "duration_bucket": _gap_bucket_name(duration_h),
                            "token_categories": {str(k): int(v) for k, v in token_categories.items()},
                        }
                    )

            if history_prefix_type_id is not None and int(win_types[wi]) == int(history_prefix_type_id):
                trajectory_shape["history_prefix_total"] += 1
                trajectory_shape["history_prefix_position_counts"][position_name] += 1
                trajectory_shape["history_prefix_next_type_counts"][str(next_type_name)] += 1
                trajectory_shape["history_prefix_prev_gap_buckets"][_gap_bucket_name(prev_gap_h)] += 1
                trajectory_shape["history_prefix_next_gap_buckets"][_gap_bucket_name(next_gap_h)] += 1
                trajectory_shape["history_prefix_duration_buckets"][_gap_bucket_name(duration_h)] += 1
                trajectory_shape["history_prefix_token_categories"].update(token_categories)
                if len(trajectory_shape["history_prefix_samples"]) < 24:
                    trajectory_shape["history_prefix_samples"].append(
                        {
                            "subject_id": int(sid),
                            "window_index": int(wi),
                            "position": position_name,
                            "opening_action": window.opening_action,
                            "closing_action": window.closing_action,
                            "next_type": str(next_type_name),
                            "prev_gap_bucket": _gap_bucket_name(prev_gap_h),
                            "next_gap_bucket": _gap_bucket_name(next_gap_h),
                            "duration_bucket": _gap_bucket_name(duration_h),
                            "token_categories": {str(k): int(v) for k, v in token_categories.items()},
                            "fallback_source": getattr(window, "fallback_window_type_source", None),
                        }
                    )

            if post_discharge_type_id is not None and int(win_types[wi]) == int(post_discharge_type_id):
                trajectory_shape["post_discharge_total"] += 1
                trajectory_shape["post_discharge_position_counts"][position_name] += 1
                trajectory_shape["post_discharge_next_type_counts"][str(next_type_name)] += 1
                trajectory_shape["post_discharge_prev_gap_buckets"][_gap_bucket_name(prev_gap_h)] += 1
                trajectory_shape["post_discharge_next_gap_buckets"][_gap_bucket_name(next_gap_h)] += 1
                trajectory_shape["post_discharge_duration_buckets"][_gap_bucket_name(duration_h)] += 1
                trajectory_shape["post_discharge_token_categories"].update(token_categories)
                carry_key = (
                    "same_day_or_short_carry"
                    if (duration_h <= 24.0 or (next_gap_h is not None and next_gap_h <= 24.0))
                    else "long_gap_or_inter_admission"
                )
                trajectory_shape["post_discharge_same_day_carry"][carry_key] += 1
                if len(trajectory_shape["post_discharge_samples"]) < 24:
                    trajectory_shape["post_discharge_samples"].append(
                        {
                            "subject_id": int(sid),
                            "window_index": int(wi),
                            "position": position_name,
                            "opening_action": window.opening_action,
                            "closing_action": window.closing_action,
                            "prev_type": str(prev_type_name),
                            "next_type": str(next_type_name),
                            "prev_gap_bucket": _gap_bucket_name(prev_gap_h),
                            "next_gap_bucket": _gap_bucket_name(next_gap_h),
                            "duration_bucket": _gap_bucket_name(duration_h),
                            "carry_class": carry_key,
                            "token_categories": {str(k): int(v) for k, v in token_categories.items()},
                        }
                    )

            next_type = win_types[wi + 1] if wi + 1 < len(win_types) else None
            next_start = win_starts[wi + 1] if wi + 1 < len(win_starts) else None
            for chunk in window.chunks:
                ids, *_ = collator._process_chunk(
                    chunk,
                    specials,
                    w_type_id=win_types[wi],
                    w_start_abs=win_starts[wi],
                    next_type_id=next_type,
                    next_start_abs=next_start,
                )
                if len(ids) > collator.max_len:
                    truncation_counts["chunks_overflow_internal_budget"] += 1
                _record_window_marker_usage(
                    ids=ids,
                    specials_count=len(specials),
                    collator=collator,
                    window_type_id=int(win_types[wi]),
                    next_type_id=(int(next_type) if next_type is not None else None),
                    chunk_is_last=bool(chunk.is_last_chunk),
                    totals=window_marker_totals,
                    prefix_type_counts=window_marker_prefix_types,
                    prefix_raw_id_counts=window_marker_prefix_raw_ids,
                    suffix_mode_counts=window_marker_suffix_modes,
                    suffix_next_type_counts=window_marker_suffix_next_types,
                    suffix_raw_id_counts=window_marker_suffix_raw_ids,
                )

        if len(example_rows) < example_subjects:
            example_rows.append(
                {
                    "subject_id": int(sid),
                    "decoded": decode_timeline_tokens(
                        timeline[:example_tokens],
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
                        structural_id2label=struct_id2label,
                        structural_id2code=struct_id2code,
                        special_id2name=SPECIAL_ID2NAME,
                    ),
                }
            )

        if len(batch_timelines) >= collate_batch_size:
            batch = collator(batch_timelines)
            numeric_mask_ones += int(batch["numeric_mask"].sum().item())
            attention_mask_ones += int(batch["attention_mask"].sum().item())
            event_numeric_mask_ones += int(batch["event_numeric_mask"].sum().item())
            event_attention_mask_ones += int(batch["event_attention_mask"].sum().item())
            window_mask_ones += int(batch["window_mask"].sum().item())
            chunk_mask_ones += int(batch["chunk_mask"].sum().item())
            win_mask = batch["window_mask"].to(dtype=torch.bool)
            window_type_zero += int(((batch["window_type_ids"] == 0) & win_mask).sum().item())
            window_type_total += int(win_mask.sum().item())
            batch_timelines = []

        _maybe_print_progress(
            "timeline",
            idx=idx,
            total=total,
            every=progress_every,
            started_at=started_at,
        )

    if batch_timelines:
        batch = collator(batch_timelines)
        numeric_mask_ones += int(batch["numeric_mask"].sum().item())
        attention_mask_ones += int(batch["attention_mask"].sum().item())
        event_numeric_mask_ones += int(batch["event_numeric_mask"].sum().item())
        event_attention_mask_ones += int(batch["event_attention_mask"].sum().item())
        window_mask_ones += int(batch["window_mask"].sum().item())
        chunk_mask_ones += int(batch["chunk_mask"].sum().item())
        win_mask = batch["window_mask"].to(dtype=torch.bool)
        window_type_zero += int(((batch["window_type_ids"] == 0) & win_mask).sum().item())
        window_type_total += int(win_mask.sum().item())

    timeline_summary = {
        "subjects_scanned": len(subject_ids),
        "total_tokens": int(total_tokens),
        "raw_events_by_category": _as_plain_counter(raw_events_by_category),
        "dropped_events_by_category": _as_plain_counter(dropped_events_by_category),
        "emitted_events_by_category": _as_plain_counter(emitted_events_by_category),
        "emitted_tokens_by_category": _as_plain_counter(emitted_tokens_by_category),
        "unknown_base_tokens_by_category": _as_plain_counter(unknown_base_tokens_by_category),
        "process_reroute_events_by_category": _as_plain_counter(process_reroute_events_by_category),
        "semantic_base_outcomes_by_category": {
            k: _as_plain_counter(v) for k, v in semantic_base_outcomes_by_category.items()
        },
        "observation_base_outcomes": _as_plain_counter(observation_base_outcomes),
        "semantic_effective_capture_by_category": {},
        "bundle_sizes_by_category": {
            k: _as_plain_counter(v) for k, v in bundle_sizes_by_category.items()
        },
        "avg_tokens_per_emitted_event_by_category": {
            k: (
                float(emitted_tokens_by_category[k]) / float(emitted_events_by_category[k])
                if emitted_events_by_category[k] > 0
                else 0.0
            )
            for k in emitted_events_by_category
        },
        "family_counts": _as_plain_counter(family_counts),
        "family_ranges": {
            family: {
                "unique_ids": len(ids),
                "min_id": int(family_min_id[family]),
                "max_id": int(family_max_id[family]),
            }
            for family, ids in family_unique_ids.items()
        },
        "frame_payload_kind_counts": _as_plain_counter(frame_payload_kind_counts),
        "frame_bundle_sizes_by_payload_kind": {
            str(kind): _as_plain_counter(counter)
            for kind, counter in frame_bundle_sizes_by_payload_kind.items()
        },
        "observation_frame_integrity": _as_plain_counter(observation_frame_integrity),
        "decoded_preview_kind_counts": _as_plain_counter(decoded_preview_kind_counts),
        "decoded_observation_integrity": _as_plain_counter(decoded_observation_integrity),
        "medication_frame_attr_presence": {
            str(section): _as_plain_counter(counter)
            for section, counter in medication_frame_attr_presence.items()
        },
        "measurement_effective_capture_by_path": {},
    }

    raw_meas_events = int(raw_events_by_category.get("MEASUREMENT", 0))
    cvae_meas_events = int(family_counts.get("measurement_code", 0))
    obs_meas_events = int(family_counts.get("observation_code", 0))
    kept_meas_events = int(cvae_meas_events + obs_meas_events)
    timeline_summary["measurement_effective_capture_by_path"] = {
        "raw_measurement_events": int(raw_meas_events),
        "cvae_events": int(cvae_meas_events),
        "obs_events": int(obs_meas_events),
        "obs_exact_events": int(observation_base_outcomes.get("exact", 0)),
        "obs_mixed_events": int(observation_base_outcomes.get("mixed", 0)),
        "obs_hash_events": int(observation_base_outcomes.get("hash", 0)),
        "obs_drop_events": int(observation_base_outcomes.get("drop", 0)),
        "kept_events": int(kept_meas_events),
        "dropped_events": int(max(0, raw_meas_events - kept_meas_events)),
        "kept_rate": (
            float(kept_meas_events) / float(raw_meas_events)
            if raw_meas_events > 0
            else 0.0
        ),
    }

    semantic_capture: Dict[str, Dict[str, Any]] = {}
    semantic_cats = ["DIAGNOSIS", "PROCEDURE", "MEDICATION"]
    for cat_name in semantic_cats:
        outcomes = semantic_base_outcomes_by_category.get(cat_name, Counter())
        raw_total = int(raw_events_by_category.get(cat_name, 0))
        process_reroute = int(process_reroute_events_by_category.get(cat_name, 0))
        semantic_total = max(0, raw_total - process_reroute)
        exact = int(outcomes.get("exact", 0))
        canonicalized = int(outcomes.get("canonicalized", 0))
        parent_lookup = int(outcomes.get("parent_lookup", 0))
        crosswalk_lookup = int(outcomes.get("crosswalk_lookup", 0))
        lexical_bridge = int(outcomes.get("lexical_bridge", 0))
        medtok_base = exact + canonicalized + parent_lookup + crosswalk_lookup + lexical_bridge
        residual_exact = int(outcomes.get("residual_exact", 0))
        residual_hash = int(outcomes.get("residual_hash", 0))
        residual_base = residual_exact + residual_hash
        unknown_base = int(outcomes.get("unk", 0))
        drop = int(outcomes.get("drop", 0))
        dropped = int(outcomes.get("dropped_or_no_encoder", 0))
        structural_only = int(outcomes.get("structural_only", 0))
        mapped = medtok_base + residual_base
        semantic_capture[cat_name] = {
            "raw_total": raw_total,
            "process_reroute": process_reroute,
            "semantic_total": semantic_total,
            "exact": exact,
            "canonicalized": canonicalized,
            "parent_lookup": parent_lookup,
            "crosswalk_lookup": crosswalk_lookup,
            "lexical_bridge": lexical_bridge,
            "explicit_medtok_base": medtok_base,
            "residual_exact": residual_exact,
            "residual_hash": residual_hash,
            "residual": residual_base,
            "mapped_non_unk": mapped,
            "unk": unknown_base,
            "drop": drop,
            "dropped_or_no_encoder": dropped,
            "structural_only": structural_only,
            "mapped_rate_over_semantic_total": (
                float(mapped) / float(semantic_total) if semantic_total > 0 else 0.0
            ),
            "medtok_only_rate_over_semantic_total": (
                float(medtok_base) / float(semantic_total) if semantic_total > 0 else 0.0
            ),
        }
    timeline_summary["semantic_effective_capture_by_category"] = semantic_capture

    collation_summary = {
        "windows": _as_plain_counter(window_counts),
        "chunks": _as_plain_counter(chunk_counts),
        "truncation": _as_plain_counter(truncation_counts),
        "numeric_mask_density_vs_attended": (
            float(numeric_mask_ones) / float(attention_mask_ones)
            if attention_mask_ones > 0
            else 0.0
        ),
        "event_numeric_mask_density_vs_attended": (
            float(event_numeric_mask_ones) / float(event_attention_mask_ones)
            if event_attention_mask_ones > 0
            else 0.0
        ),
        "avg_windows_per_subject": (
            float(window_counts["windows_total_post_cap"]) / float(window_counts["subjects"])
            if window_counts["subjects"] > 0
            else 0.0
        ),
        "window_type_unk_frac": (
            float(window_type_zero) / float(window_type_total)
            if window_type_total > 0
            else 0.0
        ),
        "window_type_raw_counts": _as_plain_counter(window_type_raw_counts),
        "window_type_clamped_counts": _as_plain_counter(window_type_clamped_counts),
        "window_type_fallback_source_counts": _as_plain_counter(window_type_fallback_sources),
        "window_marker_usage": {
            "chunks_with_markers": int(window_marker_totals.get("chunks_with_markers", 0)),
            "marker_tokens_total": int(window_marker_totals.get("marker_tokens_total", 0)),
            "prefix_type_counts": _as_plain_counter(window_marker_prefix_types),
            "prefix_raw_id_counts": _as_plain_counter(window_marker_prefix_raw_ids),
            "suffix_mode_counts": _as_plain_counter(window_marker_suffix_modes),
            "suffix_next_type_counts": _as_plain_counter(window_marker_suffix_next_types),
            "suffix_raw_id_counts": _as_plain_counter(window_marker_suffix_raw_ids),
            "end_token_id": int(collator._window_end_token_id()),
            "continue_token_id": int(collator._window_continue_token_id()),
        },
        "unknown_window_diagnostics": {
            "total_unknown_windows": int(unknown_window_total),
            "by_opening_action": _as_plain_counter(unknown_window_opening_actions),
            "by_closing_action": _as_plain_counter(unknown_window_closing_actions),
            "by_first_token_category": _as_plain_counter(unknown_window_first_categories),
            "by_first_token_family": _as_plain_counter(unknown_window_first_families),
            "first_token_transition_action_id": _as_plain_counter(unknown_window_first_transition_action),
            "first_token_transition_window_type_id": _as_plain_counter(unknown_window_first_transition_type),
            "first_token_window_type_attr": _as_plain_counter(unknown_window_first_window_type_attr),
            "first_token_struct_label_id": _as_plain_counter(unknown_window_first_struct_label),
            "samples": unknown_window_samples,
        },
        "trajectory_shape_analysis": {
            "history_prefix_total": int(trajectory_shape["history_prefix_total"]),
            "history_prefix_position_counts": _as_plain_counter(trajectory_shape["history_prefix_position_counts"]),
            "history_prefix_next_type_counts": _as_plain_counter(trajectory_shape["history_prefix_next_type_counts"]),
            "history_prefix_prev_gap_buckets": _as_plain_counter(trajectory_shape["history_prefix_prev_gap_buckets"]),
            "history_prefix_next_gap_buckets": _as_plain_counter(trajectory_shape["history_prefix_next_gap_buckets"]),
            "history_prefix_duration_buckets": _as_plain_counter(trajectory_shape["history_prefix_duration_buckets"]),
            "history_prefix_token_categories": _as_plain_counter(trajectory_shape["history_prefix_token_categories"]),
            "history_prefix_samples": list(trajectory_shape["history_prefix_samples"]),
            "post_discharge_total": int(trajectory_shape["post_discharge_total"]),
            "post_discharge_position_counts": _as_plain_counter(trajectory_shape["post_discharge_position_counts"]),
            "post_discharge_next_type_counts": _as_plain_counter(trajectory_shape["post_discharge_next_type_counts"]),
            "post_discharge_prev_gap_buckets": _as_plain_counter(trajectory_shape["post_discharge_prev_gap_buckets"]),
            "post_discharge_next_gap_buckets": _as_plain_counter(trajectory_shape["post_discharge_next_gap_buckets"]),
            "post_discharge_duration_buckets": _as_plain_counter(trajectory_shape["post_discharge_duration_buckets"]),
            "post_discharge_token_categories": _as_plain_counter(trajectory_shape["post_discharge_token_categories"]),
            "post_discharge_same_day_carry": _as_plain_counter(trajectory_shape["post_discharge_same_day_carry"]),
            "post_discharge_samples": list(trajectory_shape["post_discharge_samples"]),
            "residual_unknown_total": int(trajectory_shape["residual_unknown_total"]),
            "residual_unknown_position_counts": _as_plain_counter(trajectory_shape["residual_unknown_position_counts"]),
            "residual_unknown_prev_type_counts": _as_plain_counter(trajectory_shape["residual_unknown_prev_type_counts"]),
            "residual_unknown_next_type_counts": _as_plain_counter(trajectory_shape["residual_unknown_next_type_counts"]),
            "residual_unknown_samples": list(trajectory_shape["residual_unknown_samples"]),
        },
        "attended_tokens": int(attention_mask_ones),
        "active_windows": int(window_mask_ones),
        "active_chunks": int(chunk_mask_ones),
    }

    return {
        "timeline": timeline_summary,
        "collation": collation_summary,
        "examples": example_rows,
    }


def _print_summary(payload: Mapping[str, Any], *, top_k: int) -> None:
    raw = payload["raw"]
    timeline = payload["timeline"]
    coll = payload["collation"]

    print("Raw events by category:", raw["events_by_category"])
    print("Measurement status:", raw.get("measurement_status", {}))
    print("MedTok hits:", raw.get("medtok_hits", {}))
    print("MedTok resolution stages:", raw.get("medtok_resolution_stages", {}))
    print("Top prefixes:")
    for row in raw["top_prefixes"][:top_k]:
        print(f"  {row['key']}: {row['count']}")
    print("Timeline emitted tokens by category:", timeline["emitted_tokens_by_category"])
    print("Timeline dropped events by category:", timeline["dropped_events_by_category"])
    print("Measurement effective capture:", timeline.get("measurement_effective_capture_by_path", {}))
    if timeline.get("observation_base_outcomes"):
        print("Observation base outcomes:", timeline.get("observation_base_outcomes", {}))
    if timeline.get("observation_frame_integrity"):
        print("Observation frame integrity:", timeline.get("observation_frame_integrity", {}))
    if timeline.get("decoded_observation_integrity"):
        print("Decoded observation integrity:", timeline.get("decoded_observation_integrity", {}))
    if raw.get("medication_raw_attr_presence"):
        print("Medication raw attr presence:", raw.get("medication_raw_attr_presence", {}))
    if timeline.get("medication_frame_attr_presence"):
        print("Medication frame attr presence:", timeline.get("medication_frame_attr_presence", {}))
    print("Average tokens per emitted event:", timeline["avg_tokens_per_emitted_event_by_category"])
    if timeline.get("frame_payload_kind_counts"):
        print("Frame payload kinds:", timeline.get("frame_payload_kind_counts", {}))
    print("Semantic effective capture by category:", timeline.get("semantic_effective_capture_by_category", {}))
    print("Collation windows:", coll["windows"])
    print("Collation chunks:", coll.get("chunks", {}))
    print("Collation truncation:", coll["truncation"])
    print("Collation numeric_mask_density_vs_attended:", f"{coll['numeric_mask_density_vs_attended']:.4f}")
    print("Collation event_numeric_mask_density_vs_attended:", f"{coll['event_numeric_mask_density_vs_attended']:.4f}")
    print("Collation window_type_unk_frac:", f"{coll['window_type_unk_frac']:.4f}")
    if coll.get("window_type_fallback_source_counts"):
        print("Window type fallback sources:", coll.get("window_type_fallback_source_counts", {}))
    traj = coll.get("trajectory_shape_analysis", {})
    if traj:
        print("Trajectory shape history_prefix_total:", int(traj.get("history_prefix_total", 0)))
        print("Trajectory shape history_prefix_next_type_counts:", traj.get("history_prefix_next_type_counts", {}))
        print("Trajectory shape post_discharge_total:", int(traj.get("post_discharge_total", 0)))
        print("Trajectory shape post_discharge_same_day_carry:", traj.get("post_discharge_same_day_carry", {}))
        print("Trajectory shape residual_unknown_total:", int(traj.get("residual_unknown_total", 0)))
    print("Collation window markers:", coll.get("window_marker_usage", {}))
    unknown_diag = coll.get("unknown_window_diagnostics", {})
    if unknown_diag:
        print("Unknown windows by opening action:", unknown_diag.get("by_opening_action", {}))
        print("Unknown windows by closing action:", unknown_diag.get("by_closing_action", {}))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit raw MEDS metadata, tokenization, and collation behavior."
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=100)
    ap.add_argument("--sample_seed", type=int, default=None, help="Optional seed to randomly sample subject ids from split before truncation.")
    ap.add_argument("--subject_ids", default=None, help="Comma-separated subject ids to inspect.")
    ap.add_argument("--top_k", type=int, default=20)
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
    ap.add_argument("--code2id_pt", default=None)
    ap.add_argument("--stats_pt", default=None)
    ap.add_argument("--cvae_ckpt", default=None)
    ap.add_argument("--tokenizer_ckpt", default=None)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--max_chunks_per_window", type=int, default=8)
    ap.add_argument("--max_len_per_window", type=int, default=128)
    ap.add_argument("--collate_batch_size", type=int, default=16)
    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=40_000)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)
    ap.add_argument("--example_subjects", type=int, default=3)
    ap.add_argument("--example_tokens", type=int, default=40)
    ap.add_argument("--progress_every", type=int, default=0)
    ap.add_argument("--skip_raw_summary", action="store_true")
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
    residual_tail_policies = _resolve_residual_tail_policies(
        tokenization_contract=tokenization_contract,
    )
    semantic_resolvers = _build_semantic_resolution_encoders(
        artifacts=artifacts,
        residual_enabled=bool(residual_enabled),
        residual_buckets=int(residual_buckets),
        residual_offsets=dict(residual_offsets),
        residual_tail_policies=residual_tail_policies,
    )
    db = mr.SubjectDatabase(args.meds_reader_db)
    subject_ids = _parse_subject_ids(args.subject_ids)
    if not subject_ids:
        subject_ids = _load_subject_ids(
            args.splits_parquet,
            args.split,
            args.max_subjects,
            sample_seed=args.sample_seed,
        )
    if not subject_ids:
        raise ValueError(f"No subject IDs found for split={args.split}")

    if args.skip_raw_summary:
        raw_summary = {
            "subjects_scanned": len(subject_ids),
            "sample_seed": args.sample_seed,
            "total_events": 0,
            "events_by_category": {},
            "numeric_events_by_category": {},
            "measurement_status": {},
            "medtok_hits": {},
            "medtok_resolution_stages": {},
            "top_prefixes": [],
            "top_other_codes": [],
            "top_codes_by_category": {},
            "top_medtok_misses": {},
            "marker_like_counts": {},
            "structural_labels": {},
            "structural_boundary_labels": {},
            "structural_overlay_labels": {},
        }
        structural_raw_codes = set()
    else:
        raw_summary, structural_raw_codes = _summarize_raw_subjects(
            db,
            subject_ids,
            artifacts=artifacts,
            semantic_resolvers=semantic_resolvers,
            top_k=args.top_k,
            progress_every=args.progress_every,
        )

    struct_codes_union = set(structural_surface_vocab_codes(artifacts.structural_codebook))
    if structural_raw_codes:
        struct_codes_union.update(
            str(surface)
            for surface in (
                structural_surface_code(
                    code,
                    codebook=artifacts.structural_codebook,
                    routed_category=TokenCategory.STRUCTURAL,
                )
                for code in structural_raw_codes
            )
            if surface
        )
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)
    struct_id2code = invert_code2id(struct_vocab.code2id)

    meas_cfg = _build_measurement_config(args, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError(
            "Real measurement encoder artifacts are required: pass --code2id_pt, --stats_pt, "
            "--cvae_ckpt, and --tokenizer_ckpt."
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

    downstream = _summarize_tokenization_and_collation(
        db,
        subject_ids,
        encoders=encoders,
        artifacts=artifacts,
        struct_id2code=struct_id2code,
        max_windows=args.max_windows,
        max_chunks_per_window=args.max_chunks_per_window,
        max_len_per_window=args.max_len_per_window,
        collate_batch_size=args.collate_batch_size,
        example_subjects=args.example_subjects,
        example_tokens=args.example_tokens,
        window_markers=window_markers_cfg,
        segmentation_config=segmentation_cfg,
        progress_every=args.progress_every,
    )

    payload = {
        "config": {
            "split": args.split,
            "subjects_scanned": len(subject_ids),
            "max_windows": args.max_windows,
            "max_chunks_per_window": args.max_chunks_per_window,
            "max_len_per_window": args.max_len_per_window,
            "collate_batch_size": args.collate_batch_size,
            "measurement_num_codebooks": artifacts.measurement_num_codebooks,
            "measurement_codebook_size": artifacts.measurement_codebook_size,
            "measurement_stride": artifacts.measurement_stride,
            "medtok_parent_lookup_entries": len(artifacts.medtok_parent_lookup),
            "medtok_crosswalk_entries": {
                k: int(len(v)) for k, v in artifacts.medtok_crosswalks.items()
            },
            "residual_fallback_vocab_entries": {
                k: int(max(0, len(v.code2id) - 1))
                for k, v in artifacts.residual_fallback_vocabs.items()
            },
            "observation_vocab_entries": {
                "code": int(max(0, len(artifacts.obs_code_vocab.code2id) - 1))
                if artifacts.obs_code_vocab is not None
                else 0,
                "value": int(max(0, len(artifacts.obs_value_vocab.code2id) - 1))
                if artifacts.obs_value_vocab is not None
                else 0,
            },
            "observation_tail_policy": str(artifacts.obs_tail_policy),
            "medtok_crosswalk_json": getattr(args, "medtok_crosswalk_json", None),
            "tokenization_yaml": args.tokenization_yaml,
            "residual_fallback_enabled": bool(residual_enabled),
            "residual_fallback_buckets": int(residual_buckets),
            "residual_tail_policies": dict(residual_tail_policies),
            "diag_residual_offset": residual_offsets.get("diagnosis"),
            "proc_residual_offset": residual_offsets.get("procedure"),
            "med_residual_offset": residual_offsets.get("medication"),
            "window_markers": {
                "enabled": bool(window_markers_cfg.enabled),
                "end_mode": str(window_markers_cfg.end_mode),
                "type_token_offset": int(window_markers_cfg.type_token_offset),
                "num_types": int(window_markers_cfg.num_types),
                "end_token_id": (
                    int(window_markers_cfg.end_token_id)
                    if window_markers_cfg.end_token_id is not None
                    else None
                ),
                "continue_token_id": (
                    int(window_markers_cfg.continue_token_id)
                    if window_markers_cfg.continue_token_id is not None
                    else None
                ),
                "unk_type_id": int(window_markers_cfg.unk_type_id),
            },
            "window_segmentation": {
                "bundle_gap_hours": float(segmentation_cfg.bundle_gap_hours),
                "bundle_max_index_gap": int(segmentation_cfg.bundle_max_index_gap),
                "merge_transition_chains": bool(segmentation_cfg.merge_transition_chains),
                "chain_gap_hours": float(segmentation_cfg.chain_gap_hours),
                "chain_max_intervening_tokens": int(segmentation_cfg.chain_max_intervening_tokens),
                "rebalance_dense_windows": bool(segmentation_cfg.rebalance_dense_windows),
                "rebalance_target_frac": float(segmentation_cfg.rebalance_target_frac),
                "rebalance_min_tokens": int(segmentation_cfg.rebalance_min_tokens),
                "rebalance_tail_tokens": int(segmentation_cfg.rebalance_tail_tokens),
                "unk_window_type_id": int(segmentation_cfg.unk_window_type_id),
                "default_first_window_type_id": (
                    int(segmentation_cfg.default_first_window_type_id)
                    if segmentation_cfg.default_first_window_type_id is not None
                    else None
                ),
                "post_discharge_window_type_id": (
                    int(segmentation_cfg.post_discharge_window_type_id)
                    if segmentation_cfg.post_discharge_window_type_id is not None
                    else None
                ),
                "propagate_prev_type_for_unknown_windows": bool(
                    segmentation_cfg.propagate_prev_type_for_unknown_windows
                ),
            },
        },
        "raw": raw_summary,
        "timeline": downstream["timeline"],
        "collation": downstream["collation"],
        "examples": downstream["examples"],
    }

    _print_summary(payload, top_k=min(args.top_k, 10))

    if args.output_json:
        out_fp = Path(args.output_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote audit JSON to {out_fp}")


if __name__ == "__main__":
    main()
