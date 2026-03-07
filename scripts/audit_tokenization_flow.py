#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meds_reader as mr

from src.ehr_hier.data.demographics import (
    infer_birth_timestamp,
    infer_event_age_years,
    infer_subject_sex,
)
from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.structural_codes import (
    StructuralCodebook,
    load_structural_codebook_yaml,
)
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
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
    ensure_list,
    medication_filter,
    procedure_filter,
)
from src.ehr_hier.tokenizers.medtok_loader import (
    build_vocab_from_code2embeddings,
    CategoryVocab,
    load_attr_vocab,
    load_medtok_vocab,
)
from src.ehr_hier.tokenizers.medtok_attr_encoder import load_parent_lookup_from_codes_parquet
from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig


SPECIAL_ID2NAME = {
    0: "PAD",
    1: "PT_CLS",
    2: "SEP",
    3: "MASK",
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


def _load_manifest() -> Dict[str, Any]:
    fp = PROJECT_ROOT / "artifacts" / "vocab_manifest.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def _load_subject_ids(splits_parquet: str, split: str, max_subjects: int) -> List[int]:
    split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]]
    aliases = {"val": "tuning", "test": "held_out"}
    target_split = aliases.get(split, split)
    ids = split_df.loc[split_df["split"] == target_split, "subject_id"].astype("int64").tolist()
    if max_subjects > 0:
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


def _has_medtok_match(raw_code: object, vocab: CategoryVocab, canonicalize_fn) -> bool:
    if raw_code is None:
        return False
    for cand in ensure_list(canonicalize_fn(raw_code)):
        if vocab.maybe_encode(cand) is not None:
            return True
    return False


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
    manifest = _load_manifest()
    structural_codebook = (
        load_structural_codebook_yaml(
            args.structural_yaml,
            default_offset=_offset(manifest, "structural", 2_200_000),
        )
        if args.structural_yaml
        else None
    )

    medtok_code2embeds = Path(args.medtok_code2embeds) if args.medtok_code2embeds else None
    medtok_vocab_dir = Path(args.medtok_vocab_dir)
    medtok_attr_dir = Path(args.medtok_attr_dir)

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
    )


def _summarize_raw_subjects(
    db: mr.SubjectDatabase,
    subject_ids: List[int],
    *,
    artifacts: AuditArtifacts,
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
    medtok_miss_samples: Dict[str, Counter[str]] = {
        "diagnosis": Counter(),
        "procedure": Counter(),
        "medication": Counter(),
    }
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
                if _has_medtok_match(code, artifacts.diag_vocab, canonicalize_diagnosis_code):
                    medtok_hits["diagnosis"]["hit"] += 1
                else:
                    medtok_hits["diagnosis"]["miss"] += 1
                    medtok_miss_samples["diagnosis"][code_str] += 1
            elif category == TokenCategory.PROCEDURE:
                if _has_medtok_match(code, artifacts.proc_vocab, canonicalize_procedure_code):
                    medtok_hits["procedure"]["hit"] += 1
                else:
                    medtok_hits["procedure"]["miss"] += 1
                    medtok_miss_samples["procedure"][code_str] += 1
            elif category == TokenCategory.MEDICATION:
                if _has_medtok_match(code, artifacts.med_vocab, canonicalize_medication_code):
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
    struct_action_offset = _offset(manifest, "structural_action", 2_400_000)
    struct_entity_offset = _offset(manifest, "structural_entity", 2_420_000)

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
    if obs_value_offset <= value_id < struct_action_offset:
        return "observation_value"
    if obs_code_offset <= value_id < obs_value_offset:
        return "observation_code"
    if struct_entity_offset <= value_id:
        return "structural_entity"
    if struct_action_offset <= value_id < struct_entity_offset:
        return "structural_action"
    if value_id >= struct_offset:
        return "structural"
    return "special_or_window"


def _audit_subject_tokenization(
    db: mr.SubjectDatabase,
    subject_id: int,
    *,
    encoders: Dict[TokenCategory, Any],
    codebook: Optional[StructuralCodebook],
    artifacts: AuditArtifacts,
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
    family_counts = Counter()
    family_unique_ids: Dict[str, set[int]] = defaultdict(set)
    family_min_id: Dict[str, int] = {}
    family_max_id: Dict[str, int] = {}

    last_emitted_time = None
    timeline = build_subject_timeline(
        db=db,
        subject_id=int(subject_id),
        encoders=encoders,
        structural_codebook=codebook,
    )

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

        toks = encoder.encode_event(ev_view, dt_hours=dt_hours)
        if toks:
            emitted_for_event += len(toks)
            emitted_events_by_category[category.name] += 1
            emitted_tokens_by_category[category.name] += emitted_for_event
            bundle_sizes_by_category[category.name][emitted_for_event] += 1
            base_tok = toks[0]
            unk_gid = getattr(encoder, "unk_gid", None)
            if unk_gid is not None and int(base_tok.value_id) == int(unk_gid):
                unknown_base_tokens_by_category[category.name] += 1
            if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                if is_process_reroute:
                    semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
                elif unk_gid is not None and int(base_tok.value_id) == int(unk_gid):
                    semantic_base_outcomes_by_category[category.name]["unknown_base"] += 1
                elif int(base_tok.cat_attrs.get("residual_fallback", 0) or 0) == 1:
                    semantic_base_outcomes_by_category[category.name]["residual_base"] += 1
                else:
                    semantic_base_outcomes_by_category[category.name]["medtok_base"] += 1
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
                dropped_events_by_category[category.name] += 1
                if category in {TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION}:
                    if is_process_reroute:
                        semantic_base_outcomes_by_category[category.name]["process_reroute"] += 1
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
        "bundle_sizes_by_category": bundle_sizes_by_category,
        "family_counts": family_counts,
        "family_unique_ids": family_unique_ids,
        "family_min_id": family_min_id,
        "family_max_id": family_max_id,
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
    progress_every: int = 0,
) -> Dict[str, Any]:
    struct_id2label = _make_structural_id2label(artifacts.structural_codebook)
    collator = AETHierarchicalCollator(
        max_windows=max_windows,
        max_chunks_per_window=max_chunks_per_window,
        max_len_per_window=max_len_per_window,
        pad_id=0,
        window_markers=WindowMarkerConfig(),
    )

    total_tokens = 0
    emitted_tokens_by_category = Counter()
    raw_events_by_category = Counter()
    dropped_events_by_category = Counter()
    emitted_events_by_category = Counter()
    unknown_base_tokens_by_category = Counter()
    process_reroute_events_by_category = Counter()
    semantic_base_outcomes_by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    bundle_sizes_by_category: Dict[str, Counter[int]] = defaultdict(Counter)
    family_counts = Counter()
    family_unique_ids: Dict[str, set[int]] = defaultdict(set)
    family_min_id: Dict[str, int] = {}
    family_max_id: Dict[str, int] = {}

    window_counts = Counter()
    chunk_counts = Counter()
    truncation_counts = Counter()
    numeric_mask_ones = 0
    attention_mask_ones = 0
    window_mask_ones = 0
    chunk_mask_ones = 0
    window_type_zero = 0
    window_type_total = 0
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
        family_counts.update(audited["family_counts"])
        for family, ids in audited["family_unique_ids"].items():
            family_unique_ids[family].update(ids)
        for family, min_id in audited["family_min_id"].items():
            family_min_id[family] = min(min_id, family_min_id.get(family, min_id))
        for family, max_id in audited["family_max_id"].items():
            family_max_id[family] = max(max_id, family_max_id.get(family, max_id))
        for cat_name, bundle_counter in audited["bundle_sizes_by_category"].items():
            bundle_sizes_by_category[cat_name].update(bundle_counter)

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

        win_types = [int(window.window_type_id) for window in chunked_windows]
        win_starts = [float(window.start_time_hours) for window in chunked_windows]
        for wi, window in enumerate(chunked_windows):
            prefix_len = len(specials) + (1 if collator.window_markers.enabled else 0)
            suffix_len = 1 if collator.window_markers.enabled else 0
            budget = max(0, collator.max_len - prefix_len - suffix_len)
            if sum(len(chunk.tokens) for chunk in window.chunks) < len(window.tokens):
                truncation_counts["semantic_windows_truncated_by_max_chunks"] += 1
            if budget == 0 and len(window.tokens) > 0:
                truncation_counts["marker_only_windows"] += 1

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
                        structural_action_offset=_offset(artifacts.manifest, "structural_action", 2_400_000),
                        structural_entity_offset=_offset(artifacts.manifest, "structural_entity", 2_420_000),
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
    }

    semantic_capture: Dict[str, Dict[str, Any]] = {}
    semantic_cats = ["DIAGNOSIS", "PROCEDURE", "MEDICATION"]
    for cat_name in semantic_cats:
        outcomes = semantic_base_outcomes_by_category.get(cat_name, Counter())
        raw_total = int(raw_events_by_category.get(cat_name, 0))
        process_reroute = int(process_reroute_events_by_category.get(cat_name, 0))
        semantic_total = max(0, raw_total - process_reroute)
        medtok_base = int(outcomes.get("medtok_base", 0))
        residual_base = int(outcomes.get("residual_base", 0))
        unknown_base = int(outcomes.get("unknown_base", 0))
        dropped = int(outcomes.get("dropped_or_no_encoder", 0))
        structural_only = int(outcomes.get("structural_only", 0))
        mapped = medtok_base + residual_base
        semantic_capture[cat_name] = {
            "raw_total": raw_total,
            "process_reroute": process_reroute,
            "semantic_total": semantic_total,
            "medtok_base": medtok_base,
            "residual_base": residual_base,
            "mapped_non_unk": mapped,
            "unknown_base": unknown_base,
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
    print("Top prefixes:")
    for row in raw["top_prefixes"][:top_k]:
        print(f"  {row['key']}: {row['count']}")
    print("Timeline emitted tokens by category:", timeline["emitted_tokens_by_category"])
    print("Timeline dropped events by category:", timeline["dropped_events_by_category"])
    print("Average tokens per emitted event:", timeline["avg_tokens_per_emitted_event_by_category"])
    print("Semantic effective capture by category:", timeline.get("semantic_effective_capture_by_category", {}))
    print("Collation windows:", coll["windows"])
    print("Collation chunks:", coll.get("chunks", {}))
    print("Collation truncation:", coll["truncation"])
    print("Collation numeric_mask_density_vs_attended:", f"{coll['numeric_mask_density_vs_attended']:.4f}")
    print("Collation window_type_unk_frac:", f"{coll['window_type_unk_frac']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit raw MEDS metadata, tokenization, and collation behavior."
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=100)
    ap.add_argument("--subject_ids", default=None, help="Comma-separated subject ids to inspect.")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default="artifacts/medtok")
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
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
    db = mr.SubjectDatabase(args.meds_reader_db)
    subject_ids = _parse_subject_ids(args.subject_ids)
    if not subject_ids:
        subject_ids = _load_subject_ids(args.splits_parquet, args.split, args.max_subjects)
    if not subject_ids:
        raise ValueError(f"No subject IDs found for split={args.split}")

    if args.skip_raw_summary:
        raw_summary = {
            "subjects_scanned": len(subject_ids),
            "total_events": 0,
            "events_by_category": {},
            "numeric_events_by_category": {},
            "measurement_status": {},
            "medtok_hits": {},
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
            top_k=args.top_k,
            progress_every=args.progress_every,
        )

    struct_codes_union = set(structural_raw_codes)
    if artifacts.structural_codebook is not None:
        struct_codes_union.update(artifacts.structural_codebook.code2label.keys())
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
        enable_residual_fallback=not bool(args.disable_residual_fallback),
        residual_fallback_buckets=int(args.residual_fallback_buckets),
        residual_fallback_offsets={
            k: int(v)
            for k, v in {
                "diagnosis": args.diag_residual_offset,
                "procedure": args.proc_residual_offset,
                "medication": args.med_residual_offset,
            }.items()
            if v is not None
        },
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
            "residual_fallback_enabled": not bool(args.disable_residual_fallback),
            "residual_fallback_buckets": int(args.residual_fallback_buckets),
            "diag_residual_offset": args.diag_residual_offset,
            "proc_residual_offset": args.proc_residual_offset,
            "med_residual_offset": args.med_residual_offset,
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
