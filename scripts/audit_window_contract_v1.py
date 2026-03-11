#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import meds_reader as mr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import (
    SPECIAL_ID2NAME,
    AuditArtifacts,
    _audit_subject_tokenization,
    _build_measurement_config,
    _build_segmentation_config,
    _build_static_artifacts,
    _build_struct_vocab,
    _build_window_marker_config,
    _load_subject_ids,
    _load_tokenization_contract,
    _make_structural_id2label,
    _maybe_print_progress,
    _offset,
    _parse_subject_ids,
    _resolve_residual_policy,
    _summarize_raw_subjects,
)
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.data.window_segmentation import (
    WindowSegmentationConfig,
    _build_boundary_bundles,
    _cat_attr_int,
    _resolve_bundle_action,
    _resolve_opening_window_type,
    _token_transition_action,
    segment_event_tokens,
)
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders
from src.ehr_hier.tokenizers.decode_tokens import invert_code2id


TRANSITION_PREFIXES = {
    "TRANSFER_TO",
    "HOSPITAL_ADMISSION",
    "HOSPITAL_DISCHARGE",
    "ADMISSION",
    "DISCHARGE",
    "ICU_ADMISSION",
    "ICU_DISCHARGE",
    "ED_REGISTRATION",
    "ED_OUT",
    "MEDS_BIRTH",
    "MEDS_DEATH",
    "STRUCT_START_ADM",
    "STRUCT_END_ADM",
    "STRUCT_CAREUNIT_CHANGE",
    "STRUCT_START_OR",
    "STRUCT_END_OR",
}
ADMISSION_LIKE_PREFIXES = {
    "TRANSFER_TO",
    "HOSPITAL_ADMISSION",
    "ADMISSION",
    "ICU_ADMISSION",
    "ED_REGISTRATION",
    "STRUCT_START_ADM",
    "STRUCT_CAREUNIT_CHANGE",
    "STRUCT_START_OR",
}
DISCHARGE_LIKE_PREFIXES = {
    "HOSPITAL_DISCHARGE",
    "DISCHARGE",
    "ICU_DISCHARGE",
    "ED_OUT",
    "STRUCT_END_ADM",
    "STRUCT_END_OR",
}
PRIMARY_CLINICAL_CATEGORIES = {
    int(TokenCategory.MEASUREMENT),
    int(TokenCategory.DIAGNOSIS),
    int(TokenCategory.PROCEDURE),
    int(TokenCategory.MEDICATION),
}


def _round_hours(value: float) -> float:
    return round(float(value), 3)


def _parse_thresholds(arg: str | None) -> List[float]:
    if arg is None or not str(arg).strip():
        return [4.0, 12.0, 24.0, 48.0]
    out: List[float] = []
    for part in str(arg).split(","):
        piece = part.strip()
        if not piece:
            continue
        out.append(float(piece))
    return sorted({float(x) for x in out})


def _type_id2name(codebook: Any) -> Dict[int, str]:
    if codebook is None:
        return {0: "UNK"}
    return {int(v): str(k) for k, v in codebook.window_type2id().items()}


def _structural_label(
    tok: EventToken,
    *,
    artifacts: AuditArtifacts,
    struct_id2label: Mapping[int, str],
    struct_id2code: Mapping[int, str],
) -> str:
    value_id = int(tok.value_id)
    structural_offset = _offset(artifacts.manifest, "structural", 2_200_000)
    structural_action_offset = _offset(artifacts.manifest, "structural_action", 2_400_000)
    structural_entity_offset = _offset(artifacts.manifest, "structural_entity", 2_420_000)

    if structural_action_offset <= value_id < structural_entity_offset:
        return f"STRUCT_ACT::{value_id - structural_action_offset}"
    if value_id >= structural_entity_offset:
        return f"STRUCT_ENT::{value_id - structural_entity_offset}"
    if tok.cat_attrs and "struct_label_id" in tok.cat_attrs:
        label_id = int(tok.cat_attrs["struct_label_id"])
        return struct_id2label.get(label_id, f"STRUCT_LABEL::{label_id}")
    raw_id = value_id - structural_offset
    return struct_id2code.get(raw_id, f"STRUCTURAL::{value_id}")


def _token_label(
    tok: EventToken,
    *,
    artifacts: AuditArtifacts,
    struct_id2label: Mapping[int, str],
    struct_id2code: Mapping[int, str],
) -> str:
    category = TokenCategory(int(tok.category_id))
    if category == TokenCategory.STRUCTURAL:
        return _structural_label(
            tok,
            artifacts=artifacts,
            struct_id2label=struct_id2label,
            struct_id2code=struct_id2code,
        )
    if category == TokenCategory.SPECIAL:
        return SPECIAL_ID2NAME.get(int(tok.value_id), f"SPECIAL::{int(tok.value_id)}")
    return category.name


def _label_prefix(label: str) -> str:
    text = str(label).strip()
    if not text:
        return "<EMPTY>"
    return text.split("//", 1)[0].upper()


def _token_preview(
    tok: EventToken,
    *,
    artifacts: AuditArtifacts,
    struct_id2label: Mapping[int, str],
    struct_id2code: Mapping[int, str],
    type_id2name: Mapping[int, str],
) -> Dict[str, Any]:
    transition_type_id = _cat_attr_int(tok, "transition_window_type_id")
    window_type_attr = _cat_attr_int(tok, "window_type_id")
    return {
        "category": TokenCategory(int(tok.category_id)).name,
        "label": _token_label(
            tok,
            artifacts=artifacts,
            struct_id2label=struct_id2label,
            struct_id2code=struct_id2code,
        ),
        "t_from_start_hours": _round_hours(float(tok.t_from_start_hours)),
        "dt_from_prev_hours": _round_hours(float(tok.dt_from_prev_hours)),
        "window_hook": tok.window_hook,
        "transition_action": _token_transition_action(tok),
        "transition_window_type_id": transition_type_id,
        "transition_window_type_name": (
            type_id2name.get(int(transition_type_id), str(transition_type_id))
            if transition_type_id is not None
            else None
        ),
        "window_type_id_attr": window_type_attr,
        "window_type_name_attr": (
            type_id2name.get(int(window_type_attr), str(window_type_attr))
            if window_type_attr is not None
            else None
        ),
        "struct_label_id": _cat_attr_int(tok, "struct_label_id"),
    }


def _window_preview(
    window: Any,
    *,
    artifacts: AuditArtifacts,
    struct_id2label: Mapping[int, str],
    struct_id2code: Mapping[int, str],
    type_id2name: Mapping[int, str],
) -> Dict[str, Any]:
    first = window.tokens[0]
    last = window.tokens[-1]
    category_counts = Counter(TokenCategory(int(tok.category_id)).name for tok in window.tokens)
    clinical_count = sum(1 for tok in window.tokens if int(tok.category_id) in PRIMARY_CLINICAL_CATEGORIES)
    return {
        "window_type_id": int(window.window_type_id),
        "window_type_name": type_id2name.get(int(window.window_type_id), str(int(window.window_type_id))),
        "opening_action": window.opening_action,
        "closing_action": window.closing_action,
        "token_count": int(len(window.tokens)),
        "clinical_token_count": int(clinical_count),
        "start_time_hours": _round_hours(float(window.start_time_hours)),
        "end_time_hours": _round_hours(float(last.t_from_start_hours)),
        "duration_hours": _round_hours(float(last.t_from_start_hours) - float(first.t_from_start_hours)),
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "first_token": _token_preview(
            first,
            artifacts=artifacts,
            struct_id2label=struct_id2label,
            struct_id2code=struct_id2code,
            type_id2name=type_id2name,
        ),
        "last_token": _token_preview(
            last,
            artifacts=artifacts,
            struct_id2label=struct_id2label,
            struct_id2code=struct_id2code,
            type_id2name=type_id2name,
        ),
    }


def _maybe_add_example(store: List[Dict[str, Any]], item: Dict[str, Any], *, limit: int) -> None:
    if len(store) < int(limit):
        store.append(item)


def _summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    vals = sorted(float(v) for v in values)

    def _pct(p: float) -> float:
        idx = max(0, min(len(vals) - 1, int(round(p * (len(vals) - 1)))))
        return float(vals[idx])

    return {
        "count": float(len(vals)),
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p99": _pct(0.99),
        "max": float(max(vals)),
    }


def _count_thresholds(values: Iterable[float], thresholds: Sequence[float]) -> Dict[str, int]:
    cached = [float(v) for v in values]
    return {str(th): int(sum(1 for v in cached if v <= float(th))) for th in thresholds}


def _build_runtime_context(
    args: argparse.Namespace,
) -> tuple[mr.SubjectDatabase, List[int], AuditArtifacts, Dict[int, str], Dict[TokenCategory, Any], WindowSegmentationConfig]:
    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    artifacts = _build_static_artifacts(args)
    window_markers_cfg = _build_window_marker_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
    )
    segmentation_cfg = _build_segmentation_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=artifacts.structural_codebook,
        manifest=artifacts.manifest,
        unk_type_id=int(window_markers_cfg.unk_type_id),
    )
    residual_enabled, residual_buckets, residual_offsets = _resolve_residual_policy(
        args,
        tokenization_contract=tokenization_contract,
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

    _, structural_raw_codes = _summarize_raw_subjects(
        db,
        subject_ids,
        artifacts=artifacts,
        top_k=max(20, int(args.top_k)),
        progress_every=int(args.progress_every),
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
        enable_residual_fallback=bool(residual_enabled),
        residual_fallback_buckets=int(residual_buckets),
        residual_fallback_offsets=dict(residual_offsets),
    )
    return db, subject_ids, artifacts, struct_id2code, encoders, segmentation_cfg


def _audit_window_contract(
    db: mr.SubjectDatabase,
    subject_ids: List[int],
    *,
    encoders: Dict[TokenCategory, Any],
    artifacts: AuditArtifacts,
    struct_id2code: Mapping[int, str],
    segmentation_cfg: WindowSegmentationConfig,
    gap_thresholds: Sequence[float],
    micro_window_max_tokens: int,
    micro_window_max_duration_hours: float,
    max_examples: int,
    progress_every: int,
) -> Dict[str, Any]:
    type_id2name = _type_id2name(artifacts.structural_codebook)
    struct_id2label = _make_structural_id2label(artifacts.structural_codebook)
    started_at = time.time()

    summary = {
        "subjects_scanned": int(len(subject_ids)),
        "subjects_with_any_transition_bundle": 0,
        "subjects_with_leading_pretransition_events": 0,
        "subjects_with_prologue_backfill_opportunity": 0,
        "total_event_tokens": 0,
        "total_windows": 0,
        "window_type_counts": Counter(),
        "window_type_unknown": 0,
        "transition_bundle_counts": Counter(),
        "transition_bundle_primary_prefixes": Counter(),
        "transition_bundle_missing_transfer_primary_prefix": Counter(),
        "transition_bundle_cooccurring_with_transfer": Counter(),
        "transition_bundle_actions": Counter(),
        "leading_first_transition_type_counts": Counter(),
        "leading_current_first_window_type_counts": Counter(),
        "leading_candidate_backfill_type_counts": Counter(),
        "micro_windows": 0,
        "micro_windows_by_type": Counter(),
        "inter_admission_candidates": 0,
        "inter_admission_opening_type_counts": Counter(),
        "inter_admission_gap_hours": [],
        "inter_admission_windows": 0,
        "inter_admission_realized_gap_hours": [],
        "inter_admission_prev_type_counts": Counter(),
        "inter_admission_next_type_counts": Counter(),
    }
    examples: Dict[str, List[Dict[str, Any]]] = {
        "missing_transfer_bundles": [],
        "cooccurring_transfer_bundles": [],
        "leading_pretransition_subjects": [],
        "micro_windows": [],
        "inter_admission_candidates": [],
        "inter_admission_windows": [],
    }

    for idx, sid in enumerate(subject_ids, start=1):
        audited = _audit_subject_tokenization(
            db,
            int(sid),
            encoders=encoders,
            codebook=artifacts.structural_codebook,
            artifacts=artifacts,
        )
        timeline = list(audited["timeline"])
        event_tokens = [tok for tok in timeline if int(tok.category_id) != int(TokenCategory.SPECIAL)]
        if not event_tokens:
            _maybe_print_progress(
                "window_contract",
                idx=idx,
                total=len(subject_ids),
                every=progress_every,
                started_at=started_at,
            )
            continue

        summary["total_event_tokens"] += int(len(event_tokens))
        bundles = _build_boundary_bundles(event_tokens, config=segmentation_cfg)
        windows = segment_event_tokens(event_tokens, config=segmentation_cfg)

        if bundles:
            summary["subjects_with_any_transition_bundle"] += 1

        for window in windows:
            summary["total_windows"] += 1
            w_type_id = int(window.window_type_id)
            w_type_name = type_id2name.get(w_type_id, str(w_type_id))
            summary["window_type_counts"][w_type_name] += 1
            if w_type_id == int(segmentation_cfg.unk_window_type_id):
                summary["window_type_unknown"] += 1

            last_tok = window.tokens[-1]
            duration = max(0.0, float(last_tok.t_from_start_hours) - float(window.tokens[0].t_from_start_hours))
            clinical_count = sum(1 for tok in window.tokens if int(tok.category_id) in PRIMARY_CLINICAL_CATEGORIES)
            is_inter_window = (
                segmentation_cfg.inter_admission_window_type_id is not None
                and int(w_type_id) == int(segmentation_cfg.inter_admission_window_type_id)
            )
            if (
                not is_inter_window
                and len(window.tokens) <= int(micro_window_max_tokens)
                and duration <= float(micro_window_max_duration_hours)
                and clinical_count == 0
            ):
                summary["micro_windows"] += 1
                summary["micro_windows_by_type"][w_type_name] += 1
                _maybe_add_example(
                    examples["micro_windows"],
                    {
                        "subject_id": int(sid),
                        "window": _window_preview(
                            window,
                            artifacts=artifacts,
                            struct_id2label=struct_id2label,
                            struct_id2code=struct_id2code,
                            type_id2name=type_id2name,
                        ),
                    },
                    limit=max_examples,
                )

        inter_type_id = segmentation_cfg.inter_admission_window_type_id
        if inter_type_id is not None:
            for w_idx, window in enumerate(windows):
                if int(window.window_type_id) != int(inter_type_id):
                    continue
                summary["inter_admission_windows"] += 1
                prev_type_name = None
                next_type_name = None
                if w_idx > 0:
                    prev_type_name = type_id2name.get(
                        int(windows[w_idx - 1].window_type_id),
                        str(int(windows[w_idx - 1].window_type_id)),
                    )
                    summary["inter_admission_prev_type_counts"][str(prev_type_name)] += 1
                if w_idx + 1 < len(windows):
                    next_type_name = type_id2name.get(
                        int(windows[w_idx + 1].window_type_id),
                        str(int(windows[w_idx + 1].window_type_id)),
                    )
                    summary["inter_admission_next_type_counts"][str(next_type_name)] += 1

                gap_hours_realized = None
                if window.tokens:
                    raw_gap = window.tokens[0].num_attrs.get("numeric_value") if window.tokens[0].num_attrs else None
                    try:
                        if raw_gap is not None:
                            gap_hours_realized = float(raw_gap)
                    except (TypeError, ValueError):
                        gap_hours_realized = None
                if gap_hours_realized is None and w_idx > 0 and w_idx + 1 < len(windows):
                    gap_hours_realized = max(
                        0.0,
                        float(windows[w_idx + 1].start_time_hours)
                        - float(windows[w_idx - 1].tokens[-1].t_from_start_hours),
                    )
                if gap_hours_realized is not None:
                    summary["inter_admission_realized_gap_hours"].append(float(gap_hours_realized))

                _maybe_add_example(
                    examples["inter_admission_windows"],
                    {
                        "subject_id": int(sid),
                        "gap_hours": _round_hours(float(gap_hours_realized or 0.0)),
                        "prev_window_type": prev_type_name,
                        "next_window_type": next_type_name,
                        "window": _window_preview(
                            window,
                            artifacts=artifacts,
                            struct_id2label=struct_id2label,
                            struct_id2code=struct_id2code,
                            type_id2name=type_id2name,
                        ),
                    },
                    limit=max_examples,
                )

        for bundle in bundles:
            start_idx = int(bundle["start_idx"])
            end_idx = int(bundle["end_idx"])
            bundle_tokens = event_tokens[start_idx : end_idx + 1]
            bundle_action, _, _ = _resolve_bundle_action(
                bundle_tokens,
                list(bundle["candidate_indices"]),  # type: ignore[arg-type]
                bundle_start_idx=start_idx,
            )
            labels = [
                _token_label(
                    tok,
                    artifacts=artifacts,
                    struct_id2label=struct_id2label,
                    struct_id2code=struct_id2code,
                )
                for tok in bundle_tokens
            ]
            transition_prefixes = [_label_prefix(label) for label in labels if _label_prefix(label) in TRANSITION_PREFIXES]
            if not transition_prefixes:
                continue

            prefixes = sorted(set(transition_prefixes))
            has_transfer = "TRANSFER_TO" in prefixes
            primary_prefix = prefixes[0]
            summary["transition_bundle_counts"]["total"] += 1
            summary["transition_bundle_primary_prefixes"][primary_prefix] += 1
            summary["transition_bundle_actions"][bundle_action] += 1

            if has_transfer:
                summary["transition_bundle_counts"]["with_transfer_to"] += 1
                for prefix in prefixes:
                    if prefix != "TRANSFER_TO":
                        summary["transition_bundle_cooccurring_with_transfer"][prefix] += 1
                if len(prefixes) > 1:
                    _maybe_add_example(
                        examples["cooccurring_transfer_bundles"],
                        {
                            "subject_id": int(sid),
                            "bundle_action": bundle_action,
                            "bundle_prefixes": prefixes,
                            "bundle_tokens": [
                                _token_preview(
                                    tok,
                                    artifacts=artifacts,
                                    struct_id2label=struct_id2label,
                                    struct_id2code=struct_id2code,
                                    type_id2name=type_id2name,
                                )
                                for tok in bundle_tokens
                            ],
                        },
                        limit=max_examples,
                    )
            else:
                summary["transition_bundle_counts"]["without_transfer_to"] += 1
                summary["transition_bundle_missing_transfer_primary_prefix"][primary_prefix] += 1
                _maybe_add_example(
                    examples["missing_transfer_bundles"],
                    {
                        "subject_id": int(sid),
                        "bundle_action": bundle_action,
                        "bundle_prefixes": prefixes,
                        "bundle_tokens": [
                            _token_preview(
                                tok,
                                artifacts=artifacts,
                                struct_id2label=struct_id2label,
                                struct_id2code=struct_id2code,
                                type_id2name=type_id2name,
                            )
                            for tok in bundle_tokens
                        ],
                    },
                    limit=max_examples,
                )

        if bundles and int(bundles[0]["start_idx"]) > 0:
            summary["subjects_with_leading_pretransition_events"] += 1
            first_bundle = bundles[0]
            first_bundle_tokens = event_tokens[int(first_bundle["start_idx"]) : int(first_bundle["end_idx"]) + 1]
            bundle_action, _, opening_items = _resolve_bundle_action(
                first_bundle_tokens,
                list(first_bundle["candidate_indices"]),  # type: ignore[arg-type]
                bundle_start_idx=int(first_bundle["start_idx"]),
            )
            candidate_opening = list(opening_items) if opening_items else list(first_bundle_tokens)
            candidate_type_id = _resolve_opening_window_type(
                candidate_opening,
                previous_type_id=int(segmentation_cfg.unk_window_type_id),
                config=segmentation_cfg,
            )
            candidate_type_name = type_id2name.get(int(candidate_type_id), str(int(candidate_type_id)))
            current_type_id = int(windows[0].window_type_id) if windows else int(segmentation_cfg.unk_window_type_id)
            current_type_name = type_id2name.get(int(current_type_id), str(int(current_type_id)))

            summary["leading_first_transition_type_counts"][candidate_type_name] += 1
            summary["leading_current_first_window_type_counts"][current_type_name] += 1
            if candidate_type_id != int(segmentation_cfg.unk_window_type_id):
                summary["leading_candidate_backfill_type_counts"][candidate_type_name] += 1
            if current_type_name == "PROLOGUE" and candidate_type_name in {"ED", "INPATIENT", "ICU", "OR"}:
                summary["subjects_with_prologue_backfill_opportunity"] += 1

            leading_tokens = event_tokens[: int(first_bundle["start_idx"])]
            _maybe_add_example(
                examples["leading_pretransition_subjects"],
                {
                    "subject_id": int(sid),
                    "leading_token_count": int(len(leading_tokens)),
                    "leading_start_time_hours": _round_hours(float(leading_tokens[0].t_from_start_hours)),
                    "leading_end_time_hours": _round_hours(float(leading_tokens[-1].t_from_start_hours)),
                    "leading_duration_hours": _round_hours(
                        float(first_bundle_tokens[0].t_from_start_hours) - float(leading_tokens[0].t_from_start_hours)
                    ),
                    "current_first_window_type": current_type_name,
                    "first_transition_bundle_action": bundle_action,
                    "first_transition_candidate_type": candidate_type_name,
                    "leading_categories": {
                        str(k): int(v)
                        for k, v in Counter(TokenCategory(int(tok.category_id)).name for tok in leading_tokens).items()
                    },
                    "leading_first_tokens": [
                        _token_preview(
                            tok,
                            artifacts=artifacts,
                            struct_id2label=struct_id2label,
                            struct_id2code=struct_id2code,
                            type_id2name=type_id2name,
                        )
                        for tok in leading_tokens[: min(6, len(leading_tokens))]
                    ],
                    "first_bundle_tokens": [
                        _token_preview(
                            tok,
                            artifacts=artifacts,
                            struct_id2label=struct_id2label,
                            struct_id2code=struct_id2code,
                            type_id2name=type_id2name,
                        )
                        for tok in first_bundle_tokens
                    ],
                },
                limit=max_examples,
            )

        for left, right in zip(windows, windows[1:]):
            left_last = left.tokens[-1]
            right_first = right.tokens[0]
            gap_hours = max(0.0, float(right.start_time_hours) - float(left_last.t_from_start_hours))
            if gap_hours <= 0.0:
                continue

            left_label = _token_label(
                left_last,
                artifacts=artifacts,
                struct_id2label=struct_id2label,
                struct_id2code=struct_id2code,
            )
            right_label = _token_label(
                right_first,
                artifacts=artifacts,
                struct_id2label=struct_id2label,
                struct_id2code=struct_id2code,
            )
            left_prefix = _label_prefix(left_label)
            right_prefix = _label_prefix(right_label)
            if left_prefix not in DISCHARGE_LIKE_PREFIXES:
                continue
            if right_prefix not in ADMISSION_LIKE_PREFIXES:
                continue

            summary["inter_admission_candidates"] += 1
            summary["inter_admission_opening_type_counts"][
                type_id2name.get(int(right.window_type_id), str(int(right.window_type_id)))
            ] += 1
            summary["inter_admission_gap_hours"].append(float(gap_hours))
            _maybe_add_example(
                examples["inter_admission_candidates"],
                {
                    "subject_id": int(sid),
                    "gap_hours": _round_hours(gap_hours),
                    "left_window": _window_preview(
                        left,
                        artifacts=artifacts,
                        struct_id2label=struct_id2label,
                        struct_id2code=struct_id2code,
                        type_id2name=type_id2name,
                    ),
                    "right_window": _window_preview(
                        right,
                        artifacts=artifacts,
                        struct_id2label=struct_id2label,
                        struct_id2code=struct_id2code,
                        type_id2name=type_id2name,
                    ),
                },
                limit=max_examples,
            )

        _maybe_print_progress(
            "window_contract",
            idx=idx,
            total=len(subject_ids),
            every=progress_every,
            started_at=started_at,
        )

    total_windows = int(summary["total_windows"])
    transition_total = int(summary["transition_bundle_counts"].get("total", 0))
    gap_values = list(summary["inter_admission_gap_hours"])
    realized_gap_values = list(summary["inter_admission_realized_gap_hours"])
    return {
        "summary": {
            "subjects_scanned": int(summary["subjects_scanned"]),
            "subjects_with_any_transition_bundle": int(summary["subjects_with_any_transition_bundle"]),
            "subjects_with_leading_pretransition_events": int(summary["subjects_with_leading_pretransition_events"]),
            "subjects_with_prologue_backfill_opportunity": int(summary["subjects_with_prologue_backfill_opportunity"]),
            "total_event_tokens": int(summary["total_event_tokens"]),
            "total_windows": total_windows,
            "window_type_counts": {str(k): int(v) for k, v in summary["window_type_counts"].items()},
            "window_type_unk_frac": (
                float(summary["window_type_unknown"]) / float(total_windows) if total_windows > 0 else 0.0
            ),
            "transition_bundles": {
                "total": transition_total,
                "with_transfer_to": int(summary["transition_bundle_counts"].get("with_transfer_to", 0)),
                "without_transfer_to": int(summary["transition_bundle_counts"].get("without_transfer_to", 0)),
                "with_transfer_to_frac": (
                    float(summary["transition_bundle_counts"].get("with_transfer_to", 0)) / float(transition_total)
                    if transition_total > 0
                    else 0.0
                ),
                "primary_prefix_counts": {
                    str(k): int(v) for k, v in summary["transition_bundle_primary_prefixes"].most_common()
                },
                "missing_transfer_primary_prefix_counts": {
                    str(k): int(v) for k, v in summary["transition_bundle_missing_transfer_primary_prefix"].most_common()
                },
                "cooccurring_with_transfer_counts": {
                    str(k): int(v) for k, v in summary["transition_bundle_cooccurring_with_transfer"].most_common()
                },
                "action_counts": {str(k): int(v) for k, v in summary["transition_bundle_actions"].items()},
            },
            "leading_pretransition": {
                "subject_frac": (
                    float(summary["subjects_with_leading_pretransition_events"]) / float(summary["subjects_scanned"])
                    if int(summary["subjects_scanned"]) > 0
                    else 0.0
                ),
                "current_first_window_type_counts": {
                    str(k): int(v) for k, v in summary["leading_current_first_window_type_counts"].most_common()
                },
                "first_transition_candidate_type_counts": {
                    str(k): int(v) for k, v in summary["leading_first_transition_type_counts"].most_common()
                },
                "backfill_candidate_type_counts": {
                    str(k): int(v) for k, v in summary["leading_candidate_backfill_type_counts"].most_common()
                },
            },
            "micro_windows": {
                "count": int(summary["micro_windows"]),
                "frac": float(summary["micro_windows"]) / float(total_windows) if total_windows > 0 else 0.0,
                "by_type": {str(k): int(v) for k, v in summary["micro_windows_by_type"].most_common()},
            },
            "inter_admission": {
                "candidate_pairs": int(summary["inter_admission_candidates"]),
                "realized_windows": int(summary["inter_admission_windows"]),
                "opening_type_counts": {
                    str(k): int(v) for k, v in summary["inter_admission_opening_type_counts"].most_common()
                },
                "prev_type_counts": {
                    str(k): int(v) for k, v in summary["inter_admission_prev_type_counts"].most_common()
                },
                "next_type_counts": {
                    str(k): int(v) for k, v in summary["inter_admission_next_type_counts"].most_common()
                },
                "candidate_gap_hours": _summarize_numeric(gap_values),
                "realized_gap_hours": _summarize_numeric(realized_gap_values),
                "count_at_or_below_hours": _count_thresholds(realized_gap_values or gap_values, gap_thresholds),
            },
        },
        "examples": examples,
    }


def _print_summary(payload: Mapping[str, Any]) -> None:
    summary = payload["results"]["summary"]
    print("Window Contract Audit")
    print("Config:", json.dumps(payload["config"], indent=2))
    print("Window types:", summary["window_type_counts"])
    print("Unknown window frac:", f"{summary['window_type_unk_frac']:.4f}")
    print("Transition bundles:", summary["transition_bundles"])
    print("Leading pretransition:", summary["leading_pretransition"])
    print("Micro windows:", summary["micro_windows"])
    print("Inter-admission:", summary["inter_admission"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit a window v1 contract on real EventToken segmentation, focusing on TRANSFER_TO bundles, leading pre-transition events, and discharge/readmission gaps."
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=100)
    ap.add_argument("--sample_seed", type=int, default=None)
    ap.add_argument("--subject_ids", default=None)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default="artifacts/medtok")
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--codes_parquet_parent_lookup", default=None)
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--tokenizer_ckpt", required=True)
    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=40_000)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)
    ap.add_argument("--inter_admission_thresholds", default="4,12,24,48")
    ap.add_argument("--micro_window_max_tokens", type=int, default=4)
    ap.add_argument("--micro_window_max_duration_hours", type=float, default=1.0)
    ap.add_argument("--max_examples", type=int, default=20)
    ap.add_argument("--progress_every", type=int, default=0)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    db, subject_ids, artifacts, struct_id2code, encoders, segmentation_cfg = _build_runtime_context(args)
    gap_thresholds = _parse_thresholds(args.inter_admission_thresholds)
    results = _audit_window_contract(
        db,
        subject_ids,
        encoders=encoders,
        artifacts=artifacts,
        struct_id2code=struct_id2code,
        segmentation_cfg=segmentation_cfg,
        gap_thresholds=gap_thresholds,
        micro_window_max_tokens=int(args.micro_window_max_tokens),
        micro_window_max_duration_hours=float(args.micro_window_max_duration_hours),
        max_examples=int(args.max_examples),
        progress_every=int(args.progress_every),
    )

    payload = {
        "config": {
            "split": str(args.split),
            "subjects_scanned": int(len(subject_ids)),
            "sample_seed": args.sample_seed,
            "tokenization_yaml": str(args.tokenization_yaml),
            "structural_yaml": str(args.structural_yaml),
            "inter_admission_thresholds": list(gap_thresholds),
            "micro_window_max_tokens": int(args.micro_window_max_tokens),
            "micro_window_max_duration_hours": float(args.micro_window_max_duration_hours),
            "bundle_gap_hours": float(segmentation_cfg.bundle_gap_hours),
            "bundle_max_index_gap": int(segmentation_cfg.bundle_max_index_gap),
            "chain_gap_hours": float(segmentation_cfg.chain_gap_hours),
            "chain_max_intervening_tokens": int(segmentation_cfg.chain_max_intervening_tokens),
            "default_first_window_type_id": segmentation_cfg.default_first_window_type_id,
            "propagate_prev_type_for_unknown_windows": bool(segmentation_cfg.propagate_prev_type_for_unknown_windows),
            "enable_inter_admission_windows": bool(segmentation_cfg.enable_inter_admission_windows),
            "inter_admission_window_type_id": segmentation_cfg.inter_admission_window_type_id,
            "inter_admission_max_gap_hours": float(segmentation_cfg.inter_admission_max_gap_hours),
            "inter_admission_token_id": segmentation_cfg.inter_admission_token_id,
        },
        "results": results,
    }
    _print_summary(payload)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
