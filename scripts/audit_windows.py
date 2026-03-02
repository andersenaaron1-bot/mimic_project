#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    _build_static_artifacts,
    _build_struct_vocab,
    _load_subject_ids,
    _make_structural_id2label,
    _offset,
    _summarize_raw_subjects,
)
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders
from src.ehr_hier.tokenizers.decode_tokens import decode_timeline_tokens, invert_code2id
from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig


TRANSITION_TEXT_PARTS = (
    "STRUCT_START_ADM",
    "STRUCT_END_ADM",
    "STRUCT_CAREUNIT_CHANGE",
    "STRUCT_START_OR",
    "STRUCT_END_OR",
    "TRANSFER_TO",
    "HOSPITAL_ADMISSION",
    "HOSPITAL_DISCHARGE",
    "ED_REGISTRATION",
    "ED_OUT",
)

MARKER_TEXT_PARTS = (
    "MED_MARKER::START",
    "MED_MARKER::END",
    "MED_MARKER::STOP",
    "//START//",
    "//END//",
    "//STOP//",
    "INFUSION_START",
    "INFUSION_END",
)


@dataclass(frozen=True)
class SegmentationPolicy:
    name: str
    placement: str
    gap_hours: float
    custom_contains: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.name}__{self.placement}"


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


def _default_policies() -> List[str]:
    return ["current", "all_structural", "transition_like_plus_gap"]


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _make_policies(args: argparse.Namespace) -> List[SegmentationPolicy]:
    policy_names = _unique_preserve_order(args.policy or _default_policies())
    placements = _unique_preserve_order(args.placement or ["open_next"])
    custom_contains = tuple(str(x).upper() for x in (args.custom_boundary_contains or []))
    policies: List[SegmentationPolicy] = []
    for name in policy_names:
        for placement in placements:
            policies.append(
                SegmentationPolicy(
                    name=str(name),
                    placement=str(placement),
                    gap_hours=float(args.gap_hours),
                    custom_contains=custom_contains,
                )
            )
    return policies


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return float(vals[0])
    pct = max(0.0, min(1.0, float(pct)))
    idx = int(round(pct * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    vals = [float(v) for v in values]
    return {
        "p50": _percentile(vals, 0.50),
        "p90": _percentile(vals, 0.90),
        "p99": _percentile(vals, 0.99),
        "max": max(vals),
    }


def _annotate_token_spans(decoded: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cursor = 0
    for item in decoded:
        enriched = dict(item)
        span = enriched.get("token_span")
        if span is None:
            enriched["token_span"] = [cursor, cursor]
            cursor += 1
        else:
            lo = int(span[0])
            hi = int(span[1])
            enriched["token_span"] = [lo, hi]
            cursor = hi + 1
        out.append(enriched)
    return out


def _item_category(item: Mapping[str, Any]) -> str:
    if item.get("kind") == "measurement_bundle":
        return "MEASUREMENT"
    return str(item.get("category", "UNKNOWN"))


def _item_label(item: Mapping[str, Any]) -> str:
    if item.get("kind") == "measurement_bundle":
        return str(item.get("var_name") or f"VAR::{item.get('var_id')}")
    label = item.get("label")
    if label is not None:
        return str(label)
    return f"{_item_category(item)}::{item.get('value_id')}"


def _item_text(item: Mapping[str, Any]) -> str:
    parts = [
        _item_label(item),
        str(item.get("window_hook") or ""),
        _item_category(item),
    ]
    return " | ".join(parts).upper()


def _item_token_count(item: Mapping[str, Any]) -> int:
    span = item.get("token_span")
    if isinstance(span, list) and len(span) == 2:
        return max(1, int(span[1]) - int(span[0]) + 1)
    return 1


def _item_t(item: Mapping[str, Any]) -> float:
    return float(item.get("t_from_start_hours", 0.0) or 0.0)


def _item_dt(item: Mapping[str, Any]) -> float:
    return float(item.get("dt_from_prev_hours", 0.0) or 0.0)


def _item_preview(item: Mapping[str, Any]) -> Dict[str, Any]:
    preview: Dict[str, Any] = {
        "kind": str(item.get("kind")),
        "category": _item_category(item),
        "label": _item_label(item),
        "t_from_start_hours": round(_item_t(item), 3),
        "dt_from_prev_hours": round(_item_dt(item), 3),
        "token_count": _item_token_count(item),
        "window_hook": item.get("window_hook"),
    }
    if item.get("kind") == "measurement_bundle":
        preview["z_norm"] = item.get("z_norm")
    return preview


def _is_transition_like(item: Mapping[str, Any]) -> bool:
    text = _item_text(item)
    if _item_category(item) != "STRUCTURAL":
        return False
    return any(part in text for part in TRANSITION_TEXT_PARTS)


def _is_marker_like(item: Mapping[str, Any]) -> bool:
    text = _item_text(item)
    return any(part in text for part in MARKER_TEXT_PARTS)


def _boundary_reasons(
    item: Mapping[str, Any],
    prev_item: Mapping[str, Any] | None,
    policy: SegmentationPolicy,
) -> List[str]:
    reasons: List[str] = []
    category = _item_category(item)

    if policy.name == "current":
        if item.get("window_hook") is not None:
            reasons.append("window_hook")
    elif policy.name == "all_structural":
        if category == "STRUCTURAL":
            reasons.append("structural_any")
    elif policy.name == "gap_only":
        if prev_item is not None and _item_dt(item) >= policy.gap_hours:
            reasons.append(f"gap>={policy.gap_hours:g}h")
    elif policy.name == "current_plus_gap":
        if item.get("window_hook") is not None:
            reasons.append("window_hook")
        if prev_item is not None and _item_dt(item) >= policy.gap_hours:
            reasons.append(f"gap>={policy.gap_hours:g}h")
    elif policy.name == "structural_or_gap":
        if category == "STRUCTURAL":
            reasons.append("structural_any")
        if prev_item is not None and _item_dt(item) >= policy.gap_hours:
            reasons.append(f"gap>={policy.gap_hours:g}h")
    elif policy.name == "transition_like":
        if _is_transition_like(item):
            reasons.append("transition_like")
    elif policy.name == "transition_like_plus_gap":
        if _is_transition_like(item):
            reasons.append("transition_like")
        if prev_item is not None and _item_dt(item) >= policy.gap_hours:
            reasons.append(f"gap>={policy.gap_hours:g}h")
    elif policy.name == "marker_or_transition":
        if _is_transition_like(item):
            reasons.append("transition_like")
        if _is_marker_like(item):
            reasons.append("marker_like")
    elif policy.name == "marker_or_transition_plus_gap":
        if _is_transition_like(item):
            reasons.append("transition_like")
        if _is_marker_like(item):
            reasons.append("marker_like")
        if prev_item is not None and _item_dt(item) >= policy.gap_hours:
            reasons.append(f"gap>={policy.gap_hours:g}h")
    elif policy.name == "custom_contains":
        text = _item_text(item)
        for needle in policy.custom_contains:
            if needle in text:
                reasons.append(f"contains:{needle}")
    elif policy.name == "custom_contains_plus_gap":
        text = _item_text(item)
        for needle in policy.custom_contains:
            if needle in text:
                reasons.append(f"contains:{needle}")
        if prev_item is not None and _item_dt(item) >= policy.gap_hours:
            reasons.append(f"gap>={policy.gap_hours:g}h")
    else:
        raise ValueError(f"Unknown policy: {policy.name}")

    return _unique_preserve_order(reasons)


def _segment_items(
    items: List[Dict[str, Any]],
    *,
    policy: SegmentationPolicy,
    context_items: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    windows: List[Dict[str, Any]] = []
    boundary_examples: List[Dict[str, Any]] = []
    current_items: List[Dict[str, Any]] = []
    current_opening_reasons: List[str] = []

    for idx, item in enumerate(items):
        prev_item = items[idx - 1] if idx > 0 else None
        next_item = items[idx + 1] if idx + 1 < len(items) else None
        reasons = _boundary_reasons(item, prev_item, policy)
        is_boundary = len(reasons) > 0

        if is_boundary:
            boundary_examples.append(
                {
                    "boundary_index": int(idx),
                    "reasons": list(reasons),
                    "boundary_item": _item_preview(item),
                    "pre_context": [_item_preview(x) for x in items[max(0, idx - context_items) : idx]],
                    "post_context": [_item_preview(x) for x in items[idx + 1 : idx + 1 + context_items]],
                    "next_item": _item_preview(next_item) if next_item is not None else None,
                }
            )

        if policy.placement == "open_next":
            if is_boundary and current_items:
                windows.append(
                    {
                        "items": current_items,
                        "opening_reasons": list(current_opening_reasons),
                    }
                )
                current_items = [item]
                current_opening_reasons = list(reasons)
            else:
                current_items.append(item)
                if is_boundary and not current_opening_reasons:
                    current_opening_reasons = list(reasons)
        elif policy.placement == "close_current":
            current_items.append(item)
            if is_boundary:
                windows.append(
                    {
                        "items": current_items,
                        "opening_reasons": list(current_opening_reasons),
                        "closing_reasons": list(reasons),
                    }
                )
                current_items = []
                current_opening_reasons = []
        else:
            raise ValueError(f"Unknown placement: {policy.placement}")

    if current_items:
        windows.append(
            {
                "items": current_items,
                "opening_reasons": list(current_opening_reasons),
            }
        )

    return windows, boundary_examples


def _span_bounds(items: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    first = items[0]["token_span"]
    last = items[-1]["token_span"]
    return int(first[0]), int(last[1])


def _window_preview(
    window_items: List[Dict[str, Any]],
    *,
    timeline_tokens: List[Any],
    collator: AETHierarchicalCollator,
    special_tokens: List[Any],
    next_window_tokens: List[Any] | None,
    preview_items: int,
    opening_reasons: Sequence[str],
    closing_reasons: Sequence[str],
) -> Dict[str, Any]:
    start_idx, end_idx = _span_bounds(window_items)
    window_tokens = timeline_tokens[start_idx : end_idx + 1]
    item_count = len(window_items)
    token_count = max(0, end_idx - start_idx + 1)
    marker_slots = 2 if collator.window_markers.enabled else 0
    raw_token_budget = max(0, int(collator.max_len) - len(special_tokens) - marker_slots)
    category_counts = Counter(_item_category(item) for item in window_items)
    w_type_id = collator._infer_window_type_id(window_tokens)
    w_start_abs = float(window_tokens[0].t_from_start_hours) if window_tokens else 0.0
    next_type_id = collator._infer_window_type_id(next_window_tokens) if next_window_tokens else None
    next_start_abs = float(next_window_tokens[0].t_from_start_hours) if next_window_tokens else None
    processed_ids, *_ = collator._process_window(
        window_tokens,
        special_tokens,
        w_type_id=w_type_id,
        w_start_abs=w_start_abs,
        next_type_id=next_type_id,
        next_start_abs=next_start_abs,
    )
    duration = max(0.0, _item_t(window_items[-1]) - _item_t(window_items[0])) if window_items else 0.0
    return {
        "item_count": int(item_count),
        "token_count": int(token_count),
        "processed_seq_len": int(len(processed_ids)),
        "truncated_by_max_len": bool(token_count > raw_token_budget),
        "duration_hours": float(duration),
        "start_time_hours": float(_item_t(window_items[0])) if window_items else 0.0,
        "end_time_hours": float(_item_t(window_items[-1])) if window_items else 0.0,
        "inferred_window_type_id": int(w_type_id),
        "opening_reasons": list(opening_reasons),
        "closing_reasons": list(closing_reasons),
        "opening_item": _item_preview(window_items[0]) if window_items else None,
        "closing_item": _item_preview(window_items[-1]) if window_items else None,
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "first_items": [_item_preview(x) for x in window_items[:preview_items]],
        "last_items": [_item_preview(x) for x in window_items[-preview_items:]],
    }


def _policy_summary_bucket() -> Dict[str, Any]:
    return {
        "subjects": 0,
        "windows_total": 0,
        "boundaries_total": 0,
        "window_item_counts": [],
        "window_token_counts": [],
        "window_processed_lens": [],
        "window_durations_hours": [],
        "window_type_ids": Counter(),
        "window_type_unk": 0,
        "truncated_windows": 0,
        "opening_labels": Counter(),
        "closing_labels": Counter(),
        "opening_categories": Counter(),
        "closing_categories": Counter(),
        "boundary_reasons": Counter(),
        "boundary_labels": Counter(),
    }


def _decode_full_timeline(
    event_tokens: List[Any],
    *,
    artifacts: AuditArtifacts,
) -> List[Dict[str, Any]]:
    struct_id2label = _make_structural_id2label(artifacts.structural_codebook)
    decoded = decode_timeline_tokens(
        event_tokens,
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
        structural_offset=_offset(artifacts.manifest, "structural", 2_200_000),
        structural_id2label=struct_id2label,
        special_id2name=SPECIAL_ID2NAME,
    )
    return _annotate_token_spans(decoded)


def _collect_policy_outputs(
    db: mr.SubjectDatabase,
    subject_ids: List[int],
    *,
    encoders: Dict[Any, Any],
    artifacts: AuditArtifacts,
    max_windows: int,
    max_len_per_window: int,
    policies: List[SegmentationPolicy],
    example_subjects: int,
    example_windows: int,
    preview_items: int,
    example_boundaries: int,
    boundary_context_items: int,
) -> Dict[str, Any]:
    collator = AETHierarchicalCollator(
        max_windows=max_windows,
        max_len_per_window=max_len_per_window,
        window_markers=WindowMarkerConfig(),
    )
    summaries: Dict[str, Dict[str, Any]] = {policy.key: _policy_summary_bucket() for policy in policies}
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    boundary_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for sid in subject_ids:
        audited = _audit_subject_tokenization(
            db,
            int(sid),
            encoders=encoders,
            codebook=artifacts.structural_codebook,
            artifacts=artifacts,
        )
        timeline = audited["timeline"]
        special_tokens, event_tokens = collator._split_special(timeline)
        decoded_items = _decode_full_timeline(event_tokens, artifacts=artifacts)

        for policy in policies:
            segmented_windows, boundaries = _segment_items(
                decoded_items,
                policy=policy,
                context_items=boundary_context_items,
            )
            kept_windows = segmented_windows[:max_windows]
            bucket = summaries[policy.key]
            bucket["subjects"] += 1
            bucket["windows_total"] += len(kept_windows)
            bucket["boundaries_total"] += len(boundaries)

            if len(boundary_examples[policy.key]) < example_boundaries:
                room = max(0, example_boundaries - len(boundary_examples[policy.key]))
                for ex in boundaries[:room]:
                    ex_copy = dict(ex)
                    ex_copy["subject_id"] = int(sid)
                    boundary_examples[policy.key].append(ex_copy)

            window_previews: List[Dict[str, Any]] = []
            for wi, window_obj in enumerate(kept_windows):
                items = window_obj["items"]
                if not items:
                    continue
                next_window_tokens = None
                if wi + 1 < len(kept_windows):
                    n_start, n_end = _span_bounds(kept_windows[wi + 1]["items"])
                    next_window_tokens = event_tokens[n_start : n_end + 1]

                preview = _window_preview(
                    items,
                    timeline_tokens=event_tokens,
                    collator=collator,
                    special_tokens=special_tokens,
                    next_window_tokens=next_window_tokens,
                    preview_items=preview_items,
                    opening_reasons=window_obj.get("opening_reasons", []),
                    closing_reasons=window_obj.get("closing_reasons", []),
                )
                preview["window_index"] = int(wi)
                preview["subject_id"] = int(sid)
                window_previews.append(preview)

                bucket["window_item_counts"].append(float(preview["item_count"]))
                bucket["window_token_counts"].append(float(preview["token_count"]))
                bucket["window_processed_lens"].append(float(preview["processed_seq_len"]))
                bucket["window_durations_hours"].append(float(preview["duration_hours"]))
                bucket["window_type_ids"][str(preview["inferred_window_type_id"])] += 1
                if int(preview["inferred_window_type_id"]) == 0:
                    bucket["window_type_unk"] += 1
                if bool(preview["truncated_by_max_len"]):
                    bucket["truncated_windows"] += 1

                opening_item = preview.get("opening_item")
                if opening_item is not None:
                    bucket["opening_labels"][str(opening_item.get("label"))] += 1
                    bucket["opening_categories"][str(opening_item.get("category"))] += 1
                closing_item = preview.get("closing_item")
                if closing_item is not None:
                    bucket["closing_labels"][str(closing_item.get("label"))] += 1
                    bucket["closing_categories"][str(closing_item.get("category"))] += 1

            for boundary in boundaries:
                label = str(boundary["boundary_item"].get("label"))
                bucket["boundary_labels"][label] += 1
                for reason in boundary["reasons"]:
                    bucket["boundary_reasons"][str(reason)] += 1

            if len(examples[policy.key]) < example_subjects:
                examples[policy.key].append(
                    {
                        "subject_id": int(sid),
                        "window_count": int(len(kept_windows)),
                        "windows": window_previews[:example_windows],
                    }
                )

    out: Dict[str, Any] = {}
    for policy in policies:
        bucket = summaries[policy.key]
        win_total = int(bucket["windows_total"])
        out[policy.key] = {
            "policy": {
                "name": policy.name,
                "placement": policy.placement,
                "gap_hours": float(policy.gap_hours),
                "custom_boundary_contains": list(policy.custom_contains),
            },
            "summary": {
                "subjects_scanned": int(bucket["subjects"]),
                "windows_total": win_total,
                "boundaries_total": int(bucket["boundaries_total"]),
                "avg_windows_per_subject": (
                    float(win_total) / float(bucket["subjects"]) if bucket["subjects"] > 0 else 0.0
                ),
                "window_item_count": _summarize_numeric(bucket["window_item_counts"]),
                "window_token_count": _summarize_numeric(bucket["window_token_counts"]),
                "window_processed_seq_len": _summarize_numeric(bucket["window_processed_lens"]),
                "window_duration_hours": _summarize_numeric(bucket["window_durations_hours"]),
                "window_type_unk_frac": (
                    float(bucket["window_type_unk"]) / float(win_total) if win_total > 0 else 0.0
                ),
                "truncated_window_frac": (
                    float(bucket["truncated_windows"]) / float(win_total) if win_total > 0 else 0.0
                ),
                "window_type_counts": {str(k): int(v) for k, v in bucket["window_type_ids"].items()},
                "top_boundary_reasons": [
                    {"key": str(k), "count": int(v)} for k, v in bucket["boundary_reasons"].most_common(20)
                ],
                "top_boundary_labels": [
                    {"key": str(k), "count": int(v)} for k, v in bucket["boundary_labels"].most_common(20)
                ],
                "top_opening_labels": [
                    {"key": str(k), "count": int(v)} for k, v in bucket["opening_labels"].most_common(20)
                ],
                "top_closing_labels": [
                    {"key": str(k), "count": int(v)} for k, v in bucket["closing_labels"].most_common(20)
                ],
                "top_opening_categories": [
                    {"key": str(k), "count": int(v)} for k, v in bucket["opening_categories"].most_common(20)
                ],
                "top_closing_categories": [
                    {"key": str(k), "count": int(v)} for k, v in bucket["closing_categories"].most_common(20)
                ],
            },
            "boundary_examples": boundary_examples[policy.key],
            "subject_examples": examples[policy.key],
        }
    return out


def _print_summary(payload: Mapping[str, Any]) -> None:
    print("Window Segmentation Audit")
    print("Config:", json.dumps(payload["config"], indent=2))
    print("Raw top prefixes:")
    for row in payload["raw_excerpt"]["top_prefixes"]:
        print(" ", row["key"], row["count"])
    for policy_key, result in payload["policies"].items():
        summary = result["summary"]
        print(f"\nPolicy {policy_key}")
        print(
            " windows_total",
            summary["windows_total"],
            "| avg_windows_per_subject",
            f"{summary['avg_windows_per_subject']:.2f}",
        )
        print(
            " token_count p50/p90/max",
            f"{summary['window_token_count']['p50']:.1f}",
            f"{summary['window_token_count']['p90']:.1f}",
            f"{summary['window_token_count']['max']:.1f}",
        )
        print(
            " duration_hours p50/p90/max",
            f"{summary['window_duration_hours']['p50']:.2f}",
            f"{summary['window_duration_hours']['p90']:.2f}",
            f"{summary['window_duration_hours']['max']:.2f}",
        )
        print(
            " trunc_frac",
            f"{summary['truncated_window_frac']:.4f}",
            "| unk_type_frac",
            f"{summary['window_type_unk_frac']:.4f}",
        )
        print(" top boundary reasons:")
        for row in summary["top_boundary_reasons"][:8]:
            print("  ", row["key"], row["count"])
        print(" top opening labels:")
        for row in summary["top_opening_labels"][:8]:
            print("  ", row["key"], row["count"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Window-segmentation development audit. Builds real token timelines, decodes them, "
            "and compares heuristic boundary policies with per-window and boundary-context examples."
        )
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=100)
    ap.add_argument("--subject_ids", default=None, help="Comma-separated subject ids to inspect.")
    ap.add_argument("--top_k", type=int, default=20)

    ap.add_argument("--medtok_vocab_dir", default=str(PROJECT_ROOT / "artifacts" / "medtok"))
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_attr_dir", default=str(PROJECT_ROOT / "artifacts" / "medtok_attrs"))
    ap.add_argument("--structural_yaml", default=str(PROJECT_ROOT / "configs" / "data" / "structural_codes.yaml"))

    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--tokenizer_ckpt", required=True)

    ap.add_argument(
        "--policy",
        action="append",
        choices=[
            "current",
            "all_structural",
            "gap_only",
            "current_plus_gap",
            "structural_or_gap",
            "transition_like",
            "transition_like_plus_gap",
            "marker_or_transition",
            "marker_or_transition_plus_gap",
            "custom_contains",
            "custom_contains_plus_gap",
        ],
        help="Repeatable. Compare multiple boundary heuristics in one run.",
    )
    ap.add_argument(
        "--placement",
        action="append",
        choices=["open_next", "close_current"],
        help="Repeatable. Whether the boundary item opens the next window or closes the current one.",
    )
    ap.add_argument("--gap_hours", type=float, default=6.0)
    ap.add_argument(
        "--custom_boundary_contains",
        action="append",
        default=None,
        help="Repeatable substring trigger used by custom_contains policies.",
    )

    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--max_len_per_window", type=int, default=128)
    ap.add_argument("--example_subjects", type=int, default=4)
    ap.add_argument("--example_windows", type=int, default=6)
    ap.add_argument("--preview_items", type=int, default=8)
    ap.add_argument("--example_boundaries", type=int, default=20)
    ap.add_argument("--boundary_context_items", type=int, default=5)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    subject_ids = _parse_subject_ids(args.subject_ids)
    artifacts = _build_static_artifacts(args)
    db = mr.SubjectDatabase(args.meds_reader_db)
    if not subject_ids:
        subject_ids = _load_subject_ids(args.splits_parquet, args.split, args.max_subjects)
    if not subject_ids:
        raise ValueError(f"No subject IDs found for split={args.split}")

    raw_summary, structural_raw_codes = _summarize_raw_subjects(
        db,
        subject_ids,
        artifacts=artifacts,
        top_k=args.top_k,
    )

    struct_codes_union = set(structural_raw_codes)
    if artifacts.structural_codebook is not None:
        struct_codes_union.update(artifacts.structural_codebook.code2label.keys())
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)

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
    )

    policies = _make_policies(args)
    policy_payload = _collect_policy_outputs(
        db,
        subject_ids,
        encoders=encoders,
        artifacts=artifacts,
        max_windows=args.max_windows,
        max_len_per_window=args.max_len_per_window,
        policies=policies,
        example_subjects=args.example_subjects,
        example_windows=args.example_windows,
        preview_items=args.preview_items,
        example_boundaries=args.example_boundaries,
        boundary_context_items=args.boundary_context_items,
    )

    payload = {
        "config": {
            "split": args.split,
            "subjects_scanned": len(subject_ids),
            "subject_ids": [int(x) for x in subject_ids[: min(len(subject_ids), 50)]],
            "max_windows": args.max_windows,
            "max_len_per_window": args.max_len_per_window,
            "measurement_num_codebooks": artifacts.measurement_num_codebooks,
            "measurement_codebook_size": artifacts.measurement_codebook_size,
            "measurement_stride": artifacts.measurement_stride,
        },
        "raw_excerpt": {
            "events_by_category": raw_summary["events_by_category"],
            "measurement_status": raw_summary["measurement_status"],
            "medtok_hits": raw_summary["medtok_hits"],
            "top_prefixes": raw_summary["top_prefixes"],
            "top_other_codes": raw_summary["top_other_codes"],
            "top_medtok_misses": raw_summary["top_medtok_misses"],
        },
        "policies": policy_payload,
    }
    _print_summary(payload)

    if args.output_json:
        out_fp = Path(args.output_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote window audit JSON to {out_fp}")


if __name__ == "__main__":
    main()
