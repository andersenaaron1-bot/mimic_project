#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import (  # noqa: E402
    _build_segmentation_config,
    _build_window_marker_config,
    _load_tokenization_contract,
)
from src.ehr_hier.data.event_frames import EventFrame, flatten_event_frames  # noqa: E402
from src.ehr_hier.data.precompiled_format import deserialize_timeline_compact  # noqa: E402
from src.ehr_hier.data.structural_codes import load_structural_codebook_yaml  # noqa: E402
from src.ehr_hier.data.token_types import EventToken, TokenCategory  # noqa: E402
from src.ehr_hier.data.trajectory_splitting import split_special_tokens  # noqa: E402
from src.ehr_hier.data.window_segmentation import (  # noqa: E402
    _build_boundary_bundles,
    _cat_attr_int,
    _resolve_bundle_action,
    _token_transition_action,
    segment_event_tokens,
)


def _parse_subject_ids(arg: str | None) -> set[int]:
    if arg is None or not str(arg).strip():
        return set()
    out: set[int] = set()
    for part in str(arg).split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _type_id2name(codebook: Any | None) -> Dict[int, str]:
    out = {0: "UNK"}
    if codebook is None:
        return out
    for name, idx in codebook.window_type2id().items():
        out[int(idx)] = str(name)
    return out


class _ShardCache:
    def __init__(self, *, max_size: int = 8) -> None:
        self.max_size = max(1, int(max_size))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def load(self, path: str | Path) -> dict[str, Any]:
        key = str(path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        with gzip.open(Path(path), "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        self._cache[key] = payload
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return payload


def _load_index_rows(
    *,
    root: Path,
    index_filename: str,
    subject_ids: set[int],
    max_subjects: int,
    sample_seed: Optional[int],
) -> List[Dict[str, Any]]:
    index_path = root / str(index_filename)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    rows: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for rec in reader:
            sid = int(rec["subject_id"])
            if subject_ids and sid not in subject_ids:
                continue
            rows.append(dict(rec))

    if sample_seed is not None:
        rng = random.Random(int(sample_seed))
        rng.shuffle(rows)
    if max_subjects > 0:
        rows = rows[: int(max_subjects)]
    return rows


def _load_frames_for_row(root: Path, row: Mapping[str, Any], *, shard_cache: _ShardCache) -> List[EventFrame]:
    rel_path = str(row["rel_path"])
    file_path = root / rel_path
    subject_idx_raw = row.get("subject_idx", None)
    if subject_idx_raw is None or str(subject_idx_raw).strip() == "":
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "frame_token_offsets" in payload:
            return deserialize_timeline_compact(payload)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported standalone timeline payload at {file_path}")

    shard = shard_cache.load(file_path)
    serialized = shard["timelines"][int(subject_idx_raw)]
    return deserialize_timeline_compact(serialized)


def _frame_preview(frame: EventFrame, *, type_id2name: Mapping[int, str]) -> Dict[str, Any]:
    tok = frame.token_bundle[0]
    transition_type_id = _cat_attr_int(tok, "transition_window_type_id")
    window_type_attr = _cat_attr_int(tok, "window_type_id")
    return {
        "time_hours": round(float(frame.t_from_start_hours), 3),
        "payload_kind": str(frame.payload_kind),
        "category": TokenCategory(int(frame.category_id)).name,
        "semantic_label": frame.semantic_label,
        "source_code": frame.source_code,
        "concept_code": frame.concept_code,
        "group_code": frame.group_code,
        "token_count": int(frame.token_count),
        "window_hook": frame.window_hook,
        "transition_action": _token_transition_action(tok),
        "transition_window_type_id": transition_type_id,
        "transition_window_type_name": (
            type_id2name.get(int(transition_type_id), str(int(transition_type_id)))
            if transition_type_id is not None
            else None
        ),
        "window_type_id_attr": window_type_attr,
        "window_type_name_attr": (
            type_id2name.get(int(window_type_attr), str(int(window_type_attr)))
            if window_type_attr is not None
            else None
        ),
    }


def _ordered_frame_indices(tokens: Iterable[EventToken], *, token_to_frame_idx: Mapping[int, int]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for tok in tokens:
        idx = token_to_frame_idx.get(id(tok))
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        out.append(int(idx))
    return out


def _find_window_index(
    windows: List[Any],
    *,
    action_field: str,
    action_value: str,
    time_field: str,
    time_value: Optional[float],
    start_idx: int,
) -> tuple[Optional[int], int]:
    if time_value is None:
        return None, start_idx
    for wi in range(int(start_idx), len(windows)):
        window = windows[wi]
        if str(getattr(window, action_field, None)) != str(action_value):
            continue
        w_time = getattr(window, time_field, None)
        if w_time is None:
            continue
        if abs(float(w_time) - float(time_value)) <= 1e-6:
            return int(wi), int(wi + 1)
    return None, int(start_idx)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Inspect precompiled trajectories and emit only semantic window sequences plus the "
            "boundary frames that caused each switch action."
        )
    )
    ap.add_argument("--precompiled_root", required=True)
    ap.add_argument("--index_filename", default="index.csv")
    ap.add_argument("--subject_ids", default=None)
    ap.add_argument("--max_subjects", type=int, default=500)
    ap.add_argument("--sample_seed", type=int, default=13)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--max_boundary_frames", type=int, default=6)
    ap.add_argument("--shard_cache_size", type=int, default=8)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--output_summary_json", default=None)
    args = ap.parse_args()

    root = Path(args.precompiled_root)
    subject_ids = _parse_subject_ids(args.subject_ids)
    rows = _load_index_rows(
        root=root,
        index_filename=str(args.index_filename),
        subject_ids=subject_ids,
        max_subjects=int(args.max_subjects),
        sample_seed=int(args.sample_seed) if args.sample_seed is not None else None,
    )
    if not rows:
        raise ValueError("No precompiled rows selected.")

    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    structural_codebook = load_structural_codebook_yaml(args.structural_yaml, default_offset=2_200_000)
    window_markers_cfg = _build_window_marker_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=structural_codebook,
    )
    segmentation_cfg = _build_segmentation_config(
        tokenization_contract=tokenization_contract,
        structural_codebook=structural_codebook,
        unk_type_id=int(window_markers_cfg.unk_type_id),
    )
    type_id2name = _type_id2name(structural_codebook)
    shard_cache = _ShardCache(max_size=int(args.shard_cache_size))

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "precompiled_root": str(root),
        "index_filename": str(args.index_filename),
        "records_scanned": 0,
        "records_written": 0,
        "records_failed": 0,
        "window_type_counts": {},
        "opening_action_counts": {},
        "closing_action_counts": {},
        "switch_action_counts": {},
        "sequence_length_distribution": {},
        "failed_samples": [],
        "started_at_unix": float(time.time()),
        "elapsed_sec": 0.0,
    }
    window_type_counts: Counter[str] = Counter()
    opening_action_counts: Counter[str] = Counter()
    closing_action_counts: Counter[str] = Counter()
    switch_action_counts: Counter[str] = Counter()
    sequence_length_distribution: Counter[int] = Counter()
    failed_samples: List[Dict[str, Any]] = []

    started_at = time.time()
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            summary["records_scanned"] = int(idx)
            try:
                timeline = _load_frames_for_row(root, row, shard_cache=shard_cache)
                _, event_frames = split_special_tokens(timeline)
                event_tokens = flatten_event_frames(event_frames, clone=False)
                if not event_tokens:
                    continue

                token_to_frame_idx: Dict[int, int] = {}
                for frame_idx, frame in enumerate(event_frames):
                    for tok in frame.token_bundle:
                        token_to_frame_idx[id(tok)] = int(frame_idx)

                windows = segment_event_tokens(event_tokens, config=segmentation_cfg)
                bundles = _build_boundary_bundles(event_tokens, config=segmentation_cfg)

                prev_close_search = 0
                next_open_search = 0
                switches: List[Dict[str, Any]] = []
                for bundle_idx, bundle in enumerate(bundles):
                    start_idx = int(bundle["start_idx"])
                    end_idx = int(bundle["end_idx"])
                    bundle_tokens = event_tokens[start_idx : end_idx + 1]
                    bundle_action, closing_items, opening_items = _resolve_bundle_action(
                        bundle_tokens,
                        list(bundle["candidate_indices"]),  # type: ignore[arg-type]
                        bundle_start_idx=start_idx,
                    )
                    switch_action_counts[str(bundle_action)] += 1
                    closing_frame_ids = _ordered_frame_indices(closing_items, token_to_frame_idx=token_to_frame_idx)
                    opening_frame_ids = _ordered_frame_indices(opening_items, token_to_frame_idx=token_to_frame_idx)
                    closing_time = float(closing_items[-1].t_from_start_hours) if closing_items else None
                    opening_time = float(opening_items[0].t_from_start_hours) if opening_items else None
                    from_window_idx, prev_close_search = _find_window_index(
                        windows,
                        action_field="closing_action",
                        action_value=str(bundle_action),
                        time_field="closing_time_hours",
                        time_value=closing_time,
                        start_idx=prev_close_search,
                    )
                    to_window_idx, next_open_search = _find_window_index(
                        windows,
                        action_field="opening_action",
                        action_value=str(bundle_action),
                        time_field="opening_time_hours",
                        time_value=opening_time,
                        start_idx=next_open_search,
                    )
                    switches.append(
                        {
                            "switch_index": int(bundle_idx),
                            "action": str(bundle_action),
                            "closing_time_hours": round(float(closing_time), 3) if closing_time is not None else None,
                            "opening_time_hours": round(float(opening_time), 3) if opening_time is not None else None,
                            "from_window_index": from_window_idx,
                            "from_window_type": (
                                type_id2name.get(int(windows[from_window_idx].window_type_id), str(int(windows[from_window_idx].window_type_id)))
                                if from_window_idx is not None
                                else None
                            ),
                            "to_window_index": to_window_idx,
                            "to_window_type": (
                                type_id2name.get(int(windows[to_window_idx].window_type_id), str(int(windows[to_window_idx].window_type_id)))
                                if to_window_idx is not None
                                else None
                            ),
                            "closing_frames": [
                                _frame_preview(event_frames[frame_idx], type_id2name=type_id2name)
                                for frame_idx in closing_frame_ids[: int(args.max_boundary_frames)]
                            ],
                            "opening_frames": [
                                _frame_preview(event_frames[frame_idx], type_id2name=type_id2name)
                                for frame_idx in opening_frame_ids[: int(args.max_boundary_frames)]
                            ],
                        }
                    )

                window_sequence: List[Dict[str, Any]] = []
                for w_idx, window in enumerate(windows):
                    window_type = type_id2name.get(int(window.window_type_id), str(int(window.window_type_id)))
                    frame_ids = _ordered_frame_indices(window.tokens, token_to_frame_idx=token_to_frame_idx)
                    window_type_counts[window_type] += 1
                    opening_action_counts[str(window.opening_action or "<none>")] += 1
                    closing_action_counts[str(window.closing_action or "<none>")] += 1
                    first_frame = event_frames[frame_ids[0]] if frame_ids else None
                    last_frame = event_frames[frame_ids[-1]] if frame_ids else None
                    window_sequence.append(
                        {
                            "window_index": int(w_idx),
                            "window_type_id": int(window.window_type_id),
                            "window_type": str(window_type),
                            "opening_action": window.opening_action,
                            "closing_action": window.closing_action,
                            "start_time_hours": round(float(window.start_time_hours), 3),
                            "opening_time_hours": (
                                round(float(window.opening_time_hours), 3)
                                if window.opening_time_hours is not None
                                else None
                            ),
                            "closing_time_hours": (
                                round(float(window.closing_time_hours), 3)
                                if window.closing_time_hours is not None
                                else None
                            ),
                            "frame_count": int(len(frame_ids)),
                            "token_count": int(len(window.tokens)),
                            "first_frame": (
                                _frame_preview(first_frame, type_id2name=type_id2name)
                                if first_frame is not None
                                else None
                            ),
                            "last_frame": (
                                _frame_preview(last_frame, type_id2name=type_id2name)
                                if last_frame is not None
                                else None
                            ),
                        }
                    )

                sequence_length_distribution[int(len(window_sequence))] += 1
                record = {
                    "subject_id": int(row["subject_id"]),
                    "trajectory_ord": (
                        int(row["trajectory_ord"])
                        if row.get("trajectory_ord", "").strip() not in {"", "nan"}
                        else None
                    ),
                    "rel_path": str(row["rel_path"]),
                    "subject_idx": (
                        int(row["subject_idx"])
                        if row.get("subject_idx", "").strip() not in {"", "nan"}
                        else None
                    ),
                    "window_sequence": window_sequence,
                    "switches": switches,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                summary["records_written"] = int(summary["records_written"]) + 1
            except Exception as exc:
                summary["records_failed"] = int(summary["records_failed"]) + 1
                if len(failed_samples) < 24:
                    failed_samples.append(
                        {
                            "subject_id": row.get("subject_id"),
                            "rel_path": row.get("rel_path"),
                            "error": repr(exc),
                        }
                    )

            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                rate = float(idx) / elapsed
                print(
                    f"[precompiled-window-seq] {idx}/{len(rows)} records | "
                    f"elapsed={elapsed/60.0:.1f}m | rate={rate:.2f} rec/s"
                )

    summary["window_type_counts"] = {str(k): int(v) for k, v in window_type_counts.most_common()}
    summary["opening_action_counts"] = {str(k): int(v) for k, v in opening_action_counts.most_common()}
    summary["closing_action_counts"] = {str(k): int(v) for k, v in closing_action_counts.most_common()}
    summary["switch_action_counts"] = {str(k): int(v) for k, v in switch_action_counts.most_common()}
    summary["sequence_length_distribution"] = {str(k): int(v) for k, v in sequence_length_distribution.most_common()}
    summary["failed_samples"] = failed_samples
    summary["elapsed_sec"] = float(time.time() - started_at)

    if args.output_summary_json:
        out_summary = Path(args.output_summary_json)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote window-sequence JSONL to {out_jsonl}")
    if args.output_summary_json:
        print(f"Wrote summary JSON to {args.output_summary_json}")


if __name__ == "__main__":
    main()
