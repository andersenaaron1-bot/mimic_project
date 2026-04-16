#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _parse_csv_set(arg: str | None) -> set[str]:
    if arg is None or not str(arg).strip():
        return set()
    out: set[str] = set()
    for part in str(arg).split(","):
        part = part.strip().upper()
        if part:
            out.add(part)
    return out


def _frame_identity(frame: Mapping[str, Any] | None) -> str:
    if not frame:
        return "<missing>"
    for key in ("source_code", "semantic_label", "concept_code", "group_code", "payload_kind"):
        value = frame.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "<missing>"


def _frame_prefix(frame: Mapping[str, Any] | None) -> str:
    ident = _frame_identity(frame)
    if ident == "<missing>":
        return ident
    text = str(ident).strip()
    if not text:
        return "<missing>"
    return text.split("//", 1)[0].upper()


def _counter_to_dict(counter: Counter[Any], *, limit: int | None = None) -> Dict[str, int]:
    items = counter.most_common(limit)
    return {str(k): int(v) for k, v in items}


def _append_limited(store: List[Dict[str, Any]], item: Dict[str, Any], *, limit: int) -> None:
    if len(store) < int(limit):
        store.append(item)


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze precomputed window-sequence preview JSONL and summarize which event frames "
            "actually caused window switches."
        )
    )
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_examples", type=int, default=24)
    ap.add_argument("--max_outliers", type=int, default=12)
    ap.add_argument(
        "--allowed_close_open_openers",
        default="TRANSFER_TO,ICU_ADMISSION",
        help="Comma-separated allowed opening prefixes for close_open switches.",
    )
    ap.add_argument(
        "--allowed_close_open_closers",
        default="TRANSFER_TO,HOSPITAL_DISCHARGE,DISCHARGE,ICU_DISCHARGE,ED_OUT,MEDS_DEATH",
        help="Comma-separated allowed closing prefixes for close_open switches.",
    )
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    allowed_openers = _parse_csv_set(args.allowed_close_open_openers)
    allowed_closers = _parse_csv_set(args.allowed_close_open_closers)

    opening_prefix_by_action: Dict[str, Counter[str]] = {}
    closing_prefix_by_action: Dict[str, Counter[str]] = {}
    opening_identity_by_action: Dict[str, Counter[str]] = {}
    closing_identity_by_action: Dict[str, Counter[str]] = {}
    window_type_sequences: Counter[str] = Counter()
    suspicious_close_open_examples: List[Dict[str, Any]] = []
    outlier_records: List[Dict[str, Any]] = []
    top_outlier_windows: List[tuple[int, int, Dict[str, Any]]] = []

    total_records = 0
    total_switches = 0

    for rec in _iter_records(input_path):
        total_records += 1
        subject_id = int(rec.get("subject_id", -1))
        sequence = list(rec.get("window_sequence", []))
        switches = list(rec.get("switches", []))
        seq_types = [str(w.get("window_type", "<missing>")) for w in sequence]
        if seq_types:
            window_type_sequences[" -> ".join(seq_types)] += 1

        outlier_tuple = (len(sequence), subject_id, rec)
        top_outlier_windows.append(outlier_tuple)
        top_outlier_windows.sort(key=lambda item: (-int(item[0]), int(item[1])))
        top_outlier_windows = top_outlier_windows[: int(args.max_outliers)]

        for sw in switches:
            total_switches += 1
            action = str(sw.get("action", "<missing>"))
            opening_frames = list(sw.get("opening_frames", []))
            closing_frames = list(sw.get("closing_frames", []))
            opening_frame = opening_frames[0] if opening_frames else None
            closing_frame = closing_frames[-1] if closing_frames else None

            opening_prefix = _frame_prefix(opening_frame)
            closing_prefix = _frame_prefix(closing_frame)
            opening_identity = _frame_identity(opening_frame)
            closing_identity = _frame_identity(closing_frame)

            opening_prefix_by_action.setdefault(action, Counter())[opening_prefix] += 1
            closing_prefix_by_action.setdefault(action, Counter())[closing_prefix] += 1
            opening_identity_by_action.setdefault(action, Counter())[opening_identity] += 1
            closing_identity_by_action.setdefault(action, Counter())[closing_identity] += 1

            if action == "close_open":
                suspicious = False
                if allowed_openers and opening_prefix not in allowed_openers:
                    suspicious = True
                if allowed_closers and closing_prefix not in allowed_closers:
                    suspicious = True
                if suspicious:
                    _append_limited(
                        suspicious_close_open_examples,
                        {
                            "subject_id": int(subject_id),
                            "from_window_type": sw.get("from_window_type"),
                            "to_window_type": sw.get("to_window_type"),
                            "opening_prefix": opening_prefix,
                            "closing_prefix": closing_prefix,
                            "opening_frame": opening_frame,
                            "closing_frame": closing_frame,
                        },
                        limit=int(args.max_examples),
                    )

    outlier_records = [
        {
            "subject_id": int(subject_id),
            "window_count": int(window_count),
            "window_types": [str(w.get("window_type", "<missing>")) for w in rec.get("window_sequence", [])],
            "switch_count": int(len(rec.get("switches", []))),
            "first_switches": list(rec.get("switches", []))[:5],
        }
        for window_count, subject_id, rec in top_outlier_windows
    ]

    summary = {
        "input_jsonl": str(input_path),
        "records_scanned": int(total_records),
        "switches_scanned": int(total_switches),
        "allowed_close_open_openers": sorted(str(x) for x in allowed_openers),
        "allowed_close_open_closers": sorted(str(x) for x in allowed_closers),
        "opening_prefix_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(opening_prefix_by_action.items())
        },
        "closing_prefix_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(closing_prefix_by_action.items())
        },
        "opening_identity_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(opening_identity_by_action.items())
        },
        "closing_identity_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(closing_identity_by_action.items())
        },
        "common_window_sequences": _counter_to_dict(window_type_sequences, limit=int(args.top_k)),
        "suspicious_close_open_examples": suspicious_close_open_examples,
        "top_outlier_records": outlier_records,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote analysis JSON to {out_path}")


if __name__ == "__main__":
    main()
