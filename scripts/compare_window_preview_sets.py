#!/usr/bin/env python
"""Compare two sets of preview_chunked_trajectory JSON outputs.

Each set is expected to contain files produced by scripts/preview_chunked_trajectory.py,
typically named like:
  preview_subject_index_0.json
  preview_subject_index_1.json
  ...

This script summarizes boundary behavior and possible timing-coercion artifacts so that
window policies can be compared directly.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoundaryRecord:
    prev_type: str
    next_type: str
    prev_last_time_h: float
    next_first_time_h: float
    prev_last_label: str
    next_first_label: str
    next_first_category: str


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _token_label(tok: dict[str, Any]) -> str:
    return str(tok.get("raw_code") or tok.get("label") or tok.get("category") or "UNK")


def _non_special_abs_tokens(window: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    out: list[tuple[float, dict[str, Any]]] = []
    for chunk in window.get("chunks", []):
        base = float(chunk.get("chunk_start_abs_hours", 0.0) or 0.0)
        seq = chunk.get("decoded_sequence", [])
        times = chunk.get("processed_times", [])
        for i, tok in enumerate(seq):
            if tok.get("category") == "SPECIAL":
                continue
            rel = float(times[i]) if i < len(times) else float(tok.get("t_from_start_hours", 0.0) or 0.0)
            out.append((base + rel, tok))
    return out


def _boundary_group(label: str, category: str) -> str:
    if "TRANSFER_TO//" in label:
        return "TRANSFER_TO"
    if label in {"ED_REGISTRATION", "ED_OUT"}:
        return "ED_EVENT"
    if label.startswith("HOSPITAL_ADMISSION//"):
        return "HOSP_ADMISSION"
    if label.startswith("HOSPITAL_DISCHARGE//"):
        return "HOSP_DISCHARGE"
    if label in {"MEDS_BIRTH", "MEDS_DEATH"}:
        return label
    if category == "STRUCTURAL":
        return "OTHER_STRUCTURAL"
    if category == "DIAGNOSIS":
        return "DIAGNOSIS"
    if category == "PROCEDURE":
        return "PROCEDURE"
    if category == "MEDICATION":
        return "MEDICATION"
    if category == "MEASUREMENT":
        return "MEASUREMENT"
    return category or "UNK"


def _collect_boundaries(subject_payload: dict[str, Any]) -> list[BoundaryRecord]:
    windows = subject_payload.get("windows", [])
    records: list[BoundaryRecord] = []
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        nxt = windows[i]
        prev_tokens = _non_special_abs_tokens(prev)
        next_tokens = _non_special_abs_tokens(nxt)
        if not prev_tokens or not next_tokens:
            continue
        pt, p_tok = prev_tokens[-1]
        nt, n_tok = next_tokens[0]
        records.append(
            BoundaryRecord(
                prev_type=str(prev.get("semantic_window_type")),
                next_type=str(nxt.get("semantic_window_type")),
                prev_last_time_h=pt,
                next_first_time_h=nt,
                prev_last_label=_token_label(p_tok),
                next_first_label=_token_label(n_tok),
                next_first_category=str(n_tok.get("category") or "UNK"),
            )
        )
    return records


def _summarize(paths: list[str]) -> dict[str, Any]:
    subjects = []
    boundary_records: list[BoundaryRecord] = []
    boundary_first_group = Counter()
    boundary_first_label = Counter()
    type_transition = Counter()
    exact_same_ts = Counter()

    total_windows = 0
    total_chunks = 0
    total_split_windows = 0
    total_unk_windows = 0

    for fp in paths:
        d = _load_json(fp)
        windows = d.get("windows", [])
        w_count = len(windows)
        c_count = sum(int(w.get("chunk_count", 0) or 0) for w in windows)
        split_count = sum(1 for w in windows if int(w.get("chunk_count", 0) or 0) > 1)
        unk_count = sum(1 for w in windows if str(w.get("semantic_window_type")) == "UNK")

        total_windows += w_count
        total_chunks += c_count
        total_split_windows += split_count
        total_unk_windows += unk_count

        subjects.append(
            {
                "file": str(Path(fp).name),
                "subject_id": d.get("subject_id"),
                "windows": w_count,
                "chunks": c_count,
                "split_windows": split_count,
                "unk_windows": unk_count,
            }
        )

        b = _collect_boundaries(d)
        boundary_records.extend(b)
        for rec in b:
            g = _boundary_group(rec.next_first_label, rec.next_first_category)
            boundary_first_group[g] += 1
            boundary_first_label[rec.next_first_label] += 1
            type_transition[(rec.prev_type, rec.next_type)] += 1
            if abs(rec.next_first_time_h - rec.prev_last_time_h) < 1e-6:
                exact_same_ts[g] += 1

    n_subjects = len(subjects)
    n_boundaries = len(boundary_records)
    same_ts_total = int(sum(exact_same_ts.values()))
    same_ts_nontransfer = int(
        sum(v for k, v in exact_same_ts.items() if k != "TRANSFER_TO")
    )

    return {
        "n_subjects": n_subjects,
        "n_boundaries": n_boundaries,
        "avg_windows_per_subject": (total_windows / n_subjects) if n_subjects else 0.0,
        "avg_chunks_per_subject": (total_chunks / n_subjects) if n_subjects else 0.0,
        "avg_chunks_per_window": (total_chunks / total_windows) if total_windows else 0.0,
        "split_window_frac": (total_split_windows / total_windows) if total_windows else 0.0,
        "unk_window_frac": (total_unk_windows / total_windows) if total_windows else 0.0,
        "boundary_first_group": dict(boundary_first_group.most_common()),
        "boundary_first_label_top20": [
            {"label": k, "count": int(v)} for k, v in boundary_first_label.most_common(20)
        ],
        "window_type_transitions_top20": [
            {"from": k[0], "to": k[1], "count": int(v)} for k, v in type_transition.most_common(20)
        ],
        "same_timestamp_boundaries_total": same_ts_total,
        "same_timestamp_boundaries_non_transfer": same_ts_nontransfer,
        "subjects": subjects,
    }


def _collect_paths(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return paths


def _print_summary(name: str, s: dict[str, Any]) -> None:
    print(f"\n== {name} ==")
    print(f"subjects: {s['n_subjects']}")
    print(f"boundaries: {s['n_boundaries']}")
    print(f"avg_windows_per_subject: {s['avg_windows_per_subject']:.3f}")
    print(f"avg_chunks_per_subject: {s['avg_chunks_per_subject']:.3f}")
    print(f"avg_chunks_per_window: {s['avg_chunks_per_window']:.3f}")
    print(f"split_window_frac: {s['split_window_frac']:.3f}")
    print(f"unk_window_frac: {s['unk_window_frac']:.3f}")
    print(f"same_timestamp_boundaries_total: {s['same_timestamp_boundaries_total']}")
    print(f"same_timestamp_boundaries_non_transfer: {s['same_timestamp_boundaries_non_transfer']}")
    print("boundary_first_group:", s["boundary_first_group"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two sets of preview window JSON outputs.")
    ap.add_argument("--a_glob", required=True, help="Glob for policy A preview jsons.")
    ap.add_argument("--b_glob", required=True, help="Glob for policy B preview jsons.")
    ap.add_argument("--a_name", default="policy_a")
    ap.add_argument("--b_name", default="policy_b")
    ap.add_argument("--output_json", default=None, help="Optional output json path.")
    args = ap.parse_args()

    a_paths = _collect_paths(args.a_glob)
    b_paths = _collect_paths(args.b_glob)

    a = _summarize(a_paths)
    b = _summarize(b_paths)

    _print_summary(args.a_name, a)
    _print_summary(args.b_name, b)

    delta = {
        "avg_windows_per_subject": b["avg_windows_per_subject"] - a["avg_windows_per_subject"],
        "avg_chunks_per_subject": b["avg_chunks_per_subject"] - a["avg_chunks_per_subject"],
        "avg_chunks_per_window": b["avg_chunks_per_window"] - a["avg_chunks_per_window"],
        "split_window_frac": b["split_window_frac"] - a["split_window_frac"],
        "unk_window_frac": b["unk_window_frac"] - a["unk_window_frac"],
        "same_timestamp_boundaries_total": b["same_timestamp_boundaries_total"]
        - a["same_timestamp_boundaries_total"],
        "same_timestamp_boundaries_non_transfer": b["same_timestamp_boundaries_non_transfer"]
        - a["same_timestamp_boundaries_non_transfer"],
        "n_boundaries": b["n_boundaries"] - a["n_boundaries"],
    }

    print("\n== delta (b - a) ==")
    for k, v in delta.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    payload = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "a": a,
        "b": b,
        "delta_b_minus_a": delta,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote comparison json: {args.output_json}")


if __name__ == "__main__":
    main()

