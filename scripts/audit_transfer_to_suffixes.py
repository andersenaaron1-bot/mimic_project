#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import meds_reader as mr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import _load_subject_ids
from src.ehr_hier.data.structural_codes import (
    load_structural_codebook_yaml,
    looks_ed_location,
    looks_icu_location,
    looks_or_location,
)


def _code_suffix(code: str) -> str:
    text = str(code).strip()
    if "//" not in text:
        return ""
    return text.split("//", 1)[1].strip()


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def _pathway_flags(text: str) -> Dict[str, int]:
    upper = str(text).upper()
    return {
        "looks_ed": int(looks_ed_location(upper)),
        "looks_icu": int(looks_icu_location(upper)),
        "looks_or": int(looks_or_location(upper)),
    }


def _top_rows(counter: Counter[str], *, top_k: int) -> List[Dict[str, Any]]:
    return [
        {"key": str(key), "count": int(count)}
        for key, count in counter.most_common(int(top_k))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit TRANSFER_TO opener surfaces on a sampled split. "
            "Useful for deciding whether ICU/OR/inpatient subtyping can be safely strengthened from causal openers."
        )
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=20_000)
    ap.add_argument("--sample_seed", type=int, default=None)
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--top_k", type=int, default=250)
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    codebook = load_structural_codebook_yaml(args.structural_yaml, default_offset=2_200_000)
    subject_ids = _load_subject_ids(
        args.splits_parquet,
        args.split,
        int(args.max_subjects),
        sample_seed=args.sample_seed,
    )
    db = mr.SubjectDatabase(args.meds_reader_db)

    full_code_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    normalized_suffix_counts: Counter[str] = Counter()
    inferred_type_counts: Counter[str] = Counter()
    type_by_suffix: Dict[str, Counter[str]] = defaultdict(Counter)
    flag_counts: Counter[str] = Counter()

    started_at = time.time()
    for idx, sid in enumerate(subject_ids, start=1):
        subj = db[int(sid)]
        for ev in subj.events:
            code = getattr(ev, "code", None)
            if code is None:
                continue
            code_str = str(code).strip()
            if not code_str.upper().startswith("TRANSFER_TO"):
                continue
            suffix = _code_suffix(code_str)
            suffix_norm = _normalize_space(suffix)
            inferred = codebook.window_type_name(code=code_str, action="close_open") or "UNK"

            full_code_counts[code_str] += 1
            suffix_counts[suffix] += 1
            normalized_suffix_counts[suffix_norm] += 1
            inferred_type_counts[inferred] += 1
            type_by_suffix[suffix_norm][inferred] += 1
            for flag_name, flag_val in _pathway_flags(suffix_norm).items():
                if flag_val:
                    flag_counts[flag_name] += 1

        if args.progress_every and idx % int(args.progress_every) == 0:
            elapsed = time.time() - started_at
            rate = float(idx) / elapsed if elapsed > 0 else 0.0
            print(
                f"[transfer-to] {idx}/{len(subject_ids)} subjects | "
                f"transfer_events={int(sum(full_code_counts.values()))} | rate={rate:.1f} subj/s"
            )

    rows: List[Dict[str, Any]] = []
    for suffix, count in normalized_suffix_counts.most_common():
        type_counter = type_by_suffix.get(suffix, Counter())
        current_type = type_counter.most_common(1)[0][0] if type_counter else "UNK"
        flags = _pathway_flags(suffix)
        rows.append(
            {
                "suffix": suffix,
                "event_count": int(count),
                "current_inferred_window_type": str(current_type),
                "type_counts": {str(k): int(v) for k, v in type_counter.items()},
                **flags,
            }
        )

    payload = {
        "config": {
            "split": str(args.split),
            "subjects_scanned": int(len(subject_ids)),
            "sample_seed": int(args.sample_seed) if args.sample_seed is not None else None,
            "structural_yaml": str(args.structural_yaml),
        },
        "summary": {
            "transfer_to_events": int(sum(full_code_counts.values())),
            "unique_transfer_to_codes": int(len(full_code_counts)),
            "unique_suffixes_raw": int(len(suffix_counts)),
            "unique_suffixes_normalized": int(len(normalized_suffix_counts)),
            "inferred_window_type_counts": {str(k): int(v) for k, v in inferred_type_counts.items()},
            "flag_counts": {str(k): int(v) for k, v in flag_counts.items()},
        },
        "top_transfer_to_codes": _top_rows(full_code_counts, top_k=args.top_k),
        "top_suffixes": rows[: int(args.top_k)],
    }

    if args.output_csv:
        csv_fp = Path(args.output_csv)
        csv_fp.parent.mkdir(parents=True, exist_ok=True)
        with csv_fp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "suffix",
                    "event_count",
                    "current_inferred_window_type",
                    "looks_ed",
                    "looks_icu",
                    "looks_or",
                    "type_counts",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "suffix": row["suffix"],
                        "event_count": row["event_count"],
                        "current_inferred_window_type": row["current_inferred_window_type"],
                        "looks_ed": row["looks_ed"],
                        "looks_icu": row["looks_icu"],
                        "looks_or": row["looks_or"],
                        "type_counts": json.dumps(row["type_counts"], sort_keys=True),
                    }
                )

    if args.output_json:
        json_fp = Path(args.output_json)
        json_fp.parent.mkdir(parents=True, exist_ok=True)
        json_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print("Top suffixes:")
    for row in rows[: min(int(args.top_k), 40)]:
        print(
            f"  {row['suffix'] or '<empty>'}: {row['event_count']} "
            f"[type={row['current_inferred_window_type']}, "
            f"ed={row['looks_ed']}, icu={row['looks_icu']}, or={row['looks_or']}]"
        )
    if args.output_csv:
        print(f"Wrote CSV: {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
