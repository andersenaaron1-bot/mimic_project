#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meds_reader as mr

from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.structural_codes import StructuralCodebook, load_structural_codebook_yaml
from src.ehr_hier.data.token_types import TokenCategory
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
    CategoryVocab,
    build_vocab_from_code2embeddings,
    load_medtok_vocab,
)
from src.ehr_hier.tokenizers.vocab_contract import validate_medtok_inputs


@dataclass
class Artifacts:
    code2id: Optional[Dict[str, int]]
    diag_vocab: CategoryVocab
    proc_vocab: CategoryVocab
    med_vocab: CategoryVocab
    structural_codebook: Optional[StructuralCodebook]


def _offset(manifest: Dict[str, Any], key: str, default: int) -> int:
    return int(manifest.get(key, {}).get("offset", default))


def _load_manifest() -> Dict[str, Any]:
    fp = PROJECT_ROOT / "artifacts" / "vocab_manifest.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def _parse_subject_ids(arg: str | None) -> List[int]:
    if arg is None or not str(arg).strip():
        return []
    out: List[int] = []
    for part in str(arg).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _load_subject_ids(splits_parquet: str, split: str) -> List[int]:
    split_df = pd.read_parquet(splits_parquet)[["subject_id", "split"]]
    aliases = {"val": "tuning", "test": "held_out"}
    target_split = aliases.get(split, split)
    ids = split_df.loc[split_df["split"] == target_split, "subject_id"].astype("int64").tolist()
    return [int(x) for x in ids]


def _sample_subject_ids(
    all_ids: List[int],
    *,
    sample_subjects: int,
    sample_strategy: str,
    sample_seed: int,
) -> List[int]:
    if sample_subjects <= 0 or sample_subjects >= len(all_ids):
        return list(all_ids)
    if sample_strategy == "first":
        return list(all_ids[:sample_subjects])
    rng = random.Random(int(sample_seed))
    return sorted(rng.sample(all_ids, k=sample_subjects))


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


def _build_artifacts(args: argparse.Namespace) -> Artifacts:
    manifest = _load_manifest()
    code2id: Optional[Dict[str, int]] = None
    if args.code2id_pt:
        code2id = torch.load(args.code2id_pt, map_location="cpu")

    medtok_inputs = validate_medtok_inputs(
        medtok_code2embeds=getattr(args, "medtok_code2embeds", None),
        medtok_vocab_dir=getattr(args, "medtok_vocab_dir", None),
        allow_smoke_medtok=bool(getattr(args, "allow_smoke_medtok", False)),
    )

    if medtok_inputs["medtok_code2embeds"]:
        diag_vocab = build_vocab_from_code2embeddings(
            str(medtok_inputs["medtok_code2embeds"]),
            offset=_offset(manifest, "diagnosis", 1_000_000),
            name="diagnosis",
            filter_fn=diagnosis_filter,
        )
        proc_vocab = build_vocab_from_code2embeddings(
            str(medtok_inputs["medtok_code2embeds"]),
            offset=_offset(manifest, "procedure", 1_200_000),
            name="procedure",
            filter_fn=procedure_filter,
        )
        med_vocab = build_vocab_from_code2embeddings(
            str(medtok_inputs["medtok_code2embeds"]),
            offset=_offset(manifest, "medication", 1_400_000),
            name="medication",
            filter_fn=medication_filter,
        )
    else:
        medtok_dir = Path(str(medtok_inputs["medtok_vocab_dir"]))
        diag_vocab = load_medtok_vocab(
            str(medtok_dir / "diag_vocab.json"),
            offset=_offset(manifest, "diagnosis", 1_000_000),
            name="diagnosis",
        )
        proc_vocab = load_medtok_vocab(
            str(medtok_dir / "proc_vocab.json"),
            offset=_offset(manifest, "procedure", 1_200_000),
            name="procedure",
        )
        med_vocab = load_medtok_vocab(
            str(medtok_dir / "med_vocab.json"),
            offset=_offset(manifest, "medication", 1_400_000),
            name="medication",
        )

    structural_codebook = (
        load_structural_codebook_yaml(args.structural_yaml)
        if args.structural_yaml
        else None
    )

    return Artifacts(
        code2id=code2id,
        diag_vocab=diag_vocab,
        proc_vocab=proc_vocab,
        med_vocab=med_vocab,
        structural_codebook=structural_codebook,
    )


def _suggest_action(
    *,
    category: str,
    code: str,
    uncaptured_reason: str,
    capture_rate: float,
    numeric_rate: float,
    start_like: int,
    end_like: int,
    stop_like: int,
    in_struct_codebook: bool,
) -> tuple[str, str]:
    code_upper = code.upper()
    process_marker_like = (
        start_like > 0
        or end_like > 0
        or stop_like > 0
        or "INFUSION_START" in code_upper
        or "INFUSION_END" in code_upper
        or "//START//" in code_upper
        or "//END//" in code_upper
        or "//STOP//" in code_upper
    )

    if category == "MEASUREMENT":
        if uncaptured_reason == "code_unmapped" and numeric_rate >= 0.7:
            return ("MEASUREMENT", "add_to_measurement_mapping")
        if uncaptured_reason in {"no_numeric_value", "nonfinite_numeric_value"}:
            return ("STRUCTURAL_OR_DROP", "non_numeric_measurement_review")
        return ("MEASUREMENT", "measurement_routing_review")

    if category in {"DIAGNOSIS", "PROCEDURE", "MEDICATION"}:
        if process_marker_like:
            return ("STRUCTURAL", "move_process_markers_out_of_medtok")
        if capture_rate < 0.2:
            return ("MEDTOK", "expand_canonicalization_or_vocab")
        if capture_rate < 0.8:
            return ("MEDTOK", "canonicalization_review")
        return ("KEEP", "covered")

    if category == "STRUCTURAL":
        if in_struct_codebook:
            return ("STRUCTURAL", "covered_structural")
        return ("STRUCTURAL", "add_structural_label_or_boundary_role")

    # OTHER
    if process_marker_like:
        return ("STRUCTURAL", "candidate_transition_or_overlay_signifier")
    if numeric_rate >= 0.7:
        return ("MEASUREMENT_OR_MED_NUMERIC", "candidate_numeric_reroute")
    if any(x in code_upper for x in ("TRANSFER_TO", "ADMISSION", "DISCHARGE", "ED_", "ICU_")):
        return ("STRUCTURAL", "candidate_structural_reroute")
    return ("DROP_OR_META", "likely_low_value_or_admin")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an uncaptured-event decision table from routed raw MEDS events."
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subject_ids", default=None, help="Comma-separated subject ids (overrides split sampling).")
    ap.add_argument("--sample_subjects", type=int, default=1000)
    ap.add_argument("--sample_strategy", choices=["random", "first"], default="random")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=500)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--allow_smoke_medtok", action="store_true")
    ap.add_argument("--code2id_pt", default=None)
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--progress_every", type=int, default=100)
    args = ap.parse_args()

    artifacts = _build_artifacts(args)
    db = mr.SubjectDatabase(args.meds_reader_db)

    subject_ids = _parse_subject_ids(args.subject_ids)
    if not subject_ids:
        all_ids = _load_subject_ids(args.splits_parquet, args.split)
        subject_ids = _sample_subject_ids(
            all_ids,
            sample_subjects=int(args.sample_subjects),
            sample_strategy=str(args.sample_strategy),
            sample_seed=int(args.sample_seed),
        )
    if not subject_ids:
        raise ValueError("No subject IDs selected.")

    # Per-code aggregates.
    total_by_code = Counter()
    captured_by_code = Counter()
    uncaptured_by_code = Counter()
    reason_by_code: Dict[str, Counter[str]] = defaultdict(Counter)
    category_by_code: Dict[str, Counter[str]] = defaultdict(Counter)
    prefix_by_code = Counter()
    numeric_by_code = Counter()
    start_like_by_code = Counter()
    end_like_by_code = Counter()
    stop_like_by_code = Counter()
    subjects_by_code: Dict[str, set[int]] = defaultdict(set)

    n_subjects = len(subject_ids)
    for i, sid in enumerate(subject_ids, start=1):
        subj = db[int(sid)]
        for ev in subj.events:
            code = getattr(ev, "code", None)
            code_str = str(code) if code is not None else "<NONE>"
            prefix = code_str.split("//", 1)[0].upper()
            category = classify_code_to_category(code).name
            numeric_ok = _is_finite_numeric(getattr(ev, "numeric_value", None))

            total_by_code[code_str] += 1
            category_by_code[code_str][category] += 1
            prefix_by_code[prefix] += 1
            subjects_by_code[code_str].add(int(sid))
            if numeric_ok:
                numeric_by_code[code_str] += 1

            code_upper = code_str.upper()
            if "START" in code_upper:
                start_like_by_code[code_str] += 1
            if "END" in code_upper:
                end_like_by_code[code_str] += 1
            if "STOP" in code_upper:
                stop_like_by_code[code_str] += 1

            captured = False
            reason = "captured"
            if category == "MEASUREMENT":
                if artifacts.code2id is None:
                    reason = "measurement_mapping_missing"
                elif code_str not in artifacts.code2id:
                    reason = "code_unmapped"
                elif getattr(ev, "numeric_value", None) is None:
                    reason = "no_numeric_value"
                elif not numeric_ok:
                    reason = "nonfinite_numeric_value"
                else:
                    captured = True
            elif category == "DIAGNOSIS":
                captured = _has_medtok_match(code, artifacts.diag_vocab, canonicalize_diagnosis_code)
                reason = "captured" if captured else "medtok_no_match"
            elif category == "PROCEDURE":
                captured = _has_medtok_match(code, artifacts.proc_vocab, canonicalize_procedure_code)
                reason = "captured" if captured else "medtok_no_match"
            elif category == "MEDICATION":
                captured = _has_medtok_match(code, artifacts.med_vocab, canonicalize_medication_code)
                reason = "captured" if captured else "medtok_no_match"
            elif category == "STRUCTURAL":
                captured = True
                reason = "captured"
            else:
                reason = "routed_other"

            if captured:
                captured_by_code[code_str] += 1
            else:
                uncaptured_by_code[code_str] += 1
                reason_by_code[code_str][reason] += 1

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == n_subjects):
            print(f"[scan] {i}/{n_subjects} subjects", flush=True)

    # Build decision rows.
    rows: List[Dict[str, Any]] = []
    for code, total in total_by_code.items():
        if int(total) < int(args.min_count):
            continue

        category_counts = category_by_code[code]
        category = category_counts.most_common(1)[0][0] if category_counts else "OTHER"
        captured = int(captured_by_code.get(code, 0))
        uncaptured = int(uncaptured_by_code.get(code, 0))
        capture_rate = float(captured) / float(total) if total > 0 else 0.0
        numeric_rate = float(numeric_by_code.get(code, 0)) / float(total) if total > 0 else 0.0
        subject_count = len(subjects_by_code.get(code, set()))
        subject_coverage = float(subject_count) / float(n_subjects) if n_subjects > 0 else 0.0
        top_reason = reason_by_code[code].most_common(1)[0][0] if reason_by_code[code] else "captured"
        in_struct_codebook = (
            artifacts.structural_codebook is not None and code in artifacts.structural_codebook.code2label
        )
        suggested_family, suggested_action = _suggest_action(
            category=category,
            code=code,
            uncaptured_reason=top_reason,
            capture_rate=capture_rate,
            numeric_rate=numeric_rate,
            start_like=int(start_like_by_code.get(code, 0)),
            end_like=int(end_like_by_code.get(code, 0)),
            stop_like=int(stop_like_by_code.get(code, 0)),
            in_struct_codebook=bool(in_struct_codebook),
        )

        rows.append(
            {
                "code": code,
                "prefix": code.split("//", 1)[0].upper(),
                "routed_category": category,
                "events_total": int(total),
                "subjects_with_code": int(subject_count),
                "subject_coverage_frac": float(subject_coverage),
                "captured_events": int(captured),
                "uncaptured_events": int(uncaptured),
                "capture_rate": float(capture_rate),
                "uncaptured_reason_top": top_reason,
                "uncaptured_reason_counts": json.dumps(dict(reason_by_code[code]), ensure_ascii=True),
                "numeric_rate": float(numeric_rate),
                "start_like_count": int(start_like_by_code.get(code, 0)),
                "end_like_count": int(end_like_by_code.get(code, 0)),
                "stop_like_count": int(stop_like_by_code.get(code, 0)),
                "in_structural_codebook": int(bool(in_struct_codebook)),
                "suggested_family": suggested_family,
                "suggested_action": suggested_action,
                "decision_status": "PENDING",
                "decision_notes": "",
            }
        )

    # Prioritize meaningful misses.
    rows.sort(
        key=lambda r: (
            -int(r["uncaptured_events"]),
            -float(r["subject_coverage_frac"]),
            -int(r["events_total"]),
            str(r["code"]),
        )
    )
    if args.top_k > 0:
        rows = rows[: int(args.top_k)]

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote decision table CSV: {out_csv} (rows={len(out_df)})")

    summary = {
        "split": args.split,
        "subjects_scanned": int(n_subjects),
        "codes_seen": int(len(total_by_code)),
        "rows_written": int(len(out_df)),
        "top_uncaptured_codes": [
            {"code": r["code"], "uncaptured_events": int(r["uncaptured_events"])}
            for r in rows[:20]
        ],
    }
    print("Summary:", json.dumps(summary, indent=2))

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary JSON: {out_json}")


if __name__ == "__main__":
    main()
