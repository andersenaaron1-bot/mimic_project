#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ehr_hier.tokenizers.medtok_canonicalize import (  # noqa: E402
    canonicalize_diagnosis_code,
    canonicalize_medication_code,
    canonicalize_procedure_code,
    diagnosis_filter,
    ensure_list,
    medication_filter,
    procedure_filter,
)
from src.ehr_hier.tokenizers.medtok_loader import (  # noqa: E402
    CategoryVocab,
    build_vocab_from_code2embeddings,
    load_medtok_vocab,
)


def _offset(manifest: Dict[str, Any], key: str, default: int) -> int:
    return int(manifest.get(key, {}).get("offset", default))


def _load_manifest() -> Dict[str, Any]:
    fp = PROJECT_ROOT / "artifacts" / "vocab_manifest.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except (TypeError, ValueError):
        return float(default)


def _safe_code(row: pd.Series) -> str:
    val = row.get("code", "")
    if val is None:
        return ""
    return str(val)


def _rare_keywords_default() -> List[str]:
    return [
        "CARDIAC ARREST",
        "CODE BLUE",
        "CPR",
        "DEFIB",
        "ROSC",
        "SEPSIS",
        "SEPTIC SHOCK",
        "MASSIVE TRANSFUSION",
        "STROKE",
        "HEMORRH",
        "INTUB",
        "REINTUBATION",
        "ECMO",
        "CRRT",
        "DIALYSIS",
        "RRT",
    ]


def _is_rare_critical(raw_code: str, keywords_upper: List[str]) -> bool:
    code_u = raw_code.upper()
    return any(k in code_u for k in keywords_upper)


def _load_full_vocab(
    *,
    name: str,
    medtok_code2embeds: str | None,
    medtok_vocab_dir: Path,
    manifest: Dict[str, Any],
) -> CategoryVocab:
    if name == "diagnosis":
        offset = _offset(manifest, "diagnosis", 1_000_000)
        filter_fn = diagnosis_filter
        file_name = "diag_vocab.json"
    elif name == "procedure":
        offset = _offset(manifest, "procedure", 1_200_000)
        filter_fn = procedure_filter
        file_name = "proc_vocab.json"
    elif name == "medication":
        offset = _offset(manifest, "medication", 1_400_000)
        filter_fn = medication_filter
        file_name = "med_vocab.json"
    else:
        raise ValueError(f"Unsupported vocab family: {name}")

    if medtok_code2embeds:
        return build_vocab_from_code2embeddings(
            medtok_code2embeds,
            offset=offset,
            name=name,
            filter_fn=filter_fn,
        )
    return load_medtok_vocab(str(medtok_vocab_dir / file_name), offset=offset, name=name)


def _build_family_vocab(
    *,
    df: pd.DataFrame,
    routed_category: str,
    canonicalize_fn: Callable[[object], object],
    full_vocab: CategoryVocab,
    max_explicit: int,
    target_coverage: float,
    keywords_upper: List[str],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    fam = df[df["routed_category"].fillna("").astype(str).str.upper() == str(routed_category).upper()].copy()
    rows_considered = int(len(fam))

    code_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"events": 0.0, "score": 0.0, "rows": 0, "rare_critical": False}
    )
    total_events = 0.0
    mappable_events = 0.0
    unmappable_events = 0.0
    rare_critical_rows = 0

    for _, row in fam.iterrows():
        raw_code = _safe_code(row)
        if not raw_code:
            continue
        events_total = _safe_float(row, "events_total", default=0.0)
        if events_total <= 0.0:
            # fallback if table only has uncaptured counts
            events_total = _safe_float(row, "uncaptured_events", default=0.0) + _safe_float(
                row, "captured_events", default=0.0
            )
        if events_total <= 0.0:
            continue

        total_events += float(events_total)
        subj_cov = _safe_float(row, "subject_coverage_frac", default=0.0)
        score = float(events_total) * (1.0 + max(0.0, float(subj_cov)))
        rare_critical = _is_rare_critical(raw_code, keywords_upper)
        if rare_critical:
            rare_critical_rows += 1

        candidates = ensure_list(canonicalize_fn(raw_code))
        primary: str | None = None
        for cand in candidates:
            cand_str = str(cand)
            if cand_str in full_vocab.code2id and int(full_vocab.code2id[cand_str]) != int(full_vocab.unk_id):
                primary = cand_str
                break

        if primary is None:
            unmappable_events += float(events_total)
            continue

        mappable_events += float(events_total)
        st = code_stats[primary]
        st["events"] += float(events_total)
        st["score"] += float(score)
        st["rows"] += 1
        st["rare_critical"] = bool(st["rare_critical"] or rare_critical)

    if max_explicit <= 0:
        max_explicit = len(code_stats)
    max_explicit = max(1, int(max_explicit))
    target_coverage = min(1.0, max(0.0, float(target_coverage)))

    ranked = sorted(
        code_stats.items(),
        key=lambda kv: (
            0 if bool(kv[1]["rare_critical"]) else 1,
            -float(kv[1]["score"]),
            str(kv[0]),
        ),
    )

    selected: List[str] = []
    selected_set: set[str] = set()
    selected_events = 0.0

    for code, st in ranked:
        if len(selected) >= max_explicit:
            break
        if not bool(st["rare_critical"]):
            continue
        selected.append(code)
        selected_set.add(code)
        selected_events += float(st["events"])

    for code, st in ranked:
        if len(selected) >= max_explicit:
            break
        if code in selected_set:
            continue
        if mappable_events > 0.0 and (selected_events / mappable_events) >= target_coverage:
            break
        selected.append(code)
        selected_set.add(code)
        selected_events += float(st["events"])

    code2id = {"<UNK>": 0}
    for i, code in enumerate(selected, start=1):
        code2id[str(code)] = int(i)

    top_preview = []
    for code in selected[:50]:
        st = code_stats[code]
        top_preview.append(
            {
                "code": code,
                "events": int(st["events"]),
                "score": float(st["score"]),
                "rows": int(st["rows"]),
                "rare_critical": bool(st["rare_critical"]),
            }
        )

    report = {
        "rows_considered": rows_considered,
        "total_events": int(total_events),
        "mappable_events": int(mappable_events),
        "unmappable_events": int(unmappable_events),
        "rare_critical_rows": int(rare_critical_rows),
        "explicit_codes_selected": int(len(selected)),
        "selected_event_coverage_over_mappable": (
            float(selected_events / mappable_events) if mappable_events > 0 else 0.0
        ),
        "selected_events": int(selected_events),
        "top_selected_preview": top_preview,
    }
    return code2id, report


def _write_vocab_json(out_fp: Path, code2id: Dict[str, int]) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(code2id, indent=2, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build compressed MedTok vocabularies (diag/proc/med) from a routed decision table "
            "using explicit-cap + coverage + rare-critical keep policy."
        )
    )
    ap.add_argument("--decision_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default="artifacts/medtok")

    ap.add_argument("--diag_max_explicit", type=int, default=20_000)
    ap.add_argument("--proc_max_explicit", type=int, default=10_000)
    ap.add_argument("--med_max_explicit", type=int, default=25_000)
    ap.add_argument("--diag_target_coverage", type=float, default=0.97)
    ap.add_argument("--proc_target_coverage", type=float, default=0.97)
    ap.add_argument("--med_target_coverage", type=float, default=0.97)

    ap.add_argument("--diag_residual_buckets", type=int, default=8_000)
    ap.add_argument("--proc_residual_buckets", type=int, default=4_000)
    ap.add_argument("--med_residual_buckets", type=int, default=20_000)
    ap.add_argument("--assumed_non_medtok_vocab", type=int, default=35_000)

    ap.add_argument(
        "--rare_critical_keywords",
        default=",".join(_rare_keywords_default()),
        help="Comma-separated keyword list. Matching rows are force-prioritized in explicit selection.",
    )
    ap.add_argument("--output_report_json", default=None)
    args = ap.parse_args()

    decision_fp = Path(args.decision_csv)
    if not decision_fp.exists():
        raise FileNotFoundError(f"Decision CSV not found: {decision_fp}")
    df = pd.read_csv(decision_fp)
    if "routed_category" not in df.columns or "code" not in df.columns:
        raise ValueError("decision_csv must include at least columns: code, routed_category")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    full_diag = _load_full_vocab(
        name="diagnosis",
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=Path(args.medtok_vocab_dir),
        manifest=manifest,
    )
    full_proc = _load_full_vocab(
        name="procedure",
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=Path(args.medtok_vocab_dir),
        manifest=manifest,
    )
    full_med = _load_full_vocab(
        name="medication",
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=Path(args.medtok_vocab_dir),
        manifest=manifest,
    )

    keywords_upper = [k.strip().upper() for k in str(args.rare_critical_keywords).split(",") if k.strip()]

    diag_code2id, diag_report = _build_family_vocab(
        df=df,
        routed_category="DIAGNOSIS",
        canonicalize_fn=canonicalize_diagnosis_code,
        full_vocab=full_diag,
        max_explicit=int(args.diag_max_explicit),
        target_coverage=float(args.diag_target_coverage),
        keywords_upper=keywords_upper,
    )
    proc_code2id, proc_report = _build_family_vocab(
        df=df,
        routed_category="PROCEDURE",
        canonicalize_fn=canonicalize_procedure_code,
        full_vocab=full_proc,
        max_explicit=int(args.proc_max_explicit),
        target_coverage=float(args.proc_target_coverage),
        keywords_upper=keywords_upper,
    )
    med_code2id, med_report = _build_family_vocab(
        df=df,
        routed_category="MEDICATION",
        canonicalize_fn=canonicalize_medication_code,
        full_vocab=full_med,
        max_explicit=int(args.med_max_explicit),
        target_coverage=float(args.med_target_coverage),
        keywords_upper=keywords_upper,
    )

    _write_vocab_json(out_dir / "diag_vocab.json", diag_code2id)
    _write_vocab_json(out_dir / "proc_vocab.json", proc_code2id)
    _write_vocab_json(out_dir / "med_vocab.json", med_code2id)

    medtok_explicit = (len(diag_code2id) - 1) + (len(proc_code2id) - 1) + (len(med_code2id) - 1)
    residual_total = int(args.diag_residual_buckets) + int(args.proc_residual_buckets) + int(args.med_residual_buckets)
    projected_total_vocab = int(args.assumed_non_medtok_vocab) + int(medtok_explicit) + int(residual_total)

    report = {
        "decision_csv": str(decision_fp),
        "out_dir": str(out_dir),
        "policy": {
            "diag_max_explicit": int(args.diag_max_explicit),
            "proc_max_explicit": int(args.proc_max_explicit),
            "med_max_explicit": int(args.med_max_explicit),
            "diag_target_coverage": float(args.diag_target_coverage),
            "proc_target_coverage": float(args.proc_target_coverage),
            "med_target_coverage": float(args.med_target_coverage),
            "diag_residual_buckets": int(args.diag_residual_buckets),
            "proc_residual_buckets": int(args.proc_residual_buckets),
            "med_residual_buckets": int(args.med_residual_buckets),
            "assumed_non_medtok_vocab": int(args.assumed_non_medtok_vocab),
            "rare_critical_keywords": keywords_upper,
        },
        "selected_vocab_sizes": {
            "diag_explicit": int(len(diag_code2id) - 1),
            "proc_explicit": int(len(proc_code2id) - 1),
            "med_explicit": int(len(med_code2id) - 1),
            "medtok_explicit_total": int(medtok_explicit),
            "residual_total": int(residual_total),
            "projected_total_vocab": int(projected_total_vocab),
        },
        "families": {
            "DIAGNOSIS": diag_report,
            "PROCEDURE": proc_report,
            "MEDICATION": med_report,
        },
    }

    print(json.dumps(report, indent=2))
    out_report = Path(args.output_report_json) if args.output_report_json else (out_dir / "compression_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir / 'diag_vocab.json'}")
    print(f"Wrote: {out_dir / 'proc_vocab.json'}")
    print(f"Wrote: {out_dir / 'med_vocab.json'}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()

