#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _load_json(path: str) -> Dict[str, Any]:
    fp = Path(path)
    return json.loads(fp.read_text(encoding="utf-8"))


def _semantic_family(payload: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    return (
        payload.get("timeline", {})
        .get("semantic_effective_capture_by_category", {})
        .get(family, {})
    )


def evaluate_tokenization_freeze(
    audit_payload: Mapping[str, Any],
    *,
    runtime_payload: Optional[Mapping[str, Any]] = None,
    min_mapped_rates: Optional[Mapping[str, float]] = None,
    min_medtok_rates: Optional[Mapping[str, float]] = None,
    max_window_type_unk_frac: float = 0.0,
    require_no_residual_hash: bool = False,
    min_structural_observed_ids: int = 2,
    required_preserve_full_blocks: Optional[set[str]] = None,
) -> Dict[str, Any]:
    min_mapped_rates = dict(min_mapped_rates or {})
    min_medtok_rates = dict(min_medtok_rates or {})
    required_preserve_full_blocks = set(required_preserve_full_blocks or set())
    checks: list[Dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Mapping[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": dict(detail)})

    for family in ("DIAGNOSIS", "PROCEDURE", "MEDICATION"):
        semantic = _semantic_family(audit_payload, family)
        mapped_rate = float(semantic.get("mapped_rate_over_semantic_total", 0.0))
        medtok_rate = float(semantic.get("medtok_only_rate_over_semantic_total", 0.0))
        residual_hash = int(semantic.get("residual_hash", 0))
        min_mapped = float(min_mapped_rates.get(family, 0.99))
        min_medtok = float(min_medtok_rates.get(family, 0.0))
        record(
            f"{family.lower()}_mapped_rate",
            mapped_rate >= min_mapped,
            {"value": mapped_rate, "min_required": min_mapped},
        )
        if min_medtok > 0.0:
            record(
                f"{family.lower()}_medtok_rate",
                medtok_rate >= min_medtok,
                {"value": medtok_rate, "min_required": min_medtok},
            )
        if require_no_residual_hash:
            record(
                f"{family.lower()}_residual_hash_zero",
                residual_hash == 0,
                {"value": residual_hash, "required": 0},
            )

    window_unk_frac = float(audit_payload.get("collation", {}).get("window_type_unk_frac", 1.0))
    record(
        "window_type_unk_frac",
        window_unk_frac <= float(max_window_type_unk_frac),
        {"value": window_unk_frac, "max_allowed": float(max_window_type_unk_frac)},
    )

    if runtime_payload is not None:
        summary = dict(runtime_payload.get("summary", {}) or {})
        preserve_full_blocks = set(summary.get("preserve_full_blocks", []) or [])
        if required_preserve_full_blocks:
            record(
                "preserve_full_blocks",
                required_preserve_full_blocks.issubset(preserve_full_blocks),
                {
                    "value": sorted(preserve_full_blocks),
                    "required": sorted(required_preserve_full_blocks),
                },
            )
        structural_observed = int(
            dict(summary.get("observed_ids_per_block", {}) or {}).get("structural", 0)
        )
        record(
            "structural_observed_ids",
            structural_observed >= int(min_structural_observed_ids),
            {"value": structural_observed, "min_required": int(min_structural_observed_ids)},
        )

    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check whether tokenization/windowing artifacts meet the v1 freeze gate."
    )
    ap.add_argument("--audit_json", required=True)
    ap.add_argument("--runtime_vocab_json", default=None)
    ap.add_argument("--max_window_type_unk_frac", type=float, default=0.0)
    ap.add_argument("--require_no_residual_hash", action="store_true")
    ap.add_argument("--min_structural_observed_ids", type=int, default=2)
    ap.add_argument("--required_preserve_full_blocks", default="special,structural")
    ap.add_argument("--min_mapped_rate_diagnosis", type=float, default=0.99)
    ap.add_argument("--min_mapped_rate_procedure", type=float, default=0.99)
    ap.add_argument("--min_mapped_rate_medication", type=float, default=0.99)
    ap.add_argument("--min_medtok_rate_diagnosis", type=float, default=0.70)
    ap.add_argument("--min_medtok_rate_procedure", type=float, default=0.80)
    ap.add_argument("--min_medtok_rate_medication", type=float, default=0.20)
    args = ap.parse_args()

    result = evaluate_tokenization_freeze(
        _load_json(str(args.audit_json)),
        runtime_payload=_load_json(str(args.runtime_vocab_json)) if args.runtime_vocab_json else None,
        min_mapped_rates={
            "DIAGNOSIS": float(args.min_mapped_rate_diagnosis),
            "PROCEDURE": float(args.min_mapped_rate_procedure),
            "MEDICATION": float(args.min_mapped_rate_medication),
        },
        min_medtok_rates={
            "DIAGNOSIS": float(args.min_medtok_rate_diagnosis),
            "PROCEDURE": float(args.min_medtok_rate_procedure),
            "MEDICATION": float(args.min_medtok_rate_medication),
        },
        max_window_type_unk_frac=float(args.max_window_type_unk_frac),
        require_no_residual_hash=bool(args.require_no_residual_hash),
        min_structural_observed_ids=int(args.min_structural_observed_ids),
        required_preserve_full_blocks={
            part.strip()
            for part in str(args.required_preserve_full_blocks).split(",")
            if part.strip()
        },
    )
    print(json.dumps(result, indent=2))
    if not bool(result.get("passed", False)):
        sys.exit(1)


if __name__ == "__main__":
    main()
