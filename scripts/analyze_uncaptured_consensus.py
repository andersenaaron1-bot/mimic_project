#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _safe_float_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _safe_str_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def _top_records(df: pd.DataFrame, col: str, k: int) -> List[Dict[str, Any]]:
    if df.empty or col not in df.columns:
        return []
    cols = [
        c
        for c in (
            "code",
            "prefix",
            "routed_category",
            "uncaptured_events",
            "subject_coverage_frac",
            "capture_rate",
            "suggested_family",
            "suggested_action",
            "uncaptured_reason_top",
        )
        if c in df.columns
    ]
    out = (
        df.sort_values(col, ascending=False)
        .head(int(k))[cols]
        .to_dict(orient="records")
    )
    return out


def _parse_reason_counts(reason_col: pd.Series) -> Dict[str, int]:
    agg: Dict[str, int] = {}
    for raw in reason_col:
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for k, v in payload.items():
            try:
                agg[str(k)] = int(agg.get(str(k), 0)) + int(v)
            except (TypeError, ValueError):
                continue
    return dict(sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])))


def _set_overlap(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    sa = set(a.dropna().astype(str))
    sb = set(b.dropna().astype(str))
    inter = len(sa & sb)
    union = len(sa | sb)
    return {
        "set_a": len(sa),
        "set_b": len(sb),
        "intersection": inter,
        "union": union,
        "jaccard": (float(inter) / float(union)) if union > 0 else 1.0,
    }


def _rank_stats(consensus_codes: pd.Series, ref_df: pd.DataFrame) -> Dict[str, Any]:
    if "code" not in ref_df.columns or ref_df.empty:
        return {}
    order = ref_df.copy()
    if "uncaptured_events" in order.columns:
        order = order.sort_values("uncaptured_events", ascending=False)
    order = order.reset_index(drop=True)
    rank_map = {str(code): int(i + 1) for i, code in enumerate(order["code"].astype(str).tolist())}
    ranks = [rank_map[c] for c in consensus_codes.astype(str).tolist() if c in rank_map]
    if not ranks:
        return {"matched_codes": 0}
    ranks_sorted = sorted(ranks)
    mid = len(ranks_sorted) // 2
    median_rank = (
        ranks_sorted[mid]
        if len(ranks_sorted) % 2 == 1
        else (ranks_sorted[mid - 1] + ranks_sorted[mid]) / 2.0
    )
    return {
        "matched_codes": len(ranks),
        "mean_rank": float(sum(ranks) / len(ranks)),
        "median_rank": float(median_rank),
        "min_rank": int(min(ranks)),
        "max_rank": int(max(ranks)),
    }


def _critical_keywords_default() -> List[str]:
    return [
        "CPR",
        "CARDIAC ARREST",
        "CODE BLUE",
        "SEPSIS",
        "SEPTIC",
        "SHOCK",
        "VASOPRESS",
        "INTUB",
        "VENT",
        "ECMO",
        "CRRT",
        "DIALYSIS",
        "RRT",
        "RAPID RESPONSE",
        "RESUS",
        "STROKE",
        "HEMORRH",
        "BLEED",
        "TRANSFUS",
        "ARREST",
    ]


def _is_transition_like(code_upper: str) -> bool:
    transition_markers = (
        "TRANSFER_TO",
        "TRANSFER_FROM",
        "ADMISSION",
        "DISCHARGE",
        "ICU",
        "ED_",
        "OR_",
        "INFUSION_START",
        "INFUSION_END",
        "START",
        "END",
        "STOP",
    )
    return any(m in code_upper for m in transition_markers)


def _is_attr_or_admin_like(code_upper: str) -> bool:
    attr_markers = (
        " FLUSH",
        "ROUTE",
        "FREQ",
        "FREQUENCY",
        "FORM",
        "UNIT",
        "DOSE",
        "RATE",
        "VOLUME",
    )
    return any(m in code_upper for m in attr_markers)


def _classify_surface(row: pd.Series) -> str:
    category = str(row.get("routed_category", "")).upper()
    family = str(row.get("suggested_family", "")).upper()
    reason = str(row.get("uncaptured_reason_top", "")).lower()
    prefix = str(row.get("prefix", "")).upper()
    code_upper = str(row.get("code", "")).upper()
    numeric_rate = float(row.get("numeric_rate", 0.0) or 0.0)

    if family == "STRUCTURAL" or _is_transition_like(code_upper):
        return "structural_transition_overlay"

    if category == "MEASUREMENT":
        if reason == "code_unmapped" and numeric_rate >= 0.7:
            return "measurement_numeric_dense"
        if numeric_rate >= 0.2:
            return "measurement_numeric_sparse"
        return "measurement_non_numeric_or_invalid"

    if category in {"DIAGNOSIS", "PROCEDURE", "MEDICATION"} or family == "MEDTOK":
        if _is_attr_or_admin_like(code_upper):
            return "medication_attributes_or_admin"
        return "medtok_semantic_core"

    if family == "MEASUREMENT_OR_MED_NUMERIC":
        return "medication_numeric_or_attributes"
    if family == "DROP_OR_META":
        return "admin_or_low_value_meta"

    if prefix == "LAB":
        return "measurement_numeric_sparse"
    return "other_unclassified"


def _surface_to_path(surface: str) -> str:
    mapping = {
        "structural_transition_overlay": "STRUCTURAL path (window/boundary/overlay tokens)",
        "measurement_numeric_dense": "MEASUREMENT path (code2id + cVAE/RVQ value tokens)",
        "measurement_numeric_sparse": "MEASUREMENT path or promote to STRUCTURAL if transitional",
        "measurement_non_numeric_or_invalid": "DROP/STRUCTURAL review (not RVQ)",
        "medtok_semantic_core": "MEDTOK path (diag/proc/med canonicalization/vocab)",
        "medication_attributes_or_admin": "Medication attr stream or DROP/meta",
        "medication_numeric_or_attributes": "Medication numeric/attr side-channel",
        "admin_or_low_value_meta": "DROP/meta unless downstream task requires",
        "other_unclassified": "Manual review",
    }
    return mapping.get(surface, "Manual review")


def _mark_rare_critical(code_upper: str, keywords_upper: List[str], transition_like: bool) -> bool:
    return transition_like or any(k in code_upper for k in keywords_upper)


def _build_surface_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    grouped = (
        df.groupby("data_surface", as_index=False)
        .agg(
            codes=("code", "count"),
            uncaptured_events=("uncaptured_events", "sum"),
            avg_capture_rate=("capture_rate", "mean"),
            avg_subject_coverage=("subject_coverage_frac", "mean"),
            rare_critical_codes=("rare_critical", "sum"),
        )
        .sort_values("uncaptured_events", ascending=False)
    )
    rows: List[Dict[str, Any]] = []
    total_events = float(df["uncaptured_events"].sum()) if len(df) > 0 else 0.0
    for _, r in grouped.iterrows():
        surface = str(r["data_surface"])
        uncaptured_events = float(r["uncaptured_events"])
        rows.append(
            {
                "data_surface": surface,
                "recommended_path": _surface_to_path(surface),
                "codes": int(r["codes"]),
                "uncaptured_events": int(uncaptured_events),
                "uncaptured_event_share": (uncaptured_events / total_events) if total_events > 0 else 0.0,
                "avg_capture_rate": float(r["avg_capture_rate"]),
                "avg_subject_coverage": float(r["avg_subject_coverage"]),
                "rare_critical_codes": int(r["rare_critical_codes"]),
            }
        )
    return rows


def _select_budgeted_codes(
    df: pd.DataFrame,
    additional_budget: int,
    min_codes_per_surface: int,
    top_k_per_surface_preview: int,
) -> Dict[str, Any]:
    d = df.copy()
    d["priority"] = d["uncaptured_events"] * (1.0 + d["subject_coverage_frac"])
    d = d.sort_values("priority", ascending=False).reset_index(drop=True)

    if additional_budget <= 0:
        return {
            "additional_budget": int(additional_budget),
            "selected_codes": 0,
            "selected_uncaptured_events": 0,
            "selected_uncaptured_event_frac": 0.0,
            "by_surface": {},
            "top_selected_codes": [],
        }

    n_codes = int(len(d))
    eff_budget = int(min(additional_budget, n_codes))
    selected_idx: set[int] = set()

    # 1) Keep all rare critical first (bounded by budget)
    rare_idx = d.index[d["rare_critical"]].tolist()
    for i in rare_idx:
        if len(selected_idx) >= eff_budget:
            break
        selected_idx.add(int(i))

    remaining = eff_budget - len(selected_idx)
    if remaining > 0:
        # 2) Guarantee a small floor per data surface
        surfaces = list(d["data_surface"].value_counts().index)
        for surface in surfaces:
            if remaining <= 0:
                break
            surface_idx = [int(i) for i in d.index[d["data_surface"] == surface].tolist() if int(i) not in selected_idx]
            grant = min(int(min_codes_per_surface), len(surface_idx), remaining)
            for i in surface_idx[:grant]:
                selected_idx.add(i)
            remaining = eff_budget - len(selected_idx)

    if remaining > 0:
        # 3) Allocate remainder by sqrt(weighted uncaptured events) per surface
        avail = d.loc[[i for i in d.index if int(i) not in selected_idx]].copy()
        if not avail.empty:
            surface_totals = (
                avail.groupby("data_surface", as_index=False)["uncaptured_events"]
                .sum()
                .sort_values("uncaptured_events", ascending=False)
            )
            weights = {
                str(r["data_surface"]): math.sqrt(float(r["uncaptured_events"]))
                for _, r in surface_totals.iterrows()
            }
            w_sum = float(sum(weights.values()))
            quotas: Dict[str, int] = {}
            fracs: List[tuple[float, str]] = []
            if w_sum > 0:
                for surface, w in weights.items():
                    q = (remaining * w) / w_sum
                    base = int(math.floor(q))
                    quotas[surface] = base
                    fracs.append((q - base, surface))
            else:
                for surface in weights:
                    quotas[surface] = 0
            assigned = sum(quotas.values())
            left = remaining - assigned
            fracs.sort(reverse=True)
            for _, surface in fracs:
                if left <= 0:
                    break
                quotas[surface] += 1
                left -= 1

            for surface, q in quotas.items():
                if q <= 0:
                    continue
                surface_idx = (
                    avail[avail["data_surface"] == surface]
                    .sort_values("priority", ascending=False)
                    .index.tolist()
                )
                for i in surface_idx[: int(q)]:
                    if len(selected_idx) >= eff_budget:
                        break
                    selected_idx.add(int(i))
                if len(selected_idx) >= eff_budget:
                    break

    # Final fill by global priority if rounding left holes.
    if len(selected_idx) < eff_budget:
        for i in d.index.tolist():
            ii = int(i)
            if ii not in selected_idx:
                selected_idx.add(ii)
            if len(selected_idx) >= eff_budget:
                break

    selected = d.loc[sorted(selected_idx)].copy()
    selected_uncaptured = float(selected["uncaptured_events"].sum()) if not selected.empty else 0.0
    total_uncaptured = float(d["uncaptured_events"].sum()) if not d.empty else 0.0

    by_surface = {}
    if not selected.empty:
        grouped = (
            selected.groupby("data_surface", as_index=False)
            .agg(
                selected_codes=("code", "count"),
                selected_uncaptured_events=("uncaptured_events", "sum"),
                selected_rare_critical=("rare_critical", "sum"),
            )
            .sort_values("selected_uncaptured_events", ascending=False)
        )
        for _, r in grouped.iterrows():
            surface = str(r["data_surface"])
            tmp = selected[selected["data_surface"] == surface].sort_values("priority", ascending=False)
            by_surface[surface] = {
                "selected_codes": int(r["selected_codes"]),
                "selected_uncaptured_events": int(r["selected_uncaptured_events"]),
                "selected_rare_critical": int(r["selected_rare_critical"]),
                "recommended_path": _surface_to_path(surface),
                "top_codes_preview": tmp.head(int(top_k_per_surface_preview))[
                    [
                        "code",
                        "uncaptured_events",
                        "subject_coverage_frac",
                        "capture_rate",
                        "suggested_family",
                        "suggested_action",
                        "rare_critical",
                    ]
                ].to_dict(orient="records"),
            }

    return {
        "additional_budget": int(additional_budget),
        "effective_budget_after_code_cap": int(eff_budget),
        "selected_codes": int(len(selected)),
        "selected_uncaptured_events": int(selected_uncaptured),
        "selected_uncaptured_event_frac": (
            float(selected_uncaptured / total_uncaptured) if total_uncaptured > 0 else 0.0
        ),
        "by_surface": by_surface,
        "top_selected_codes": selected.sort_values("priority", ascending=False).head(100)[
            [
                "code",
                "data_surface",
                "uncaptured_events",
                "subject_coverage_frac",
                "capture_rate",
                "suggested_family",
                "suggested_action",
                "rare_critical",
            ]
        ].to_dict(orient="records"),
    }


def analyze(
    consensus_df: pd.DataFrame,
    top_k: int,
    base_vocab_size: int,
    target_vocab_min: int,
    target_vocab_max: int,
    min_codes_per_surface: int,
    top_k_per_surface_preview: int,
    critical_keywords: List[str],
) -> Dict[str, Any]:
    if consensus_df.empty:
        raise ValueError("Consensus CSV is empty.")

    df = consensus_df.copy()
    df["uncaptured_events"] = _safe_float_col(df, "uncaptured_events", 0.0)
    df["subject_coverage_frac"] = _safe_float_col(df, "subject_coverage_frac", 0.0)
    df["capture_rate"] = _safe_float_col(df, "capture_rate", 0.0)
    df["numeric_rate"] = _safe_float_col(df, "numeric_rate", 0.0)
    df["in_structural_codebook"] = _safe_float_col(df, "in_structural_codebook", 0.0)
    df["priority_score"] = df["uncaptured_events"] * df["subject_coverage_frac"]

    prefix = _safe_str_col(df, "prefix", default="<NONE>")
    category = _safe_str_col(df, "routed_category", default="<NONE>")
    fam = _safe_str_col(df, "suggested_family", default="<NONE>")
    action = _safe_str_col(df, "suggested_action", default="<NONE>")
    reason_top = _safe_str_col(df, "uncaptured_reason_top", default="<NONE>")
    code = _safe_str_col(df, "code", default="<NONE>")
    reason_counts_col = _safe_str_col(df, "uncaptured_reason_counts", default="")

    unk_mask = code.str.upper().str.contains("UNK", regex=False)
    start_stop_mask = code.str.upper().str.contains("START|STOP|END", regex=True)
    code_upper = code.str.upper()
    transition_like = code_upper.apply(_is_transition_like)
    keywords_upper = [k.upper() for k in critical_keywords if str(k).strip()]
    df["data_surface"] = df.apply(_classify_surface, axis=1)
    df["rare_critical"] = [
        _mark_rare_critical(cu, keywords_upper, bool(t))
        for cu, t in zip(code_upper.tolist(), transition_like.tolist())
    ]

    additional_budget_min = max(0, int(target_vocab_min) - int(base_vocab_size))
    additional_budget_max = max(0, int(target_vocab_max) - int(base_vocab_size))

    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "uncaptured_events_total": int(df["uncaptured_events"].sum()),
        "avg_uncaptured_events_per_code": float(df["uncaptured_events"].mean()),
        "median_uncaptured_events_per_code": float(df["uncaptured_events"].median()),
        "codes_with_unk_label": int(unk_mask.sum()),
        "codes_with_start_stop_end_pattern": int(start_stop_mask.sum()),
        "codes_marked_rare_critical": int(df["rare_critical"].sum()),
        "counts_by_prefix_top30": prefix.value_counts().head(30).to_dict(),
        "counts_by_routed_category": category.value_counts().to_dict(),
        "counts_by_suggested_family": fam.value_counts().to_dict(),
        "counts_by_suggested_action_top30": action.value_counts().head(30).to_dict(),
        "counts_by_uncaptured_reason_top": reason_top.value_counts().to_dict(),
        "counts_by_data_surface": _safe_str_col(df, "data_surface").value_counts().to_dict(),
        "surface_summary": _build_surface_summary(df),
        "aggregated_uncaptured_reason_counts": _parse_reason_counts(reason_counts_col),
        "top_by_uncaptured_events": _top_records(df, "uncaptured_events", top_k),
        "top_by_subject_coverage_frac": _top_records(df, "subject_coverage_frac", top_k),
        "top_by_priority_score": _top_records(df, "priority_score", top_k),
        "vocab_budget_context": {
            "base_vocab_size": int(base_vocab_size),
            "target_vocab_min": int(target_vocab_min),
            "target_vocab_max": int(target_vocab_max),
            "additional_budget_min": int(additional_budget_min),
            "additional_budget_max": int(additional_budget_max),
            "min_codes_per_surface": int(min_codes_per_surface),
            "critical_keywords": keywords_upper,
        },
        "budget_simulation_min_target": _select_budgeted_codes(
            df=df,
            additional_budget=additional_budget_min,
            min_codes_per_surface=int(min_codes_per_surface),
            top_k_per_surface_preview=int(top_k_per_surface_preview),
        ),
        "budget_simulation_max_target": _select_budgeted_codes(
            df=df,
            additional_budget=additional_budget_max,
            min_codes_per_surface=int(min_codes_per_surface),
            top_k_per_surface_preview=int(top_k_per_surface_preview),
        ),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze consensus uncaptured-code decisions and emit compact summary stats."
    )
    ap.add_argument("--consensus_csv", required=True)
    ap.add_argument("--seed42_csv", default=None)
    ap.add_argument("--seed1337_csv", default=None)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--base_vocab_size", type=int, default=0)
    ap.add_argument("--target_vocab_min", type=int, default=60000)
    ap.add_argument("--target_vocab_max", type=int, default=100000)
    ap.add_argument("--min_codes_per_surface", type=int, default=50)
    ap.add_argument("--top_k_per_surface_preview", type=int, default=20)
    ap.add_argument(
        "--critical_keywords",
        default=",".join(_critical_keywords_default()),
        help="Comma-separated keyword list to protect rare but clinically important codes.",
    )
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    consensus_fp = Path(args.consensus_csv)
    if not consensus_fp.exists():
        raise FileNotFoundError(f"Consensus CSV not found: {consensus_fp}")

    consensus_df = pd.read_csv(consensus_fp)
    critical_keywords = [s.strip() for s in str(args.critical_keywords).split(",") if s.strip()]
    summary = analyze(
        consensus_df=consensus_df,
        top_k=int(args.top_k),
        base_vocab_size=int(args.base_vocab_size),
        target_vocab_min=int(args.target_vocab_min),
        target_vocab_max=int(args.target_vocab_max),
        min_codes_per_surface=int(args.min_codes_per_surface),
        top_k_per_surface_preview=int(args.top_k_per_surface_preview),
        critical_keywords=critical_keywords,
    )
    summary["consensus_csv"] = str(consensus_fp)

    if args.seed42_csv and args.seed1337_csv:
        seed42_fp = Path(args.seed42_csv)
        seed1337_fp = Path(args.seed1337_csv)
        if not seed42_fp.exists():
            raise FileNotFoundError(f"Seed42 CSV not found: {seed42_fp}")
        if not seed1337_fp.exists():
            raise FileNotFoundError(f"Seed1337 CSV not found: {seed1337_fp}")
        s42 = pd.read_csv(seed42_fp)
        s13 = pd.read_csv(seed1337_fp)
        if "code" in s42.columns and "code" in s13.columns and "code" in consensus_df.columns:
            summary["seed_overlap"] = _set_overlap(s42["code"], s13["code"])
            summary["consensus_vs_seed42"] = _set_overlap(consensus_df["code"], s42["code"])
            summary["consensus_vs_seed1337"] = _set_overlap(consensus_df["code"], s13["code"])
            summary["consensus_rank_in_seed42"] = _rank_stats(consensus_df["code"], s42)
            summary["consensus_rank_in_seed1337"] = _rank_stats(consensus_df["code"], s13)

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out_fp = Path(args.output_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote: {out_fp}")


if __name__ == "__main__":
    main()
