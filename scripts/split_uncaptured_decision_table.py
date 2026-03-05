#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _normalize_str_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split uncaptured decision table CSV into actionable review buckets."
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_uncaptured_events", type=int, default=100)
    ap.add_argument("--min_subject_coverage", type=float, default=0.005)
    ap.add_argument("--drop_max_uncaptured_events", type=int, default=200)
    ap.add_argument("--write_summary_json", action="store_true")
    args = ap.parse_args()

    in_fp = Path(args.input_csv)
    if not in_fp.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_fp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_fp)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {in_fp}")

    # Ensure expected columns exist even if upstream script changes.
    for col, default in (
        ("uncaptured_events", 0),
        ("subject_coverage_frac", 0.0),
        ("capture_rate", 0.0),
    ):
        if col not in df.columns:
            df[col] = default

    suggested_family = _normalize_str_col(df, "suggested_family")
    suggested_action = _normalize_str_col(df, "suggested_action")
    decision_status = _normalize_str_col(df, "decision_status", default="PENDING")

    unresolved = decision_status.str.upper().isin({"", "PENDING", "TODO"})
    high_impact = (
        (df["uncaptured_events"] >= int(args.min_uncaptured_events))
        & (df["subject_coverage_frac"] >= float(args.min_subject_coverage))
    )

    immediate_families = {"MEASUREMENT", "MEDTOK", "STRUCTURAL", "MEASUREMENT_OR_MED_NUMERIC"}
    likely_drop_families = {"DROP_OR_META", "STRUCTURAL_OR_DROP"}

    immediate_mask = unresolved & high_impact & suggested_family.isin(immediate_families)
    likely_drop_mask = (
        unresolved
        & suggested_family.isin(likely_drop_families)
        & (df["uncaptured_events"] <= int(args.drop_max_uncaptured_events))
    )
    manual_mask = unresolved & ~(immediate_mask | likely_drop_mask)

    # Sort for review convenience.
    sort_cols = [c for c in ("uncaptured_events", "subject_coverage_frac", "events_total") if c in df.columns]
    ascending = [False] * len(sort_cols)

    immediate_df = df.loc[immediate_mask].sort_values(sort_cols, ascending=ascending)
    manual_df = df.loc[manual_mask].sort_values(sort_cols, ascending=ascending)
    likely_drop_df = df.loc[likely_drop_mask].sort_values(sort_cols, ascending=ascending)

    immediate_fp = out_dir / "decision_immediate_action.csv"
    manual_fp = out_dir / "decision_manual_review.csv"
    drop_fp = out_dir / "decision_likely_drop.csv"

    immediate_df.to_csv(immediate_fp, index=False)
    manual_df.to_csv(manual_fp, index=False)
    likely_drop_df.to_csv(drop_fp, index=False)

    summary = {
        "input_csv": str(in_fp),
        "rows_total": int(len(df)),
        "rows_unresolved": int(unresolved.sum()),
        "rows_immediate_action": int(len(immediate_df)),
        "rows_manual_review": int(len(manual_df)),
        "rows_likely_drop": int(len(likely_drop_df)),
        "min_uncaptured_events": int(args.min_uncaptured_events),
        "min_subject_coverage": float(args.min_subject_coverage),
        "drop_max_uncaptured_events": int(args.drop_max_uncaptured_events),
        "top_immediate_actions": (
            immediate_df[["code", "uncaptured_events", "subject_coverage_frac", "suggested_family", "suggested_action"]]
            .head(25)
            .to_dict(orient="records")
            if not immediate_df.empty
            else []
        ),
        "counts_by_suggested_family_unresolved": (
            suggested_family[unresolved].value_counts().to_dict()
        ),
        "counts_by_suggested_action_unresolved": (
            suggested_action[unresolved].value_counts().head(30).to_dict()
        ),
    }

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {immediate_fp}")
    print(f"Wrote: {manual_fp}")
    print(f"Wrote: {drop_fp}")

    if args.write_summary_json:
        summary_fp = out_dir / "decision_split_summary.json"
        summary_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote: {summary_fp}")


if __name__ == "__main__":
    main()

