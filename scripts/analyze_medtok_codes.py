"""
Inspect the MedTok parquet mapping to understand which code systems and ranges
are available. This focuses on light-weight metadata (code strings + systems)
so it can run without loading the heavy embedding columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow.parquet as pq

# Only ICD code systems encode explicit range nodes such as "A00-A09".
RANGE_CODE_SYSTEMS = {"icd9", "icd10"}


def load_codes(parquet_path: Path) -> pd.DataFrame:
    """Load only the columns we need from the MedTok parquet file."""
    table = pq.read_table(parquet_path, columns=["med_code", "code_system"])
    return table.to_pandas()


def split_range(code: str) -> Tuple[str, str] | None:
    """Return the start/end of a range code like 'A00-A09.9'; otherwise None."""
    if "-" not in code:
        return None
    start, end = code.split("-", 1)
    if not start or not end:
        return None
    return start, end


def summarize_system(
    code_system: str,
    codes: pd.Series,
    top_examples: int,
    top_ranges: int,
) -> Dict[str, object]:
    med_codes = codes.dropna()
    is_range_system = code_system in RANGE_CODE_SYSTEMS

    range_codes: List[str] = []
    range_pairs: List[Tuple[str, str]] = []
    if is_range_system:
        range_codes = sorted(med_codes[med_codes.str.contains("-", regex=False)].unique())
        for code in range_codes:
            parsed = split_range(code)
            if parsed:
                range_pairs.append(parsed)

    if is_range_system and range_codes:
        range_set = set(range_codes)
        leaf_codes = med_codes[~med_codes.isin(range_set)]
    else:
        leaf_codes = med_codes

    sample_leaf = sorted(leaf_codes.unique())[:top_examples]
    prefix_counts = med_codes.str[0].value_counts().head(10).to_dict()

    return {
        "code_system": code_system,
        "rows": int(len(med_codes)),
        "unique_codes": int(med_codes.nunique()),
        "min_code": med_codes.min(),
        "max_code": med_codes.max(),
        "sample_leaf_codes": sample_leaf,
        "prefix_counts": prefix_counts,
        "num_range_entries": len(range_codes),
        "range_entries": range_codes[:top_ranges],
        "range_pairs": range_pairs[:top_ranges],
    }


def summarize_all(
    df: pd.DataFrame,
    top_examples: int,
    top_ranges: int,
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for code_system, subset in df.groupby("code_system"):
        summaries.append(
            summarize_system(
                code_system=code_system,
                codes=subset["med_code"],
                top_examples=top_examples,
                top_ranges=top_ranges,
            )
        )
    return sorted(summaries, key=lambda item: item["code_system"])


def print_summary(parquet_path: Path, df: pd.DataFrame, summaries: List[Dict[str, object]]) -> None:
    print(f"Loaded {len(df):,} rows from {parquet_path}")
    systems = ", ".join(summary["code_system"] for summary in summaries)
    print(f"Code systems: {systems}\n")

    for summary in summaries:
        print(f"[{summary['code_system']}] rows={summary['rows']:,} unique_codes={summary['unique_codes']:,}")
        print(f"  min_code={summary['min_code']} | max_code={summary['max_code']}")
        if summary["prefix_counts"]:
            prefix_str = ", ".join(f"{k}:{v}" for k, v in summary["prefix_counts"].items())
            print(f"  leading char counts (top 10): {prefix_str}")
        if summary["sample_leaf_codes"]:
            samples = ", ".join(summary["sample_leaf_codes"])
            print(f"  sample atomic codes (up to requested limit): {samples}")
        if summary["num_range_entries"]:
            pairs = " | ".join(f"{start} -> {end}" for start, end in summary["range_pairs"])
            ranges = ", ".join(summary["range_entries"])
            print(f"  range entries: {summary['num_range_entries']} total")
            print(f"    first ranges (as strings): {ranges}")
            print(f"    first ranges (start -> end): {pairs}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MedTok-supported code systems and any explicit range nodes."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("artifacts/medtok/all_codes_mappings.parquet"),
        help="Path to the MedTok all_codes_mappings.parquet file.",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=10,
        help="Number of atomic code examples to display per code system.",
    )
    parser.add_argument(
        "--top-ranges",
        type=int,
        default=20,
        help="Number of range entries to display for ICD code systems.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump the summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(f"Could not find parquet at {args.parquet}")

    df = load_codes(args.parquet)
    summaries = summarize_all(df, top_examples=args.top_examples, top_ranges=args.top_ranges)
    print_summary(args.parquet, df, summaries)

    if args.json_out:
        payload = {
            "parquet_path": str(args.parquet),
            "num_rows": len(df),
            "summaries": summaries,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
