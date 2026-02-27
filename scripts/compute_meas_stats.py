#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd
import torch


def load_split_subjects(splits_parquet: str, split: str) -> set[int]:
    df = pd.read_parquet(splits_parquet)[["subject_id", "split"]]
    keep = set(df.loc[df["split"] == split, "subject_id"].astype("int64").tolist())
    if not keep:
        raise ValueError(f"No subjects for split='{split}' in {splits_parquet}")
    return keep


def build_meas_code2id(
    meds_reader_db: str,
    allowed_subjects: set[int],
    *,
    min_count: int = 1,
    sort_by_freq: bool = True,
) -> Dict[str, int]:
    import meds_reader as mr

    db = mr.SubjectDatabase(meds_reader_db)
    freq = Counter()

    for sid in db:
        sid_i = int(sid)
        if sid_i not in allowed_subjects:
            continue
        subj = db[sid_i]
        for ev in subj.events:
            v = getattr(ev, "numeric_value", None)
            if v is None:
                continue
            c = getattr(ev, "code", None)
            if c is None:
                continue
            freq[str(c)] += 1

    items = [(code, cnt) for code, cnt in freq.items() if cnt >= min_count]
    if sort_by_freq:
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    # Reserve 0 for unknown/pad.
    return {code: i + 1 for i, (code, _) in enumerate(items)}


def compute_var_stats(
    meds_reader_db: str,
    allowed_subjects: set[int],
    code2id: Dict[str, int],
    *,
    min_std: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import meds_reader as mr

    db = mr.SubjectDatabase(meds_reader_db)

    sum_by_var = defaultdict(float)
    sumsq_by_var = defaultdict(float)
    count_by_var = defaultdict(int)

    for sid in db:
        sid_i = int(sid)
        if sid_i not in allowed_subjects:
            continue
        subj = db[sid_i]
        for ev in subj.events:
            c = getattr(ev, "code", None)
            if c is None:
                continue
            vid = code2id.get(str(c))
            if vid is None:
                continue

            v = getattr(ev, "numeric_value", None)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue

            sum_by_var[vid] += fv
            sumsq_by_var[vid] += fv * fv
            count_by_var[vid] += 1

    max_vid = max(code2id.values(), default=0)
    mean = torch.zeros(max_vid + 1, dtype=torch.float32)
    std = torch.ones(max_vid + 1, dtype=torch.float32)
    count = torch.zeros(max_vid + 1, dtype=torch.long)

    for vid in range(1, max_vid + 1):
        c = int(count_by_var.get(vid, 0))
        count[vid] = c
        if c == 0:
            mean[vid] = 0.0
            std[vid] = 1.0
            continue
        m = sum_by_var[vid] / c
        var = max(0.0, (sumsq_by_var[vid] / c) - (m * m))
        mean[vid] = float(m)
        std[vid] = float(max(min_std, math.sqrt(var)))

    return mean, std, count


def coverage_check(
    meds_reader_db: str,
    allowed_subjects: set[int],
    code2id: Dict[str, int],
    *,
    max_subjects: int | None = None,
) -> Counter:
    import meds_reader as mr

    db = mr.SubjectDatabase(meds_reader_db)
    cnt = Counter()
    seen_subjects = 0

    for sid in db:
        sid_i = int(sid)
        if sid_i not in allowed_subjects:
            continue
        seen_subjects += 1
        if max_subjects is not None and seen_subjects > max_subjects:
            break

        subj = db[sid_i]
        for ev in subj.events:
            v = getattr(ev, "numeric_value", None)
            if v is None:
                continue
            cnt["total_numeric"] += 1

            c = getattr(ev, "code", None)
            if c is None:
                continue
            if str(c) in code2id:
                cnt["mapped"] += 1
            else:
                cnt["unmapped"] += 1

    return cnt


def main() -> None:
    ap = argparse.ArgumentParser(description="Build frozen measurement code2id mapping + per-var stats.")
    ap.add_argument("--meds_reader_db", required=True, help="Path to meds_reader SubjectDatabase")
    ap.add_argument("--splits_parquet", required=True, help="Path to metadata/subject_splits.parquet")
    ap.add_argument("--split", default="train", help="Split label to use (default: train)")
    ap.add_argument("--code2id_pt", required=True, help="Output torch file for code2id mapping")
    ap.add_argument("--stats_pt", required=True, help="Output torch file for mean/std stats")
    ap.add_argument("--min_count", type=int, default=1, help="Minimum event count per code to keep")
    ap.add_argument("--sort_by_freq", action="store_true", default=True, help="Sort mapping by frequency")
    ap.add_argument("--no_sort_by_freq", dest="sort_by_freq", action="store_false", help="Sort mapping lexicographically")
    ap.add_argument("--min_std", type=float, default=1e-3, help="Minimum std floor per variable")
    ap.add_argument("--coverage_subject_cap", type=int, default=100, help="Max subjects for lightweight coverage report")
    args = ap.parse_args()

    allowed = load_split_subjects(args.splits_parquet, args.split)
    code2id = build_meas_code2id(
        meds_reader_db=args.meds_reader_db,
        allowed_subjects=allowed,
        min_count=args.min_count,
        sort_by_freq=args.sort_by_freq,
    )
    mean_by_var, std_by_var, count_by_var = compute_var_stats(
        meds_reader_db=args.meds_reader_db,
        allowed_subjects=allowed,
        code2id=code2id,
        min_std=args.min_std,
    )

    Path(args.code2id_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_pt).parent.mkdir(parents=True, exist_ok=True)

    torch.save(code2id, args.code2id_pt)
    torch.save(
        {
            "mean_by_var": mean_by_var,
            "std_by_var": std_by_var,
            "count_by_var": count_by_var,
            "split": args.split,
            "n_codes": len(code2id),
        },
        args.stats_pt,
    )

    cov = coverage_check(
        meds_reader_db=args.meds_reader_db,
        allowed_subjects=allowed,
        code2id=code2id,
        max_subjects=args.coverage_subject_cap,
    )
    mapped = int(cov.get("mapped", 0))
    total = int(cov.get("total_numeric", 0))
    frac = (mapped / total) if total > 0 else 0.0

    print(f"split={args.split} subjects={len(allowed)}")
    print(f"codes kept={len(code2id)}")
    print(f"saved code2id -> {args.code2id_pt}")
    print(f"saved stats   -> {args.stats_pt}")
    print(f"coverage(sample): mapped={mapped} total_numeric={total} frac={frac:.3f}")


if __name__ == "__main__":
    main()
