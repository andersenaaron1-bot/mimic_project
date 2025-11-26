# scripts/build_meas_code2id.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

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
    splits_parquet: str,
    split: str = "train",
    *,
    min_count: int = 1,        # drop ultra-rare codes if you want (>=1 keeps all)
    sort_by_freq: bool = True, # stable, frequent-first id assignment
) -> Dict[str, int]:
    import meds_reader as mr
    from collections import Counter

    allowed = load_split_subjects(splits_parquet, split)
    db = mr.SubjectDatabase(meds_reader_db)

    freq = Counter()
    # count numeric codes only, within the split
    for sid in db:
        sid = int(sid)
        if sid not in allowed:
            continue
        subj = db[sid]
        for ev in subj.events:
            v = getattr(ev, "numeric_value", None)
            if v is None:
                continue
            c = getattr(ev, "code", None)
            if c is None:
                continue
            freq[str(c)] += 1

    # filter & order
    items = [(code, cnt) for code, cnt in freq.items() if cnt >= min_count]
    if sort_by_freq:
        items.sort(key=lambda x: (-x[1], x[0]))  # freq desc, then lexicographic
    else:
        items.sort(key=lambda x: x[0])

    code2id = {code: i + 1 for i, (code, _) in enumerate(items)}  # 1..N (reserve 0)
    return code2id

def save_mapping(code2id: Dict[str, int], out_pt: str):
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(code2id, out_pt)

def coverage_check(
    meds_reader_db: str,
    splits_parquet: str,
    split: str,
    code2id: Dict[str, int],
    max_subjects: Optional[int] = None,
):
    import meds_reader as mr
    from collections import Counter

    allowed = load_split_subjects(splits_parquet, split)
    db = mr.SubjectDatabase(meds_reader_db)

    cnt = Counter()
    n = 0
    for sid in db:
        sid = int(sid)
        if sid not in allowed:
            continue
        n += 1
        if max_subjects and n > max_subjects:
            break
        subj = db[sid]
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
