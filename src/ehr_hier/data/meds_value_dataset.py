from typing import Iterator, Dict, Any, Optional, Callable, List, Set
import math, random, pathlib
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from src.ehr_hier.data.demographics import (
    infer_subject_sex,
    infer_birth_timestamp,
    infer_event_age_years,
)


try:
    import meds_reader as mr  # pip install meds_reader
except Exception:
    mr = None


class ValueEventsDataset(IterableDataset):
    """
    Streams standardized numeric-value events from a MEDS dataset converted to a meds_reader
    SubjectDatabase. Each yielded sample is a dict with keys:
      - value:   standardized numeric value (float)
      - var_id:  integer code id for the measurement variable
      - dt_prev: hours since previous event of the same var within subject (float)
      - age:     subject age in years at the event time (if derivable, else 0.0)
      - sex:     1.0 for male / 0.0 otherwise (if derivable)
    """
    def __init__(
        self,
        meds_reader_db: str,                  # path produced by: meds_reader_convert <MEDS_root> <meds_reader_db>
        split: Optional[str] = "train",       # "train"|"tuning"|"held_out"|None
        splits_parquet: Optional[str] = None, # path to MEDS metadata/subject_splits.parquet (recommended)
        codes_parquet: Optional[str] = None,  # path to MEDS metadata/codes.parquet (recommended)
        strict_codes: bool = True,
        include_code_fn: Optional[Callable[[str], bool]] = None,
        allowed_var_ids: Optional[Set[int]] = None,
        shuffle_subjects: bool = True,
        fixed_code2id: dict[str, int] | None = None,
    ):
        super().__init__()
        if mr is None:
            raise ImportError("meds_reader is not installed. `pip install meds_reader`")

        # open meds_reader database
        self.db = mr.SubjectDatabase(meds_reader_db)
        self.include_code_fn = include_code_fn
        self.allowed_var_ids = allowed_var_ids
        self.shuffle_subjects = shuffle_subjects
        self.strict_codes = strict_codes
        self.codes_parquet = codes_parquet

        # subject split filtering (recommended)
        self.subject_ids = list(self.db)  # iterable of subject_ids
        if split is not None:
            # if a splits file is provided or can be found, filter ids
            if splits_parquet is None:
                # try to find it relative to the db path (metadata next to db)
                p = pathlib.Path(meds_reader_db).parent / "metadata" / "subject_splits.parquet"
                splits_parquet = str(p) if p.exists() else None
            if splits_parquet is not None:
                sp = pd.read_parquet(splits_parquet)[["subject_id", "split"]]
                keep = set(sp.loc[sp["split"] == split, "subject_id"].astype("int64").tolist())
                self.subject_ids = [sid for sid in self.subject_ids if int(sid) in keep]

        # code → id mapping (for stable integer var_ids)
        self.strict_codes = strict_codes
        if fixed_code2id is not None:
            # Frozen mapping (preferred): keep exactly what was used to train the cVAE.
            self.code2id = dict(fixed_code2id)
        elif codes_parquet is not None and pathlib.Path(codes_parquet).exists():
            # Fallback: build mapping from MEDS metadata/codes.parquet.
            df_codes = pd.read_parquet(codes_parquet)
            code_col = "code" if "code" in df_codes.columns else "text"
            id_col = "code_id" if "code_id" in df_codes.columns else None
            if id_col is None:
                # fall back to enumerating if no code_id column exists
                uniq = df_codes[code_col].astype(str).tolist()
                self.code2id = {c: i + 1 for i, c in enumerate(sorted(set(uniq)))}  # reserve 0 if needed
            else:
                self.code2id = {str(c): int(i) for c, i in zip(df_codes[code_col], df_codes[id_col])}
        else:
            self.code2id = {}

    def _get_var_id(self, code_str: str) -> int:
        vid = self.code2id.get(code_str)
        if vid is None:
            if self.strict_codes:
                # unknown code → skip by signaling 0 (caller should ignore)
                return 0
            # else: dynamically grow (demo mode)
            vid = len(self.code2id) + 1
            self.code2id[code_str] = vid
        return vid

    def _infer_sex_and_birth(self, subj) -> (float, Optional[float]):
        """
        Infer subject-level sex (1.0 male / 0.0 otherwise) and birth timestamp
        (seconds since epoch) using MEDS conventions, with fallbacks for
        alternative event schemas.
        """
        events = getattr(subj, "events", [])
        sex_val = infer_subject_sex(events, default=0.0)
        birth_ts = infer_birth_timestamp(events)
        return sex_val, birth_ts

    def _subject_iter(self, subj) -> Iterator[Dict[str, Any]]:
        """
        Iterate over a meds_reader.Subject and yield per-event samples.

        - subj.events is already sorted by time according to MEDS spec.
        - sex is subject-level (from attribute or GENDER code).
        - age is computed per-event from MEDS_BIRTH if possible.
        """
        last_time_by_var: Dict[int, float] = {}

        # Subject-level sex + birth time
        sex_val, birth_ts = self._infer_sex_and_birth(subj)

        for ev in subj.events:
            # time
            t = getattr(ev, "time", None)
            if t is None or not hasattr(t, "timestamp"):
                continue
            t_secs = t.timestamp()
            if not math.isfinite(t_secs):
                continue

            # code
            code_str = getattr(ev, "code", None)
            if code_str is None:
                continue
            if self.include_code_fn is not None and not self.include_code_fn(code_str):
                continue

            # numeric value
            v = getattr(ev, "numeric_value", None)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                continue
            v = float(v)

            var_id = self._get_var_id(str(code_str))
            if var_id <= 0:
                continue  # unknown code when strict
            if self.allowed_var_ids is not None and var_id not in self.allowed_var_ids:
                continue



            # delta-time since previous same var (in hours)
            dt_prev = 0.0
            if var_id in last_time_by_var:
                dt_prev = max(0.0, (t_secs - last_time_by_var[var_id]) / 3600.0)
            last_time_by_var[var_id] = t_secs

            # age in years at this event (if birth time known)
            age_years = infer_event_age_years(ev, birth_ts=birth_ts)

            yield {
                "value": v,
                "var_id": var_id,
                "dt_prev": dt_prev,
                "age": age_years,
                "sex": sex_val,
            }

    def __iter__(self):
        ids = list(self.subject_ids)
        if self.shuffle_subjects:
            random.shuffle(ids)
        for sid in ids:
            subj = self.db[int(sid)]
            yield from self._subject_iter(subj)


def collate_value_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    value = torch.tensor([b["value"] for b in batch], dtype=torch.float32)
    var_id = torch.tensor([b["var_id"] for b in batch], dtype=torch.long)
    dt_prev = torch.tensor([b["dt_prev"] for b in batch], dtype=torch.float32)
    age = torch.tensor([b["age"] for b in batch], dtype=torch.float32)
    sex = torch.tensor([b["sex"] for b in batch], dtype=torch.float32)

    return {"value": value, "var_id": var_id, "dt_prev": dt_prev, "age": age, "sex": sex}


