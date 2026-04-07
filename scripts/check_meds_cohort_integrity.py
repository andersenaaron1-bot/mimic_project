#!/usr/bin/env python
"""
Strict integrity and metadata checks for a MEDS_cohort directory.

Examples:
  python scripts/check_meds_cohort_integrity.py --cohort C:\\path\\to\\MEDS_cohort

  # MedTok-oriented strict mode (requires parent_codes in codes.parquet)
  python scripts/check_meds_cohort_integrity.py \
    --cohort C:\\path\\to\\MEDS_cohort \
    --require-parent-codes \
    --require-codes-columns code description parent_codes

  # Deep (slower): verify all event codes are present in metadata/codes.parquet
  python scripts/check_meds_cohort_integrity.py \
    --cohort C:\\path\\to\\MEDS_cohort \
    --deep-code-check
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _require_pyarrow():
    try:
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required for this script. Install with: pip install pyarrow"
        ) from exc
    return pq, pc


@dataclass
class CheckReport:
    cohort: str

    # Structure
    metadata_entries: List[str] = field(default_factory=list)
    required_metadata_missing: List[str] = field(default_factory=list)
    parquet_files: int = 0
    metadata_files: int = 0
    lock_files: int = 0

    # Data shard consistency
    expected_shards: int = 0
    present_shards: int = 0
    missing_shards: int = 0
    extra_shards: int = 0
    total_rows: int = 0
    zero_row_files: int = 0
    missing_required_columns: Dict[str, List[str]] = field(default_factory=dict)
    unreadable_parquet_refs: List[str] = field(default_factory=list)

    # dataset.json
    dataset_json_present: bool = False
    dataset_json_fields_missing: List[str] = field(default_factory=list)
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    etl_name: Optional[str] = None
    etl_version: Optional[str] = None
    meds_version: Optional[str] = None
    created_at: Optional[str] = None

    # subject_splits
    subject_splits_present: bool = False
    subject_splits_columns: List[str] = field(default_factory=list)
    subject_splits_rows: Optional[int] = None
    subject_splits_unique_subjects: Optional[int] = None
    subject_splits_split_counts: Dict[str, int] = field(default_factory=dict)
    subject_splits_invalid_splits: List[str] = field(default_factory=list)

    # .shards.json
    shard_manifest_split_counts: Dict[str, int] = field(default_factory=dict)
    shard_manifest_shard_counts: Dict[str, int] = field(default_factory=dict)
    shard_manifest_subjects_total: Optional[int] = None
    shard_manifest_duplicate_subject_ids: int = 0
    shard_manifest_invalid_subject_ids: int = 0

    # subject alignment
    subject_ids_missing_in_shards: int = 0
    subject_ids_missing_in_subject_splits: int = 0
    split_count_mismatches: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # codes.parquet
    codes_parquet_present: bool = False
    codes_columns: List[str] = field(default_factory=list)
    codes_rows: Optional[int] = None
    codes_unique: Optional[int] = None
    codes_duplicate_count: Optional[int] = None
    parent_codes_column_present: Optional[bool] = None
    parent_codes_null_fraction: Optional[float] = None
    parent_codes_empty_fraction: Optional[float] = None

    # Coverage of metadata sources declared in event config
    event_conversion_config_present: bool = False
    expected_metadata_sources: List[str] = field(default_factory=list)
    extracted_metadata_sources: List[str] = field(default_factory=list)
    missing_expected_metadata_sources: List[str] = field(default_factory=list)

    # Deep code coverage check (optional)
    deep_code_check_enabled: bool = False
    data_unique_codes: Optional[int] = None
    metadata_unique_codes: Optional[int] = None
    data_codes_missing_in_metadata: Optional[int] = None
    metadata_codes_unused_in_data: Optional[int] = None
    data_codes_missing_examples: List[str] = field(default_factory=list)
    data_code_prefix_counts: Dict[str, int] = field(default_factory=dict)
    missing_code_prefix_counts: Dict[str, int] = field(default_factory=dict)
    critical_prefixes_missing_counts: Dict[str, int] = field(default_factory=dict)

    # Outcome
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_shard_manifest(shards_fp: Path) -> Dict[str, Any]:
    raw = _load_json(shards_fp)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {shards_fp}, got {type(raw)}")
    return raw


def _compute_shard_sets(cohort: Path, shard_map: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    expected = {str((cohort / "data" / f"{k}.parquet").resolve()) for k in shard_map}
    got = {str(p.resolve()) for p in (cohort / "data").glob("*/*.parquet")}
    return expected, got


def _safe_count_distinct_arrow(col, pc) -> int:
    try:
        return int(pc.count_distinct(col).as_py())
    except Exception:
        return int(len(set(col.to_pylist())))


def _iter_unique_codes_from_parquet(data_files: Iterable[Path], pq, pc) -> Set[str]:
    uniq: Set[str] = set()
    for fp in data_files:
        pf = pq.ParquetFile(str(fp))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["code"])
            carr = tbl.column("code")
            for chunk in carr.chunks:
                try:
                    vals = pc.unique(chunk).to_pylist()
                except Exception:
                    vals = chunk.to_pylist()
                for v in vals:
                    if v is not None:
                        uniq.add(str(v))
    return uniq


def _code_prefix(code: str) -> str:
    if "//" in code:
        return code.split("//", 1)[0]
    return code


def _parse_metadata_sources_from_event_config(cfg_fp: Path) -> List[str]:
    """
    Parse metadata source keys from event_conversion_config.yaml without extra deps.
    Looks for blocks like:
      _metadata:
        hosp/d_icd_diagnoses:
        d_labitems_to_loinc:
    """
    srcs: Set[str] = set()
    lines = cfg_fp.read_text(encoding="utf-8").splitlines()
    in_block = False
    base_indent = 0

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        if stripped == "_metadata:":
            in_block = True
            base_indent = indent
            continue

        if in_block:
            if indent <= base_indent:
                in_block = False
                # continue processing this line outside block
            else:
                # Keep only metadata source keys (not list items/templates).
                if stripped.endswith(":") and not stripped.startswith("- "):
                    key = stripped[:-1].strip()
                    if key and key not in {"description", "parent_codes", "itemid", "valueuom", "possibly_cpt_code"}:
                        srcs.add(key)
                continue

    return sorted(srcs)


def run_checks(
    cohort: Path,
    required_metadata_files: Set[str],
    required_dataset_fields: Set[str],
    allowed_splits: Set[str],
    required_data_columns: Set[str],
    required_codes_columns: Set[str],
    require_parent_codes: bool,
    max_parent_codes_null_fraction: float,
    deep_code_check: bool,
    critical_prefixes: Set[str],
    max_bad_files_printed: int = 20,
) -> CheckReport:
    pq, pc = _require_pyarrow()
    report = CheckReport(cohort=str(cohort.resolve()))

    data_dir = cohort / "data"
    metadata_dir = cohort / "metadata"
    shards_fp = metadata_dir / ".shards.json"
    dataset_fp = metadata_dir / "dataset.json"
    splits_fp = metadata_dir / "subject_splits.parquet"
    codes_fp = metadata_dir / "codes.parquet"

    if not cohort.exists():
        report.errors.append(f"Cohort path does not exist: {cohort}")
        return report

    if not data_dir.is_dir():
        report.errors.append(f"Missing required directory: {data_dir}")
    if not metadata_dir.is_dir():
        report.errors.append(f"Missing required directory: {metadata_dir}")
    if report.errors:
        return report

    report.metadata_entries = sorted(p.name for p in metadata_dir.iterdir())
    report.metadata_files = len(report.metadata_entries)
    report.parquet_files = len(list(data_dir.glob("*/*.parquet")))
    report.lock_files = len(list(cohort.rglob("*.lock")))

    missing_meta = sorted(required_metadata_files - set(report.metadata_entries))
    report.required_metadata_missing = missing_meta
    if missing_meta:
        report.errors.append(f"Missing required metadata files: {missing_meta}")

    # dataset.json checks
    if dataset_fp.is_file():
        report.dataset_json_present = True
        try:
            ds = _load_json(dataset_fp)
            if not isinstance(ds, dict):
                report.errors.append("metadata/dataset.json is not a JSON object")
            else:
                report.dataset_name = str(ds.get("dataset_name")) if ds.get("dataset_name") is not None else None
                report.dataset_version = (
                    str(ds.get("dataset_version")) if ds.get("dataset_version") is not None else None
                )
                report.etl_name = str(ds.get("etl_name")) if ds.get("etl_name") is not None else None
                report.etl_version = str(ds.get("etl_version")) if ds.get("etl_version") is not None else None
                report.meds_version = str(ds.get("meds_version")) if ds.get("meds_version") is not None else None
                report.created_at = str(ds.get("created_at")) if ds.get("created_at") is not None else None

                miss = sorted(required_dataset_fields - set(ds.keys()))
                report.dataset_json_fields_missing = miss
                if miss:
                    report.errors.append(f"dataset.json missing required fields: {miss}")
        except Exception as exc:
            report.errors.append(f"Failed reading metadata/dataset.json: {exc}")
    else:
        report.errors.append("Missing metadata/dataset.json")

    # shard manifest checks
    shard_subject_ids: Set[int] = set()
    shard_subject_counts = Counter()
    shard_counts = Counter()
    try:
        shard_map = _load_shard_manifest(shards_fp)
        expected, got = _compute_shard_sets(cohort, shard_map)
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        report.expected_shards = len(expected)
        report.present_shards = len(expected & got)
        report.missing_shards = len(missing)
        report.extra_shards = len(extra)
        if missing:
            report.errors.append(f"Missing shard parquet files: {len(missing)}")
            report.unreadable_parquet_refs.extend(missing[:max_bad_files_printed])
        if extra:
            report.warnings.append(f"Unexpected extra shard parquet files: {len(extra)}")
            report.unreadable_parquet_refs.extend(extra[:max_bad_files_printed])

        for shard_key, subjects in shard_map.items():
            split = str(shard_key).split("/", 1)[0]
            shard_counts[split] += 1
            if split not in allowed_splits:
                report.subject_splits_invalid_splits.append(split)
            if not isinstance(subjects, list):
                report.errors.append(f"Shard map value for '{shard_key}' is not a list.")
                continue
            shard_subject_counts[split] += len(subjects)
            for sid in subjects:
                try:
                    sid_i = int(sid)
                except Exception:
                    report.shard_manifest_invalid_subject_ids += 1
                    continue
                if sid_i in shard_subject_ids:
                    report.shard_manifest_duplicate_subject_ids += 1
                shard_subject_ids.add(sid_i)

        report.shard_manifest_split_counts = dict(sorted(shard_subject_counts.items()))
        report.shard_manifest_shard_counts = dict(sorted(shard_counts.items()))
        report.shard_manifest_subjects_total = len(shard_subject_ids)
        if report.shard_manifest_duplicate_subject_ids > 0:
            report.errors.append(
                f"Duplicate subject IDs in shard manifest: {report.shard_manifest_duplicate_subject_ids}"
            )
        if report.shard_manifest_invalid_subject_ids > 0:
            report.errors.append(
                f"Invalid non-integer subject IDs in shard manifest: "
                f"{report.shard_manifest_invalid_subject_ids}"
            )
    except Exception as exc:
        report.errors.append(f"Failed reading metadata/.shards.json: {exc}")
        return report

    # Per-file parquet checks
    for fp in data_dir.glob("*/*.parquet"):
        try:
            pf = pq.ParquetFile(str(fp))
            num_rows = int(pf.metadata.num_rows)
            report.total_rows += num_rows
            if num_rows == 0:
                report.zero_row_files += 1
            cols = set(pf.schema.names)
            miss = sorted(required_data_columns - cols)
            if miss:
                report.missing_required_columns[str(fp.resolve())] = miss
        except Exception as exc:
            report.unreadable_parquet_refs.append(f"{fp.resolve()} :: {exc}")

    if report.unreadable_parquet_refs:
        report.errors.append(
            f"Unreadable/invalid parquet references: {len(report.unreadable_parquet_refs)}"
        )
    if report.missing_required_columns:
        report.errors.append(
            f"Files missing required data columns {sorted(required_data_columns)}: "
            f"{len(report.missing_required_columns)}"
        )

    # subject_splits checks
    split_subject_ids: Set[int] = set()
    split_counts: Counter[str] = Counter()
    if splits_fp.is_file():
        report.subject_splits_present = True
        try:
            split_pf = pq.ParquetFile(str(splits_fp))
            split_cols = list(split_pf.schema.names)
            report.subject_splits_columns = split_cols
            need = {"subject_id", "split"}
            miss = sorted(need - set(split_cols))
            if miss:
                report.errors.append(f"subject_splits.parquet missing columns: {miss}")
            else:
                tbl = pq.read_table(str(splits_fp), columns=["subject_id", "split"])
                report.subject_splits_rows = int(tbl.num_rows)
                sid_col = tbl.column("subject_id")
                split_col = tbl.column("split")

                sid_vals = sid_col.to_pylist()
                split_vals = [str(x) if x is not None else "None" for x in split_col.to_pylist()]
                for sid, split in zip(sid_vals, split_vals):
                    try:
                        sid_i = int(sid)
                    except Exception:
                        report.errors.append(f"Invalid subject_id in subject_splits.parquet: {sid}")
                        continue
                    split_subject_ids.add(sid_i)
                    split_counts[split] += 1

                report.subject_splits_unique_subjects = len(split_subject_ids)
                report.subject_splits_split_counts = dict(sorted(split_counts.items()))

                invalid = sorted(set(split_counts.keys()) - allowed_splits)
                report.subject_splits_invalid_splits.extend(invalid)
                if invalid:
                    report.errors.append(f"subject_splits contains unexpected split labels: {invalid}")

                if report.subject_splits_rows is not None and report.subject_splits_unique_subjects is not None:
                    if report.subject_splits_rows != report.subject_splits_unique_subjects:
                        report.errors.append(
                            "subject_splits.parquet has duplicate subject rows "
                            f"(rows={report.subject_splits_rows}, unique={report.subject_splits_unique_subjects})"
                        )
        except Exception as exc:
            report.errors.append(f"Failed reading subject_splits.parquet: {exc}")
    else:
        report.errors.append("Missing metadata/subject_splits.parquet")

    # Cross-check shard manifest vs subject_splits subjects
    if split_subject_ids:
        missing_in_shards = split_subject_ids - shard_subject_ids
        missing_in_splits = shard_subject_ids - split_subject_ids
        report.subject_ids_missing_in_shards = len(missing_in_shards)
        report.subject_ids_missing_in_subject_splits = len(missing_in_splits)
        if missing_in_shards:
            report.errors.append(
                f"Subjects in subject_splits but not in .shards.json: {len(missing_in_shards)}"
            )
        if missing_in_splits:
            report.errors.append(
                f"Subjects in .shards.json but not in subject_splits: {len(missing_in_splits)}"
            )

        mismatches: Dict[str, Dict[str, int]] = {}
        all_split_labels = sorted(set(split_counts.keys()) | set(shard_subject_counts.keys()))
        for s in all_split_labels:
            a = int(split_counts.get(s, 0))
            b = int(shard_subject_counts.get(s, 0))
            if a != b:
                mismatches[s] = {"subject_splits": a, "shards_json": b}
        report.split_count_mismatches = mismatches
        if mismatches:
            report.errors.append(f"Per-split subject count mismatches: {mismatches}")

    # codes.parquet checks
    codes_set: Optional[Set[str]] = None
    if codes_fp.is_file():
        report.codes_parquet_present = True
        try:
            # Use Arrow schema (top-level field names), not parquet leaf names.
            schema = pq.read_schema(str(codes_fp))
            cols = list(schema.names)
            report.codes_columns = cols

            miss_codes_cols = sorted(required_codes_columns - set(cols))
            if miss_codes_cols:
                report.errors.append(f"codes.parquet missing required columns: {miss_codes_cols}")

            if "code" in cols:
                ctbl = pq.read_table(str(codes_fp), columns=["code"])
                report.codes_rows = int(ctbl.num_rows)
                code_col = ctbl.column("code")
                report.codes_unique = _safe_count_distinct_arrow(code_col, pc)
                if report.codes_rows is not None and report.codes_unique is not None:
                    dups = report.codes_rows - report.codes_unique
                    report.codes_duplicate_count = dups
                    if dups > 0:
                        report.errors.append(f"codes.parquet has duplicate code rows: {dups}")
                codes_set = {str(x) for x in code_col.to_pylist() if x is not None}
            else:
                report.errors.append("codes.parquet missing 'code' column.")

            parent_codes_probe_ok = False
            if "parent_codes" in cols:
                parent_codes_probe_ok = True
            else:
                # Fallback probe for edge parquet schema encodings.
                try:
                    pq.read_table(str(codes_fp), columns=["parent_codes"])
                    parent_codes_probe_ok = True
                    # Keep report of what is actually readable in this environment.
                    if "parent_codes" not in cols:
                        report.codes_columns.append("parent_codes(readable)")
                except Exception:
                    parent_codes_probe_ok = False

            if parent_codes_probe_ok:
                report.parent_codes_column_present = True
                ptbl = pq.read_table(str(codes_fp), columns=["parent_codes"])
                parr = ptbl.column("parent_codes")
                n = int(ptbl.num_rows)
                nulls = int(parr.null_count)
                report.parent_codes_null_fraction = (nulls / n) if n > 0 else 0.0

                # Empty-parent check (covers list and string variants).
                vals = parr.to_pylist()
                empty = 0
                for v in vals:
                    if v is None:
                        continue
                    if isinstance(v, str) and v.strip() == "":
                        empty += 1
                    elif isinstance(v, list) and len(v) == 0:
                        empty += 1
                denom = max(1, n - nulls)
                report.parent_codes_empty_fraction = empty / denom

                if report.parent_codes_null_fraction > max_parent_codes_null_fraction:
                    report.errors.append(
                        "parent_codes null fraction too high: "
                        f"{report.parent_codes_null_fraction:.6f} > {max_parent_codes_null_fraction:.6f}"
                    )
            else:
                report.parent_codes_column_present = False
                if require_parent_codes:
                    report.errors.append("Required parent_codes column not found in codes.parquet.")
        except Exception as exc:
            report.errors.append(f"Failed reading metadata/codes.parquet: {exc}")
    else:
        report.errors.append("Missing metadata/codes.parquet")

    # Expected metadata-source coverage from event conversion config
    event_cfg_fp = cohort / "extract_code_metadata" / "event_conversion_config.yaml"
    extract_dir = cohort / "extract_code_metadata"
    if event_cfg_fp.is_file():
        report.event_conversion_config_present = True
        try:
            exp_src = _parse_metadata_sources_from_event_config(event_cfg_fp)
            report.expected_metadata_sources = exp_src
            got_src = []
            for src in exp_src:
                p = extract_dir / f"{src}.parquet"
                if p.is_file():
                    got_src.append(src)
            report.extracted_metadata_sources = sorted(got_src)
            missing_src = sorted(set(exp_src) - set(got_src))
            report.missing_expected_metadata_sources = missing_src
            if missing_src:
                report.errors.append(
                    "Missing extract_code_metadata outputs for expected sources: "
                    f"{missing_src}"
                )
        except Exception as exc:
            report.errors.append(f"Failed parsing event_conversion_config.yaml: {exc}")
    else:
        report.warnings.append("extract_code_metadata/event_conversion_config.yaml not found.")

    # Deep code coverage check (optional, slow)
    if deep_code_check:
        report.deep_code_check_enabled = True
        if codes_set is None:
            report.errors.append("Deep code check requested, but metadata codes could not be loaded.")
        else:
            data_files = sorted(data_dir.glob("*/*.parquet"))
            try:
                data_codes = _iter_unique_codes_from_parquet(data_files, pq, pc)
                report.data_unique_codes = len(data_codes)
                report.metadata_unique_codes = len(codes_set)
                missing = sorted(data_codes - codes_set)
                unused = sorted(codes_set - data_codes)
                report.data_codes_missing_in_metadata = len(missing)
                report.metadata_codes_unused_in_data = len(unused)
                report.data_codes_missing_examples = missing[:20]

                all_pref = Counter(_code_prefix(c) for c in data_codes)
                miss_pref = Counter(_code_prefix(c) for c in missing)
                report.data_code_prefix_counts = dict(sorted(all_pref.items()))
                report.missing_code_prefix_counts = dict(sorted(miss_pref.items()))
                report.critical_prefixes_missing_counts = {
                    p: int(miss_pref.get(p, 0)) for p in sorted(critical_prefixes)
                }

                if missing:
                    report.errors.append(
                        f"Data contains codes not present in metadata/codes.parquet: {len(missing)}"
                    )
                critical_missing = {k: v for k, v in report.critical_prefixes_missing_counts.items() if v > 0}
                if critical_missing:
                    report.errors.append(
                        "Critical code families missing metadata coverage: "
                        f"{critical_missing}"
                    )
                if unused:
                    report.warnings.append(
                        f"Metadata contains codes unused in data shards: {len(unused)}"
                    )
            except Exception as exc:
                report.errors.append(f"Deep code check failed: {exc}")

    return report


def print_report(report: CheckReport, max_bad_files_printed: int) -> None:
    print(f"cohort: {report.cohort}")
    print(f"parquet_files: {report.parquet_files}")
    print(f"metadata_files: {report.metadata_files}")
    print(f"metadata_entries: {report.metadata_entries}")
    print(f"lock_files: {report.lock_files}")
    print(
        "shards: expected={expected} present={present} missing={missing} extra={extra}".format(
            expected=report.expected_shards,
            present=report.present_shards,
            missing=report.missing_shards,
            extra=report.extra_shards,
        )
    )
    print(f"total_rows: {report.total_rows}")
    print(f"zero_row_files: {report.zero_row_files}")

    print("dataset_json:")
    print(f"  present={report.dataset_json_present}")
    print(f"  dataset_name={report.dataset_name}")
    print(f"  dataset_version={report.dataset_version}")
    print(f"  etl_name={report.etl_name}")
    print(f"  etl_version={report.etl_version}")
    print(f"  meds_version={report.meds_version}")
    print(f"  created_at={report.created_at}")
    if report.dataset_json_fields_missing:
        print(f"  missing_fields={report.dataset_json_fields_missing}")

    print("subject_splits:")
    print(f"  present={report.subject_splits_present}")
    print(f"  columns={report.subject_splits_columns}")
    print(f"  rows={report.subject_splits_rows}")
    print(f"  unique_subjects={report.subject_splits_unique_subjects}")
    print(f"  split_counts={report.subject_splits_split_counts}")

    print("shard_manifest:")
    print(f"  split_subject_counts={report.shard_manifest_split_counts}")
    print(f"  split_shard_counts={report.shard_manifest_shard_counts}")
    print(f"  unique_subjects={report.shard_manifest_subjects_total}")
    print(f"  duplicate_subject_ids={report.shard_manifest_duplicate_subject_ids}")
    print(f"  invalid_subject_ids={report.shard_manifest_invalid_subject_ids}")

    print("subject_alignment:")
    print(f"  missing_in_shards={report.subject_ids_missing_in_shards}")
    print(f"  missing_in_subject_splits={report.subject_ids_missing_in_subject_splits}")
    print(f"  split_count_mismatches={report.split_count_mismatches}")

    print("codes_metadata:")
    print(f"  present={report.codes_parquet_present}")
    print(f"  columns={report.codes_columns}")
    print(f"  rows={report.codes_rows}")
    print(f"  unique_codes={report.codes_unique}")
    print(f"  duplicate_codes={report.codes_duplicate_count}")
    print(f"  parent_codes_column_present={report.parent_codes_column_present}")
    if report.parent_codes_null_fraction is not None:
        print(f"  parent_codes_null_fraction={report.parent_codes_null_fraction:.6f}")
    if report.parent_codes_empty_fraction is not None:
        print(f"  parent_codes_empty_fraction={report.parent_codes_empty_fraction:.6f}")

    print("event_config_metadata_sources:")
    print(f"  event_conversion_config_present={report.event_conversion_config_present}")
    if report.event_conversion_config_present:
        print(f"  expected_sources={report.expected_metadata_sources}")
        print(f"  extracted_sources={report.extracted_metadata_sources}")
        print(f"  missing_expected_sources={report.missing_expected_metadata_sources}")

    if report.deep_code_check_enabled:
        print("deep_code_check:")
        print(f"  data_unique_codes={report.data_unique_codes}")
        print(f"  metadata_unique_codes={report.metadata_unique_codes}")
        print(f"  data_codes_missing_in_metadata={report.data_codes_missing_in_metadata}")
        print(f"  metadata_codes_unused_in_data={report.metadata_codes_unused_in_data}")
        print(f"  critical_prefixes_missing_counts={report.critical_prefixes_missing_counts}")
        if report.missing_code_prefix_counts:
            top_missing = sorted(
                report.missing_code_prefix_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:20]
            print(f"  top_missing_prefixes={top_missing}")
        if report.data_codes_missing_examples:
            print(f"  missing_examples={report.data_codes_missing_examples}")

    if report.missing_required_columns:
        print(f"missing_required_columns_files: {len(report.missing_required_columns)}")
        shown = 0
        for fp, cols in report.missing_required_columns.items():
            print(f"  - {fp}: missing {cols}")
            shown += 1
            if shown >= max_bad_files_printed:
                break

    if report.unreadable_parquet_refs:
        print(f"bad_parquet_refs (showing up to {max_bad_files_printed}):")
        for item in report.unreadable_parquet_refs[:max_bad_files_printed]:
            print(f"  - {item}")

    if report.warnings:
        print("warnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.errors:
        print("status: FAIL")
        print("errors:")
        for e in report.errors:
            print(f"  - {e}")
    else:
        print("status: OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run strict integrity and metadata checks on a MEDS_cohort directory."
    )
    p.add_argument("--cohort", type=Path, required=True, help="Path to MEDS_cohort directory.")
    p.add_argument("--json-out", type=Path, default=None, help="Optional path for JSON report.")
    p.add_argument(
        "--max-bad-files-printed",
        type=int,
        default=20,
        help="Max bad/missing entries to print.",
    )

    p.add_argument(
        "--require-parent-codes",
        action="store_true",
        help="Fail if codes.parquet lacks parent_codes column.",
    )
    p.add_argument(
        "--max-parent-codes-null-fraction",
        type=float,
        default=1.0,
        help="Fail if parent_codes null fraction exceeds this threshold.",
    )
    p.add_argument(
        "--deep-code-check",
        action="store_true",
        help="Slow check: ensure all event codes in data are present in metadata/codes.parquet.",
    )
    p.add_argument(
        "--critical-prefixes",
        nargs="+",
        default=["DIAGNOSIS", "PROCEDURE", "MEDICATION", "LAB", "HCPCS"],
        help=(
            "When --deep-code-check is enabled, fail if these code families "
            "have missing metadata coverage."
        ),
    )
    p.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat warnings as failures (non-zero exit).",
    )
    p.add_argument(
        "--require-codes-columns",
        nargs="+",
        default=["code"],
        help="Required columns in metadata/codes.parquet.",
    )
    p.add_argument(
        "--require-data-columns",
        nargs="+",
        default=["subject_id", "time", "code"],
        help="Required columns in each data parquet.",
    )
    p.add_argument(
        "--require-dataset-fields",
        nargs="+",
        default=["dataset_name", "dataset_version", "etl_name", "etl_version", "meds_version", "created_at"],
        help="Required fields in metadata/dataset.json.",
    )
    p.add_argument(
        "--require-metadata-files",
        nargs="+",
        default=[".shards.json", "dataset.json", "codes.parquet", "subject_splits.parquet"],
        help="Required entries under metadata/.",
    )
    p.add_argument(
        "--allowed-splits",
        nargs="+",
        default=["train", "tuning", "held_out"],
        help="Allowed split labels.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_checks(
        cohort=args.cohort,
        required_metadata_files=set(args.require_metadata_files),
        required_dataset_fields=set(args.require_dataset_fields),
        allowed_splits=set(args.allowed_splits),
        required_data_columns=set(args.require_data_columns),
        required_codes_columns=set(args.require_codes_columns),
        require_parent_codes=bool(args.require_parent_codes),
        max_parent_codes_null_fraction=float(args.max_parent_codes_null_fraction),
        deep_code_check=bool(args.deep_code_check),
        critical_prefixes=set(args.critical_prefixes),
        max_bad_files_printed=int(args.max_bad_files_printed),
    )
    print_report(report, max_bad_files_printed=int(args.max_bad_files_printed))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"wrote_json: {args.json_out}")

    has_fail = bool(report.errors) or (bool(args.fail_on_warning) and bool(report.warnings))
    if has_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
