#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import meds_reader as mr
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import _load_subject_ids
from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.observation_vocab import OBS_RESERVED_VALUE_IDS, observation_surfaces
from src.ehr_hier.data.token_types import TokenCategory


def _is_finite_numeric(value: object) -> bool:
    try:
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    except (TypeError, ValueError):
        return False


def _build_vocab_from_counter(
    *,
    counts: Counter[str],
    target_coverage: float,
    max_explicit: int,
    seed_ids: Dict[str, int] | None = None,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    total = int(sum(int(v) for v in counts.values()))
    code2id: Dict[str, int] = {"<UNK>": 0}
    if seed_ids:
        for key, raw_id in sorted(seed_ids.items(), key=lambda kv: int(kv[1])):
            code2id[str(key)] = int(raw_id)
    covered = int(sum(int(counts.get(code, 0)) for code in code2id if code != "<UNK>"))
    selected = len([k for k in code2id if k != "<UNK>"])

    for code, count in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        if code in code2id:
            continue
        if selected >= int(max_explicit):
            break
        code2id[str(code)] = max(code2id.values(), default=0) + 1
        selected += 1
        covered += int(count)
        if total > 0 and (float(covered) / float(total)) >= float(target_coverage):
            break

    report = {
        "total_events": total,
        "covered_events": int(covered),
        "covered_rate": (float(covered) / float(total)) if total > 0 else 0.0,
        "selected_entries": int(max(0, len(code2id) - 1)),
        "max_explicit": int(max_explicit),
        "target_coverage": float(target_coverage),
        "top_uncovered": [
            {"key": str(code), "count": int(count)}
            for code, count in sorted(
                ((code, count) for code, count in counts.items() if code not in code2id),
                key=lambda kv: (-int(kv[1]), str(kv[0])),
            )[:25]
        ],
    }
    return code2id, report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build exact OBS_CODE / OBS_VALUE vocabularies from train-set fallback observation mass. "
            "These vocabularies replace hash-only qualitative observation encoding for the critical mass."
        )
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--max_subjects", type=int, default=20_000)
    ap.add_argument("--sample_seed", type=int, default=None)
    ap.add_argument("--obs_code_target_coverage", type=float, default=0.99)
    ap.add_argument("--obs_value_target_coverage", type=float, default=0.99)
    ap.add_argument("--obs_code_max_explicit", type=int, default=8_192)
    ap.add_argument("--obs_value_max_explicit", type=int, default=32_768)
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--output_report_json", default=None)
    args = ap.parse_args()

    code2id = torch.load(args.code2id_pt, map_location="cpu")
    db = mr.SubjectDatabase(args.meds_reader_db)
    subject_ids = _load_subject_ids(args.splits_parquet, args.split)
    if args.sample_seed is not None:
        rnd = random.Random(int(args.sample_seed))
        rnd.shuffle(subject_ids)
    if args.max_subjects and len(subject_ids) > int(args.max_subjects):
        subject_ids = subject_ids[: int(args.max_subjects)]

    code_counts: Counter[str] = Counter()
    value_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    observed_events = 0

    for idx, sid in enumerate(subject_ids, start=1):
        subj = db[int(sid)]
        for ev in subj.events:
            code = getattr(ev, "code", None)
            code_str = str(code) if code is not None else None
            if classify_code_to_category(code) != TokenCategory.MEASUREMENT:
                continue
            surfaces = observation_surfaces(ev, code_value=code_str)
            if surfaces is None:
                continue
            if code_str in code2id and _is_finite_numeric(getattr(ev, "numeric_value", None)):
                continue
            code_counts[surfaces.code_surface] += 1
            value_counts[surfaces.value_surface] += 1
            pair_counts[(surfaces.code_surface, surfaces.value_surface)] += 1
            observed_events += 1
        if args.progress_every and idx % int(args.progress_every) == 0:
            print(
                f"[obs-vocab] {idx}/{len(subject_ids)} subjects | "
                f"observed_obs_events={observed_events}"
            )

    seed_value_ids = {
        "UNK": int(OBS_RESERVED_VALUE_IDS["UNK"]),
        "N/A": int(OBS_RESERVED_VALUE_IDS["N/A"]),
        "NONE": int(OBS_RESERVED_VALUE_IDS["NONE"]),
        "": int(OBS_RESERVED_VALUE_IDS[""]),
    }
    obs_code_vocab, obs_code_report = _build_vocab_from_counter(
        counts=code_counts,
        target_coverage=float(args.obs_code_target_coverage),
        max_explicit=int(args.obs_code_max_explicit),
    )
    obs_value_vocab, obs_value_report = _build_vocab_from_counter(
        counts=value_counts,
        target_coverage=float(args.obs_value_target_coverage),
        max_explicit=int(args.obs_value_max_explicit),
        seed_ids=seed_value_ids,
    )

    jointly_covered = int(
        sum(
            int(count)
            for (code_surface, value_surface), count in pair_counts.items()
            if code_surface in obs_code_vocab and value_surface in obs_value_vocab
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "obs_code_vocab.json").write_text(json.dumps(obs_code_vocab, indent=2), encoding="utf-8")
    (out_dir / "obs_value_vocab.json").write_text(json.dumps(obs_value_vocab, indent=2), encoding="utf-8")

    report = {
        "subjects_scanned": int(len(subject_ids)),
        "observed_obs_events": int(observed_events),
        "obs_code": obs_code_report,
        "obs_value": obs_value_report,
        "joint_event_coverage": {
            "covered_events": jointly_covered,
            "covered_rate": (float(jointly_covered) / float(observed_events)) if observed_events > 0 else 0.0,
        },
    }
    report_path = Path(args.output_report_json) if args.output_report_json else (out_dir / "obs_vocab_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote: {out_dir / 'obs_code_vocab.json'}")
    print(f"Wrote: {out_dir / 'obs_value_vocab.json'}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
