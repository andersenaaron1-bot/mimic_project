"""
Smoke-level coverage check for the subject timeline builder against a meds_reader DB.

Run with:
  MEDS_READER_DB=/path/to/meds_reader.db pytest -q tests/test_subject_timeline_db_coverage.py -s
Optionally set MEDTOK_VOCAB_DIR to point at MedTok vocab JSONs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pytest

pytest.importorskip("meds_reader")
import meds_reader as mr  # noqa: E402

from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.medtok_attr_encoder import MedTokenWithAttrsEncoder
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_medication_code,
    canonicalize_procedure_code,
    diagnosis_filter,
    procedure_filter,
    medication_filter,
)
from src.ehr_hier.tokenizers.medtok_loader import (
    CategoryVocab,
    load_medtok_vocab,
    build_vocab_from_code2embeddings,
)
from src.ehr_hier.tokenizers.simple_categorical_encoders import (
    OtherNoOpEncoder,
    SimpleCategoricalEncoder,
)


def _offset(manifest: Dict, key: str, default: int) -> int:
    return int(manifest.get(key, {}).get("offset", default))


def _load_manifest() -> Dict:
    fp = Path("artifacts/vocab_manifest.json")
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


def _build_medtok_vocab(
    name: str,
    *,
    offset: int,
    code2embeds: Optional[Path],
) -> CategoryVocab:
    vocab_dir = Path(os.getenv("MEDTOK_VOCAB_DIR", "artifacts/medtok"))
    if code2embeds is not None and code2embeds.exists():
        filter_map = {
            "diagnosis": diagnosis_filter,
            "procedure": procedure_filter,
            "medication": medication_filter,
        }
        return build_vocab_from_code2embeddings(
            str(code2embeds),
            offset=offset,
            name=name,
            filter_fn=filter_map[name],
        )

    fp_map = {"diagnosis": "diag_vocab.json", "procedure": "proc_vocab.json", "medication": "med_vocab.json"}
    fp = vocab_dir / fp_map[name]
    if fp.exists():
        return load_medtok_vocab(str(fp), offset=offset, name=name)

    pytest.skip(f"Missing MedTok vocab for {name}: {fp} and MEDTOK_CODE2EMBEDS not provided")


def _scan_vocab_from_db(
    db: mr.SubjectDatabase,
    subject_ids: Iterable[int],
    target_category: TokenCategory,
    *,
    offset: int,
) -> CategoryVocab:
    seen: set[str] = set()
    for sid in subject_ids:
        for ev in db[int(sid)].events:
            if classify_code_to_category(getattr(ev, "code", None)) != target_category:
                continue
            code = getattr(ev, "code", None)
            if code is None:
                continue
            seen.add(str(code))

    code2id = {c: i + 1 for i, c in enumerate(sorted(seen))}
    code2id["<UNK>"] = 0
    return CategoryVocab(name=target_category.name.lower(), offset=offset, code2id=code2id)


def _build_encoders(db: mr.SubjectDatabase, subject_ids: List[int]) -> Dict[TokenCategory, object]:
    manifest = _load_manifest()
    code2embeds_env = os.getenv("MEDTOK_CODE2EMBEDS")
    code2embeds = Path(code2embeds_env) if code2embeds_env else None
    if code2embeds is not None and not code2embeds.exists():
        code2embeds = None

    diag_vocab = _build_medtok_vocab(
        "diagnosis",
        offset=_offset(manifest, "diagnosis", 1_000_000),
        code2embeds=code2embeds,
    )
    proc_vocab = _build_medtok_vocab(
        "procedure",
        offset=_offset(manifest, "procedure", 1_200_000),
        code2embeds=code2embeds,
    )
    med_vocab = _build_medtok_vocab(
        "medication",
        offset=_offset(manifest, "medication", 1_400_000),
        code2embeds=code2embeds,
    )
    meas_vocab = _scan_vocab_from_db(db, subject_ids, TokenCategory.MEASUREMENT, offset=_offset(manifest, "measurement_code", 2_000_000))
    struct_vocab = _scan_vocab_from_db(db, subject_ids, TokenCategory.STRUCTURAL, offset=_offset(manifest, "structural", 2_200_000))

    encoders: Dict[TokenCategory, object] = {
        TokenCategory.MEASUREMENT: SimpleCategoricalEncoder(TokenCategory.MEASUREMENT, meas_vocab),
        TokenCategory.DIAGNOSIS: MedTokenWithAttrsEncoder(
            TokenCategory.DIAGNOSIS,
            diag_vocab,
            canonicalize_fn=canonicalize_diagnosis_code,
            drop_unknowns=False,
        ),
        TokenCategory.PROCEDURE: MedTokenWithAttrsEncoder(
            TokenCategory.PROCEDURE,
            proc_vocab,
            canonicalize_fn=canonicalize_procedure_code,
            drop_unknowns=False,
        ),
        TokenCategory.MEDICATION: MedTokenWithAttrsEncoder(
            TokenCategory.MEDICATION,
            med_vocab,
            canonicalize_fn=canonicalize_medication_code,
            drop_unknowns=False,
        ),
        TokenCategory.STRUCTURAL: SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab),
        TokenCategory.OTHER: OtherNoOpEncoder(),
    }
    return encoders


@pytest.mark.integration
def test_subject_timeline_db_coverage():
    db_path = os.getenv("MEDS_READER_DB")
    if not db_path or not Path(db_path).exists():
        pytest.skip("MEDS_READER_DB not set or missing; provide a meds_reader DB path to run this test.")

    db = mr.SubjectDatabase(db_path)
    subject_ids = [int(s) for s in list(db)[:5]]
    if not subject_ids:
        pytest.skip("No subjects found in MEDS_READER_DB.")

    encoders = _build_encoders(db, subject_ids)
    stats = {cat: {"events": 0, "tokens": 0, "unk": 0} for cat in encoders.keys()}
    unk_gid = {
        TokenCategory.MEASUREMENT: encoders[TokenCategory.MEASUREMENT].vocab.offset + encoders[TokenCategory.MEASUREMENT].vocab.unk_id,
        TokenCategory.DIAGNOSIS: encoders[TokenCategory.DIAGNOSIS].base_vocab.offset + encoders[TokenCategory.DIAGNOSIS].base_vocab.unk_id,
        TokenCategory.PROCEDURE: encoders[TokenCategory.PROCEDURE].base_vocab.offset + encoders[TokenCategory.PROCEDURE].base_vocab.unk_id,
        TokenCategory.MEDICATION: encoders[TokenCategory.MEDICATION].base_vocab.offset + encoders[TokenCategory.MEDICATION].base_vocab.unk_id,
        TokenCategory.STRUCTURAL: encoders[TokenCategory.STRUCTURAL].vocab.offset + encoders[TokenCategory.STRUCTURAL].vocab.unk_id,
    }

    for sid in subject_ids:
        subj = db[int(sid)]
        for ev in subj.events:
            cat = classify_code_to_category(getattr(ev, "code", None))
            if cat in stats:
                stats[cat]["events"] += 1

        tokens = build_subject_timeline(db, subject_id=int(sid), encoders=encoders)
        for tok in tokens:
            cat = TokenCategory(tok.category_id)
            if cat not in stats:
                continue
            stats[cat]["tokens"] += 1
            if tok.value_id == unk_gid.get(cat):
                stats[cat]["unk"] += 1
            assert tok.dt_from_prev_hours >= 0.0
            assert tok.t_from_start_hours >= 0.0

    # Emit compact coverage stats for debugging
    for cat, vals in stats.items():
        if vals["events"] == 0:
            continue
        unk_rate = (vals["unk"] / vals["tokens"]) if vals["tokens"] else 0.0
        print(
            f"{cat.name}: events={vals['events']}, tokens={vals['tokens']}, "
            f"unk={vals['unk']} ({unk_rate:.1%})"
        )

    # Require that at least one category produced tokens
    assert any(v["tokens"] > 0 for v in stats.values())
