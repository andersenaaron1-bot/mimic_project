"""
Lightweight smoke test to inspect MedTok-backed EventTokens end-to-end against a meds_reader DB.

Run with:
    MEDS_READER_DB=/path/to/meds_reader.db \\
    MEDTOK_CODE2EMBEDS=/path/to/code2embeddings.json \\
    pytest -q tests/test_medtok_smoke_db.py -s

If you prefer a tiny vocab, set MEDTOK_VOCAB_DIR (must contain diag_vocab.json / proc_vocab.json / med_vocab.json)
instead of MEDTOK_CODE2EMBEDS to avoid loading a large embeddings file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

import pytest

import meds_reader as mr

# Ensure repo root is importable when PYTHONPATH is not set
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_attr_encoder import MedTokenWithAttrsEncoder
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_procedure_code,
    canonicalize_medication_code,
    diagnosis_filter,
    procedure_filter,
    medication_filter,
)
from src.ehr_hier.tokenizers.medtok_loader import (
    CategoryVocab,
    build_vocab_from_code2embeddings,
    load_medtok_vocab,
    load_attr_vocab,
)
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig


def _offset(manifest: Dict, key: str, default: int) -> int:
    return int(manifest.get(key, {}).get("offset", default))


def _load_manifest() -> Dict:
    fp = Path("artifacts/vocab_manifest.json")
    if fp.exists():
        import json

        return json.loads(fp.read_text())
    return {}


def _build_category_vocab(
    name: str,
    *,
    offset: int,
    code2embeds_path: Optional[str],
    vocab_dir: Optional[Path],
    filter_fn,
) -> CategoryVocab:
    if code2embeds_path:
        return build_vocab_from_code2embeddings(
            code2embeds_path, offset=offset, name=name, filter_fn=filter_fn
        )
    if vocab_dir is None:
        raise FileNotFoundError("MEDTOK_VOCAB_DIR not provided and MEDTOK_CODE2EMBEDS missing")
    fp = vocab_dir / f"{name[:4]}_vocab.json" if name in {"diagnosis", "procedure"} else vocab_dir / "med_vocab.json"
    if name == "diagnosis":
        fp = vocab_dir / "diag_vocab.json"
    elif name == "procedure":
        fp = vocab_dir / "proc_vocab.json"
    elif name == "medication":
        fp = vocab_dir / "med_vocab.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing vocab file: {fp}")
    return load_medtok_vocab(str(fp), offset=offset, name=name)


def _maybe_attr_vocab(path: Path, offset: int, name: str) -> Optional[CategoryVocab]:
    return load_attr_vocab(str(path), offset=offset, name=name) if path.exists() else None


def _build_medication_attrs(manifest: Dict) -> Dict[str, CategoryVocab]:
    attr_dir = Path(os.getenv("MEDTOK_ATTR_DIR", "artifacts/medtok_attrs"))
    attr_vocabs = {}
    route = _maybe_attr_vocab(attr_dir / "route_vocab.json", _offset(manifest, "med_route", 1_600_000), "route")
    form = _maybe_attr_vocab(attr_dir / "form_vocab.json", _offset(manifest, "med_form", 1_620_000), "form")
    freq = _maybe_attr_vocab(attr_dir / "freq_vocab.json", _offset(manifest, "med_freq", 1_640_000), "freq")
    unit = _maybe_attr_vocab(attr_dir / "unit_vocab.json", _offset(manifest, "med_unit", 1_660_000), "unit")
    for name, vocab in (("route", route), ("form", form), ("freq", freq), ("unit", unit)):
        if vocab is not None:
            attr_vocabs[name] = vocab
    return attr_vocabs


def _med_numeric_cfg(manifest: Dict) -> Dict[str, NumericBinConfig]:
    return {
        # Wide ranges to avoid clipping most realistic values; normalization happens via .normalize()
        "dosage": NumericBinConfig(offset=_offset(manifest, "med_dosage", 1_680_000), bins=8, min_val=0.1, max_val=2000.0, log=True),
        "rate": NumericBinConfig(offset=_offset(manifest, "med_rate", 1_700_000), bins=6, min_val=0.1, max_val=2000.0, log=True),
        "duration_hours": NumericBinConfig(offset=_offset(manifest, "med_duration", 1_720_000), bins=6, min_val=0.1, max_val=240.0, log=True),
    }


def _build_encoders() -> Dict[TokenCategory, MedTokenWithAttrsEncoder]:
    manifest = _load_manifest()
    db_path = os.getenv("MEDS_READER_DB")
    if not db_path or not Path(db_path).exists():
        pytest.skip("MEDS_READER_DB not set or missing; set to your meds_reader DB path.")

    code2emb = os.getenv("MEDTOK_CODE2EMBEDS")
    vocab_dir_env = os.getenv("MEDTOK_VOCAB_DIR")
    vocab_dir = Path(vocab_dir_env) if vocab_dir_env else Path("artifacts/medtok")
    if not code2emb and not vocab_dir.exists():
        pytest.skip("Provide MEDTOK_CODE2EMBEDS or MEDTOK_VOCAB_DIR to run this smoke test.")

    diag_vocab = _build_category_vocab(
        "diagnosis",
        offset=_offset(manifest, "diagnosis", 1_000_000),
        code2embeds_path=code2emb,
        vocab_dir=vocab_dir if vocab_dir.exists() else None,
        filter_fn=diagnosis_filter,
    )
    proc_vocab = _build_category_vocab(
        "procedure",
        offset=_offset(manifest, "procedure", 1_200_000),
        code2embeds_path=code2emb,
        vocab_dir=vocab_dir if vocab_dir.exists() else None,
        filter_fn=procedure_filter,
    )
    med_vocab = _build_category_vocab(
        "medication",
        offset=_offset(manifest, "medication", 1_400_000),
        code2embeds_path=code2emb,
        vocab_dir=vocab_dir if vocab_dir.exists() else None,
        filter_fn=medication_filter,
    )

    med_attr_vocabs = _build_medication_attrs(manifest)
    med_numeric_attrs = _med_numeric_cfg(manifest)

    encoders: Dict[TokenCategory, MedTokenWithAttrsEncoder] = {
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
            categorical_attrs=med_attr_vocabs,
            numeric_attrs=med_numeric_attrs,
            drop_unknowns=False,
        ),
    }
    return encoders


def _collect_tokens_with_codes(
    db: mr.SubjectDatabase,
    encoders: Dict[TokenCategory, MedTokenWithAttrsEncoder],
    max_per_cat: int = 10,
) -> Dict[TokenCategory, List[dict]]:
    """
    Replicates timeline building for DIAG/PROC/MED only, keeping raw codes alongside tokens.
    """
    collected: Dict[TokenCategory, List[dict]] = {
        TokenCategory.DIAGNOSIS: [],
        TokenCategory.PROCEDURE: [],
        TokenCategory.MEDICATION: [],
    }

    for sid in db:
        subj = db[int(sid)]
        events = list(subj.events)
        # establish start
        timeline_start = None
        for ev in events:
            t = getattr(ev, "time", None)
            if t is not None:
                timeline_start = t
                break

        def t_from_start_hours(t):
            if timeline_start is None or t is None:
                return 0.0
            return max(0.0, (t - timeline_start).total_seconds() / 3600.0)

        last_emitted_time = None
        for ev in events:
            cat = classify_code_to_category(getattr(ev, "code", None))
            if cat not in collected:
                continue
            enc = encoders.get(cat)
            if enc is None:
                continue
            t = getattr(ev, "time", None)
            dt = 0.0
            if t is not None and last_emitted_time is not None:
                dt = max(0.0, (t - last_emitted_time).total_seconds() / 3600.0)
            toks = enc.encode_event(ev, dt_hours=dt)
            if not toks:
                continue
            if t is not None:
                last_emitted_time = t
            t_start = t_from_start_hours(t)
            for tok in toks:
                tok.t_from_start_hours = t_start
                tok.raw_time = t
                collected[cat].append(
                    {"token": tok, "raw_code": getattr(ev, "code", None)}
                )
                if all(len(v) >= max_per_cat for v in collected.values()):
                    return collected
    return collected


@pytest.mark.integration
def test_medtok_event_tokens_smoke():
    db_path = os.getenv("MEDS_READER_DB")
    if not db_path or not Path(db_path).exists():
        pytest.skip("MEDS_READER_DB not set or missing; set to your meds_reader DB path.")

    encoders = _build_encoders()
    db = mr.SubjectDatabase(db_path)
    tokens_by_cat = _collect_tokens_with_codes(db, encoders, max_per_cat=5)

    # Require at least one token per category; skip if dataset lacks the category
    for cat, toks in tokens_by_cat.items():
        if not toks:
            pytest.skip(f"No tokens found for category {cat.name}; check DB contents or vocabs.")

    # Basic shape/monotonicity checks
    for cat, pairs in tokens_by_cat.items():
        for pair in pairs:
            tok = pair["token"]
            raw_code = pair["raw_code"]
            assert isinstance(tok.value_id, int) and tok.value_id >= 0
            assert tok.category_id == int(cat)
            assert tok.dt_from_prev_hours >= 0.0
            assert tok.t_from_start_hours >= 0.0
            # Diagnostics: print a compact view for manual inspection
            print(
                f"{cat.name}: raw_code={raw_code}, value_id={tok.value_id}, dt={tok.dt_from_prev_hours:.2f}, "
                f"t_start={tok.t_from_start_hours:.2f}, cat_attrs={tok.cat_attrs}, num_attrs={tok.num_attrs}"
            )

    # Medication-specific metadata sanity
    for pair in tokens_by_cat[TokenCategory.MEDICATION]:
        tok = pair["token"]
        assert isinstance(tok.cat_attrs, dict)
        assert isinstance(tok.num_attrs, dict)
        # categorical attrs should map to global ids when present
        for v in tok.cat_attrs.values():
            assert isinstance(v, int) and v >= 0
        # numeric attrs are normalized to [0,1]
        for v in tok.num_attrs.values():
            assert 0.0 <= float(v) <= 1.0


@pytest.mark.integration
def test_medtok_vocab_coverage():
    """
    Report hit/miss coverage for MedTok vocab against the DB.
    """
    db_path = os.getenv("MEDS_READER_DB")
    if not db_path or not Path(db_path).exists():
        pytest.skip("MEDS_READER_DB not set or missing; set to your meds_reader DB path.")
    code2emb = os.getenv("MEDTOK_CODE2EMBEDS")
    vocab_dir_env = os.getenv("MEDTOK_VOCAB_DIR")
    vocab_dir = Path(vocab_dir_env) if vocab_dir_env else Path("artifacts/medtok")
    if not code2emb and not vocab_dir.exists():
        pytest.skip("Provide MEDTOK_CODE2EMBEDS or MEDTOK_VOCAB_DIR to run this coverage test.")

    manifest = _load_manifest()
    diag_vocab = _build_category_vocab(
        "diagnosis",
        offset=_offset(manifest, "diagnosis", 1_000_000),
        code2embeds_path=code2emb,
        vocab_dir=vocab_dir if vocab_dir.exists() else None,
        filter_fn=diagnosis_filter,
    )
    proc_vocab = _build_category_vocab(
        "procedure",
        offset=_offset(manifest, "procedure", 1_200_000),
        code2embeds_path=code2emb,
        vocab_dir=vocab_dir if vocab_dir.exists() else None,
        filter_fn=procedure_filter,
    )
    med_vocab = _build_category_vocab(
        "medication",
        offset=_offset(manifest, "medication", 1_400_000),
        code2embeds_path=code2emb,
        vocab_dir=vocab_dir if vocab_dir.exists() else None,
        filter_fn=medication_filter,
    )

    db = mr.SubjectDatabase(db_path)
    import collections

    stats = {
        "diag": {"hit": 0, "miss": 0, "samples": collections.Counter()},
        "proc": {"hit": 0, "miss": 0, "samples": collections.Counter()},
        "med": {"hit": 0, "miss": 0, "samples": collections.Counter()},
    }

    def check(ev, vocab, canon_fn, bucket):
        cands = canon_fn(getattr(ev, "code", None))
        gid = None
        for c in cands:
            gid = vocab.maybe_encode(c)
            if gid is not None:
                break
        if gid is None:
            stats[bucket]["miss"] += 1
            stats[bucket]["samples"][str(getattr(ev, "code", None))] += 1
        else:
            stats[bucket]["hit"] += 1

    for sid in db:
        for ev in db[int(sid)].events:
            code = getattr(ev, "code", None)
            if code is None:
                continue
            cs = str(code).upper()
            if diagnosis_filter(cs):
                check(ev, diag_vocab, canonicalize_diagnosis_code, "diag")
            elif procedure_filter(cs):
                check(ev, proc_vocab, canonicalize_procedure_code, "proc")
            elif medication_filter(cs):
                check(ev, med_vocab, canonicalize_medication_code, "med")

    for k, v in stats.items():
        total = v["hit"] + v["miss"]
        if total == 0:
            continue
        hit_rate = v["hit"] / total
        print(f"{k}: hit {v['hit']} / {total} ({hit_rate:.1%})")
        for code, count in v["samples"].most_common(10):
            print(f"  miss {count}x {code}")
