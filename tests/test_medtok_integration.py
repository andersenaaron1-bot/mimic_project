import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers import measurement_encoder as meas_mod
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders
from src.ehr_hier.tokenizers.medtok_loader import (
    build_vocab_from_code2embeddings,
    load_attr_vocab,
    CategoryVocab,
)
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_procedure_code,
    canonicalize_medication_code,
    diagnosis_filter,
    procedure_filter,
    medication_filter,
)
from src.ehr_hier.tokenizers.medtok_attr_encoder import MedTokenWithAttrsEncoder
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig


class FakeEvent:
    def __init__(self, code: str, time: dt.datetime, **kwargs):
        self.code = code
        self.time = time
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeSubject:
    def __init__(self, events):
        self.events = events


class FakeDB:
    def __init__(self, subjects):
        self._subjects = subjects

    def __iter__(self):
        return iter(self._subjects.keys())

    def __getitem__(self, sid: int):
        return self._subjects[int(sid)]


class DummyMeasurementEncoder:
    category = TokenCategory.MEASUREMENT

    def __init__(self, cfg):
        self.cfg = cfg

    def reset_state(self):
        return None

    def encode_event(self, ev, dt_hours: float):
        return []  # unused for this integration check


def _write_vocab(tmp_path: Path, name: str, entries: list[str]) -> Path:
    """
    Create a tiny vocab file with <UNK>=0 and provided entries starting at 1.
    """
    payload = {"<UNK>": 0}
    for idx, token in enumerate(entries, start=1):
        payload[token] = idx
    fp = tmp_path / f"{name}_vocab.json"
    fp.write_text(json.dumps(payload))
    return fp


@pytest.fixture
def tiny_vocabs(tmp_path):
    # Minimal MedTok-style embeddings file
    code2emb_fp = tmp_path / "code2embeddings.json"
    payload = {
        "<UNK>": [0.0],
        "ICD10CM//A123": [0.1],
        "CPT//00100": [0.2],
        "NDC//00000-0000": [0.3],
    }
    code2emb_fp.write_text(json.dumps(payload))

    diag = build_vocab_from_code2embeddings(
        str(code2emb_fp), offset=1_000_000, name="diag", filter_fn=diagnosis_filter
    )
    proc = build_vocab_from_code2embeddings(
        str(code2emb_fp), offset=1_200_000, name="proc", filter_fn=procedure_filter
    )
    med = build_vocab_from_code2embeddings(
        str(code2emb_fp), offset=1_400_000, name="med", filter_fn=medication_filter
    )
    struct = CategoryVocab(name="struct", offset=2_200_000, code2id={"<UNK>": 0})
    med_attr_vocabs = {
        "route": load_attr_vocab(
            str(_write_vocab(tmp_path, "route", ["PO", "IV"])), offset=1_600_000, name="route"
        ),
        "form": load_attr_vocab(
            str(_write_vocab(tmp_path, "form", ["TABLET", "INJECTION"])), offset=1_620_000, name="form"
        ),
        "freq": load_attr_vocab(
            str(_write_vocab(tmp_path, "freq", ["BID", "QD"])), offset=1_640_000, name="freq"
        ),
        "unit": load_attr_vocab(
            str(_write_vocab(tmp_path, "unit", ["MG", "ML_HR"])), offset=1_660_000, name="unit"
        ),
    }
    med_numeric_attrs = {
        "dosage": NumericBinConfig(offset=1_680_000, bins=4, min_val=1.0, max_val=400.0, log=True),
        "rate": NumericBinConfig(offset=1_700_000, bins=4, min_val=1.0, max_val=200.0, log=True),
        "duration_hours": NumericBinConfig(offset=1_720_000, bins=4, min_val=0.5, max_val=72.0, log=True),
    }
    return diag, proc, med, struct, med_attr_vocabs, med_numeric_attrs


def test_medtok_offsets_and_dt(monkeypatch, tiny_vocabs):
    diag_vocab, proc_vocab, med_vocab, struct_vocab, med_attr_vocabs, med_numeric_attrs = tiny_vocabs

    # Patch out heavy measurement encoder with a stub
    monkeypatch.setattr(meas_mod, "MeasurementTokenEncoder", DummyMeasurementEncoder)
    import src.ehr_hier.tokenizers.base_encoder as base_enc
    monkeypatch.setattr(base_enc, "MeasurementTokenEncoder", DummyMeasurementEncoder)

    encoders = build_base_encoders(
        meas_cfg=SimpleNamespace(),
        diag_vocab=diag_vocab,
        proc_vocab=proc_vocab,
        med_vocab=med_vocab,
        struct_vocab=struct_vocab,
        med_attr_vocabs=None,
        med_numeric_attrs=None,
    )

    start = dt.datetime(2024, 1, 1, 0, 0, 0)
    events = [
        FakeEvent("ICD10CM//A123", start),                      # known diag
        FakeEvent("CPT//00100", start + dt.timedelta(hours=1)), # known proc
        FakeEvent("RXNORM//999", start + dt.timedelta(hours=3)),# unknown med -> UNK
        FakeEvent("NDC//00000-0000", start + dt.timedelta(hours=5)),  # known med
    ]
    db = FakeDB({42: FakeSubject(events)})

    tokens = build_subject_timeline(db, subject_id=42, encoders=encoders)

    assert len(tokens) == 4
    assert [t.category_id for t in tokens] == [
        int(TokenCategory.DIAGNOSIS),
        int(TokenCategory.PROCEDURE),
        int(TokenCategory.MEDICATION),
        int(TokenCategory.MEDICATION),
    ]

    # Value offsets and UNK fallback
    assert tokens[0].value_id == diag_vocab.offset + diag_vocab.code2id["ICD10CM//A123"]
    assert tokens[1].value_id == proc_vocab.offset + proc_vocab.code2id["CPT//00100"]
    assert tokens[2].value_id == med_vocab.offset + med_vocab.unk_id  # unknown RXNORM
    assert tokens[3].value_id == med_vocab.offset + med_vocab.code2id["NDC//00000-0000"]

    # dt_hours progression
    assert [t.dt_from_prev_hours for t in tokens] == [0.0, 1.0, 2.0, 2.0]


def test_canonicalizers_basic():
    diag_cands = canonicalize_diagnosis_code("dx ICD10CM A12.3")
    assert "ICD10CM//A12.3" in diag_cands
    assert "A12.3" in diag_cands

    proc_cands = canonicalize_procedure_code("Procedure CPT 00100")
    assert "CPT//00100" in proc_cands

    med_cands = canonicalize_medication_code("rxnorm 12345")
    assert "RXNORM//12345" in med_cands


def test_drop_unknown_diagnosis(monkeypatch, tiny_vocabs):
    diag_vocab, proc_vocab, med_vocab, struct_vocab, med_attr_vocabs, med_numeric_attrs = tiny_vocabs

    encoder = MedTokenWithAttrsEncoder(
        TokenCategory.DIAGNOSIS,
        diag_vocab,
        canonicalize_fn=canonicalize_diagnosis_code,
        drop_unknowns=True,
    )
    out = encoder.encode_event(SimpleNamespace(code="UNKNOWN"), dt_hours=1.0)
    assert out == []


def test_medtok_metadata_bundle(monkeypatch, tiny_vocabs):
    (
        _diag_vocab,
        _proc_vocab,
        med_vocab,
        struct_vocab,
        med_attr_vocabs,
        med_numeric_attrs,
    ) = tiny_vocabs

    # Patch measurement encoder out
    monkeypatch.setattr(meas_mod, "MeasurementTokenEncoder", DummyMeasurementEncoder)
    import src.ehr_hier.tokenizers.base_encoder as base_enc
    monkeypatch.setattr(base_enc, "MeasurementTokenEncoder", DummyMeasurementEncoder)

    encoders = build_base_encoders(
        meas_cfg=SimpleNamespace(),
        diag_vocab=_diag_vocab,
        proc_vocab=_proc_vocab,
        med_vocab=med_vocab,
        struct_vocab=struct_vocab,
        med_attr_vocabs=med_attr_vocabs,
        med_numeric_attrs=med_numeric_attrs,
    )

    start = dt.datetime(2024, 1, 1, 0, 0, 0)
    events = [
        FakeEvent(
            "RXNORM//123",
            start,
            route="PO",
            form="TABLET",
            freq="BID",
            unit="MG",
            dosage=50.0,
            rate=10.0,
            duration_hours=1.0,
        ),
        FakeEvent(
            "NDC//00000-0000",
            start + dt.timedelta(hours=4),
            route="IV",
            form="INJECTION",
            freq="QD",
            unit="ML_HR",
            dosage=5.0,
            rate=2.0,
            duration_hours=12.0,
        ),
    ]
    db = FakeDB({101: FakeSubject(events)})
    tokens = build_subject_timeline(db, subject_id=101, encoders=encoders)

    assert len(tokens) == 2  # one EventToken per med event
    first, second = tokens

    # dt progression
    assert first.dt_from_prev_hours == 0.0
    assert second.dt_from_prev_hours == 4.0

    # Category IDs remain MEDICATION
    assert all(t.category_id == int(TokenCategory.MEDICATION) for t in tokens)

    # Offsets: base tokens in med band; attrs stored inside EventToken
    assert med_vocab.offset <= first.value_id < 1_600_000
    assert first.cat_attrs == {
        "route": med_attr_vocabs["route"].encode("PO"),
        "form": med_attr_vocabs["form"].encode("TABLET"),
        "freq": med_attr_vocabs["freq"].encode("BID"),
        "unit": med_attr_vocabs["unit"].encode("MG"),
    }

    # Numeric metadata normalized to [0,1]
    assert first.num_attrs["dosage"] == pytest.approx(med_numeric_attrs["dosage"].normalize(50.0))
    assert first.num_attrs["rate"] == pytest.approx(med_numeric_attrs["rate"].normalize(10.0))
    assert first.num_attrs["duration_hours"] == pytest.approx(
        med_numeric_attrs["duration_hours"].normalize(1.0)
    )

    # Unknown attr fallback: values missing in vocab land on UNK id and 0.0 norm
    assert second.cat_attrs["freq"] == med_attr_vocabs["freq"].encode("QD")
    assert second.num_attrs["dosage"] == pytest.approx(med_numeric_attrs["dosage"].normalize(5.0))
