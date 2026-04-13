import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.event_frames import flatten_event_frames
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
from src.ehr_hier.tokenizers.medtok_crosswalk import (
    build_medtok_crosswalk_artifact,
    load_resolved_crosswalk_lookup,
)
from scripts.build_compressed_medtok_vocabs import _build_fallback_vocab


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
    proc_snomed_cands = canonicalize_procedure_code("SNOMED/77477000")
    assert "77477000" in proc_snomed_cands

    med_cands = canonicalize_medication_code("rxnorm 12345")
    assert "RXNORM//12345" in med_cands
    med_name_cands = canonicalize_medication_code("MEDICATION//Acetaminophen//Administered")
    assert "MEDICATION//ACETAMINOPHEN" in med_name_cands
    assert "ACETAMINOPHEN" in med_name_cands


def test_medtok_resolution_stages_basic(tiny_vocabs):
    diag_vocab, _proc_vocab, med_vocab, _struct_vocab, _med_attr_vocabs, _med_numeric_attrs = tiny_vocabs

    diag_encoder = MedTokenWithAttrsEncoder(
        TokenCategory.DIAGNOSIS,
        diag_vocab,
        canonicalize_fn=canonicalize_diagnosis_code,
    )
    med_encoder = MedTokenWithAttrsEncoder(
        TokenCategory.MEDICATION,
        med_vocab,
        canonicalize_fn=canonicalize_medication_code,
    )

    exact = diag_encoder.resolve_event(SimpleNamespace(code="ICD10CM//A123"))
    assert exact.stage == "exact"
    assert exact.matched_code == "ICD10CM//A123"

    canonicalized = diag_encoder.resolve_event(SimpleNamespace(code="dx A12.3"))
    assert canonicalized.stage == "canonicalized"
    assert canonicalized.matched_code == "ICD10CM//A123"

    unk = med_encoder.resolve_event(SimpleNamespace(code="RXNORM//999"))
    assert unk.stage == "unk"
    assert unk.base_gid == med_vocab.offset + med_vocab.unk_id


def test_medtok_resolution_parent_lookup_and_lexical_bridge():
    med_vocab = CategoryVocab(
        name="med",
        offset=1_400_000,
        code2id={
            "<UNK>": 0,
            "RXNORM//123": 1,
            "MEDICATION//ACETAMINOPHEN TABLET//ADMINISTERED": 2,
        },
    )
    med_encoder = MedTokenWithAttrsEncoder(
        TokenCategory.MEDICATION,
        med_vocab,
        canonicalize_fn=canonicalize_medication_code,
    )

    parent_hit = med_encoder.resolve_event(
        SimpleNamespace(
            code="MEDICATION//UNKNOWN DRUG//ADMINISTERED",
            parent_codes=["RXNORM//123"],
        )
    )
    assert parent_hit.stage == "parent_lookup"
    assert parent_hit.matched_code == "RXNORM//123"

    lexical = med_encoder.resolve_event(SimpleNamespace(code="Acetaminophen Tablet"))
    assert lexical.stage == "lexical_bridge"
    assert lexical.matched_code == "MEDICATION//ACETAMINOPHEN TABLET//ADMINISTERED"


def test_medication_crosswalk_lookup_stage(tmp_path):
    concept_dir = tmp_path / "concept_map"
    concept_dir.mkdir(parents=True, exist_ok=True)
    (concept_dir / "inputevents_to_rxnorm.csv").write_text(
        "\n".join(
            [
                "itemid (omop_source_code),label,ordercategorydescription,amountuom,omop_concept_id,omop_concept_name,omop_domain_id,omop_vocabulary_id,omop_concept_class_id,omop_standard_concept,omop_concept_code",
                "221833,Hydromorphone (Dilaudid),Drug Push,mg,35603598,hydromorphone Injection [Dilaudid],Drug,RxNorm,Branded Drug Form,S,1724273",
            ]
        ),
        encoding="utf-8",
    )
    artifact = build_medtok_crosswalk_artifact(concept_map_dir=concept_dir)
    artifact_fp = tmp_path / "crosswalk.json"
    artifact_fp.write_text(json.dumps(artifact), encoding="utf-8")

    med_vocab = CategoryVocab(
        name="med",
        offset=1_400_000,
        code2id={"<UNK>": 0, "RXNORM//1724273": 1},
    )
    lookup = load_resolved_crosswalk_lookup(
        artifact_fp,
        "medication",
        available_codes=med_vocab.code2id.keys(),
    )
    med_encoder = MedTokenWithAttrsEncoder(
        TokenCategory.MEDICATION,
        med_vocab,
        canonicalize_fn=canonicalize_medication_code,
        crosswalk_lookup=lookup,
    )
    resolution = med_encoder.resolve_event(
        SimpleNamespace(code="MEDICATION//Hydromorphone (Dilaudid)//Administered")
    )
    assert resolution.stage == "crosswalk_lookup"
    assert resolution.matched_code == "RXNORM//1724273"


def test_procedure_crosswalk_lookup_stage(tmp_path):
    concept_dir = tmp_path / "concept_map"
    concept_dir.mkdir(parents=True, exist_ok=True)
    (concept_dir / "proc_itemid.csv").write_text(
        "\n".join(
            [
                "itemid (omop_source_code),label,omop_concept_id,omop_concept_name,omop_domain_id,omop_vocabulary_id,omop_concept_class_id,omop_standard_concept,omop_concept_code",
                "221214,CT scan,4300757,Computerized axial tomography,Procedure,SNOMED,Procedure,S,77477000",
            ]
        ),
        encoding="utf-8",
    )
    artifact = build_medtok_crosswalk_artifact(concept_map_dir=concept_dir)
    artifact_fp = tmp_path / "crosswalk.json"
    artifact_fp.write_text(json.dumps(artifact), encoding="utf-8")

    proc_vocab = CategoryVocab(
        name="proc",
        offset=1_200_000,
        code2id={"<UNK>": 0, "77477000": 1},
    )
    lookup = load_resolved_crosswalk_lookup(
        artifact_fp,
        "procedure",
        available_codes=proc_vocab.code2id.keys(),
    )
    proc_encoder = MedTokenWithAttrsEncoder(
        TokenCategory.PROCEDURE,
        proc_vocab,
        canonicalize_fn=canonicalize_procedure_code,
        crosswalk_lookup=lookup,
    )
    resolution = proc_encoder.resolve_event(SimpleNamespace(code="PROCEDURE//CT scan"))
    assert resolution.stage == "crosswalk_lookup"
    assert resolution.matched_code == "77477000"


def test_build_fallback_vocab_prefers_normalized_medication_surface():
    df = pd.DataFrame(
        [
            {
                "code": "MEDICATION//Acetaminophen//Administered",
                "routed_category": "MEDICATION",
                "events_total": 10,
                "subject_coverage_frac": 0.2,
            },
            {
                "code": "MEDICATION//Acetaminophen//Confirmed",
                "routed_category": "MEDICATION",
                "events_total": 5,
                "subject_coverage_frac": 0.1,
            },
            {
                "code": "MEDICATION//RareDrug//Administered",
                "routed_category": "MEDICATION",
                "events_total": 1,
                "subject_coverage_frac": 0.0,
            },
        ]
    )
    med_vocab = CategoryVocab(
        name="med",
        offset=1_400_000,
        code2id={"<UNK>": 0},
    )
    code2id, report = _build_fallback_vocab(
        df=df,
        routed_category="MEDICATION",
        canonicalize_fn=canonicalize_medication_code,
        full_vocab=med_vocab,
        crosswalk_candidates={},
        max_explicit=1,
        target_coverage=0.90,
        drop_low_specificity_med=False,
    )
    assert code2id == {
        "<UNK>": 0,
        "MEDICATION//ACETAMINOPHEN": 1,
    }
    assert report["unresolved_events"] == 16
    assert report["selected_unresolved_events"] == 15


def test_drop_unknown_diagnosis(monkeypatch, tiny_vocabs):
    diag_vocab, proc_vocab, med_vocab, struct_vocab, med_attr_vocabs, med_numeric_attrs = tiny_vocabs

    encoder = MedTokenWithAttrsEncoder(
        TokenCategory.DIAGNOSIS,
        diag_vocab,
        canonicalize_fn=canonicalize_diagnosis_code,
        drop_unknowns=True,
    )
    resolution = encoder.resolve_event(SimpleNamespace(code="UNKNOWN"))
    assert resolution.stage == "drop"
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


def test_medtok_start_stop_markers(monkeypatch, tiny_vocabs):
    diag_vocab, proc_vocab, med_vocab, struct_vocab, med_attr_vocabs, med_numeric_attrs = tiny_vocabs

    # Patch measurement encoder out
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
        FakeEvent("MEDICATION//START//NDC//00000-0000", start),
        FakeEvent("MEDICATION//STOP//NDC//00000-0000", start + dt.timedelta(hours=2)),
    ]
    db = FakeDB({7: FakeSubject(events)})

    frames = build_subject_timeline(db, subject_id=7, encoders=encoders)
    tokens = flatten_event_frames(frames)

    # Two frames, each containing a base-med token plus a marker token.
    assert len(frames) == 2
    assert len(tokens) == 4
    assert all(t.category_id == int(TokenCategory.MEDICATION) for t in tokens)

    base_id = med_vocab.offset + med_vocab.code2id["NDC//00000-0000"]
    marker_id = med_vocab.offset + med_vocab.unk_id

    # First event
    assert tokens[0].value_id == base_id
    assert tokens[0].dt_from_prev_hours == 0.0
    assert tokens[1].value_id == marker_id
    assert tokens[1].dt_from_prev_hours == 0.0
    assert tokens[1].cat_attrs.get("event_marker") == 1  # START

    # Second event (dt progresses)
    assert tokens[2].value_id == base_id
    assert tokens[2].dt_from_prev_hours == 2.0
    assert tokens[3].value_id == marker_id
    assert tokens[3].dt_from_prev_hours == 0.0
    assert tokens[3].cat_attrs.get("event_marker") == 3  # STOP
