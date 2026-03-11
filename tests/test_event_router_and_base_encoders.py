from __future__ import annotations
from types import SimpleNamespace
from typing import Dict

import pytest

from src.ehr_hier.data.event_router import classify_code_to_category, is_rare_critical_other_code
from src.ehr_hier.data.token_types import EventToken, TokenCategory
import src.ehr_hier.tokenizers.base_encoder as base_encoder
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab


@pytest.mark.parametrize(
    "code,expected",
    [
        ("LAB//GLUCOSE", TokenCategory.MEASUREMENT),
        ("Blood Pressure", TokenCategory.MEASUREMENT),
        ("BMI (kg/m2)", TokenCategory.OTHER),
        ("ICD10PCS//0JH", TokenCategory.PROCEDURE),
        ("ICD10CM//A000", TokenCategory.DIAGNOSIS),
        ("NDC//1234-5678", TokenCategory.MEDICATION),
        ("HOSPITAL_ADMISSION", TokenCategory.STRUCTURAL),
        ("Event//Code Blue activated", TokenCategory.STRUCTURAL),
        ("UNKNOWN_PREFIX//X", TokenCategory.OTHER),
        (None, TokenCategory.OTHER),
    ],
)
def test_classify_code_to_category_handles_common_prefixes(code, expected):
    assert classify_code_to_category(code) == expected


def test_rare_critical_detection_only_for_other_surface():
    assert is_rare_critical_other_code("Event//Code Blue activated")
    assert not is_rare_critical_other_code("LAB//CPR")


def _dummy_encoders(monkeypatch, drop_unknowns: bool) -> Dict[TokenCategory, object]:
    class DummyMeasurementEncoder:
        category = TokenCategory.MEASUREMENT

        def __init__(self, cfg):
            self.cfg = cfg

        def reset_state(self):
            return None

        def encode_event(self, ev, dt_hours: float):
            return [
                EventToken(
                    value_id=42,
                    category_id=int(self.category),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=dt_hours,
                    cat_attrs={},
                    num_attrs={},
                )
            ]

    monkeypatch.setattr(base_encoder, "MeasurementTokenEncoder", DummyMeasurementEncoder)

    diag_vocab = CategoryVocab(name="diag", offset=10, code2id={"ICD10CM//A000": 1, "<UNK>": 0})
    proc_vocab = CategoryVocab(name="proc", offset=20, code2id={"CPT//00100": 1, "<UNK>": 0})
    med_vocab = CategoryVocab(name="med", offset=30, code2id={"RXNORM//1": 1, "<UNK>": 0})
    struct_vocab = CategoryVocab(name="struct", offset=40, code2id={"<UNK>": 0})

    enc = base_encoder.build_base_encoders(
        object(),
        diag_vocab=diag_vocab,
        proc_vocab=proc_vocab,
        med_vocab=med_vocab,
        struct_vocab=struct_vocab,
        include_other_noop=False,
        drop_unknowns=drop_unknowns,
    )
    return enc


def test_base_encoders_emit_unk_instead_of_drop(monkeypatch):
    enc = _dummy_encoders(monkeypatch, drop_unknowns=False)
    diag_enc = enc[TokenCategory.DIAGNOSIS]
    resolution = diag_enc.resolve_event(SimpleNamespace(code="ICD10CM//ZZZ"))
    assert resolution.stage == "unk"
    out = diag_enc.encode_event(SimpleNamespace(code="ICD10CM//ZZZ"), dt_hours=1.5)
    assert out, "Unknown diagnosis codes should map to UNK token, not be dropped"
    assert out[0].value_id == enc[TokenCategory.DIAGNOSIS].base_vocab.offset + enc[TokenCategory.DIAGNOSIS].base_vocab.unk_id


def test_base_encoders_can_drop_unknowns_when_requested(monkeypatch):
    enc = _dummy_encoders(monkeypatch, drop_unknowns=True)
    diag_enc = enc[TokenCategory.DIAGNOSIS]
    resolution = diag_enc.resolve_event(SimpleNamespace(code="ICD10CM//ZZZ"))
    assert resolution.stage == "drop"
    out = diag_enc.encode_event(SimpleNamespace(code="ICD10CM//ZZZ"), dt_hours=1.5)
    assert out == []


def test_medication_residual_fallback_enabled(monkeypatch):
    class DummyMeasurementEncoder:
        category = TokenCategory.MEASUREMENT

        def __init__(self, cfg):
            self.cfg = cfg

        def reset_state(self):
            return None

        def encode_event(self, ev, dt_hours: float):
            return []

    monkeypatch.setattr(base_encoder, "MeasurementTokenEncoder", DummyMeasurementEncoder)

    diag_vocab = CategoryVocab(name="diag", offset=1_000_000, code2id={"ICD10CM//A000": 1, "<UNK>": 0})
    proc_vocab = CategoryVocab(name="proc", offset=1_200_000, code2id={"CPT//00100": 1, "<UNK>": 0})
    med_vocab = CategoryVocab(name="med", offset=1_400_000, code2id={"RXNORM//1": 1, "<UNK>": 0})
    struct_vocab = CategoryVocab(name="struct", offset=2_200_000, code2id={"<UNK>": 0})

    enc = base_encoder.build_base_encoders(
        object(),
        diag_vocab=diag_vocab,
        proc_vocab=proc_vocab,
        med_vocab=med_vocab,
        struct_vocab=struct_vocab,
        include_other_noop=False,
        drop_unknowns=False,
        enable_residual_fallback=True,
        residual_fallback_buckets=128,
    )
    med_enc = enc[TokenCategory.MEDICATION]
    out = med_enc.encode_event(
        SimpleNamespace(code="MEDICATION//Acetaminophen//Administered"),
        dt_hours=0.0,
    )
    resolution = med_enc.resolve_event(
        SimpleNamespace(code="MEDICATION//Acetaminophen//Administered")
    )
    assert resolution.stage == "residual"
    assert out, "Unknown medication should emit a residual fallback token when enabled"
    tok = out[0]
    assert tok.value_id != med_vocab.offset + med_vocab.unk_id
    assert tok.cat_attrs.get("residual_fallback") == 1
