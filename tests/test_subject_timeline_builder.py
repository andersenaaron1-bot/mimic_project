from __future__ import annotations
from datetime import datetime, timedelta
import pytest
from types import SimpleNamespace
from typing import Dict, List

from src.ehr_hier.data.structural_codes import (
    TRANSITION_ACTION_TO_ID,
    StructuralCodebook,
    load_structural_codebook_yaml,
)
from src.ehr_hier.data.subject_timeline_builder import build_subject_timeline
from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.tokenizers.simple_categorical_encoders import SimpleCategoricalEncoder


class DummySubject:
    def __init__(self, events: List[object]):
        self.events = events


class DummyDB(dict):
    def __iter__(self):
        return iter(self.keys())


class DummyEncoder:
    def __init__(self, category: TokenCategory, value_id: int):
        self.category = category
        self.value_id = value_id
        self.reset_calls = 0

    def reset_state(self):
        self.reset_calls += 1

    def encode_event(self, ev, dt_hours: float):
        return [
            EventToken(
                value_id=self.value_id,
                category_id=int(self.category),
                t_from_start_hours=0.0,
                dt_from_prev_hours=dt_hours,
                cat_attrs={},
                num_attrs={},
            )
        ]


def test_subject_timeline_orders_and_attaches_numeric():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [
        SimpleNamespace(code="LAB//GLUCOSE", time=t0),
        SimpleNamespace(code="NDC//123", time=t0 + timedelta(hours=2), numeric_value=7.5),
        SimpleNamespace(code="HOSPITAL_ADMISSION", time=t0 + timedelta(hours=5)),
    ]
    db = DummyDB({1: DummySubject(events)})

    meas_enc = DummyEncoder(TokenCategory.MEASUREMENT, 100)
    med_enc = DummyEncoder(TokenCategory.MEDICATION, 200)
    struct_enc = DummyEncoder(TokenCategory.STRUCTURAL, 300)

    summary = EventToken(
        value_id=9,
        category_id=int(TokenCategory.SPECIAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={},
        num_attrs={},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={
            TokenCategory.MEASUREMENT: meas_enc,
            TokenCategory.MEDICATION: med_enc,
            TokenCategory.STRUCTURAL: struct_enc,
        },
        add_summary_tokens=[summary],
        structural_event_map={"HOSPITAL_ADMISSION": 1},
        window_hook_label="episode",
    )

    # Reset called for every encoder we passed in
    assert meas_enc.reset_calls == 1
    assert med_enc.reset_calls == 1
    assert struct_enc.reset_calls == 1

    # Summary token + 3 events -> 4 tokens in order
    assert len(tokens) == 4
    assert tokens[0].category_id == int(TokenCategory.SPECIAL)
    assert tokens[1].dt_from_prev_hours == 0.0  # first event
    assert pytest.approx(tokens[2].dt_from_prev_hours, rel=1e-6) == 2.0
    assert pytest.approx(tokens[3].dt_from_prev_hours, rel=1e-6) == 3.0
    assert tokens[1].t_from_start_hours == 0.0
    assert pytest.approx(tokens[2].t_from_start_hours, rel=1e-6) == 2.0
    assert pytest.approx(tokens[3].t_from_start_hours, rel=1e-6) == 5.0

    # Medication numeric_value should be attached when encoder omits it
    assert tokens[2].num_attrs["numeric_value"] == 7.5
    # Structural hook should be applied from the map
    assert tokens[3].window_hook == "episode"


def test_structural_codebook_inserts_token_and_splits_dt():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="CPT//00100", time=t0 + timedelta(hours=1))]
    db = DummyDB({7: DummySubject(events)})

    proc_vocab = CategoryVocab(name="proc", offset=20, code2id={"CPT//00100": 1, "<UNK>": 0})
    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0})
    proc_enc = SimpleCategoricalEncoder(TokenCategory.PROCEDURE, proc_vocab)
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={"CPT//00100": "or_procedure"},
        structural_only=set(),
        keep_original={"CPT//00100"},
        offset=100,
    )

    tokens = build_subject_timeline(
        db,
        subject_id=7,
        encoders={
            TokenCategory.PROCEDURE: proc_enc,
            TokenCategory.STRUCTURAL: struct_enc,
        },
        structural_codebook=codebook,
    )

    # Structural token emitted first, then procedure token with dt split
    assert len(tokens) == 2
    struct_tok, proc_tok = tokens
    assert struct_tok.category_id == int(TokenCategory.STRUCTURAL)
    assert proc_tok.category_id == int(TokenCategory.PROCEDURE)
    assert struct_tok.dt_from_prev_hours == pytest.approx(0.0)
    assert proc_tok.dt_from_prev_hours == 0.0  # dt consumed by structural token
    assert struct_tok.value_id == struct_vocab.offset + codebook.label2id()["or_procedure"]
    assert proc_tok.t_from_start_hours == pytest.approx(0.0)


def test_structural_codebook_can_emit_overlay_without_window_hook():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [
        SimpleNamespace(code="EVT_BOUNDARY", time=t0),
        SimpleNamespace(code="EVT_OVERLAY", time=t0 + timedelta(hours=1)),
    ]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0})
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={
            "EVT_BOUNDARY": "STRUCT_START_ADM",
            "EVT_OVERLAY": "STRUCT_START_MECH",
        },
        boundary_labels={"STRUCT_START_ADM"},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        structural_codebook=codebook,
        window_hook_label="episode",
    )

    assert len(tokens) == 2
    boundary_tok, overlay_tok = tokens
    assert boundary_tok.window_hook == "episode"
    assert overlay_tok.window_hook is None
    # Ensure struct_label_id is present even when it's 0.
    assert boundary_tok.cat_attrs.get("struct_label_id") == codebook.label2id()["STRUCT_START_ADM"]
    assert overlay_tok.cat_attrs.get("struct_label_id") == codebook.label2id()["STRUCT_START_MECH"]


def test_structural_codebook_emits_transition_metadata_for_typed_windows():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="ED_REGISTRATION", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0})
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={"ED_REGISTRATION": "STRUCT_START_ADM"},
        transition_map={"ED_REGISTRATION": "open_next"},
        window_type2id_map={"UNK": 0, "ED": 2},
        window_type_map={"ED_REGISTRATION": "ED"},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        structural_codebook=codebook,
        window_hook_label="episode",
    )

    assert len(tokens) == 1
    tok = tokens[0]
    assert tok.cat_attrs["transition_action_id"] == TRANSITION_ACTION_TO_ID["open_next"]
    assert tok.cat_attrs["transition_window_type_id"] == 2
    assert tok.cat_attrs["window_type_id"] == 2
    assert tok.window_hook == "episode"


def test_structural_codebook_suppresses_duplicate_legacy_hook_on_original_structural_token():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="HOSPITAL_ADMISSION", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct_raw",
        offset=50,
        code2id={"<UNK>": 0, "HOSPITAL_ADMISSION": 1},
    )
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={"HOSPITAL_ADMISSION": "STRUCT_START_ADM"},
        boundary_labels={"STRUCT_START_ADM"},
        transition_map={"HOSPITAL_ADMISSION": "open_next"},
        window_type2id_map={"UNK": 0, "INPATIENT": 3},
        window_type_map={"HOSPITAL_ADMISSION": "INPATIENT"},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        structural_codebook=codebook,
        window_hook_label="episode",
    )

    assert len(tokens) == 1
    assert sum(1 for tok in tokens if tok.window_hook is not None) == 1
    assert tokens[0].window_hook == "episode"


def test_routed_structural_transition_events_get_metadata_without_structural_map_hit():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct_raw",
        offset=50,
        code2id={"<UNK>": 0, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM": 1},
    )
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={},
        transition_map={"HOSPITAL_ADMISSION": "open_next"},
        window_type2id_map={"UNK": 0, "ED": 2, "INPATIENT": 3},
        window_type_map={"HOSPITAL_ADMISSION": "INPATIENT"},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        structural_codebook=codebook,
        window_hook_label="episode",
    )

    assert len(tokens) == 1
    tok = tokens[0]
    assert tok.cat_attrs["transition_action_id"] == TRANSITION_ACTION_TO_ID["open_next"]
    assert tok.cat_attrs["transition_window_type_id"] == 3
    assert tok.cat_attrs["window_type_id"] == 3
    assert tok.window_hook == "episode"


def test_routed_transfer_event_uses_prefix_transition_action_and_infers_ed_window_type():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="TRANSFER_TO//ED//Emergency Department", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct_raw",
        offset=50,
        code2id={"<UNK>": 0, "TRANSFER_TO//ED//Emergency Department": 1},
    )
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={},
        transition_map={"TRANSFER_TO": "close_open"},
        window_type2id_map={"UNK": 0, "ED": 2, "INPATIENT": 3},
        window_type_map={"TRANSFER_TO": "INPATIENT"},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        structural_codebook=codebook,
        window_hook_label="episode",
    )

    assert len(tokens) == 1
    tok = tokens[0]
    assert tok.cat_attrs["transition_action_id"] == TRANSITION_ACTION_TO_ID["close_open"]
    assert tok.cat_attrs["transition_window_type_id"] == 2
    assert tok.cat_attrs["window_type_id"] == 2
    assert tok.window_hook == "episode"


def test_routed_meds_birth_transition_is_suppressed_and_not_marked_as_boundary():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="MEDS_BIRTH", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct_raw",
        offset=50,
        code2id={"<UNK>": 0, "MEDS_BIRTH": 1},
    )
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={},
        transition_map={"MEDS_BIRTH": "suppress"},
        window_type2id_map={"UNK": 0, "PROLOGUE": 1},
        window_type_map={"MEDS_BIRTH": "PROLOGUE"},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=1,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        structural_codebook=codebook,
        window_hook_label="episode",
    )

    assert len(tokens) == 1
    tok = tokens[0]
    assert tok.cat_attrs["transition_action_id"] == TRANSITION_ACTION_TO_ID["suppress"]
    assert tok.cat_attrs["transition_window_type_id"] == 1
    assert "window_type_id" not in tok.cat_attrs
    assert tok.window_hook is None


def test_load_structural_codebook_yaml_respects_boundary_labels_and_soft_signifiers(tmp_path):
    yaml_fp = tmp_path / "structural_codes.yaml"
    yaml_fp.write_text(
        "\n".join(
            [
                "structural_map:",
                "  EVT_BOUNDARY: STRUCT_START_ADM",
                "window_boundary_labels:",
                "  - STRUCT_START_ADM",
                "transition_map:",
                "  EVT_BOUNDARY: open_next",
                "  TRANSFER_TO: close_open",
                "window_types:",
                "  UNK: 0",
                "  ED: 2",
                "window_type_map:",
                "  EVT_BOUNDARY: ED",
                "soft_signifiers:",
                "  - CPR_EVENT",
            ]
        ),
        encoding="utf-8",
    )

    codebook = load_structural_codebook_yaml(str(yaml_fp))
    assert codebook.code2label["EVT_BOUNDARY"] == "STRUCT_START_ADM"
    assert codebook.boundary_labels == {"STRUCT_START_ADM"}
    assert codebook.code2label["CPR_EVENT"].startswith("SOFT::")
    assert codebook.transition_action(code="EVT_BOUNDARY", label="STRUCT_START_ADM") == "open_next"
    assert codebook.transition_action(code="TRANSFER_TO//ED//Emergency Department") == "close_open"
    assert codebook.window_type_id(code="EVT_BOUNDARY") == 2


def test_subject_timeline_injects_age_and_sex_for_measurement_encoders():
    t_birth = datetime(2000, 1, 1, 0, 0, 0)
    t_meas = datetime(2020, 1, 1, 0, 0, 0)
    events = [
        # MEDS demographic conventions used by ValueEventsDataset
        SimpleNamespace(code="MEDS_BIRTH", time=t_birth),
        SimpleNamespace(code="GENDER//M", time=t_birth),
        # Actual measurement event
        SimpleNamespace(code="LAB//GLUCOSE", time=t_meas),
    ]
    db = DummyDB({3: DummySubject(events)})

    class CaptureEncoder(DummyEncoder):
        def __init__(self):
            super().__init__(TokenCategory.MEASUREMENT, 123)
            self.seen = []

        def encode_event(self, ev, dt_hours: float):
            self.seen.append((getattr(ev, "age_years", None), getattr(ev, "sex", None)))
            return super().encode_event(ev, dt_hours)

    meas_enc = CaptureEncoder()

    _ = build_subject_timeline(
        db,
        subject_id=3,
        encoders={TokenCategory.MEASUREMENT: meas_enc},
    )

    assert len(meas_enc.seen) == 1
    age_years, sex = meas_enc.seen[0]
    assert sex == pytest.approx(1.0)
    assert age_years == pytest.approx(20.0, rel=1e-6)
