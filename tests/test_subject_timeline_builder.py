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
    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0, "CPT//00100": 1})
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
    assert struct_tok.value_id == struct_vocab.offset + struct_vocab.code2id["CPT//00100"]
    assert struct_tok.cat_attrs.get("struct_label_id") == codebook.label2id()["or_procedure"]
    assert proc_tok.t_from_start_hours == pytest.approx(0.0)


def test_structural_codebook_can_emit_overlay_without_window_hook():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [
        SimpleNamespace(code="EVT_BOUNDARY", time=t0),
        SimpleNamespace(code="EVT_OVERLAY", time=t0 + timedelta(hours=1)),
    ]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct",
        offset=50,
        code2id={"<UNK>": 0, "EVT_BOUNDARY": 1, "EVT_OVERLAY": 2},
    )
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
    struct_vocab.code2id["ED_REGISTRATION"] = 1
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
        code2id={"<UNK>": 0, "HOSPITAL_ADMISSION": 1},
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
    assert tok.value_id == struct_vocab.offset + struct_vocab.code2id["HOSPITAL_ADMISSION"]


def test_routed_transfer_event_uses_prefix_transition_action_and_infers_ed_window_type():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="TRANSFER_TO//ED//Emergency Department", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct_raw",
        offset=50,
        code2id={"<UNK>": 0, "TRANSFER_TO": 1},
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
    assert tok.value_id == struct_vocab.offset + struct_vocab.code2id["TRANSFER_TO"]


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


def test_routed_structural_event_does_not_use_legacy_boundary_prefix_when_codebook_present():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="HOSPITAL_ADMISSION//GENERIC", time=t0)]
    db = DummyDB({1: DummySubject(events)})

    struct_vocab = CategoryVocab(
        name="struct_raw",
        offset=50,
        code2id={"<UNK>": 0, "HOSPITAL_ADMISSION": 1},
    )
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    codebook = StructuralCodebook(
        code2label={},
        transition_map={},
        window_type2id_map={"UNK": 0, "INPATIENT": 3},
        window_type_map={},
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
    assert tok.window_hook is None
    assert "transition_action_id" not in tok.cat_attrs


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


def test_subject_timeline_emits_global_demographic_special_tokens():
    t_birth = datetime(1980, 1, 1, 0, 0, 0)
    t_obs = datetime(2020, 1, 1, 0, 0, 0)
    events = [
        SimpleNamespace(code="MEDS_BIRTH", time=t_birth),
        SimpleNamespace(code="GENDER//F", time=t_birth),
        SimpleNamespace(code="BMI (kg/m2)", time=t_obs, numeric_value=31.2),
    ]
    db = DummyDB({8: DummySubject(events)})

    tokens = build_subject_timeline(
        db,
        subject_id=8,
        encoders={},
    )
    ids = {int(tok.value_id) for tok in tokens if tok.category_id == int(TokenCategory.SPECIAL)}

    # SEX_F, AGE_40_64, BMI_OBESE_1 in GLOBAL_DEMOGRAPHIC_TOKEN_IDS.
    assert 30 in ids
    assert 34 in ids
    assert 40 in ids


def test_subject_timeline_emits_height_and_weight_numeric_specials():
    t_birth = datetime(1980, 1, 1, 0, 0, 0)
    t_adm = datetime(2020, 1, 1, 0, 0, 0)
    events = [
        SimpleNamespace(code="MEDS_BIRTH", time=t_birth),
        SimpleNamespace(code="HOSPITAL_ADMISSION", time=t_adm),
        SimpleNamespace(code="HEIGHT (INCHES)", time=t_adm, numeric_value=70.0),
        SimpleNamespace(code="WEIGHT (LBS)", time=t_adm, numeric_value=154.0),
    ]
    db = DummyDB({12: DummySubject(events)})

    tokens = build_subject_timeline(
        db,
        subject_id=12,
        encoders={},
    )
    special_tokens = [tok for tok in tokens if tok.category_id == int(TokenCategory.SPECIAL)]
    by_id = {int(tok.value_id): tok for tok in special_tokens}

    assert 43 in by_id
    assert 44 in by_id
    assert by_id[43].num_attrs["numeric_value"] == pytest.approx(177.8, rel=1e-6)
    assert by_id[44].num_attrs["numeric_value"] == pytest.approx(69.85322498, rel=1e-6)


def test_blood_pressure_routes_to_measurement_obs_fallback():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="Blood Pressure", time=t0, value="120/80")]
    db = DummyDB({9: DummySubject(events)})

    class EmptyMeasEncoder(DummyEncoder):
        def __init__(self):
            super().__init__(TokenCategory.MEASUREMENT, 123)

        def encode_event(self, ev, dt_hours: float):
            return []

    tokens = build_subject_timeline(
        db,
        subject_id=9,
        encoders={TokenCategory.MEASUREMENT: EmptyMeasEncoder()},
    )

    obs_tokens = [tok for tok in tokens if tok.category_id == int(TokenCategory.MEASUREMENT)]
    assert len(obs_tokens) == 2
    assert obs_tokens[0].cat_attrs.get("obs_bundle_pos") == 1
    assert obs_tokens[1].cat_attrs.get("obs_bundle_pos") == 2


def test_blood_pressure_obs_uses_exact_vocabs_when_present():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="Blood Pressure", time=t0, value="120/80")]
    db = DummyDB({91: DummySubject(events)})

    class EmptyMeasEncoder(DummyEncoder):
        def __init__(self):
            super().__init__(TokenCategory.MEASUREMENT, 123)

        def encode_event(self, ev, dt_hours: float):
            return []

    obs_code_vocab = CategoryVocab(
        name="obs_code",
        offset=2_300_000,
        code2id={"<UNK>": 0, "BLOOD PRESSURE::Blood Pressure": 1},
    )
    obs_value_vocab = CategoryVocab(
        name="obs_value",
        offset=2_320_000,
        code2id={"<UNK>": 0, "UNK": 1, "N/A": 2, "NONE": 3, "": 4, "120/80": 5},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=91,
        encoders={TokenCategory.MEASUREMENT: EmptyMeasEncoder()},
        qual_obs_code_vocab=obs_code_vocab,
        qual_obs_value_vocab=obs_value_vocab,
        qual_obs_tail_policy="drop",
    )

    obs_tokens = [tok for tok in tokens if tok.category_id == int(TokenCategory.MEASUREMENT)]
    assert [int(tok.value_id) for tok in obs_tokens] == [2_300_001, 2_320_005]
    assert obs_tokens[0].cat_attrs.get("obs_code_exact") == 1
    assert obs_tokens[1].cat_attrs.get("obs_value_exact") == 1
    assert obs_tokens[0].cat_attrs.get("obs_stage_exact") == 1
    assert obs_tokens[1].cat_attrs.get("obs_stage_exact") == 1


def test_blood_pressure_obs_tail_drops_when_value_missing_from_exact_vocab():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    events = [SimpleNamespace(code="Blood Pressure", time=t0, value="120/80")]
    db = DummyDB({92: DummySubject(events)})

    class EmptyMeasEncoder(DummyEncoder):
        def __init__(self):
            super().__init__(TokenCategory.MEASUREMENT, 123)

        def encode_event(self, ev, dt_hours: float):
            return []

    obs_code_vocab = CategoryVocab(
        name="obs_code",
        offset=2_300_000,
        code2id={"<UNK>": 0, "BLOOD PRESSURE::Blood Pressure": 1},
    )
    obs_value_vocab = CategoryVocab(
        name="obs_value",
        offset=2_320_000,
        code2id={"<UNK>": 0, "UNK": 1, "N/A": 2, "NONE": 3, "": 4},
    )

    tokens = build_subject_timeline(
        db,
        subject_id=92,
        encoders={TokenCategory.MEASUREMENT: EmptyMeasEncoder()},
        qual_obs_code_vocab=obs_code_vocab,
        qual_obs_value_vocab=obs_value_vocab,
        qual_obs_tail_policy="drop",
    )

    obs_tokens = [tok for tok in tokens if tok.category_id == int(TokenCategory.MEASUREMENT)]
    assert obs_tokens == []


def test_rare_critical_structural_event_does_not_force_window_boundary():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    code = "Event//Code Blue activated"
    events = [SimpleNamespace(code=code, time=t0)]
    db = DummyDB({10: DummySubject(events)})

    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0, code: 1})
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    tokens = build_subject_timeline(
        db,
        subject_id=10,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
        window_hook_label="episode",
    )

    assert len(tokens) == 1
    assert tokens[0].category_id == int(TokenCategory.STRUCTURAL)
    assert tokens[0].window_hook is None


def test_demographic_tokens_use_admission_anchor_only():
    t_birth = datetime(1980, 1, 1, 0, 0, 0)
    t_adm = datetime(2020, 1, 10, 0, 0, 0)
    t_late = datetime(2020, 1, 11, 0, 0, 0)
    events = [
        SimpleNamespace(code="MEDS_BIRTH", time=t_birth),
        SimpleNamespace(code="HOSPITAL_ADMISSION", time=t_adm),
        SimpleNamespace(code="BMI (kg/m2)", time=t_late, numeric_value=33.0),
    ]
    db = DummyDB({11: DummySubject(events)})

    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0, "HOSPITAL_ADMISSION": 1})
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    tokens = build_subject_timeline(
        db,
        subject_id=11,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
    )
    special_ids = {int(tok.value_id) for tok in tokens if tok.category_id == int(TokenCategory.SPECIAL)}

    # Age token can exist via MEDS_BIRTH, but BMI should not leak from post-admission measurement.
    assert 34 in special_ids  # AGE_40_64
    assert 40 not in special_ids  # BMI_OBESE_1


def test_height_weight_specials_do_not_leak_from_post_admission_measurements():
    t_birth = datetime(1980, 1, 1, 0, 0, 0)
    t_adm = datetime(2020, 1, 10, 0, 0, 0)
    t_late = datetime(2020, 1, 11, 0, 0, 0)
    events = [
        SimpleNamespace(code="MEDS_BIRTH", time=t_birth),
        SimpleNamespace(code="HOSPITAL_ADMISSION", time=t_adm),
        SimpleNamespace(code="HEIGHT (INCHES)", time=t_late, numeric_value=70.0),
        SimpleNamespace(code="WEIGHT (LBS)", time=t_late, numeric_value=154.0),
    ]
    db = DummyDB({13: DummySubject(events)})

    struct_vocab = CategoryVocab(name="struct", offset=50, code2id={"<UNK>": 0, "HOSPITAL_ADMISSION": 1})
    struct_enc = SimpleCategoricalEncoder(TokenCategory.STRUCTURAL, struct_vocab)

    tokens = build_subject_timeline(
        db,
        subject_id=13,
        encoders={TokenCategory.STRUCTURAL: struct_enc},
    )
    special_ids = {int(tok.value_id) for tok in tokens if tok.category_id == int(TokenCategory.SPECIAL)}

    assert 43 not in special_ids
    assert 44 not in special_ids
