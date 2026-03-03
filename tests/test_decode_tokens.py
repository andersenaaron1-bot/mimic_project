from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.decode_tokens import (
    decode_measurement_tokens,
    decode_timeline_tokens,
)


def test_decode_measurement_tokens_respects_num_codebooks():
    tokens = [
        EventToken(2000005, int(TokenCategory.MEASUREMENT), 1.0, 1.0, {}, {"z": 0.25}),
        EventToken(2100007, int(TokenCategory.MEASUREMENT), 1.0, 0.0, {"codebook": 0}, {}),
        EventToken(2100260, int(TokenCategory.MEASUREMENT), 1.0, 0.0, {"codebook": 1}, {}),
        EventToken(2000009, int(TokenCategory.MEASUREMENT), 2.0, 1.0, {}, {"z": -0.5}),
        EventToken(2100011, int(TokenCategory.MEASUREMENT), 2.0, 0.0, {"codebook": 0}, {}),
        EventToken(2100280, int(TokenCategory.MEASUREMENT), 2.0, 0.0, {"codebook": 1}, {}),
    ]

    decoded = decode_measurement_tokens(
        tokens,
        code_token_offset=2000000,
        rvq_token_offset=2100000,
        rvq_codebook_stride=256,
        num_codebooks=2,
        code2name={5: "LAB//A", 9: "LAB//B"},
    )

    assert len(decoded) == 2
    assert decoded[0]["var_id"] == 5
    assert decoded[0]["var_name"] == "LAB//A"
    assert decoded[0]["rvq_indices"] == [7, 4]
    assert decoded[1]["var_id"] == 9
    assert decoded[1]["rvq_indices"] == [11, 24]


def test_decode_timeline_tokens_labels_med_markers_and_structural():
    tokens = [
        EventToken(1, int(TokenCategory.SPECIAL), 0.0, 0.0, {}, {}),
        EventToken(1400003, int(TokenCategory.MEDICATION), 1.0, 1.0, {}, {}),
        EventToken(1400000, int(TokenCategory.MEDICATION), 1.0, 0.0, {"event_marker": 1}, {}),
        EventToken(2200002, int(TokenCategory.STRUCTURAL), 2.0, 1.0, {"struct_label_id": 2}, {}),
    ]

    decoded = decode_timeline_tokens(
        tokens,
        medication_offset=1400000,
        medication_id2code={3: "NDC//123"},
        structural_offset=2200000,
        structural_id2label={2: "STRUCT_START_MECH"},
        special_id2name={1: "PT_CLS"},
    )

    assert decoded[0]["label"] == "PT_CLS"
    assert decoded[1]["label"] == "NDC//123"
    assert decoded[2]["label"] == "MED_MARKER::START"
    assert decoded[3]["label"] == "STRUCT_START_MECH"


def test_decode_timeline_tokens_decodes_raw_structural_code_ids():
    tokens = [
        EventToken(2200007, int(TokenCategory.STRUCTURAL), 2.0, 1.0, {}, {}),
    ]

    decoded = decode_timeline_tokens(
        tokens,
        structural_offset=2200000,
        structural_id2code={7: "TRANSFER_TO//ED//Emergency Department"},
    )

    assert decoded[0]["label"] == "TRANSFER_TO//ED//Emergency Department"
    assert decoded[0]["raw_code"] == "TRANSFER_TO//ED//Emergency Department"


def test_decode_timeline_tokens_does_not_misread_rvq_as_code_token():
    tokens = [
        EventToken(2100007, int(TokenCategory.MEASUREMENT), 1.0, 0.0, {}, {}),
    ]

    decoded = decode_timeline_tokens(
        tokens,
        code_token_offset=2000000,
        rvq_token_offset=2100000,
        rvq_codebook_stride=256,
        measurement_num_codebooks=2,
    )

    assert decoded[0]["kind"] == "token"
    assert decoded[0]["value_id"] == 2100007
