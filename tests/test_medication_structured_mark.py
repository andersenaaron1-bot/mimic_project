import torch


def _build_medication_timeline():
    from src.ehr_hier.data.event_frames import EventPayloadKind, build_event_frame
    from src.ehr_hier.data.token_types import EventToken, TokenCategory

    summary = build_event_frame(
        [
            EventToken(
                value_id=1,
                category_id=int(TokenCategory.SPECIAL),
                t_from_start_hours=0.0,
                dt_from_prev_hours=0.0,
                cat_attrs={},
                num_attrs={},
            )
        ],
        payload_kind=EventPayloadKind.SPECIAL,
    )
    diagnosis = build_event_frame(
        [
            EventToken(
                value_id=1_000_001,
                category_id=int(TokenCategory.DIAGNOSIS),
                t_from_start_hours=1.0,
                dt_from_prev_hours=1.0,
                cat_attrs={"window_type_id": 2},
                num_attrs={},
            )
        ],
        payload_kind=EventPayloadKind.SYMBOLIC_CODE,
    )
    medication = build_event_frame(
        [
            EventToken(
                value_id=1_400_001,
                category_id=int(TokenCategory.MEDICATION),
                t_from_start_hours=2.0,
                dt_from_prev_hours=1.0,
                cat_attrs={
                    "window_type_id": 2,
                    "med_group": 1_670_001,
                    "route": 1_600_001,
                    "form": 1_620_001,
                    "freq": 1_640_001,
                    "unit": 1_660_001,
                    "event_marker": 1,
                },
                num_attrs={
                    "dosage": 0.25,
                    "rate": 0.5,
                    "duration_hours": 0.75,
                },
            )
        ],
        payload_kind=EventPayloadKind.SYMBOLIC_CODE,
        concept_code="NDC//00000-0000",
        group_code="MED_GROUP::FUROSEMIDE",
        semantic_label="FUROSEMIDE",
    )
    return [summary, diagnosis, medication]


def test_collator_emits_medication_structured_mark_targets() -> None:
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4),
    )
    batch = collator([_build_medication_timeline()])

    med_event_idx = 3  # summary, win_type marker, diagnosis, medication, win_end
    assert batch["event_med_group_ids"][0, 0, 0, med_event_idx].item() == 1_670_001
    assert batch["event_med_group_mask"][0, 0, 0, med_event_idx].item() == 1
    assert batch["event_med_route_ids"][0, 0, 0, med_event_idx].item() == 1_600_001
    assert batch["event_med_form_ids"][0, 0, 0, med_event_idx].item() == 1_620_001
    assert batch["event_med_freq_ids"][0, 0, 0, med_event_idx].item() == 1_640_001
    assert batch["event_med_unit_ids"][0, 0, 0, med_event_idx].item() == 1_660_001
    assert batch["event_med_marker_ids"][0, 0, 0, med_event_idx].item() == 1
    assert batch["event_med_dosage_mask"][0, 0, 0, med_event_idx].item() == 1
    assert batch["event_med_dosage_values"][0, 0, 0, med_event_idx, 0].item() == 0.25
    assert batch["event_med_rate_values"][0, 0, 0, med_event_idx, 0].item() == 0.5
    assert batch["event_med_duration_values"][0, 0, 0, med_event_idx, 0].item() == 0.75


def test_loss_adds_medication_structured_mark_terms() -> None:
    from src.ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "total_size": 128,
        "size_special": 32,
        "size_rvq": 8,
        "size_meas_labels": 16,
        "size_meds": 32,
        "offsets": {"SPECIAL": 0, "RVQ": 100, "MEAS": 200, "MED": 1000},
        "sparse_vocab_contract": {
            "families": {
                "med_group": {"offset": 1_670_000, "source_size": 4},
                "med_route": {"offset": 1_600_000, "source_size": 4},
                "med_form": {"offset": 1_620_000, "source_size": 4},
                "med_freq": {"offset": 1_640_000, "source_size": 4},
                "med_unit": {"offset": 1_660_000, "source_size": 4},
            }
        },
    }

    targets = {
        "input_ids": torch.zeros((1, 1, 1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1, 1, 1), dtype=torch.long),
        "token_type_ids": torch.ones((1, 1, 1, 1), dtype=torch.long),
        "event_input_ids": torch.tensor([[[[1, 1_000_001, 1_400_001]]]], dtype=torch.long),
        "event_attention_mask": torch.ones((1, 1, 1, 3), dtype=torch.long),
        "event_type_ids": torch.tensor([[[[0, 2, 4]]]], dtype=torch.long),
        "event_med_group_ids": torch.tensor([[[[0, 0, 1_670_001]]]], dtype=torch.long),
        "event_med_group_mask": torch.tensor([[[[0, 0, 1]]]], dtype=torch.long),
        "event_med_marker_ids": torch.tensor([[[[0, 0, 1]]]], dtype=torch.long),
        "event_med_marker_mask": torch.tensor([[[[0, 0, 1]]]], dtype=torch.long),
        "event_med_dosage_values": torch.tensor([[[[[0.0], [0.0], [0.25]]]]], dtype=torch.float),
        "event_med_dosage_mask": torch.tensor([[[[0, 0, 1]]]], dtype=torch.long),
    }
    head_outputs = {
        "logits_event_med_group": torch.zeros((1, 1, 1, 3, 4), dtype=torch.float32),
        "logits_event_med_marker": torch.zeros((1, 1, 1, 3, 4), dtype=torch.float32),
        "pred_event_med_dosage_mu": torch.zeros((1, 1, 1, 3), dtype=torch.float32),
        "pred_event_med_dosage_sigma": torch.ones((1, 1, 1, 3), dtype=torch.float32),
    }
    # The diagnosis event at index 1 predicts the next medication mark at index 2.
    head_outputs["logits_event_med_group"][0, 0, 0, 1, 1] = 20.0
    head_outputs["logits_event_med_marker"][0, 0, 0, 1, 1] = 20.0
    head_outputs["pred_event_med_dosage_mu"][0, 0, 0, 1] = 0.25

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={
            "event_med_group": 1.0,
            "event_med_marker": 1.0,
            "event_med_dosage": 1.0,
        },
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_event_med_group"] < 1e-6
    assert logs["loss_event_med_marker"] < 1e-6
    assert logs["loss_event_med_dosage_nll"] < 1e-6
    assert logs["n_event_med_group_supervised"] == 1
    assert logs["n_event_med_marker_supervised"] == 1
    assert logs["n_event_med_dosage_supervised"] == 1
