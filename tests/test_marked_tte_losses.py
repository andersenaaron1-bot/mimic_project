import torch


def test_loss_adds_event_numeric_value_nll_term() -> None:
    from ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "size_special": 10,
        "size_rvq": 20,
        "size_meas_labels": 30,
        "size_meds": 40,
        "offsets": {
            "SPECIAL": 0,
            "RVQ": 100,
            "MEAS": 200,
            "MED": 1000,
        },
    }

    numeric_payload = int(EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"])
    special_payload = int(EVENT_PAYLOAD_KIND_TO_ID["special"])

    event_attention_mask = torch.ones((1, 1, 1, 4), dtype=torch.long)
    event_type_ids = torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.long)
    event_payload_ids = torch.tensor(
        [[[[special_payload, numeric_payload, numeric_payload, special_payload]]]],
        dtype=torch.long,
    )
    event_numeric_values = torch.tensor([[[[[0.0], [1.25], [2.5], [0.0]]]]], dtype=torch.float)
    event_numeric_mask = torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.long)

    head_outputs = {
        "pred_event_value_mu": torch.tensor([[[[0.0, 2.5, 0.0, 0.0]]]], dtype=torch.float),
        "pred_event_value_sigma": torch.ones((1, 1, 1, 4), dtype=torch.float),
    }
    targets = {
        "input_ids": torch.zeros((1, 1, 1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1, 1, 1), dtype=torch.long),
        "token_type_ids": torch.ones((1, 1, 1, 1), dtype=torch.long),
        "event_attention_mask": event_attention_mask,
        "event_type_ids": event_type_ids,
        "event_payload_ids": event_payload_ids,
        "event_numeric_values": event_numeric_values,
        "event_numeric_mask": event_numeric_mask,
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"event_value": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_event_value_nll"] < 1e-6
    assert logs["n_event_value_supervised"] == 1


def test_loss_adds_next_window_gap_nll_term() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "size_special": 10,
        "size_rvq": 20,
        "size_meas_labels": 30,
        "size_meds": 40,
        "offsets": {
            "SPECIAL": 0,
            "RVQ": 100,
            "MEAS": 200,
            "MED": 1000,
        },
    }

    window_start_times = torch.tensor([[0.0, 10.0]], dtype=torch.float)
    semantic_duration_hours = torch.tensor([[4.0, 3.0]], dtype=torch.float)
    window_mask = torch.tensor([[1, 1]], dtype=torch.long)

    head_outputs = {
        "pred_next_window_gap_mu": torch.log1p(torch.tensor([[6.0, 0.0]], dtype=torch.float)),
        "pred_next_window_gap_sigma": torch.ones((1, 2), dtype=torch.float),
    }
    targets = {
        "input_ids": torch.zeros((1, 2, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 2, 1), dtype=torch.long),
        "token_type_ids": torch.ones((1, 2, 1), dtype=torch.long),
        "window_start_times": window_start_times,
        "semantic_duration_hours": semantic_duration_hours,
        "window_mask": window_mask,
        "semantic_token_counts": torch.tensor([[1.0, 1.0]], dtype=torch.float),
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"next_window_gap": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_next_window_gap_nll"] < 1e-6
    assert logs["n_next_window_gap_supervised"] == 1


def test_loss_adds_next_window_duration_nll_term() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "size_special": 10,
        "size_rvq": 20,
        "size_meas_labels": 30,
        "size_meds": 40,
        "offsets": {
            "SPECIAL": 0,
            "RVQ": 100,
            "MEAS": 200,
            "MED": 1000,
        },
    }

    semantic_duration_hours = torch.tensor([[4.0, 3.0]], dtype=torch.float)
    window_mask = torch.tensor([[1, 1]], dtype=torch.long)

    head_outputs = {
        "pred_next_window_duration_mu": torch.log1p(torch.tensor([[3.0, 0.0]], dtype=torch.float)),
        "pred_next_window_duration_sigma": torch.ones((1, 2), dtype=torch.float),
    }
    targets = {
        "input_ids": torch.zeros((1, 2, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 2, 1), dtype=torch.long),
        "token_type_ids": torch.ones((1, 2, 1), dtype=torch.long),
        "semantic_duration_hours": semantic_duration_hours,
        "window_mask": window_mask,
        "semantic_token_counts": torch.tensor([[1.0, 1.0]], dtype=torch.float),
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"next_window_duration": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_next_window_duration_nll"] < 1e-6
    assert logs["n_next_window_duration_supervised"] == 1


def test_loss_adds_next_window_support_bce_term() -> None:
    from ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from ehr_hier.data.token_types import TokenCategory
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "size_special": 10,
        "size_rvq": 20,
        "size_meas_labels": 30,
        "size_meds": 40,
        "offsets": {
            "SPECIAL": 0,
            "RVQ": 100,
            "MEAS": 200,
            "MED": 1000,
        },
    }

    numeric_payload = int(EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"])
    symbolic_payload = int(EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"])
    special_payload = int(EVENT_PAYLOAD_KIND_TO_ID["special"])

    head_outputs = {
        "logits_next_window_support": torch.tensor(
            [[[-20.0, 20.0, 20.0, -20.0, -20.0, -20.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float,
        )
    }
    targets = {
        "input_ids": torch.zeros((1, 2, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 2, 1), dtype=torch.long),
        "token_type_ids": torch.ones((1, 2, 1), dtype=torch.long),
        "window_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "event_type_ids": torch.tensor(
            [[
                [[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)]],
                [[int(TokenCategory.DIAGNOSIS), int(TokenCategory.PROCEDURE)]],
            ]],
            dtype=torch.long,
        ),
        "event_payload_ids": torch.tensor(
            [[
                [[special_payload, numeric_payload]],
                [[symbolic_payload, symbolic_payload]],
            ]],
            dtype=torch.long,
        ),
        "event_attention_mask": torch.ones((1, 2, 1, 2), dtype=torch.long),
        "event_memory_chronic_flags": torch.tensor(
            [[[[0, 0]], [[1, 0]]]],
            dtype=torch.long,
        ),
        "event_numeric_values": torch.tensor(
            [[[[[0.0], [0.1]]], [[[0.0], [0.0]]]]],
            dtype=torch.float,
        ),
        "event_numeric_mask": torch.tensor(
            [[[[0, 1]], [[0, 0]]]],
            dtype=torch.long,
        ),
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"next_window_support": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_next_window_support_bce"] < 1e-6
    assert logs["n_next_window_support_supervised"] == 1
    assert logs["acc_next_window_support"] == 1.0
