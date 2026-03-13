import torch


def test_unified_token_loss_shifts_targets_and_ignores_nonmarker_specials() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "total_size": 32,
        "size_special": 16,
        "size_rvq": 4,
        "size_meas_labels": 4,
        "size_meds": 8,
        "offsets": {"SPECIAL": 0, "RVQ": 100, "MEAS": 200, "MED": 1000},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
    }

    # [demo1, demo2, WIN_TYPE(=11), content(=6), WIN_END(=14)]
    target_ids = torch.tensor([[[[2, 3, 11, 6, 14]]]], dtype=torch.long)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 1]]]], dtype=torch.long)
    token_type_ids = torch.tensor([[[[0, 0, 0, 1, 0]]]], dtype=torch.long)

    logits_token = torch.zeros((1, 1, 1, 5, 32), dtype=torch.float)
    # pos0 -> target demo2 should be ignored
    logits_token[0, 0, 0, 0, 0] = 20.0
    # pos1 predicts WIN_TYPE marker
    logits_token[0, 0, 0, 1, 11] = 20.0
    # pos2 predicts content
    logits_token[0, 0, 0, 2, 6] = 20.0
    # pos3 predicts WIN_END marker
    logits_token[0, 0, 0, 3, 14] = 20.0

    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={"token": 1.0, "val": 0.0},
    )
    loss, logs = criterion(
        {"logits_token": logits_token, "pred_values": torch.zeros((1, 1, 1, 5, 1), dtype=torch.float)},
        {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "numeric_values": torch.zeros((1, 1, 1, 5, 1), dtype=torch.float),
            "numeric_mask": torch.zeros((1, 1, 1, 5), dtype=torch.long),
        },
    )

    assert loss.item() < 1e-3
    assert logs["loss_token"] < 1e-3
    assert logs["n_token_supervised"] == 3
    assert logs["candidate_nonmarker_special_targets"] == 1
    assert logs["ignored_nonmarker_special_targets"] == 1
    assert logs["acc_token"] == 1.0


def test_unified_value_loss_predicts_next_numeric_value() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "total_size": 32,
        "size_special": 16,
        "size_rvq": 4,
        "size_meas_labels": 4,
        "size_meds": 8,
        "offsets": {"SPECIAL": 0, "RVQ": 100, "MEAS": 200, "MED": 1000},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
    }

    target_ids = torch.tensor([[[[11, 20, 21]]]], dtype=torch.long)
    attention_mask = torch.tensor([[[[1, 1, 1]]]], dtype=torch.long)
    token_type_ids = torch.tensor([[[[0, 1, 1]]]], dtype=torch.long)
    numeric_values = torch.tensor([[[[[0.0], [1.5], [2.5]]]]], dtype=torch.float)
    numeric_mask = torch.tensor([[[[0, 1, 1]]]], dtype=torch.long)

    logits_token = torch.zeros((1, 1, 1, 3, 32), dtype=torch.float)
    logits_token[0, 0, 0, 0, 20] = 20.0
    logits_token[0, 0, 0, 1, 21] = 20.0
    pred_values = torch.zeros((1, 1, 1, 3, 1), dtype=torch.float)
    pred_values[0, 0, 0, 0, 0] = 1.5
    pred_values[0, 0, 0, 1, 0] = 2.5

    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={"token": 1.0, "val": 1.0},
    )
    loss, logs = criterion(
        {"logits_token": logits_token, "pred_values": pred_values},
        {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
        },
    )

    assert loss.item() < 1e-3
    assert logs["loss_token"] < 1e-3
    assert logs["loss_val"] < 1e-6


def test_unified_value_loss_is_skipped_when_weight_zero() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "total_size": 32,
        "size_special": 16,
        "size_rvq": 4,
        "size_meas_labels": 4,
        "size_meds": 8,
        "offsets": {"SPECIAL": 0, "RVQ": 100, "MEAS": 200, "MED": 1000},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
    }

    target_ids = torch.tensor([[[[11, 20, 21]]]], dtype=torch.long)
    attention_mask = torch.tensor([[[[1, 1, 1]]]], dtype=torch.long)
    token_type_ids = torch.tensor([[[[0, 1, 1]]]], dtype=torch.long)
    numeric_values = torch.tensor([[[[[0.0], [1.5], [2.5]]]]], dtype=torch.float)
    numeric_mask = torch.tensor([[[[0, 1, 1]]]], dtype=torch.long)

    logits_token = torch.zeros((1, 1, 1, 3, 32), dtype=torch.float)
    logits_token[0, 0, 0, 0, 20] = 20.0
    logits_token[0, 0, 0, 1, 21] = 20.0
    pred_values = torch.randn((1, 1, 1, 3, 1), dtype=torch.float)

    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={"token": 1.0, "val": 0.0},
    )
    _, logs = criterion(
        {"logits_token": logits_token, "pred_values": pred_values},
        {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
        },
    )

    assert "loss_val" not in logs
