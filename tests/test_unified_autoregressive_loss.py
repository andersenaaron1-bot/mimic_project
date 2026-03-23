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


def test_unified_token_logs_family_stratified_metrics() -> None:
    from ehr_hier.data.token_types import TokenCategory
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "total_size": 54,
        "size_special": 16,
        "size_rvq": 4,
        "size_meas_labels": 16,
        "size_meds": 18,
        "offsets": {"SPECIAL": 0, "RVQ": 16, "MEAS": 20, "MED": 36},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
        "dense_blocks": [
            {
                "name": "special",
                "dense_offset": 0,
                "dense_size": 16,
                "global_offset": 0,
                "sparse_global_ids": list(range(16)),
            },
            {
                "name": "structural",
                "dense_offset": 16,
                "dense_size": 4,
                "global_offset": 2_200_000,
                "sparse_global_ids": [2_200_000, 2_200_001, 2_200_002, 2_200_003],
            },
            {
                "name": "measurement_code",
                "dense_offset": 20,
                "dense_size": 4,
                "global_offset": 2_000_000,
                "sparse_global_ids": [2_000_000, 2_000_001, 2_000_002, 2_000_003],
            },
            {
                "name": "measurement_value",
                "dense_offset": 24,
                "dense_size": 4,
                "global_offset": 2_100_000,
                "sparse_global_ids": [2_100_000, 2_100_001, 2_100_002, 2_100_003],
            },
            {
                "name": "observation_code",
                "dense_offset": 28,
                "dense_size": 4,
                "global_offset": 2_300_000,
                "sparse_global_ids": [2_300_000, 2_300_001, 2_300_002, 2_300_003],
            },
            {
                "name": "observation_value",
                "dense_offset": 32,
                "dense_size": 4,
                "global_offset": 2_320_000,
                "sparse_global_ids": [2_320_000, 2_320_001, 2_320_002, 2_320_003],
            },
            {
                "name": "diagnosis",
                "dense_offset": 36,
                "dense_size": 3,
                "global_offset": 1_000_000,
                "sparse_global_ids": [1_000_000, 1_000_001, 1_000_002],
            },
            {
                "name": "diagnosis_residual",
                "dense_offset": 39,
                "dense_size": 3,
                "global_offset": 1_100_000,
                "sparse_global_ids": [1_100_000, 1_100_001, 1_100_002],
            },
            {
                "name": "procedure",
                "dense_offset": 42,
                "dense_size": 3,
                "global_offset": 1_200_000,
                "sparse_global_ids": [1_200_000, 1_200_001, 1_200_002],
            },
            {
                "name": "procedure_residual",
                "dense_offset": 45,
                "dense_size": 3,
                "global_offset": 1_300_000,
                "sparse_global_ids": [1_300_000, 1_300_001, 1_300_002],
            },
            {
                "name": "medication",
                "dense_offset": 48,
                "dense_size": 3,
                "global_offset": 1_400_000,
                "sparse_global_ids": [1_400_000, 1_400_001, 1_400_002],
            },
            {
                "name": "medication_residual",
                "dense_offset": 51,
                "dense_size": 3,
                "global_offset": 1_500_000,
                "sparse_global_ids": [1_500_000, 1_500_001, 1_500_002],
            },
        ],
        "sparse_vocab_contract": {
            "families": {
                "special": {"offset": 0},
                "structural": {"offset": 2_200_000},
                "measurement_code": {"offset": 2_000_000},
                "measurement_value": {"offset": 2_100_000},
                "observation_code": {"offset": 2_300_000},
                "observation_value": {"offset": 2_320_000},
                "diagnosis": {"offset": 1_000_000},
                "diagnosis_residual": {"offset": 1_100_000},
                "procedure": {"offset": 1_200_000},
                "procedure_residual": {"offset": 1_300_000},
                "medication": {"offset": 1_400_000},
                "medication_residual": {"offset": 1_500_000},
            }
        },
    }

    target_ids = torch.tensor(
        [[[[2, 11, 37, 40, 43, 46, 49, 52, 21, 25, 29, 33, 17, 0]]]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(target_ids, dtype=torch.long)
    token_type_ids = torch.tensor(
        [[[
            [
                int(TokenCategory.SPECIAL),
                int(TokenCategory.SPECIAL),
                int(TokenCategory.DIAGNOSIS),
                int(TokenCategory.DIAGNOSIS),
                int(TokenCategory.PROCEDURE),
                int(TokenCategory.PROCEDURE),
                int(TokenCategory.MEDICATION),
                int(TokenCategory.MEDICATION),
                int(TokenCategory.MEASUREMENT),
                int(TokenCategory.MEASUREMENT),
                int(TokenCategory.MEASUREMENT),
                int(TokenCategory.MEASUREMENT),
                int(TokenCategory.STRUCTURAL),
                int(TokenCategory.DIAGNOSIS),
            ]
        ]]],
        dtype=torch.long,
    )

    logits_token = torch.zeros((1, 1, 1, target_ids.shape[-1], vocab_config["total_size"]), dtype=torch.float)
    for pos in range(target_ids.shape[-1] - 1):
        logits_token[0, 0, 0, pos, int(target_ids[0, 0, 0, pos + 1].item())] = 20.0

    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={"token": 1.0, "val": 0.0},
    )
    loss, logs = criterion(
        {"logits_token": logits_token, "pred_values": torch.zeros((1, 1, 1, target_ids.shape[-1], 1), dtype=torch.float)},
        {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "numeric_values": torch.zeros((1, 1, 1, target_ids.shape[-1], 1), dtype=torch.float),
            "numeric_mask": torch.zeros((1, 1, 1, target_ids.shape[-1]), dtype=torch.long),
        },
    )

    assert loss.item() < 1e-3
    assert logs["n_token_supervised"] == 13
    assert logs["candidate_nonmarker_special_targets"] == 0
    assert logs["acc_token"] == 1.0

    for group_name in (
        "special_marker",
        "diagnosis",
        "diagnosis_residual",
        "procedure",
        "procedure_residual",
        "medication",
        "medication_residual",
        "measurement_code",
        "measurement_value",
        "observation_code",
        "observation_value",
        "structural",
        "unk",
    ):
        assert logs[f"n_token_family_{group_name}"] == 1
        assert logs[f"acc_token_family_{group_name}"] == 1.0
        assert logs[f"loss_token_family_{group_name}"] < 1e-3


def test_unified_token_family_weights_affect_optimized_loss_not_reported_accuracy() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "total_size": 18,
        "size_special": 16,
        "size_rvq": 0,
        "size_meas_labels": 0,
        "size_meds": 2,
        "offsets": {"SPECIAL": 0, "RVQ": 16, "MEAS": 16, "MED": 16},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
        "dense_blocks": [
            {
                "name": "special",
                "dense_offset": 0,
                "dense_size": 16,
                "global_offset": 0,
                "sparse_global_ids": list(range(16)),
            },
            {
                "name": "structural",
                "dense_offset": 16,
                "dense_size": 2,
                "global_offset": 2_200_000,
                "sparse_global_ids": [2_200_000, 2_200_001],
            },
        ],
        "sparse_vocab_contract": {
            "families": {
                "special": {"offset": 0},
                "structural": {"offset": 2_200_000},
            }
        },
    }

    target_ids = torch.tensor([[[[2, 11, 17]]]], dtype=torch.long)
    attention_mask = torch.tensor([[[[1, 1, 1]]]], dtype=torch.long)
    token_type_ids = torch.tensor([[[[0, 0, 5]]]], dtype=torch.long)
    logits_token = torch.zeros((1, 1, 1, 3, 18), dtype=torch.float)
    logits_token[0, 0, 0, 0, 0] = 10.0
    logits_token[0, 0, 0, 1, 17] = 20.0

    baseline = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={"token": 1.0, "val": 0.0},
    )
    weighted = AETLossModule(
        vocab_config=vocab_config,
        token_family_weights={"special_marker": 5.0},
        strict_routing=True,
        weights={"token": 1.0, "val": 0.0},
    )

    batch = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "numeric_values": torch.zeros((1, 1, 1, 3, 1), dtype=torch.float),
        "numeric_mask": torch.zeros((1, 1, 1, 3), dtype=torch.long),
    }
    head_outputs = {
        "logits_token": logits_token,
        "pred_values": torch.zeros((1, 1, 1, 3, 1), dtype=torch.float),
    }

    loss_base, logs_base = baseline(head_outputs, batch)
    loss_weighted, logs_weighted = weighted(head_outputs, batch)

    assert logs_base["acc_token"] == logs_weighted["acc_token"]
    assert logs_base["loss_token"] == logs_weighted["loss_token"]
    assert logs_weighted["loss_token_weighted"] > logs_base["loss_token"]
    assert loss_weighted.item() > loss_base.item()
