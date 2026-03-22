import torch


def test_transition_boundary_losses_use_chunk_end_positions_only() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "size_special": 32,
        "size_rvq": 8,
        "size_meas_labels": 8,
        "size_meds": 8,
        "offsets": {
            "SPECIAL": 0,
            "RVQ": 100,
            "MEAS": 200,
            "MED": 1000,
        },
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
    }

    # (B=1, W=1, C=1, L=4)
    # idx0: WIN_TYPE prefix
    # idx2: last content token, which should carry the auxiliary supervision
    # idx3: WIN_TYPE suffix marker, which must NOT be supervised directly
    target_ids = torch.tensor([[[[11, 6, 7, 12]]]], dtype=torch.long)
    attention_mask = torch.tensor([[[[1, 1, 1, 1]]]], dtype=torch.long)

    logits_struct = torch.zeros((1, 1, 1, 4, 32), dtype=torch.float)
    logits_struct[0, 0, 0, 1, 6] = 20.0
    logits_struct[0, 0, 0, 2, 7] = 20.0
    # Deliberately wrong on marker tokens; marker positions must be excluded from struct CE.
    logits_struct[0, 0, 0, 0, 0] = 20.0
    logits_struct[0, 0, 0, 3, 0] = 20.0

    logits_transition = torch.zeros((1, 1, 1, 4, 2), dtype=torch.float)
    # The last content token predicts an END transition because the next token is a suffix type marker.
    logits_transition[0, 0, 0, 2, 1] = 20.0

    logits_boundary_type = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float)
    # The last content token predicts the next window type carried by the suffix marker id=12 -> class 2.
    logits_boundary_type[0, 0, 0, 2, 2] = 20.0

    head_outputs = {
        "logits_struct": logits_struct,
        "logits_transition_boundary": logits_transition,
        "logits_boundary_next_window_type": logits_boundary_type,
    }
    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=True,
        weights={
            "struct": 1.0,
            "rvq": 0.0,
            "meas": 0.0,
            "med": 0.0,
            "val": 0.0,
            "transition": 1.0,
            "win_boundary": 1.0,
            "win": 0.0,
            "len": 0.0,
            "chunk": 0.0,
            "time": 0.0,
            "dt": 0.0,
        },
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-3
    assert logs["loss_struct"] < 1e-3
    assert logs["loss_transition_boundary"] < 1e-3
    assert logs["loss_next_window_type_boundary"] < 1e-3
    assert logs["n_transition_boundary_supervised"] == 1
    assert logs["n_next_window_type_boundary_supervised"] == 1


def test_boundary_next_window_type_loss_is_supervised_in_end_token_mode() -> None:
    from ehr_hier.transformer.loss import AETLossModule

    vocab_config = {
        "size_special": 32,
        "size_rvq": 8,
        "size_meas_labels": 8,
        "size_meds": 8,
        "offsets": {
            "SPECIAL": 0,
            "RVQ": 100,
            "MEAS": 200,
            "MED": 1000,
        },
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
            "end_mode": "end_token",
        },
    }

    # Two semantic windows, each with one chunk:
    # window 0: WIN_TYPE(1), content, WIN_END
    # window 1: WIN_TYPE(2), content, WIN_END
    target_ids = torch.tensor([[[[11, 6, 7, 14]], [[12, 8, 9, 14]]]], dtype=torch.long)
    attention_mask = torch.ones_like(target_ids)
    window_type_ids = torch.tensor([[1, 2]], dtype=torch.long)
    window_mask = torch.tensor([[1, 1]], dtype=torch.long)

    logits_boundary_type = torch.zeros((1, 2, 1, 4, 4), dtype=torch.float)
    # Window 0's last content token should predict window 1 type = 2.
    logits_boundary_type[0, 0, 0, 2, 2] = 20.0

    criterion = AETLossModule(
        vocab_config=vocab_config,
        strict_routing=False,
        weights={
            "struct": 0.0,
            "rvq": 0.0,
            "meas": 0.0,
            "med": 0.0,
            "val": 0.0,
            "transition": 0.0,
            "win_boundary": 1.0,
            "win": 0.0,
            "len": 0.0,
            "chunk": 0.0,
            "time": 0.0,
            "dt": 0.0,
        },
    )
    loss, logs = criterion(
        {"logits_boundary_next_window_type": logits_boundary_type},
        {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "window_type_ids": window_type_ids,
            "window_mask": window_mask,
            "token_type_ids": torch.tensor([[[[0, 1, 1, 0]], [[0, 1, 1, 0]]]], dtype=torch.long),
        },
    )

    assert loss.item() < 1e-3
    assert logs["loss_next_window_type_boundary"] < 1e-3
    assert logs["n_next_window_type_boundary_supervised"] == 1
