import torch


def test_loss_adds_window_duration_nll_term() -> None:
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

    B, W, L = 1, 1, 3
    target_ids = torch.zeros((B, W, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, L), dtype=torch.long)
    token_type_ids = torch.ones((B, W, L), dtype=torch.long)  # non-special
    time_ids = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float)

    # True duration is max(time_ids) = 2.0 hours => y = log1p(2).
    mu = torch.log1p(torch.tensor([[2.0]], dtype=torch.float))
    sigma = torch.ones_like(mu)

    head_outputs = {
        "pred_window_dur_mu": mu,
        "pred_window_dur_sigma": sigma,
    }
    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "time_ids": time_ids,
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"time": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_window_dur_nll"] < 1e-6


def test_loss_adds_window_duration_nll_term_for_chunked_semantic_windows() -> None:
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

    B, W, C, L = 1, 1, 2, 2
    target_ids = torch.zeros((B, W, C, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, C, L), dtype=torch.long)
    token_type_ids = torch.ones((B, W, C, L), dtype=torch.long)
    time_ids = torch.tensor([[[[0.0, 1.0], [0.0, 2.0]]]], dtype=torch.float)
    chunk_start_offsets = torch.tensor([[[0.0, 3.0]]], dtype=torch.float)

    # Semantic duration is max(chunk_start_offset + chunk time) = 5.0 hours.
    mu = torch.log1p(torch.tensor([[5.0]], dtype=torch.float))
    sigma = torch.ones_like(mu)

    head_outputs = {
        "pred_window_dur_mu": mu,
        "pred_window_dur_sigma": sigma,
    }
    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "time_ids": time_ids,
        "chunk_start_offsets": chunk_start_offsets,
        "window_mask": torch.tensor([[1]], dtype=torch.long),
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"time": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_window_dur_nll"] < 1e-6
