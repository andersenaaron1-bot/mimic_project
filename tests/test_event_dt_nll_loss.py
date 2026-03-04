import torch


def test_loss_adds_event_dt_nll_term() -> None:
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
    token_type_ids = torch.ones((B, W, L), dtype=torch.long)  # all content
    time_ids = torch.tensor([[[0.0, 1.0, 3.0]]], dtype=torch.float)

    # dt targets for content tokens:
    #   i=0 -> 1h, i=1 -> 2h, i=2 -> none (masked out)
    y0 = torch.log1p(torch.tensor(1.0))
    y1 = torch.log1p(torch.tensor(2.0))
    mu = torch.tensor([[[y0, y1, 0.0]]], dtype=torch.float)
    sigma = torch.ones_like(mu)

    head_outputs = {
        "pred_dt_next_mu": mu,
        "pred_dt_next_sigma": sigma,
    }
    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "time_ids": time_ids,
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"dt": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_dt_nll"] < 1e-6


def test_loss_adds_event_dt_nll_term_across_chunk_boundaries() -> None:
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

    # Flattened semantic times are [0, 1, 3, 5], so dt targets are [1, 2, 2, none].
    mu = torch.log1p(torch.tensor([[[[1.0, 2.0], [2.0, 0.0]]]], dtype=torch.float))
    sigma = torch.ones_like(mu)

    head_outputs = {
        "pred_dt_next_mu": mu,
        "pred_dt_next_sigma": sigma,
    }
    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "time_ids": time_ids,
        "chunk_start_offsets": chunk_start_offsets,
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"dt": 1.0},
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-6
    assert logs["loss_dt_nll"] < 1e-6
