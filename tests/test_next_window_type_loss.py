import torch


def test_loss_adds_next_window_type_aux_term() -> None:
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

    # No token loss (all padded), but two real windows to form one transition.
    target_ids = torch.zeros((1, 3, 4), dtype=torch.long)  # (B=1,W=3,L=4)
    attention_mask = torch.zeros_like(target_ids)

    window_type_ids = torch.tensor([[0, 2, 1]], dtype=torch.long)  # predict 2 then 1
    window_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    # logits_next_window_type is (B,W,K) and uses positions [:, :-1] to predict next window type.
    K = 3
    logits_next = torch.zeros((1, 3, K), dtype=torch.float)
    logits_next[0, 0, 2] = 10.0  # window 0 predicts window 1 type = 2
    logits_next[0, 1, 1] = 10.0  # window 1 predicts window 2 type = 1

    head_outputs = {"logits_next_window_type": logits_next}
    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "window_type_ids": window_type_ids,
        "window_mask": window_mask,
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={"struct": 0.0, "rvq": 0.0, "meas": 0.0, "med": 0.0, "val": 0.0, "win": 1.0},
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-3
    assert logs["loss_next_window_type"] < 1e-3
    assert logs["acc_next_window_type"] == 1.0

