import torch


def test_loss_routes_by_token_id_ranges_and_uses_meas_head() -> None:
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

    # One token from each lane + one padded token.
    target_ids = torch.tensor([[[0, 105, 207, 1012, 0]]], dtype=torch.long)  # (B=1,W=1,L=5)
    attention_mask = torch.tensor([[[1, 1, 1, 1, 0]]], dtype=torch.long)

    # Build logits that are "correct" for each routed lane.
    B, W, L = target_ids.shape
    logits_struct = torch.zeros(B, W, L, vocab_config["size_special"])
    logits_rvq = torch.zeros(B, W, L, vocab_config["size_rvq"])
    logits_meas = torch.zeros(B, W, L, vocab_config["size_meas_labels"])
    logits_medtok = torch.zeros(B, W, L, vocab_config["size_meds"])

    # Strongly prefer the correct class at each position (only for its lane).
    logits_struct[0, 0, 0, 0] = 50.0  # SPECIAL token 0 -> local 0
    logits_rvq[0, 0, 1, 5] = 50.0  # 105 -> local 5
    logits_meas[0, 0, 2, 7] = 50.0  # 207 -> local 7
    logits_medtok[0, 0, 3, 12] = 50.0  # 1012 -> local 12

    head_outputs = {
        "logits_struct": logits_struct,
        "logits_rvq": logits_rvq,
        "logits_meas": logits_meas,
        "logits_medtok": logits_medtok,
        # Regression head not used in this test
        "pred_values": torch.zeros(B, W, L, 1),
    }

    targets = {
        "input_ids": target_ids,
        "attention_mask": attention_mask,
        "numeric_values": torch.zeros(B, W, L, 1),
        "numeric_mask": torch.zeros(B, W, L, dtype=torch.long),
    }

    criterion = AETLossModule(vocab_config=vocab_config, weights={"struct": 1.0, "rvq": 1.0, "meas": 1.0, "med": 1.0, "val": 0.0})
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-3
    assert logs["loss_meas"] < 1e-3
    assert logs["frac_unrouted"] == 0.0
