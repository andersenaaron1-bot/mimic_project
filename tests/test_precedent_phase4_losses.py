import torch


def test_loss_adds_precedent_phase4_terms() -> None:
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

    future_dim = 12
    d_model = 8
    target_future = torch.linspace(0.1, 1.2, future_dim, dtype=torch.float32).view(1, 1, future_dim)
    target_embedding = torch.linspace(0.2, 1.6, d_model, dtype=torch.float32).view(1, 1, d_model)

    head_outputs = {
        "precedent_future_summary": target_future.clone(),
        "precedent_future_embedding": target_embedding.clone(),
        "precedent_retrieval_scores": torch.tensor([[[20.0, -20.0]]], dtype=torch.float32),
        "precedent_candidate_future_summaries": torch.stack(
            [
                target_future.clone(),
                torch.zeros_like(target_future),
            ],
            dim=2,
        ),
        "precedent_candidate_future_embeddings": torch.stack(
            [
                target_embedding.clone(),
                torch.zeros_like(target_embedding),
            ],
            dim=2,
        ),
        "precedent_matched_item_ids": torch.tensor([[[5, 6]]], dtype=torch.long),
        "precedent_target_future_summary": target_future.clone(),
        "precedent_target_future_embedding": target_embedding.clone(),
        "precedent_target_future_mask": torch.tensor([[1]], dtype=torch.long),
        "precedent_anchor_item_ids": torch.tensor([[5]], dtype=torch.long),
    }
    targets = {
        "input_ids": torch.zeros((1, 1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1, 1), dtype=torch.long),
        "token_type_ids": torch.ones((1, 1, 1), dtype=torch.long),
    }

    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={
            "precedent_future": 1.0,
            "precedent_contrast": 1.0,
            "precedent_anchor": 0.25,
        },
        strict_routing=False,
    )
    loss, logs = criterion(head_outputs, targets)

    assert loss.item() < 1e-4
    assert logs["loss_precedent_future"] < 1e-6
    assert logs["loss_precedent_contrast"] < 1e-6
    assert logs["loss_precedent_anchor"] < 1e-6
    assert logs["n_precedent_future_supervised"] == 1
    assert logs["n_precedent_contrast_supervised"] == 1
    assert logs["n_precedent_anchor_supervised"] == 1
    assert logs["acc_precedent_contrast"] == 1.0
    assert logs["acc_precedent_anchor"] == 1.0
