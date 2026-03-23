import torch


def test_window_type_segment_embedding_excludes_special_tokens() -> None:
    from ehr_hier.transformer.embeddings import AETEmbeddings

    emb = AETEmbeddings(
        vocab_size=8,
        d_model=4,
        dropout=0.0,
        num_window_types=3,
        special_type_id=0,
        exclude_special_from_window_type=True,
    )
    emb.layer_norm = torch.nn.Identity()

    with torch.no_grad():
        emb.token_embedding.weight.zero_()
        # value_encoder = Linear(1->D) + Tanh, force it to emit zeros
        emb.value_encoder[0].weight.zero_()
        emb.value_encoder[0].bias.zero_()
        emb.window_type_embedding.weight.zero_()
        emb.window_type_embedding.weight[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        emb.window_type_embedding.weight[2] = torch.tensor([-1.0, 0.0, 1.0, 2.0])

    B, W, L = 1, 2, 3
    input_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.long)
    numeric_values = torch.zeros((B, W, L, 1), dtype=torch.float)
    window_type_ids = torch.tensor([[1, 2]], dtype=torch.long)
    token_type_ids = torch.tensor([[[0, 1, 0], [1, 1, 0]]], dtype=torch.long)

    out = emb(input_ids, numeric_values, window_type_ids=window_type_ids, token_type_ids=token_type_ids)

    # Window 0 type=1: only token 1 is non-special
    assert torch.allclose(out[0, 0, 0], torch.zeros(4))
    assert torch.allclose(out[0, 0, 1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(out[0, 0, 2], torch.zeros(4))

    # Window 1 type=2: tokens 0 and 1 are non-special
    assert torch.allclose(out[0, 1, 0], torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert torch.allclose(out[0, 1, 1], torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert torch.allclose(out[0, 1, 2], torch.zeros(4))


def test_numeric_value_projection_respects_explicit_mask() -> None:
    from ehr_hier.transformer.embeddings import AETEmbeddings

    emb = AETEmbeddings(
        vocab_size=8,
        d_model=4,
        dropout=0.0,
        num_window_types=0,
        special_type_id=0,
        exclude_special_from_window_type=True,
    )
    emb.layer_norm = torch.nn.Identity()

    with torch.no_grad():
        emb.token_embedding.weight.zero_()
        emb.value_encoder[0].weight.fill_(1.0)
        emb.value_encoder[0].bias.fill_(0.25)

    input_ids = torch.tensor([[[1, 2]]], dtype=torch.long)
    numeric_values = torch.tensor([[[[2.0], [7.0]]]], dtype=torch.float)
    numeric_mask = torch.tensor([[[1, 0]]], dtype=torch.long)
    token_type_ids = torch.tensor([[[1, 0]]], dtype=torch.long)

    out = emb(
        input_ids,
        numeric_values,
        numeric_mask=numeric_mask,
        token_type_ids=token_type_ids,
    )

    expected_numeric = torch.tanh(torch.tensor([2.25, 2.25, 2.25, 2.25]))
    assert torch.allclose(out[0, 0, 0], expected_numeric)
    assert torch.allclose(out[0, 0, 1], torch.zeros(4))
