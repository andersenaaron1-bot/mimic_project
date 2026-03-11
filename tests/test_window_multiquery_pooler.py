import torch


def test_window_multiquery_pooler_can_select_different_tokens_per_query() -> None:
    from ehr_hier.transformer.encoder import WindowMultiQueryPooler

    pooler = WindowMultiQueryPooler(d_model=2, num_queries=2, temperature=0.01, dropout=0.0)
    with torch.no_grad():
        pooler.scorer.weight.zero_()
        pooler.scorer.bias.zero_()
        # query 0 scores feature 0; query 1 scores feature 1
        pooler.scorer.weight[0, 0] = 1.0
        pooler.scorer.weight[1, 1] = 1.0

    hidden = torch.tensor([[[10.0, 0.0], [0.0, 9.0], [1.0, 1.0]]])  # (N=1, L=3, D=2)
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    summaries = pooler(hidden, mask)  # (N, Q, D)
    assert summaries.shape == (1, 2, 2)
    # query 0 should pick token 0, query 1 should pick token 1
    assert torch.allclose(summaries[0, 0], hidden[0, 0], atol=1e-4)
    assert torch.allclose(summaries[0, 1], hidden[0, 1], atol=1e-4)

