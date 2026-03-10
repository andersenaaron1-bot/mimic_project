import torch


def test_window_attention_pooler_respects_mask_and_temperature() -> None:
    from ehr_hier.transformer.encoder import WindowAttentionPooler

    pooler = WindowAttentionPooler(d_model=3, temperature=0.1, dropout=0.0)
    with torch.no_grad():
        pooler.scorer.weight.zero_()
        pooler.scorer.weight[0, 0] = 1.0  # score = first feature
        pooler.scorer.bias.zero_()

    hidden = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [-5.0, 0.0, 0.0],
            ]
        ]
    )  # (N=1, L=4, D=3)

    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    summary = pooler(hidden, mask)
    assert torch.allclose(summary, hidden[:, 2, :], atol=1e-4)

    mask_exclude_best = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    summary2 = pooler(hidden, mask_exclude_best)
    assert torch.allclose(summary2, hidden[:, 1, :], atol=1e-4)


def test_local_encoder_returns_zero_summary_for_padded_windows() -> None:
    from ehr_hier.transformer.embeddings import ContinuousRotaryPositionalEmbedding
    from ehr_hier.transformer.encoder import AETLocalEncoder

    class _Cfg:
        d_model = 4
        num_heads = 1
        d_ff = 8
        num_local_layers = 0
        dropout = 0.0
        summary_pool_temperature = 1.0
        summary_pool_dropout = 0.0
        exclude_special_from_summary = True
        special_type_id = 0

    rope = ContinuousRotaryPositionalEmbedding(dim=_Cfg.d_model // _Cfg.num_heads, max_period=10000.0)
    enc = AETLocalEncoder(_Cfg, rope)

    x = torch.randn(1, 2, 3, _Cfg.d_model)
    times = torch.zeros(1, 2, 3)
    attn = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.long)
    types = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.long)

    _, summaries = enc(x, times, attn, token_type_ids=types)
    assert summaries.shape == (1, 2, 1, _Cfg.d_model)
    assert torch.allclose(summaries[:, 1, 0, :], torch.zeros_like(summaries[:, 1, 0, :]))
