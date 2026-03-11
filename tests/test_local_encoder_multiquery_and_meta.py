import torch


def test_local_encoder_multiquery_summary_and_meta_keep_padded_windows_zero() -> None:
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
        summary_num_queries = 4
        enable_window_meta = True

    rope = ContinuousRotaryPositionalEmbedding(dim=_Cfg.d_model // _Cfg.num_heads, max_period=10000.0)
    enc = AETLocalEncoder(_Cfg, rope)

    x = torch.randn(1, 2, 3, _Cfg.d_model)
    times = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]], dtype=torch.float)
    attn = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.long)
    types = torch.tensor([[[0, 1, 5], [0, 0, 0]]], dtype=torch.long)

    _, summaries = enc(x, times, attn, token_type_ids=types)
    assert summaries.shape == (1, 2, _Cfg.d_model)
    assert torch.allclose(summaries[:, 1, :], torch.zeros_like(summaries[:, 1, :]))

