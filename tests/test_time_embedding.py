import torch


def test_time_embedding_clamps_and_shapes() -> None:
    from ehr_hier.transformer.embeddings import TimeEmbedding

    emb = TimeEmbedding(d_model=8, max_hours=10.0, dropout=0.0)

    # -1 clamps to 0; 20 clamps to 10.
    t = torch.tensor([-1.0, 0.0, 10.0, 20.0], dtype=torch.float)
    out = emb(t)

    assert out.shape == (4, 8)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[2], out[3])


def test_multiscale_time_embedding_clamps_each_clock() -> None:
    from ehr_hier.transformer.embeddings import MultiScaleTimeEmbedding

    emb = MultiScaleTimeEmbedding(
        d_model=8,
        local_max_hours=4.0,
        semantic_max_hours=10.0,
        global_max_hours=100.0,
        dropout=0.0,
    )

    local = torch.tensor([-1.0, 0.0, 4.0, 9.0], dtype=torch.float)
    semantic = torch.tensor([0.0, 0.0, 10.0, 50.0], dtype=torch.float)
    global_t = torch.tensor([0.0, 0.0, 100.0, 500.0], dtype=torch.float)
    out = emb(
        local_time_hours=local,
        semantic_time_hours=semantic,
        global_time_hours=global_t,
    )

    assert out.shape == (4, 8)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[2], out[3])
