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

