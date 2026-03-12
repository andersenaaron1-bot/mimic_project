import torch


def test_model_emits_next_window_type_logits() -> None:
    from ehr_hier.transformer.model import AdaptiveEpisodicTransformer

    class _Cfg:
        d_model = 8
        num_heads = 1
        d_ff = 16
        num_local_layers = 0
        num_global_layers = 0
        rope_max_period = 10000.0
        dropout = 0.0
        summary_pool_temperature = 1.0
        summary_pool_dropout = 0.0
        exclude_special_from_summary = True
        special_type_id = 0
        num_window_types = 3

    vocab_config = {
        "total_size": 128,
        "size_special": 16,
        "size_rvq": 8,
        "size_meas_labels": 8,
        "size_meds": 16,
    }

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    B, W, L = 2, 3, 4
    input_ids = torch.randint(0, vocab_config["total_size"], (B, W, L))
    time_ids = torch.zeros((B, W, L), dtype=torch.float)
    numeric_values = torch.zeros((B, W, L, 1), dtype=torch.float)
    token_type_ids = torch.zeros((B, W, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, L), dtype=torch.long)
    window_start_times = torch.tensor([[0.0, 5.0, 10.0], [1.0, 2.0, 3.0]], dtype=torch.float)
    window_mask = torch.ones((B, W), dtype=torch.long)

    logits, final_state = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        window_start_times=window_start_times,
        window_mask=window_mask,
    )

    assert logits["logits_token"].shape == (B, W, L, vocab_config["total_size"])
    assert logits["logits_next_window_type"].shape == (B, W, _Cfg.num_window_types)
    assert final_state.shape == (B, _Cfg.d_model)


def test_model_accepts_chunked_semantic_windows() -> None:
    from ehr_hier.transformer.model import AdaptiveEpisodicTransformer

    class _Cfg:
        d_model = 8
        num_heads = 1
        d_ff = 16
        num_local_layers = 0
        num_chunk_layers = 0
        num_global_layers = 0
        rope_max_period = 10000.0
        dropout = 0.0
        summary_pool_temperature = 1.0
        summary_pool_dropout = 0.0
        exclude_special_from_summary = True
        special_type_id = 0
        num_window_types = 3

    vocab_config = {
        "total_size": 128,
        "size_special": 16,
        "size_rvq": 8,
        "size_meas_labels": 8,
        "size_meds": 16,
    }

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    B, W, C, L = 1, 2, 3, 4
    input_ids = torch.randint(0, vocab_config["total_size"], (B, W, C, L))
    time_ids = torch.zeros((B, W, C, L), dtype=torch.float)
    numeric_values = torch.zeros((B, W, C, L, 1), dtype=torch.float)
    token_type_ids = torch.zeros((B, W, C, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, C, L), dtype=torch.long)
    chunk_mask = torch.ones((B, W, C), dtype=torch.long)
    chunk_start_offsets = torch.tensor([[[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]]], dtype=torch.float)
    chunk_is_last = torch.tensor([[[0, 0, 1], [0, 0, 1]]], dtype=torch.long)
    window_start_times = torch.tensor([[0.0, 10.0]], dtype=torch.float)
    window_mask = torch.ones((B, W), dtype=torch.long)
    window_type_ids = torch.tensor([[1, 2]], dtype=torch.long)

    logits, final_state = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        chunk_mask=chunk_mask,
        chunk_start_offsets=chunk_start_offsets,
        chunk_is_last=chunk_is_last,
        window_start_times=window_start_times,
        window_mask=window_mask,
        window_type_ids=window_type_ids,
    )

    assert logits["logits_token"].shape == (B, W, C, L, vocab_config["total_size"])
    assert logits["logits_struct"].shape[:4] == (B, W, C, L)
    assert logits["logits_next_window_type"].shape == (B, W, _Cfg.num_window_types)
    assert final_state.shape == (B, _Cfg.d_model)
