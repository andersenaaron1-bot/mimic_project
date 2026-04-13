import torch


def test_latent_health_state_emits_window_states() -> None:
    from ehr_hier.transformer.global_state import AETLatentHealthState

    class _Cfg:
        d_model = 8
        dropout = 0.0
        num_window_types = 4
        latent_state_use_window_type = True
        latent_state_use_absolute_time = True

    module = AETLatentHealthState(_Cfg)
    window_summaries = torch.randn((2, 3, 8), dtype=torch.float32)
    window_start_times = torch.tensor([[0.0, 5.0, 9.0], [1.0, 2.0, 4.0]], dtype=torch.float32)
    semantic_duration_hours = torch.tensor([[2.0, 1.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    window_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    window_type_ids = torch.tensor([[1, 2, 3], [0, 1, 0]], dtype=torch.long)

    out = module(
        window_summaries=window_summaries,
        window_start_times=window_start_times,
        padding_mask=window_mask,
        semantic_duration_hours=semantic_duration_hours,
        window_type_ids=window_type_ids,
    )

    assert out.shape == window_summaries.shape
    assert torch.isfinite(out).all()
    assert torch.all(out[1, 2, :] == 0)


def test_latent_health_state_is_gap_sensitive() -> None:
    from ehr_hier.transformer.global_state import AETLatentHealthState

    class _Cfg:
        d_model = 8
        dropout = 0.0
        num_window_types = 2
        latent_state_use_window_type = True
        latent_state_use_absolute_time = True

    torch.manual_seed(0)
    module = AETLatentHealthState(_Cfg)
    window_summaries = torch.randn((1, 2, 8), dtype=torch.float32)
    semantic_duration_hours = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    window_mask = torch.tensor([[1, 1]], dtype=torch.long)
    window_type_ids = torch.tensor([[1, 1]], dtype=torch.long)

    near = module(
        window_summaries=window_summaries,
        window_start_times=torch.tensor([[0.0, 3.0]], dtype=torch.float32),
        padding_mask=window_mask,
        semantic_duration_hours=semantic_duration_hours,
        window_type_ids=window_type_ids,
    )
    far = module(
        window_summaries=window_summaries,
        window_start_times=torch.tensor([[0.0, 48.0]], dtype=torch.float32),
        padding_mask=window_mask,
        semantic_duration_hours=semantic_duration_hours,
        window_type_ids=window_type_ids,
    )

    assert not torch.allclose(near[:, 1, :], far[:, 1, :])


def test_model_accepts_latent_state_global_mode() -> None:
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
        global_context_mode = "latent_state"
        enable_next_window_gap_nll_head = True

    vocab_config = {
        "total_size": 128,
        "size_special": 16,
        "size_rvq": 8,
        "size_meas_labels": 8,
        "size_meds": 16,
    }

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    B, W, L = 1, 3, 4
    input_ids = torch.randint(0, vocab_config["total_size"], (B, W, L))
    time_ids = torch.zeros((B, W, L), dtype=torch.float32)
    numeric_values = torch.zeros((B, W, L, 1), dtype=torch.float32)
    token_type_ids = torch.ones((B, W, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, L), dtype=torch.long)
    window_start_times = torch.tensor([[0.0, 6.0, 18.0]], dtype=torch.float32)
    window_mask = torch.ones((B, W), dtype=torch.long)
    semantic_duration_hours = torch.tensor([[2.0, 3.0, 1.0]], dtype=torch.float32)
    window_type_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    logits, final_state = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        window_start_times=window_start_times,
        window_mask=window_mask,
        semantic_duration_hours=semantic_duration_hours,
        window_type_ids=window_type_ids,
    )

    assert logits["logits_next_window_type"].shape == (B, W, _Cfg.num_window_types)
    assert logits["pred_next_window_gap_mu"].shape == (B, W)
    assert final_state.shape == (B, _Cfg.d_model)
    assert torch.isfinite(final_state).all()
