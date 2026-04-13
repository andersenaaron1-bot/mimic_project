import torch


def test_window_state_packet_builder_emits_structured_packet() -> None:
    from ehr_hier.transformer.aggregator import AETWindowStatePacketBuilder

    class _Cfg:
        d_model = 8
        window_packet_write_tokens = 2

    builder = AETWindowStatePacketBuilder(_Cfg)

    chunk_states = torch.tensor(
        [
            [
                [[1.0] * 8, [2.0] * 8, [0.0] * 8],
                [[3.0] * 8, [5.0] * 8, [7.0] * 8],
            ]
        ],
        dtype=torch.float32,
    )
    chunk_mask = torch.tensor([[[1, 1, 0], [1, 1, 1]]], dtype=torch.long)
    base_summary = torch.tensor(
        [[[1.5] * 8, [5.0] * 8]],
        dtype=torch.float32,
    )
    chunk_start_offsets = torch.tensor(
        [[[0.0, 2.0, 4.0], [0.0, 8.0, 16.0]]],
        dtype=torch.float32,
    )
    window_start_times = torch.tensor([[0.0, 24.0]], dtype=torch.float32)
    semantic_duration_hours = torch.tensor([[4.0, 24.0]], dtype=torch.float32)
    chunk_token_counts = torch.tensor([[[5.0, 4.0, 0.0], [9.0, 7.0, 3.0]]], dtype=torch.float32)
    chunk_duration_hours = torch.tensor([[[2.0, 2.0, 0.0], [8.0, 8.0, 8.0]]], dtype=torch.float32)
    window_mask = torch.tensor([[1, 1]], dtype=torch.long)
    window_type_ids = torch.tensor([[0, 2]], dtype=torch.long)

    packet = builder(
        base_summary=base_summary,
        chunk_states=chunk_states,
        chunk_mask=chunk_mask,
        chunk_start_offsets=chunk_start_offsets,
        window_start_times=window_start_times,
        semantic_duration_hours=semantic_duration_hours,
        chunk_token_counts=chunk_token_counts,
        chunk_duration_hours=chunk_duration_hours,
        window_mask=window_mask,
        window_type_ids=window_type_ids,
    )

    assert packet.slot_tokens.shape == (1, 2, 6, 8)
    assert packet.slot_mask.shape == (1, 2, 6)
    assert packet.query_token.shape == (1, 2, 8)
    assert packet.write_tokens is not None
    assert packet.write_tokens.shape == (1, 2, 2, 8)
    assert packet.write_mask is not None
    assert packet.write_mask.shape == (1, 2, 2)
    assert torch.isfinite(packet.summary()).all()
    assert torch.equal(packet.window_type_ids, window_type_ids)
    assert packet.gap_prev_hours is not None
    assert packet.gap_prev_hours.shape == (1, 2)


def test_model_emits_window_state_packet_aux() -> None:
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
        window_packet_write_tokens = 2

    vocab_config = {
        "total_size": 128,
        "size_special": 16,
        "size_rvq": 8,
        "size_meas_labels": 8,
        "size_meds": 16,
    }

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    B, W, C, L = 1, 3, 2, 4
    input_ids = torch.randint(0, vocab_config["total_size"], (B, W, C, L))
    time_ids = torch.zeros((B, W, C, L), dtype=torch.float32)
    numeric_values = torch.zeros((B, W, C, L, 1), dtype=torch.float32)
    token_type_ids = torch.ones((B, W, C, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, C, L), dtype=torch.long)
    window_start_times = torch.tensor([[0.0, 6.0, 18.0]], dtype=torch.float32)
    window_mask = torch.ones((B, W), dtype=torch.long)
    chunk_mask = torch.ones((B, W, C), dtype=torch.long)
    chunk_start_offsets = torch.tensor([[[0.0, 2.0], [0.0, 3.0], [0.0, 1.0]]], dtype=torch.float32)
    semantic_duration_hours = torch.tensor([[4.0, 6.0, 2.0]], dtype=torch.float32)
    chunk_token_counts = torch.full((B, W, C), 4.0, dtype=torch.float32)
    chunk_duration_hours = torch.tensor([[[2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]], dtype=torch.float32)
    window_type_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

    logits, aux = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        window_start_times=window_start_times,
        window_mask=window_mask,
        chunk_mask=chunk_mask,
        chunk_start_offsets=chunk_start_offsets,
        semantic_duration_hours=semantic_duration_hours,
        chunk_token_counts=chunk_token_counts,
        chunk_duration_hours=chunk_duration_hours,
        window_type_ids=window_type_ids,
        return_aux_state=True,
    )

    packet = aux["window_state_packet"]
    summary = aux["window_packet_summary"]

    assert logits["logits_next_window_type"].shape == (B, W, _Cfg.num_window_types)
    assert summary.shape == (B, W, _Cfg.d_model)
    assert packet.slot_tokens.shape == (B, W, 6, _Cfg.d_model)
    assert packet.query_token is not None
    assert packet.write_tokens is not None
    assert aux["global_state"].shape == (B, _Cfg.d_model)
    assert aux["window_global_states"].shape == (B, W, _Cfg.d_model)
    assert aux["patient_memory_context"] is None
    assert aux["patient_memory_context_by_bank"] is None
    assert aux["patient_memory_state_digests_by_bank"] is None
