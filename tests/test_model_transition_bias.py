import torch


def test_model_biases_transition_control_heads_at_boundary_markers_only() -> None:
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
        enable_window_len_head = True
        enable_transition_bias = True

    vocab_config = {
        "total_size": 64,
        "size_special": 32,
        "size_rvq": 4,
        "size_meas_labels": 4,
        "size_meds": 4,
        "offsets": {"SPECIAL": 0, "RVQ": 100, "MEAS": 200, "MED": 1000},
        "window_markers": {"type_token_offset": 10, "num_types": 3, "end_token_id": 13, "end_mode": "next_type"},
    }

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.next_window_type_head.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
        model.transition_prior_scale.fill_(1.0)
        model.transition_hazard_scale.fill_(1.0)

    # One semantic window, one chunk:
    # idx0: WIN_TYPE prefix, idx3: WIN_TYPE suffix (boundary marker).
    B, W, L = 1, 1, 4
    input_ids = torch.tensor([[[10, 6, 7, 11]]], dtype=torch.long)
    time_ids = torch.tensor([[[0.0, 0.5, 1.0, 2.0]]], dtype=torch.float)
    numeric_values = torch.zeros((B, W, L, 1), dtype=torch.float)
    token_type_ids = torch.tensor([[[0, 1, 1, 0]]], dtype=torch.long)  # SPECIAL, content, content, SPECIAL
    attention_mask = torch.ones((B, W, L), dtype=torch.long)
    window_type_ids = torch.tensor([[1]], dtype=torch.long)

    logits, _ = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        window_type_ids=window_type_ids,
    )

    # Boundary transition head receives bias only at the boundary marker (idx=3).
    assert logits["logits_transition_boundary"][0, 0, 3, 1].item() != 0.0
    assert logits["logits_transition_boundary"][0, 0, 1, 1].item() == 0.0

    # Next-window-type boundary head receives prior+hazard only at boundary type marker.
    v = logits["logits_boundary_next_window_type"][0, 0, 3, :]
    assert torch.any(v.abs() > 1e-6)
    v_nonboundary = logits["logits_boundary_next_window_type"][0, 0, 1, :]
    assert torch.all(v_nonboundary == 0)

    # No global broadcast into struct token logits.
    assert torch.all(logits["logits_struct"] == 0)
