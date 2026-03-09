import torch


def test_model_emits_transition_control_heads() -> None:
    from ehr_hier.transformer.model import AdaptiveEpisodicTransformer

    class _Cfg:
        d_model = 16
        num_heads = 2
        d_ff = 32
        num_local_layers = 0
        num_global_layers = 0
        num_chunk_layers = 0
        rope_max_period = 10000.0
        dropout = 0.0
        summary_pool_temperature = 1.0
        summary_pool_dropout = 0.0
        exclude_special_from_summary = True
        special_type_id = 0
        num_window_types = 4
        enable_transition_control_heads = True

    vocab_config = {
        "total_size": 128,
        "size_special": 32,
        "size_rvq": 8,
        "size_meas_labels": 16,
        "size_meds": 16,
        "offsets": {"SPECIAL": 0, "RVQ": 64, "MEAS": 72, "MED": 88},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
            "end_mode": "next_type",
        },
    }
    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    B, W, C, L = 2, 3, 2, 5
    input_ids = torch.zeros((B, W, C, L), dtype=torch.long)
    time_ids = torch.zeros((B, W, C, L), dtype=torch.float)
    numeric_values = torch.zeros((B, W, C, L, 1), dtype=torch.float)
    token_type_ids = torch.ones((B, W, C, L), dtype=torch.long)
    attention_mask = torch.ones((B, W, C, L), dtype=torch.long)

    out, _ = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )

    assert out["logits_transition_boundary"].shape == (B, W, C, L, 2)
    assert out["logits_boundary_next_window_type"].shape == (B, W, C, L, _Cfg.num_window_types)

