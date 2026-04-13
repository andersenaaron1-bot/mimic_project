import torch


def _build_bundle_timeline():
    from src.ehr_hier.data.event_frames import EventPayloadKind, build_event_frame
    from src.ehr_hier.data.token_types import EventToken, TokenCategory

    summary = build_event_frame(
        [
            EventToken(
                value_id=1,
                category_id=int(TokenCategory.SPECIAL),
                t_from_start_hours=0.0,
                dt_from_prev_hours=0.0,
                cat_attrs={},
                num_attrs={},
            )
        ],
        payload_kind=EventPayloadKind.SPECIAL,
    )
    measurement = build_event_frame(
        [
            EventToken(
                value_id=100,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=2.0,
                dt_from_prev_hours=2.0,
                cat_attrs={"window_type_id": 2, "var_id": 17},
                num_attrs={"z": 1.5},
            ),
            EventToken(
                value_id=101,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=2.0,
                dt_from_prev_hours=0.0,
                cat_attrs={"window_type_id": 2, "var_id": 17, "codebook": 0},
                num_attrs={},
            ),
        ],
        payload_kind=EventPayloadKind.NUMERIC_MEASUREMENT,
    )
    return [summary, measurement]


def _build_two_event_timeline():
    from src.ehr_hier.data.event_frames import EventPayloadKind, build_event_frame
    from src.ehr_hier.data.token_types import EventToken, TokenCategory

    summary = build_event_frame(
        [
            EventToken(
                value_id=1,
                category_id=int(TokenCategory.SPECIAL),
                t_from_start_hours=0.0,
                dt_from_prev_hours=0.0,
                cat_attrs={},
                num_attrs={},
            )
        ],
        payload_kind=EventPayloadKind.SPECIAL,
    )
    measurement_a = build_event_frame(
        [
            EventToken(
                value_id=100,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=2.0,
                dt_from_prev_hours=2.0,
                cat_attrs={"window_type_id": 2, "var_id": 17},
                num_attrs={"z": 1.5},
            )
        ],
        payload_kind=EventPayloadKind.NUMERIC_MEASUREMENT,
    )
    measurement_b = build_event_frame(
        [
            EventToken(
                value_id=102,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=5.0,
                dt_from_prev_hours=3.0,
                cat_attrs={"window_type_id": 2, "var_id": 18},
                num_attrs={"z": 0.5},
            )
        ],
        payload_kind=EventPayloadKind.NUMERIC_MEASUREMENT,
    )
    return [summary, measurement_a, measurement_b]


def test_collator_emits_event_level_metadata_for_bundle_sequences() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4),
    )

    batch = collator([_build_bundle_timeline()])

    assert batch["input_ids"][0, 0, 0, :5].tolist() == [1, 12, 100, 101, 14]
    assert batch["event_input_ids"][0, 0, 0, :4].tolist() == [1, 12, 100, 14]
    assert batch["token_event_index"][0, 0, 0, :5].tolist() == [0, 1, 2, 2, 3]
    assert batch["token_event_slot_ids"][0, 0, 0, :5].tolist() == [0, 0, 0, 1, 0]
    assert batch["event_attention_mask"][0, 0, 0, :4].tolist() == [1, 1, 1, 1]
    assert batch["event_type_ids"][0, 0, 0, :4].tolist() == [
        int(TokenCategory.SPECIAL),
        int(TokenCategory.SPECIAL),
        int(TokenCategory.MEASUREMENT),
        int(TokenCategory.SPECIAL),
    ]
    assert batch["event_payload_ids"][0, 0, 0, :4].tolist() == [
        EVENT_PAYLOAD_KIND_TO_ID["special"],
        EVENT_PAYLOAD_KIND_TO_ID["special"],
        EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"],
        EVENT_PAYLOAD_KIND_TO_ID["special"],
    ]
    assert batch["event_numeric_mask"][0, 0, 0, :4].tolist() == [0, 0, 1, 0]
    assert batch["event_numeric_values"][0, 0, 0, 2, 0].item() == 1.5


def test_model_accepts_event_composer_path_and_keeps_token_logits() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_ORDER
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig
    from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer

    class _Cfg:
        d_model = 16
        num_heads = 1
        d_ff = 32
        num_local_layers = 0
        num_chunk_layers = 0
        num_global_layers = 0
        rope_max_period = 10000.0
        dropout = 0.0
        summary_pool_temperature = 1.0
        summary_pool_dropout = 0.0
        exclude_special_from_summary = True
        special_type_id = 0
        num_window_types = 4
        num_token_types = 8
        use_unified_token_head = True
        emit_switched_heads = False
        use_event_composer = True
        enable_event_time_nll_head = True
        enable_next_window_gap_nll_head = True
        enable_time_embedding = False
        enable_chunk_meta_sidechannel = False
        enable_window_sequence_meta = False
        condition_numeric_on_token_type = False
        numeric_value_transform = "identity"

    vocab_config = {
        "total_size": 128,
        "size_special": 32,
        "size_rvq": 16,
        "size_meas_labels": 64,
        "size_meds": 32,
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
        "offsets": {"SPECIAL": 0, "RVQ": 32, "MEAS": 48, "MED": 112},
        "dense_blocks": [
            {"name": "special", "dense_offset": 0, "dense_size": 32, "global_offset": 0},
            {"name": "measurement_code", "dense_offset": 48, "dense_size": 64, "global_offset": 48},
        ],
    }

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4),
    )
    batch = collator([_build_bundle_timeline()])
    torch.manual_seed(0)
    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    logits, final_state = model(
        input_ids=batch["input_ids"],
        time_ids=batch["time_ids"],
        numeric_values=batch["numeric_values"],
        numeric_mask=batch["numeric_mask"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
        window_start_times=batch["window_start_times"],
        window_mask=batch["window_mask"],
        window_type_ids=batch["window_type_ids"],
        chunk_mask=batch["chunk_mask"],
        chunk_start_offsets=batch["chunk_start_offsets"],
        chunk_is_last=batch["chunk_is_last"],
        semantic_token_counts=batch["semantic_token_counts"],
        semantic_duration_hours=batch["semantic_duration_hours"],
        chunk_token_counts=batch["chunk_token_counts"],
        chunk_duration_hours=batch["chunk_duration_hours"],
        token_event_index=batch["token_event_index"],
        token_event_slot_ids=batch["token_event_slot_ids"],
        event_input_ids=batch["event_input_ids"],
        event_time_ids=batch["event_time_ids"],
        event_numeric_values=batch["event_numeric_values"],
        event_numeric_mask=batch["event_numeric_mask"],
        event_type_ids=batch["event_type_ids"],
        event_payload_ids=batch["event_payload_ids"],
        event_attention_mask=batch["event_attention_mask"],
    )
    logits_unconditioned, _ = model(
        input_ids=batch["input_ids"],
        time_ids=batch["time_ids"],
        numeric_values=batch["numeric_values"],
        numeric_mask=batch["numeric_mask"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
        window_start_times=batch["window_start_times"],
        window_mask=batch["window_mask"],
        window_type_ids=batch["window_type_ids"],
        chunk_mask=batch["chunk_mask"],
        chunk_start_offsets=batch["chunk_start_offsets"],
        chunk_is_last=batch["chunk_is_last"],
        semantic_token_counts=batch["semantic_token_counts"],
        semantic_duration_hours=batch["semantic_duration_hours"],
        chunk_token_counts=batch["chunk_token_counts"],
        chunk_duration_hours=batch["chunk_duration_hours"],
        token_event_index=batch["token_event_index"],
        token_event_slot_ids=batch["token_event_slot_ids"],
        event_time_ids=batch["event_time_ids"],
        event_numeric_values=batch["event_numeric_values"],
        event_numeric_mask=batch["event_numeric_mask"],
        event_type_ids=batch["event_type_ids"],
        event_payload_ids=batch["event_payload_ids"],
        event_attention_mask=batch["event_attention_mask"],
    )

    assert logits["logits_token"].shape == (1, 1, 1, 8, vocab_config["total_size"])
    assert logits["logits_event_token"].shape == (1, 1, 1, 4, vocab_config["total_size"])
    assert logits["logits_event_family"].shape == (1, 1, 1, 4, _Cfg.num_token_types)
    assert logits["logits_event_payload"].shape == (1, 1, 1, 4, len(EVENT_PAYLOAD_KIND_ORDER))
    assert logits["logits_event_concept_special"].shape == (1, 1, 1, 4, 32)
    assert logits["logits_event_concept_measurement"].shape == (1, 1, 1, 4, 64)
    assert logits["pred_event_value_mu"].shape == (1, 1, 1, 4)
    assert logits["pred_event_value_sigma"].shape == (1, 1, 1, 4)
    assert logits["pred_event_dt_next_mu"].shape == (1, 1, 1, 4)
    assert logits["pred_event_dt_next_sigma"].shape == (1, 1, 1, 4)
    assert logits["pred_next_window_gap_mu"].shape == (1, 1)
    assert logits["pred_next_window_gap_sigma"].shape == (1, 1)
    assert final_state.shape == (1, _Cfg.d_model)
    assert torch.isfinite(logits["logits_token"]).all()
    assert not torch.allclose(logits["pred_event_value_mu"], logits_unconditioned["pred_event_value_mu"])


def test_loss_accepts_event_token_supervision() -> None:
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig
    from src.ehr_hier.transformer.loss import AETLossModule
    from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer

    class _Cfg:
        d_model = 16
        num_heads = 1
        d_ff = 32
        num_local_layers = 0
        num_chunk_layers = 0
        num_global_layers = 0
        rope_max_period = 10000.0
        dropout = 0.0
        summary_pool_temperature = 1.0
        summary_pool_dropout = 0.0
        exclude_special_from_summary = True
        special_type_id = 0
        num_window_types = 4
        num_token_types = 8
        use_unified_token_head = True
        emit_switched_heads = False
        use_event_composer = True
        enable_event_time_nll_head = True
        enable_next_window_gap_nll_head = True
        enable_time_embedding = False
        enable_chunk_meta_sidechannel = False
        enable_window_sequence_meta = False
        condition_numeric_on_token_type = False
        numeric_value_transform = "identity"

    vocab_config = {
        "total_size": 128,
        "size_special": 32,
        "size_rvq": 16,
        "size_meas_labels": 64,
        "size_meds": 32,
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
        "offsets": {"SPECIAL": 0, "RVQ": 32, "MEAS": 48, "MED": 112},
        "dense_blocks": [
            {"name": "special", "dense_offset": 0, "dense_size": 32, "global_offset": 0},
            {"name": "measurement_code", "dense_offset": 48, "dense_size": 64, "global_offset": 48},
        ],
    }

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4),
    )
    batch = collator([_build_two_event_timeline()])
    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)
    criterion = AETLossModule(
        vocab_config=vocab_config,
        weights={
            "token": 0.0,
            "event_token": 1.0,
            "event_family": 1.0,
            "event_payload": 1.0,
            "event_concept": 1.0,
            "event_dt": 1.0,
            "event_value": 1.0,
        },
    )

    logits, _ = model(
        input_ids=batch["input_ids"],
        time_ids=batch["time_ids"],
        numeric_values=batch["numeric_values"],
        numeric_mask=batch["numeric_mask"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
        window_start_times=batch["window_start_times"],
        window_mask=batch["window_mask"],
        window_type_ids=batch["window_type_ids"],
        chunk_mask=batch["chunk_mask"],
        chunk_start_offsets=batch["chunk_start_offsets"],
        chunk_is_last=batch["chunk_is_last"],
        semantic_token_counts=batch["semantic_token_counts"],
        semantic_duration_hours=batch["semantic_duration_hours"],
        chunk_token_counts=batch["chunk_token_counts"],
        chunk_duration_hours=batch["chunk_duration_hours"],
        token_event_index=batch["token_event_index"],
        token_event_slot_ids=batch["token_event_slot_ids"],
        event_input_ids=batch["event_input_ids"],
        event_time_ids=batch["event_time_ids"],
        event_numeric_values=batch["event_numeric_values"],
        event_numeric_mask=batch["event_numeric_mask"],
        event_type_ids=batch["event_type_ids"],
        event_payload_ids=batch["event_payload_ids"],
        event_attention_mask=batch["event_attention_mask"],
    )
    loss, logs = criterion(logits, batch)

    assert torch.isfinite(loss)
    assert "loss_event_token" in logs
    assert "loss_event_family" in logs
    assert "loss_event_payload" in logs
    assert "loss_event_concept" in logs
    assert "loss_event_dt_nll" in logs
    assert "loss_event_value_nll" in logs
    assert "n_event_token_supervised" in logs
    assert "n_event_family_supervised" in logs
    assert "n_event_payload_supervised" in logs
    assert "n_event_concept_supervised" in logs
    assert "n_event_dt_supervised" in logs
    assert "n_event_value_supervised" in logs
    assert int(logs["n_event_token_supervised"]) > 0
    assert int(logs["n_event_family_supervised"]) > 0
    assert int(logs["n_event_payload_supervised"]) > 0
    assert int(logs["n_event_concept_supervised"]) > 0
    assert int(logs["n_event_dt_supervised"]) > 0
    assert int(logs["n_event_value_supervised"]) > 0
