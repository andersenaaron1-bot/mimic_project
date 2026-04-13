import torch


def test_episodic_memory_is_causal_and_tracks_exact_event_ids() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.episodic_memory import AETEpisodicMemory
    from src.ehr_hier.transformer.memory_rules import EventMemoryGroup

    class _Cfg:
        d_model = 4
        special_type_id = 0
        exact_memory_slots = 2
        exact_memory_write_per_window = 1
        exact_memory_retrieve_k = 1
        exact_memory_max_same_group = 2
        exact_memory_age_decay = 0.0
        exact_memory_rule_write_scale = 1.0
        exact_memory_rule_retrieval_scale = 0.0
        exact_memory_learned_write_scale = 0.0
        exact_memory_first_occurrence_bonus = 0.0
        exact_memory_chronic_bonus = 0.0

    module = AETEpisodicMemory(_Cfg)

    event_states = torch.tensor(
        [
            [
                [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
                [[[0.0, 0.0, 1.0, 0.0], [0.2, 0.8, 0.0, 0.0]]],
            ]
        ],
        dtype=torch.float32,
    )  # (1,3,1,2,4)
    event_input_ids = torch.tensor([[[[10, 20]], [[30, 40]], [[50, 60]]]], dtype=torch.long)
    event_attention_mask = torch.ones((1, 3, 1, 2), dtype=torch.long)
    event_type_ids = torch.tensor(
        [
            [
                [[int(TokenCategory.STRUCTURAL), int(TokenCategory.MEASUREMENT)]],
                [[int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]],
                [[int(TokenCategory.MEASUREMENT), int(TokenCategory.MEASUREMENT)]],
            ]
        ],
        dtype=torch.long,
    )
    event_payload_ids = torch.tensor(
        [
            [
                [[EVENT_PAYLOAD_KIND_TO_ID["structural"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]]],
                [[EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]]],
                [[EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]]],
            ]
        ],
        dtype=torch.long,
    )
    event_time_ids = torch.tensor(
        [[[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]]],
        dtype=torch.float32,
    )
    query_states = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    window_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    event_memory_rule_scores = torch.tensor(
        [[[[5.0, 0.5]], [[4.0, 0.25]], [[0.75, 0.1]]]],
        dtype=torch.float32,
    )
    event_memory_group_ids = torch.tensor(
        [
            [
                [[int(EventMemoryGroup.STRUCTURAL), int(EventMemoryGroup.EXTREME_MEASUREMENT)]],
                [[int(EventMemoryGroup.CHRONIC_DIAGNOSIS), int(EventMemoryGroup.EXTREME_MEASUREMENT)]],
                [[int(EventMemoryGroup.EXTREME_MEASUREMENT), int(EventMemoryGroup.EXTREME_MEASUREMENT)]],
            ]
        ],
        dtype=torch.long,
    )
    event_memory_first_flags = torch.tensor([[[[1, 0]], [[1, 0]], [[0, 0]]]], dtype=torch.long)
    event_memory_chronic_flags = torch.tensor([[[[0, 0]], [[1, 0]], [[0, 0]]]], dtype=torch.long)

    out = module(
        event_states=event_states,
        event_input_ids=event_input_ids,
        event_time_ids=event_time_ids,
        event_attention_mask=event_attention_mask,
        event_type_ids=event_type_ids,
        event_payload_ids=event_payload_ids,
        query_states=query_states,
        window_mask=window_mask,
        event_memory_rule_scores=event_memory_rule_scores,
        event_memory_group_ids=event_memory_group_ids,
        event_memory_first_flags=event_memory_first_flags,
        event_memory_chronic_flags=event_memory_chronic_flags,
    )

    assert out.context.shape == (1, 3, 4)
    assert torch.isfinite(out.context).all()
    assert out.retrieval_counts.tolist() == [[0, 1, 1]]
    assert out.write_counts.tolist() == [[1, 1, 1]]
    assert out.written_event_ids[0, 0, 0].item() == 10
    assert out.retrieved_event_ids[0, 1, 0].item() == 10
    assert out.written_event_ids[0, 1, 0].item() == 30
    assert out.retrieved_event_ids[0, 2, 0].item() == 30


def test_collator_emits_memory_rule_features() -> None:
    from src.ehr_hier.data.event_frames import EventPayloadKind, build_event_frame
    from src.ehr_hier.data.token_types import EventToken, TokenCategory
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig
    from src.ehr_hier.transformer.memory_rules import EventMemoryGroup

    timeline = [
        build_event_frame(
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
        ),
        build_event_frame(
            [
                EventToken(
                    value_id=50,
                    category_id=int(TokenCategory.STRUCTURAL),
                    t_from_start_hours=1.0,
                    dt_from_prev_hours=1.0,
                    cat_attrs={"window_type_id": 1},
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.STRUCTURAL,
            semantic_label="STRUCT_START_MECH",
            concept_code="TRANSFER_TO",
        ),
        build_event_frame(
            [
                EventToken(
                    value_id=60,
                    category_id=int(TokenCategory.DIAGNOSIS),
                    t_from_start_hours=2.0,
                    dt_from_prev_hours=1.0,
                    cat_attrs={"window_type_id": 1},
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.SYMBOLIC_CODE,
            concept_code="ICD10CM//I50.9",
        ),
    ]

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4),
    )
    batch = collator([timeline])

    event_groups = batch["event_memory_group_ids"][0, 0, 0, :5].tolist()
    event_scores = batch["event_memory_rule_scores"][0, 0, 0, :5].tolist()
    chronic_flags = batch["event_memory_chronic_flags"][0, 0, 0, :5].tolist()
    first_flags = batch["event_memory_first_flags"][0, 0, 0, :5].tolist()

    assert event_groups[2] == int(EventMemoryGroup.STRUCTURAL)
    assert event_scores[2] > 0.0
    assert event_groups[3] == int(EventMemoryGroup.CHRONIC_DIAGNOSIS)
    assert chronic_flags[3] == 1
    assert first_flags[3] == 1


def test_collator_emits_demographic_feature_ids_for_global_specials() -> None:
    from src.ehr_hier.data.event_frames import EventPayloadKind, build_event_frame
    from src.ehr_hier.data.token_types import EventToken, TokenCategory
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig

    timeline = [
        build_event_frame(
            [
                EventToken(
                    value_id=30,
                    category_id=int(TokenCategory.SPECIAL),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,
                    cat_attrs={
                        "global_demographic": 1,
                        "demographic_feature_id": 1,
                    },
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.SPECIAL,
        ),
        build_event_frame(
            [
                EventToken(
                    value_id=34,
                    category_id=int(TokenCategory.SPECIAL),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,
                    cat_attrs={
                        "global_demographic": 1,
                        "demographic_feature_id": 2,
                    },
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.SPECIAL,
        ),
        build_event_frame(
            [
                EventToken(
                    value_id=50,
                    category_id=int(TokenCategory.STRUCTURAL),
                    t_from_start_hours=1.0,
                    dt_from_prev_hours=1.0,
                    cat_attrs={"window_type_id": 1},
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.STRUCTURAL,
            semantic_label="ADMISSION",
        ),
    ]

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=8,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4),
    )
    batch = collator([timeline])

    feature_ids = batch["event_demographic_feature_ids"][0, 0, 0, :4].tolist()
    assert feature_ids[:2] == [1, 2]
    assert feature_ids[2:] == [0, 0]


def test_collator_preserves_subject_segment_metadata() -> None:
    from src.ehr_hier.data.event_frames import EventPayloadKind, build_event_frame
    from src.ehr_hier.data.token_types import EventToken, TokenCategory
    from src.ehr_hier.transformer.collator import AETHierarchicalCollator

    timeline = [
        build_event_frame(
            [
                EventToken(
                    value_id=10,
                    category_id=int(TokenCategory.STRUCTURAL),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,
                    cat_attrs={"window_type_id": 1},
                    num_attrs={},
                )
            ],
            payload_kind=EventPayloadKind.STRUCTURAL,
            semantic_label="ADMISSION",
        )
    ]
    collator = AETHierarchicalCollator(max_windows=2, max_chunks_per_window=1, max_len_per_window=4, pad_id=0)
    batch = collator([{"timeline": timeline, "subject_id": 42, "trajectory_ord": 3}])

    assert batch["subject_ids"].tolist() == [42]
    assert batch["trajectory_ords"].tolist() == [3]


def test_episodic_memory_carries_across_segments() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.episodic_memory import AETEpisodicMemory
    from src.ehr_hier.transformer.memory_rules import EventMemoryGroup

    class _Cfg:
        d_model = 4
        special_type_id = 0
        exact_memory_slots = 2
        exact_memory_write_per_window = 1
        exact_memory_retrieve_k = 1
        exact_memory_max_same_group = 2
        exact_memory_age_decay = 0.0
        exact_memory_rule_write_scale = 1.0
        exact_memory_rule_retrieval_scale = 0.0
        exact_memory_learned_write_scale = 0.0
        exact_memory_first_occurrence_bonus = 0.0
        exact_memory_chronic_bonus = 0.0

    module = AETEpisodicMemory(_Cfg)

    seg1 = module(
        event_states=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], dtype=torch.float32),
        event_input_ids=torch.tensor([[[[10]]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[[0.0]]]], dtype=torch.float32),
        event_attention_mask=torch.ones((1, 1, 1, 1), dtype=torch.long),
        event_type_ids=torch.tensor([[[[int(TokenCategory.STRUCTURAL)]]]], dtype=torch.long),
        event_payload_ids=torch.tensor(
            [[[[EVENT_PAYLOAD_KIND_TO_ID["structural"]]]]],
            dtype=torch.long,
        ),
        query_states=torch.zeros((1, 1, 4), dtype=torch.float32),
        window_mask=torch.tensor([[1]], dtype=torch.long),
        event_memory_rule_scores=torch.tensor([[[[5.0]]]], dtype=torch.float32),
        event_memory_group_ids=torch.tensor(
            [[[[int(EventMemoryGroup.STRUCTURAL)]]]],
            dtype=torch.long,
        ),
        event_memory_first_flags=torch.tensor([[[[1]]]], dtype=torch.long),
        event_memory_chronic_flags=torch.tensor([[[[0]]]], dtype=torch.long),
    )

    assert seg1.next_state is not None
    assert seg1.next_state.persistent.valid_mask[0, 0].item() is True
    assert seg1.next_state.persistent.event_ids[0, 0].item() == 10

    seg2 = module(
        event_states=torch.tensor([[[[[0.0, 0.0, 1.0, 0.0]]]]], dtype=torch.float32),
        event_input_ids=torch.tensor([[[[30]]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[[0.0]]]], dtype=torch.float32),
        event_attention_mask=torch.ones((1, 1, 1, 1), dtype=torch.long),
        event_type_ids=torch.tensor([[[[int(TokenCategory.MEASUREMENT)]]]], dtype=torch.long),
        event_payload_ids=torch.tensor(
            [[[[EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]]]]],
            dtype=torch.long,
        ),
        query_states=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        window_mask=torch.tensor([[1]], dtype=torch.long),
        prev_memory_state=seg1.next_state,
        event_memory_rule_scores=torch.tensor([[[[0.5]]]], dtype=torch.float32),
        event_memory_group_ids=torch.tensor(
            [[[[int(EventMemoryGroup.EXTREME_MEASUREMENT)]]]],
            dtype=torch.long,
        ),
        event_memory_first_flags=torch.tensor([[[[0]]]], dtype=torch.long),
        event_memory_chronic_flags=torch.tensor([[[[0]]]], dtype=torch.long),
    )

    assert seg2.retrieval_counts.tolist() == [[1]]
    assert seg2.retrieved_event_ids[0, 0, 0].item() == 10
    assert seg2.retrieval_counts_by_bank is not None
    assert seg2.retrieval_counts_by_bank["persistent"].tolist() == [[1]]
    assert seg2.retrieval_counts_by_bank["episodic"].tolist() == [[0]]


def test_patient_memory_routes_structural_to_persistent_and_measurement_to_episodic() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.episodic_memory import AETEpisodicMemory
    from src.ehr_hier.transformer.memory_rules import EventMemoryGroup

    class _Cfg:
        d_model = 4
        special_type_id = 0
        exact_memory_slots = 2
        exact_memory_persistent_slots = 2
        exact_memory_episodic_slots = 2
        exact_memory_write_per_window = 2
        exact_memory_retrieve_k = 1
        exact_memory_persistent_retrieve_k = 1
        exact_memory_episodic_retrieve_k = 1
        exact_memory_max_same_group = 2
        exact_memory_age_decay = 0.0
        exact_memory_rule_write_scale = 1.0
        exact_memory_rule_retrieval_scale = 0.0
        exact_memory_learned_write_scale = 0.0
        exact_memory_first_occurrence_bonus = 0.0
        exact_memory_chronic_bonus = 0.0

    module = AETEpisodicMemory(_Cfg)

    out = module(
        event_states=torch.tensor(
            [[[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]]],
            dtype=torch.float32,
        ),
        event_input_ids=torch.tensor([[[[10, 20]]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32),
        event_attention_mask=torch.ones((1, 1, 1, 2), dtype=torch.long),
        event_type_ids=torch.tensor(
            [[[[int(TokenCategory.STRUCTURAL), int(TokenCategory.MEASUREMENT)]]]],
            dtype=torch.long,
        ),
        event_payload_ids=torch.tensor(
            [[[[EVENT_PAYLOAD_KIND_TO_ID["structural"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]]]]],
            dtype=torch.long,
        ),
        query_states=torch.zeros((1, 1, 4), dtype=torch.float32),
        window_mask=torch.tensor([[1]], dtype=torch.long),
        event_memory_rule_scores=torch.tensor([[[[5.0, 1.0]]]], dtype=torch.float32),
        event_memory_group_ids=torch.tensor(
            [[[[int(EventMemoryGroup.STRUCTURAL), int(EventMemoryGroup.EXTREME_MEASUREMENT)]]]],
            dtype=torch.long,
        ),
        event_memory_first_flags=torch.tensor([[[[1, 0]]]], dtype=torch.long),
        event_memory_chronic_flags=torch.tensor([[[[0, 0]]]], dtype=torch.long),
    )

    assert out.next_state is not None
    assert out.next_state.persistent.valid_mask[0, 0].item() is True
    assert out.next_state.persistent.event_ids[0, 0].item() == 10
    assert out.next_state.episodic.valid_mask[0, 0].item() is True
    assert out.next_state.episodic.event_ids[0, 0].item() == 20
    assert out.write_counts_by_bank is not None
    assert out.write_counts_by_bank["persistent"].tolist() == [[1]]
    assert out.write_counts_by_bank["episodic"].tolist() == [[1]]


def test_patient_memory_seeds_static_bank_from_demographic_specials() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.episodic_memory import AETEpisodicMemory

    class _Cfg:
        d_model = 4
        special_type_id = 0
        exact_memory_static_slots = 2
        exact_memory_static_retrieve_k = 2
        exact_memory_write_per_window = 0
        exact_memory_retrieve_k = 0
        exact_memory_max_same_group = 4
        exact_memory_age_decay = 0.0
        exact_memory_rule_write_scale = 1.0
        exact_memory_rule_retrieval_scale = 0.0
        exact_memory_learned_write_scale = 0.0
        exact_memory_first_occurrence_bonus = 0.0
        exact_memory_chronic_bonus = 0.0
        exact_memory_static_feature_ids = (1, 2)

    module = AETEpisodicMemory(_Cfg)

    event_seed_states = torch.tensor(
        [[[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]]],
        dtype=torch.float32,
    )
    event_states = event_seed_states.clone()

    out = module(
        event_states=event_states,
        event_seed_states=event_seed_states,
        event_input_ids=torch.tensor([[[[30, 34]]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float32),
        event_attention_mask=torch.ones((1, 1, 1, 2), dtype=torch.long),
        event_type_ids=torch.tensor(
            [[[[int(TokenCategory.SPECIAL), int(TokenCategory.SPECIAL)]]]],
            dtype=torch.long,
        ),
        event_payload_ids=torch.tensor(
            [[[[EVENT_PAYLOAD_KIND_TO_ID["special"], EVENT_PAYLOAD_KIND_TO_ID["special"]]]]],
            dtype=torch.long,
        ),
        event_demographic_feature_ids=torch.tensor([[[[1, 2]]]], dtype=torch.long),
        query_states=torch.zeros((1, 1, 4), dtype=torch.float32),
        window_mask=torch.tensor([[1]], dtype=torch.long),
        event_memory_rule_scores=torch.zeros((1, 1, 1, 2), dtype=torch.float32),
        event_memory_group_ids=torch.zeros((1, 1, 1, 2), dtype=torch.long),
        event_memory_first_flags=torch.zeros((1, 1, 1, 2), dtype=torch.long),
        event_memory_chronic_flags=torch.zeros((1, 1, 1, 2), dtype=torch.long),
    )

    assert out.next_state is not None
    assert out.next_state.static.valid_mask[0, :2].tolist() == [True, True]
    assert out.next_state.static.event_ids[0, :2].tolist() == [30, 34]
    assert out.retrieval_counts_by_bank is not None
    assert out.retrieval_counts_by_bank["static"].tolist() == [[2]]


def test_model_accepts_exact_memory_event_path() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_ORDER, EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer

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
        num_token_types = 8
        global_context_mode = "latent_state"
        use_unified_token_head = True
        emit_switched_heads = False
        use_event_composer = True
        enable_exact_memory = True
        exact_memory_slots = 2
        exact_memory_write_per_window = 1
        exact_memory_retrieve_k = 1
        exact_memory_max_same_group = 2
        enable_event_time_nll_head = True
        enable_next_window_gap_nll_head = True
        enable_time_embedding = False
        enable_chunk_meta_sidechannel = False
        enable_window_sequence_meta = False
        condition_numeric_on_token_type = False
        numeric_value_transform = "identity"
        global_fusion_mode = "add"
        exclude_special_from_global_fusion = True

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

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)

    input_ids = torch.tensor([[[10, 20], [30, 40]]], dtype=torch.long)
    time_ids = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
    numeric_values = torch.zeros((1, 2, 2, 1), dtype=torch.float32)
    numeric_mask = torch.zeros((1, 2, 2), dtype=torch.long)
    token_type_ids = torch.tensor(
        [[[int(TokenCategory.STRUCTURAL), int(TokenCategory.MEASUREMENT)],
          [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
        dtype=torch.long,
    )
    attention_mask = torch.ones((1, 2, 2), dtype=torch.long)
    window_start_times = torch.tensor([[0.0, 12.0]], dtype=torch.float32)
    window_mask = torch.tensor([[1, 1]], dtype=torch.long)
    window_type_ids = torch.tensor([[0, 1]], dtype=torch.long)
    semantic_duration_hours = torch.tensor([[2.0, 3.0]], dtype=torch.float32)

    token_event_index = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
    token_event_slot_ids = torch.zeros((1, 2, 2), dtype=torch.long)
    event_input_ids = torch.tensor([[[10, 20], [30, 40]]], dtype=torch.long)
    event_time_ids = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
    event_numeric_values = torch.tensor(
        [[[[0.0], [0.2]], [[0.0], [0.3]]]],
        dtype=torch.float32,
    )
    event_numeric_mask = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
    event_type_ids = token_type_ids.clone()
    event_payload_ids = torch.tensor(
        [[
            [EVENT_PAYLOAD_KIND_TO_ID["structural"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
            [EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
        ]],
        dtype=torch.long,
    )
    event_attention_mask = torch.ones((1, 2, 2), dtype=torch.long)
    event_memory_rule_scores = torch.tensor(
        [[[4.0, 0.5], [3.5, 0.25]]],
        dtype=torch.float32,
    )
    event_memory_group_ids = torch.tensor(
        [[[1, 5], [2, 5]]],
        dtype=torch.long,
    )
    event_memory_first_flags = torch.tensor(
        [[[1, 0], [1, 0]]],
        dtype=torch.long,
    )
    event_memory_chronic_flags = torch.tensor(
        [[[0, 0], [1, 0]]],
        dtype=torch.long,
    )

    logits, final_state = model(
        input_ids=input_ids,
        time_ids=time_ids,
        numeric_values=numeric_values,
        numeric_mask=numeric_mask,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        window_start_times=window_start_times,
        window_mask=window_mask,
        window_type_ids=window_type_ids,
        semantic_duration_hours=semantic_duration_hours,
        token_event_index=token_event_index,
        token_event_slot_ids=token_event_slot_ids,
        event_input_ids=event_input_ids,
        event_time_ids=event_time_ids,
        event_numeric_values=event_numeric_values,
        event_numeric_mask=event_numeric_mask,
        event_type_ids=event_type_ids,
        event_payload_ids=event_payload_ids,
        event_attention_mask=event_attention_mask,
        event_memory_rule_scores=event_memory_rule_scores,
        event_memory_group_ids=event_memory_group_ids,
        event_memory_first_flags=event_memory_first_flags,
        event_memory_chronic_flags=event_memory_chronic_flags,
    )

    assert logits["logits_token"].shape == (1, 2, 2, vocab_config["total_size"])
    assert logits["logits_event_token"].shape == (1, 2, 2, vocab_config["total_size"])
    assert logits["logits_event_family"].shape == (1, 2, 2, _Cfg.num_token_types)
    assert logits["logits_event_payload"].shape == (1, 2, 2, len(EVENT_PAYLOAD_KIND_ORDER))
    assert logits["pred_event_dt_next_mu"].shape == (1, 2, 2)
    assert logits["pred_next_window_gap_mu"].shape == (1, 2)
    assert final_state.shape == (1, _Cfg.d_model)
    assert torch.isfinite(final_state).all()
