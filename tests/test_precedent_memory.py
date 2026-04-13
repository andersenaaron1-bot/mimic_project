from __future__ import annotations

from pathlib import Path

import torch


def test_build_future_summary_captures_typed_future_shape() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.precedent_memory import build_future_summary

    window_type_ids = torch.tensor([0, 2, 1], dtype=torch.long)
    window_start_times = torch.tensor([0.0, 6.0, 18.0], dtype=torch.float32)
    semantic_duration_hours = torch.tensor([4.0, 5.0, 3.0], dtype=torch.float32)
    event_type_ids = torch.tensor(
        [
            [[int(TokenCategory.STRUCTURAL), int(TokenCategory.MEASUREMENT), 0]],
            [[int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT), int(TokenCategory.PROCEDURE)]],
            [[int(TokenCategory.MEDICATION), int(TokenCategory.MEASUREMENT), 0]],
        ],
        dtype=torch.long,
    )
    event_payload_ids = torch.tensor(
        [
            [[EVENT_PAYLOAD_KIND_TO_ID["structural"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"], 0]],
            [[EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"], EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"]]],
            [[EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"], 0]],
        ],
        dtype=torch.long,
    )
    event_attention_mask = torch.tensor(
        [[[1, 1, 0]], [[1, 1, 1]], [[1, 1, 0]]],
        dtype=torch.long,
    )
    event_memory_chronic_flags = torch.tensor(
        [[[0, 0, 0]], [[1, 0, 0]], [[0, 0, 0]]],
        dtype=torch.long,
    )
    event_numeric_values = torch.tensor(
        [
            [[[0.0], [1.0], [0.0]]],
            [[[0.0], [3.2], [0.0]]],
            [[[0.0], [1.5], [0.0]]],
        ],
        dtype=torch.float32,
    )
    event_numeric_mask = torch.tensor(
        [[[0, 1, 0]], [[0, 1, 0]], [[0, 1, 0]]],
        dtype=torch.long,
    )

    summary = build_future_summary(
        current_window_type_id=0,
        current_window_end_h=4.0,
        future_window_indices=[1, 2],
        future_truncated=True,
        window_type_ids=window_type_ids,
        window_start_times=window_start_times,
        semantic_duration_hours=semantic_duration_hours,
        event_type_ids=event_type_ids,
        event_payload_ids=event_payload_ids,
        event_attention_mask=event_attention_mask,
        event_memory_chronic_flags=event_memory_chronic_flags,
        event_numeric_values=event_numeric_values,
        event_numeric_mask=event_numeric_mask,
    )

    assert int(summary.next_window_type_id.item()) == 2
    assert float(summary.next_window_gap_h.item()) == 2.0
    assert float(summary.future_window_count.item()) == 2.0
    assert float(summary.measurement_count.item()) == 2.0
    assert float(summary.extreme_measurement_count.item()) == 1.0
    assert float(summary.support_flags[1].item()) == 1.0
    assert float(summary.support_flags[2].item()) == 1.0
    assert float(summary.support_flags[5].item()) == 1.0
    assert float(summary.transition_flags[0].item()) == 1.0
    assert float(summary.transition_flags[3].item()) == 1.0


def test_precedent_memory_roundtrip_and_retrieval(tmp_path: Path) -> None:
    from src.ehr_hier.transformer.precedent_memory import (
        AETPrecedentMemory,
        materialize_precedent_index_store,
        save_precedent_index_store,
    )
    from src.ehr_hier.transformer.world_model_contract import (
        FutureSnippetRef,
        NextWindowHeader,
        PrecedentIndexItem,
        WindowStatePacket,
        compose_precedent_key_state,
        empty_future_summary,
    )

    d_model = 4
    packet_a = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    latent_a = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    memory_a = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    packet_b = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    latent_b = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    memory_b = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def _item(item_id: int, subject_id: int, packet: torch.Tensor, latent: torch.Tensor, memory: torch.Tensor) -> PrecedentIndexItem:
        key_state = compose_precedent_key_state(
            packet_query=packet.view(1, -1),
            latent_state=latent.view(1, -1),
            memory_digest=memory.view(1, -1),
            window_type_ids=torch.tensor([1], dtype=torch.long),
            gap_prev_hours=torch.tensor([2.0], dtype=torch.float32),
            duration_hours=torch.tensor([3.0], dtype=torch.float32),
        ).squeeze(0)
        summary = empty_future_summary(device=torch.device("cpu"), dtype=torch.float32)
        summary.next_window_type_id = torch.tensor(2, dtype=torch.long)
        summary.future_window_count = torch.tensor(1.0, dtype=torch.float32)
        return PrecedentIndexItem(
            item_id=torch.tensor(item_id, dtype=torch.long),
            subject_id=torch.tensor(subject_id, dtype=torch.long),
            trajectory_ord=torch.tensor(0, dtype=torch.long),
            boundary_ord=torch.tensor(0, dtype=torch.long),
            anchor_window_ord=torch.tensor(0, dtype=torch.long),
            current_window_type_id=torch.tensor(1, dtype=torch.long),
            current_window_start_h=torch.tensor(0.0, dtype=torch.float32),
            current_window_duration_h=torch.tensor(3.0, dtype=torch.float32),
            gap_prev_h=torch.tensor(2.0, dtype=torch.float32),
            support_flags=torch.zeros((6,), dtype=torch.float32),
            anchor_mask_flags=torch.zeros((4,), dtype=torch.float32),
            key_state=key_state,
            key_packet=packet,
            key_memory=memory,
            future_summary_h1=summary,
            future_summary_h2=summary,
            future_summary_h3=summary,
            future_prefix_prompt=torch.stack([packet, packet + 0.5], dim=0),
            future_snippet_ref=FutureSnippetRef(
                rel_path_id=torch.tensor(0, dtype=torch.long),
                subject_idx=torch.tensor(0, dtype=torch.long),
                trajectory_ord=torch.tensor(0, dtype=torch.long),
                start_boundary_ord=torch.tensor(1, dtype=torch.long),
                stop_boundary_ord=torch.tensor(2, dtype=torch.long),
            ),
        )

    store = materialize_precedent_index_store(
        items=[
            _item(0, 101, packet_a, latent_a, memory_a),
            _item(1, 202, packet_b, latent_b, memory_b),
        ],
        rel_path_vocab=["shards/000000.ptz"],
        num_window_types=4,
    )
    index_path = tmp_path / "precedent_index.pt"
    save_precedent_index_store(index_path, store)

    class _Cfg:
        d_model = 4
        precedent_retrieve_k = 1
        precedent_strict_window_type_match = True
        precedent_support_overlap_bias = 0.0
        precedent_score_temperature = 1.0

    module = AETPrecedentMemory(_Cfg)
    module.load_index(index_path)

    readout = module(
        state_packet=WindowStatePacket(
            slot_tokens=torch.zeros((1, 1, 6, d_model), dtype=torch.float32),
            query_token=packet_a.view(1, 1, d_model),
            window_type_ids=torch.tensor([[1]], dtype=torch.long),
            gap_prev_hours=torch.tensor([[2.0]], dtype=torch.float32),
            duration_hours=torch.tensor([[3.0]], dtype=torch.float32),
        ),
        query_state=latent_a.view(1, 1, d_model),
        memory_digest=memory_a.view(1, 1, d_model),
    )

    assert readout.context_tokens.shape == (1, 1, 4, d_model)
    assert readout.context_summary.shape == (1, 1, d_model)
    assert readout.future_summary.shape[0:2] == (1, 1)
    assert readout.future_embedding.shape == (1, 1, d_model)
    assert readout.query_embedding.shape == (1, 1, d_model)
    assert readout.candidate_future_summaries.shape[0:3] == (1, 1, 1)
    assert readout.candidate_future_embeddings.shape == (1, 1, 1, d_model)
    assert int(readout.matched_item_ids[0, 0, 0].item()) in {0, 1}
    assert int(readout.matched_subject_ids[0, 0, 0].item()) in {101, 202}
    assert torch.isfinite(readout.context_summary).all()

    anchor_ids = module.lookup_anchor_item_ids(
        subject_ids=torch.tensor([[101]], dtype=torch.long),
        trajectory_ords=torch.tensor([[0]], dtype=torch.long),
        boundary_ords=torch.tensor([[0]], dtype=torch.long),
    )
    assert anchor_ids.tolist() == [[0]]

    generation = module.query_generation_prompt(
        state_packet=WindowStatePacket(
            slot_tokens=torch.zeros((1, 1, 6, d_model), dtype=torch.float32),
            query_token=packet_a.view(1, 1, d_model),
            window_type_ids=torch.tensor([[1]], dtype=torch.long),
            gap_prev_hours=torch.tensor([[2.0]], dtype=torch.float32),
            duration_hours=torch.tensor([[3.0]], dtype=torch.float32),
        ),
        query_state=latent_a.view(1, 1, d_model),
        memory_digest=memory_a.view(1, 1, d_model),
        next_window_header=NextWindowHeader(
            window_type_ids=torch.tensor([[2]], dtype=torch.long),
            gap_hours=torch.tensor([[1.0]], dtype=torch.float32),
            duration_hours=torch.tensor([[4.0]], dtype=torch.float32),
            support_flags=torch.zeros((1, 1, 6), dtype=torch.float32),
        ),
    )
    assert generation.summary_prior.shape[0:2] == (1, 1)
    assert generation.prompt_tokens.shape == (1, 1, 2, d_model)
    assert generation.prompt_summary.shape == (1, 1, d_model)
    assert generation.candidate_weights.shape == (1, 1, 1)


def test_model_aux_exports_memory_digests_by_bank() -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
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
        exact_memory_static_slots = 2
        exact_memory_persistent_slots = 2
        exact_memory_episodic_slots = 2
        exact_memory_write_per_window = 1
        exact_memory_retrieve_k = 1
        exact_memory_static_retrieve_k = 1
        exact_memory_persistent_retrieve_k = 1
        exact_memory_episodic_retrieve_k = 1
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
    logits, aux = model(
        input_ids=torch.tensor([[[10, 20], [30, 40]]], dtype=torch.long),
        time_ids=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        numeric_values=torch.zeros((1, 2, 2, 1), dtype=torch.float32),
        numeric_mask=torch.zeros((1, 2, 2), dtype=torch.long),
        token_type_ids=torch.tensor(
            [[[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)],
              [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
            dtype=torch.long,
        ),
        attention_mask=torch.ones((1, 2, 2), dtype=torch.long),
        window_start_times=torch.tensor([[0.0, 12.0]], dtype=torch.float32),
        window_mask=torch.tensor([[1, 1]], dtype=torch.long),
        window_type_ids=torch.tensor([[0, 1]], dtype=torch.long),
        semantic_duration_hours=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        token_event_index=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        token_event_slot_ids=torch.zeros((1, 2, 2), dtype=torch.long),
        event_input_ids=torch.tensor([[[30, 20], [50, 40]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        event_numeric_values=torch.tensor(
            [[[[0.0], [0.2]], [[0.0], [3.0]]]],
            dtype=torch.float32,
        ),
        event_numeric_mask=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        event_type_ids=torch.tensor(
            [[[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)],
              [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
            dtype=torch.long,
        ),
        event_payload_ids=torch.tensor(
            [[
                [EVENT_PAYLOAD_KIND_TO_ID["special"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
                [EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
            ]],
            dtype=torch.long,
        ),
        event_demographic_feature_ids=torch.tensor([[[1, 0], [0, 0]]], dtype=torch.long),
        event_attention_mask=torch.ones((1, 2, 2), dtype=torch.long),
        event_memory_rule_scores=torch.tensor([[[4.0, 0.5], [3.5, 0.25]]], dtype=torch.float32),
        event_memory_group_ids=torch.tensor([[[1, 5], [2, 5]]], dtype=torch.long),
        event_memory_first_flags=torch.tensor([[[1, 0], [1, 0]]], dtype=torch.long),
        event_memory_chronic_flags=torch.tensor([[[0, 0], [1, 0]]], dtype=torch.long),
        return_aux_state=True,
    )

    assert logits["logits_next_window_type"].shape == (1, 2, _Cfg.num_window_types)
    assert aux["patient_memory_context_by_bank"] is not None
    assert aux["patient_memory_state_digests_by_bank"] is not None
    assert set(aux["patient_memory_context_by_bank"]) == {"static", "persistent", "episodic"}
    assert set(aux["patient_memory_state_digests_by_bank"]) == {"static", "persistent", "episodic"}
    assert aux["patient_memory_context_by_bank"]["static"].shape == (1, 2, _Cfg.d_model)
    assert aux["patient_memory_state_digests_by_bank"]["persistent"].shape == (1, 2, _Cfg.d_model)


def test_model_emits_phase5_precedent_generation_outputs(tmp_path: Path) -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer
    from src.ehr_hier.transformer.precedent_memory import (
        materialize_precedent_index_store,
        save_precedent_index_store,
    )
    from src.ehr_hier.transformer.world_model_contract import (
        FutureSnippetRef,
        PrecedentIndexItem,
        compose_precedent_key_state,
        empty_future_summary,
    )

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
        enable_exact_memory = False
        enable_precedent_memory = True
        precedent_retrieve_k = 1
        precedent_strict_window_type_match = False
        precedent_support_overlap_bias = 0.0
        precedent_score_temperature = 1.0
        enable_event_time_nll_head = True
        enable_next_window_gap_nll_head = True
        enable_next_window_duration_nll_head = True
        enable_next_window_support_head = True
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

    packet = torch.arange(_Cfg.d_model, dtype=torch.float32)
    latent = torch.flip(packet, dims=[0])
    memory = torch.ones((_Cfg.d_model,), dtype=torch.float32)
    key_state = compose_precedent_key_state(
        packet_query=packet.view(1, -1),
        latent_state=latent.view(1, -1),
        memory_digest=memory.view(1, -1),
        window_type_ids=torch.tensor([0], dtype=torch.long),
        gap_prev_hours=torch.tensor([0.0], dtype=torch.float32),
        duration_hours=torch.tensor([2.0], dtype=torch.float32),
    ).squeeze(0)
    summary = empty_future_summary(device=torch.device("cpu"), dtype=torch.float32)
    summary.next_window_type_id = torch.tensor(1, dtype=torch.long)
    summary.future_window_count = torch.tensor(1.0, dtype=torch.float32)

    store = materialize_precedent_index_store(
        items=[
            PrecedentIndexItem(
                item_id=torch.tensor(0, dtype=torch.long),
                subject_id=torch.tensor(111, dtype=torch.long),
                trajectory_ord=torch.tensor(0, dtype=torch.long),
                boundary_ord=torch.tensor(0, dtype=torch.long),
                anchor_window_ord=torch.tensor(0, dtype=torch.long),
                current_window_type_id=torch.tensor(0, dtype=torch.long),
                current_window_start_h=torch.tensor(0.0, dtype=torch.float32),
                current_window_duration_h=torch.tensor(2.0, dtype=torch.float32),
                gap_prev_h=torch.tensor(0.0, dtype=torch.float32),
                support_flags=torch.zeros((6,), dtype=torch.float32),
                anchor_mask_flags=torch.zeros((4,), dtype=torch.float32),
                key_state=key_state,
                key_packet=packet,
                key_memory=memory,
                future_summary_h1=summary,
                future_summary_h2=summary,
                future_summary_h3=summary,
                future_prefix_prompt=torch.stack([packet, packet + 1.0], dim=0),
                future_snippet_ref=FutureSnippetRef(
                    rel_path_id=torch.tensor(0, dtype=torch.long),
                    subject_idx=torch.tensor(0, dtype=torch.long),
                    trajectory_ord=torch.tensor(0, dtype=torch.long),
                    start_boundary_ord=torch.tensor(1, dtype=torch.long),
                    stop_boundary_ord=torch.tensor(2, dtype=torch.long),
                ),
            )
        ],
        rel_path_vocab=["shards/000000.ptz"],
        num_window_types=_Cfg.num_window_types,
    )
    index_path = tmp_path / "precedent_index.pt"
    save_precedent_index_store(index_path, store)

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)
    model.load_precedent_index(str(index_path))

    logits, aux = model(
        input_ids=torch.tensor([[[10, 20], [30, 40]]], dtype=torch.long),
        time_ids=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        numeric_values=torch.zeros((1, 2, 2, 1), dtype=torch.float32),
        numeric_mask=torch.zeros((1, 2, 2), dtype=torch.long),
        token_type_ids=torch.tensor(
            [[[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)],
              [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
            dtype=torch.long,
        ),
        attention_mask=torch.ones((1, 2, 2), dtype=torch.long),
        window_start_times=torch.tensor([[0.0, 12.0]], dtype=torch.float32),
        window_mask=torch.tensor([[1, 1]], dtype=torch.long),
        window_type_ids=torch.tensor([[0, 1]], dtype=torch.long),
        semantic_duration_hours=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        token_event_index=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        token_event_slot_ids=torch.zeros((1, 2, 2), dtype=torch.long),
        event_input_ids=torch.tensor([[[30, 20], [50, 40]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        event_numeric_values=torch.tensor(
            [[[[0.0], [0.2]], [[0.0], [3.0]]]],
            dtype=torch.float32,
        ),
        event_numeric_mask=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        event_type_ids=torch.tensor(
            [[[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)],
              [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
            dtype=torch.long,
        ),
        event_payload_ids=torch.tensor(
            [[
                [EVENT_PAYLOAD_KIND_TO_ID["special"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
                [EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
            ]],
            dtype=torch.long,
        ),
        event_attention_mask=torch.ones((1, 2, 2), dtype=torch.long),
        subject_ids=torch.tensor([111], dtype=torch.long),
        trajectory_ords=torch.tensor([0], dtype=torch.long),
        return_aux_state=True,
    )

    assert "precedent_generation_prompt_tokens" in logits
    assert logits["precedent_generation_prompt_tokens"].shape[-2:] == (2, _Cfg.d_model)
    assert "precedent_generation_prompt_summary" in logits
    assert logits["precedent_generation_prompt_summary"].shape == (1, 2, _Cfg.d_model)
    assert "next_window_header_type_ids" in logits
    assert logits["next_window_header_type_ids"].shape == (1, 2)
    assert "pred_next_window_duration_mu" in logits
    assert logits["pred_next_window_duration_mu"].shape == (1, 2)
    assert "logits_next_window_support" in logits
    assert logits["logits_next_window_support"].shape == (1, 2, 6)
    assert aux["precedent_generation"] is not None
    assert aux["precedent_generation_prompt_context"].shape == (1, 2, 1, 2, _Cfg.d_model)
    assert aux["next_window_header"] is not None


def test_phase5_header_fallback_uses_predicted_duration_gap_and_support(tmp_path: Path) -> None:
    from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
    from src.ehr_hier.data.token_types import TokenCategory
    from src.ehr_hier.transformer.model import AdaptiveEpisodicTransformer
    from src.ehr_hier.transformer.precedent_memory import (
        materialize_precedent_index_store,
        save_precedent_index_store,
    )
    from src.ehr_hier.transformer.world_model_contract import (
        FutureSnippetRef,
        PrecedentIndexItem,
        compose_precedent_key_state,
        empty_future_summary,
    )

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
        enable_exact_memory = False
        enable_precedent_memory = True
        precedent_retrieve_k = 1
        precedent_strict_window_type_match = False
        precedent_support_overlap_bias = 0.0
        precedent_score_temperature = 1.0
        enable_event_time_nll_head = True
        enable_next_window_gap_nll_head = True
        enable_next_window_duration_nll_head = True
        enable_next_window_support_head = True
        next_window_gap_nll_min_sigma = 0.1
        next_window_duration_nll_min_sigma = 0.1
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

    packet = torch.arange(_Cfg.d_model, dtype=torch.float32)
    latent = torch.flip(packet, dims=[0])
    memory = torch.ones((_Cfg.d_model,), dtype=torch.float32)
    key_state = compose_precedent_key_state(
        packet_query=packet.view(1, -1),
        latent_state=latent.view(1, -1),
        memory_digest=memory.view(1, -1),
        window_type_ids=torch.tensor([0], dtype=torch.long),
        gap_prev_hours=torch.tensor([0.0], dtype=torch.float32),
        duration_hours=torch.tensor([2.0], dtype=torch.float32),
    ).squeeze(0)
    summary = empty_future_summary(device=torch.device("cpu"), dtype=torch.float32)
    summary.next_window_type_id = torch.tensor(1, dtype=torch.long)
    summary.future_window_count = torch.tensor(1.0, dtype=torch.float32)

    store = materialize_precedent_index_store(
        items=[
            PrecedentIndexItem(
                item_id=torch.tensor(0, dtype=torch.long),
                subject_id=torch.tensor(111, dtype=torch.long),
                trajectory_ord=torch.tensor(0, dtype=torch.long),
                boundary_ord=torch.tensor(0, dtype=torch.long),
                anchor_window_ord=torch.tensor(0, dtype=torch.long),
                current_window_type_id=torch.tensor(0, dtype=torch.long),
                current_window_start_h=torch.tensor(0.0, dtype=torch.float32),
                current_window_duration_h=torch.tensor(2.0, dtype=torch.float32),
                gap_prev_h=torch.tensor(0.0, dtype=torch.float32),
                support_flags=torch.zeros((6,), dtype=torch.float32),
                anchor_mask_flags=torch.zeros((4,), dtype=torch.float32),
                key_state=key_state,
                key_packet=packet,
                key_memory=memory,
                future_summary_h1=summary,
                future_summary_h2=summary,
                future_summary_h3=summary,
                future_prefix_prompt=torch.stack([packet, packet + 1.0], dim=0),
                future_snippet_ref=FutureSnippetRef(
                    rel_path_id=torch.tensor(0, dtype=torch.long),
                    subject_idx=torch.tensor(0, dtype=torch.long),
                    trajectory_ord=torch.tensor(0, dtype=torch.long),
                    start_boundary_ord=torch.tensor(1, dtype=torch.long),
                    stop_boundary_ord=torch.tensor(2, dtype=torch.long),
                ),
            )
        ],
        rel_path_vocab=["shards/000000.ptz"],
        num_window_types=_Cfg.num_window_types,
    )
    index_path = tmp_path / "precedent_index.pt"
    save_precedent_index_store(index_path, store)

    model = AdaptiveEpisodicTransformer(_Cfg, vocab_config)
    model.load_precedent_index(str(index_path))
    with torch.no_grad():
        for module in (
            model.next_window_gap_nll_head,
            model.next_window_duration_nll_head,
        ):
            assert module is not None
            for param in module.parameters():
                param.zero_()
        assert model.next_window_type_head is not None
        model.next_window_type_head.weight.zero_()
        model.next_window_type_head.bias.copy_(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32))
        model.next_window_gap_nll_head[-1].bias.copy_(
            torch.tensor([torch.log1p(torch.tensor(5.0)).item(), -20.0], dtype=torch.float32)
        )
        model.next_window_duration_nll_head[-1].bias.copy_(
            torch.tensor([torch.log1p(torch.tensor(4.0)).item(), -20.0], dtype=torch.float32)
        )
        assert model.next_window_support_head is not None
        model.next_window_support_head.weight.zero_()
        model.next_window_support_head.bias.copy_(
            torch.tensor([10.0, -10.0, 10.0, -10.0, 10.0, -10.0], dtype=torch.float32)
        )

    logits, aux = model(
        input_ids=torch.tensor([[[10, 20], [30, 40]]], dtype=torch.long),
        time_ids=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        numeric_values=torch.zeros((1, 2, 2, 1), dtype=torch.float32),
        numeric_mask=torch.zeros((1, 2, 2), dtype=torch.long),
        token_type_ids=torch.tensor(
            [[[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)],
              [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
            dtype=torch.long,
        ),
        attention_mask=torch.ones((1, 2, 2), dtype=torch.long),
        token_event_index=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        token_event_slot_ids=torch.zeros((1, 2, 2), dtype=torch.long),
        event_input_ids=torch.tensor([[[30, 20], [50, 40]]], dtype=torch.long),
        event_time_ids=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        event_numeric_values=torch.tensor(
            [[[[0.0], [0.2]], [[0.0], [3.0]]]],
            dtype=torch.float32,
        ),
        event_numeric_mask=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        event_type_ids=torch.tensor(
            [[[int(TokenCategory.SPECIAL), int(TokenCategory.MEASUREMENT)],
              [int(TokenCategory.DIAGNOSIS), int(TokenCategory.MEASUREMENT)]]],
            dtype=torch.long,
        ),
        event_payload_ids=torch.tensor(
            [[
                [EVENT_PAYLOAD_KIND_TO_ID["special"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
                [EVENT_PAYLOAD_KIND_TO_ID["symbolic_code"], EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"]],
            ]],
            dtype=torch.long,
        ),
        event_attention_mask=torch.ones((1, 2, 2), dtype=torch.long),
        return_aux_state=True,
    )

    header = aux["next_window_header"]
    assert header is not None
    assert torch.equal(header.window_type_ids, torch.tensor([[2, 2]], dtype=torch.long))
    assert torch.allclose(header.gap_hours, torch.full((1, 2), 5.0), atol=0.2)
    assert torch.allclose(header.duration_hours, torch.full((1, 2), 4.0), atol=0.2)
    assert header.support_flags is not None
    expected_support = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose((header.support_flags[:, :1, :] > 0.5).to(dtype=torch.float32), expected_support)
    assert "precedent_generation_prompt_tokens" in logits
