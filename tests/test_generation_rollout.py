import torch


def test_rollout_state_from_collated_batch_trims_trailing_marker() -> None:
    from ehr_hier.data.token_types import EventToken, TokenCategory
    from ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig
    from ehr_hier.transformer.generation import RolloutSubjectState, WindowGenerationGrammar

    summary = EventToken(
        value_id=1,
        category_id=int(TokenCategory.SPECIAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={},
        num_attrs={},
    )
    token = EventToken(
        value_id=30,
        category_id=int(TokenCategory.MEASUREMENT),
        t_from_start_hours=1.0,
        dt_from_prev_hours=1.0,
        cat_attrs={"window_type_id": 1},
        num_attrs={"numeric_value": 1.0},
    )
    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=6,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, type_token_offset=10, num_types=4, end_token_id=14),
    )
    batch = collator([[summary, token]])
    grammar = WindowGenerationGrammar.from_vocab_config(
        {
            "offsets": {"SPECIAL": 0},
            "window_markers": {"type_token_offset": 10, "num_types": 4, "end_token_id": 14, "continue_token_id": 15},
        }
    )
    state = RolloutSubjectState.from_collated_batch(
        batch,
        sample_idx=0,
        grammar=grammar,
        trim_trailing_marker=True,
    )

    assert state.global_special_ids == [1]
    assert state.input_ids == [[[1, 11, 30]]]
    assert state.token_type_ids == [[[0, 0, 1]]]


def test_rollout_state_truncate_to_window_prefix_trims_last_marker() -> None:
    from ehr_hier.transformer.generation import RolloutSubjectState, WindowGenerationGrammar

    grammar = WindowGenerationGrammar.from_vocab_config(
        {
            "offsets": {"SPECIAL": 0},
            "window_markers": {"type_token_offset": 10, "num_types": 4, "end_token_id": 14, "continue_token_id": 15},
        }
    )
    state = RolloutSubjectState(
        input_ids=[[[1, 11, 30, 14]], [[1, 12, 31, 14]]],
        time_ids=[[[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]],
        numeric_values=[[[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]],
        numeric_mask=[[[0, 0, 0, 0]], [[0, 0, 0, 0]]],
        token_type_ids=[[[0, 0, 1, 0]], [[0, 0, 1, 0]]],
        window_type_ids=[1, 2],
        window_start_times=[0.0, 2.0],
        chunk_start_offsets=[[0.0], [0.0]],
        chunk_start_times=[[0.0], [2.0]],
        chunk_is_last=[[1], [1]],
        global_special_ids=[1],
        pad_id=0,
    )

    state.truncate_to_window_prefix(1, grammar=grammar, trim_trailing_marker=True)

    assert state.window_type_ids == [1]
    assert state.input_ids == [[[1, 11, 30]]]
    assert state.token_type_ids == [[[0, 0, 1]]]


class _ScriptedRolloutModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int, num_window_types: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_window_types = int(num_window_types)
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        input_ids,
        time_ids,
        numeric_values,
        numeric_mask,
        token_type_ids,
        attention_mask,
        window_start_times,
        window_mask,
        window_type_ids,
        chunk_mask,
        chunk_start_offsets,
        chunk_is_last,
    ):
        B, W, C, L = input_ids.shape
        logits_token = torch.full((B, W, C, L, self.vocab_size), -50.0, dtype=torch.float32, device=input_ids.device)
        logits_next_window_type = torch.zeros((B, W, self.num_window_types), dtype=torch.float32, device=input_ids.device)
        logits_transition_boundary = torch.zeros((B, W, C, L, 2), dtype=torch.float32, device=input_ids.device)
        logits_boundary_next_window_type = torch.zeros(
            (B, W, C, L, self.num_window_types),
            dtype=torch.float32,
            device=input_ids.device,
        )

        active_w = int(window_mask[0].to(dtype=torch.long).sum().item()) - 1
        active_c = int(chunk_mask[0, active_w].to(dtype=torch.long).sum().item()) - 1
        seq_len = int(attention_mask[0, active_w, active_c].to(dtype=torch.long).sum().item())
        pos = seq_len - 1
        content_count = int((token_type_ids[0, active_w, active_c, :seq_len] != 0).to(dtype=torch.long).sum().item())

        if active_w == 0 and active_c == 0 and content_count == 0:
            logits_token[0, active_w, active_c, pos, 30] = 10.0
        elif active_w == 0 and active_c == 0 and content_count == 1:
            logits_token[0, active_w, active_c, pos, 31] = 9.0
            logits_token[0, active_w, active_c, pos, 15] = 12.0
            logits_transition_boundary[0, active_w, active_c, pos, 0] = 3.0
        elif active_w == 0 and active_c == 1 and content_count == 0:
            logits_token[0, active_w, active_c, pos, 32] = 10.0
        elif active_w == 0 and active_c == 1 and content_count == 1:
            logits_token[0, active_w, active_c, pos, 33] = 8.0
            logits_token[0, active_w, active_c, pos, 14] = 12.0
            logits_transition_boundary[0, active_w, active_c, pos, 1] = 3.0
            logits_next_window_type[0, active_w, 2] = 10.0
            logits_boundary_next_window_type[0, active_w, active_c, pos, 2] = 5.0
        else:
            logits_token[0, active_w, active_c, pos, 34] = 10.0

        return {
            "logits_token": logits_token,
            "logits_next_window_type": logits_next_window_type,
            "logits_transition_boundary": logits_transition_boundary,
            "logits_boundary_next_window_type": logits_boundary_next_window_type,
        }, torch.zeros((B, 1), dtype=torch.float32, device=input_ids.device)


def test_rollout_subject_with_model_opens_new_chunk_and_window() -> None:
    from ehr_hier.transformer.generation import (
        RolloutConfig,
        RolloutSubjectState,
        rollout_subject_with_model,
    )

    vocab_config = {
        "total_size": 64,
        "dense_blocks": [
            {
                "name": "special",
                "head": "logits_struct",
                "global_offset": 0,
                "source_size": 20,
                "dense_offset": 0,
                "dense_size": 20,
                "sparse_global_ids": list(range(20)),
            },
            {
                "name": "measurement_code",
                "head": "logits_meas",
                "global_offset": 100,
                "source_size": 44,
                "dense_offset": 20,
                "dense_size": 44,
                "sparse_global_ids": list(range(100, 144)),
            },
        ],
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
            "end_mode": "end_token",
        },
    }
    state = RolloutSubjectState(
        input_ids=[[[1, 11]]],
        time_ids=[[[0.0, 0.0]]],
        numeric_values=[[[0.0, 0.0]]],
        numeric_mask=[[[0, 0]]],
        token_type_ids=[[[0, 0]]],
        window_type_ids=[1],
        window_start_times=[0.0],
        chunk_start_offsets=[[0.0]],
        chunk_start_times=[[0.0]],
        chunk_is_last=[[1]],
        global_special_ids=[1],
        pad_id=0,
    )
    model = _ScriptedRolloutModel(vocab_size=64, num_window_types=4)
    result = rollout_subject_with_model(
        model=model,
        vocab_config=vocab_config,
        subject_state=state,
        config=RolloutConfig(
            max_new_tokens=4,
            max_new_windows=1,
            max_chunks_per_window=4,
            max_content_tokens_per_chunk=2,
            min_content_tokens_per_chunk=1,
            boundary_logit_margin=0.0,
            trim_trailing_marker=False,
        ),
    )

    assert result["num_generated_tokens"] == 4
    assert result["num_opened_windows"] == 1
    assert [step["token_kind"] for step in result["steps"]] == [
        "content",
        "chunk_continue",
        "content",
        "window_end",
    ]
    assert result["steps"][1]["inserted_prefix_token_ids"] == [1, 11]
    assert result["steps"][3]["inserted_prefix_token_ids"] == [1, 12]
    final_state = result["final_state"]
    assert len(final_state) == 2
    assert [item["token_id"] for item in final_state[0]["chunks"][0]["sequence"]] == [1, 11, 30, 15]
    assert [item["token_id"] for item in final_state[0]["chunks"][1]["sequence"]] == [1, 11, 32, 14]
    assert [item["token_id"] for item in final_state[1]["chunks"][0]["sequence"]] == [1, 12]
