import torch


def _grammar():
    from ehr_hier.transformer.generation import WindowGenerationGrammar

    vocab_config = {
        "offsets": {"SPECIAL": 0},
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
        },
    }
    return WindowGenerationGrammar.from_vocab_config(vocab_config)


def test_generation_mask_inside_chunk_blocks_marker_tokens() -> None:
    from ehr_hier.transformer.generation import GenerationState

    g = _grammar()
    mask = g.legal_token_mask(vocab_size=64, state=GenerationState.INSIDE_CHUNK)
    assert bool(mask[5].item()) is True
    assert bool(mask[10].item()) is False
    assert bool(mask[14].item()) is False
    assert bool(mask[15].item()) is False


def test_generation_mask_chunk_end_boundary_candidate_allows_transition_tokens() -> None:
    from ehr_hier.transformer.generation import GenerationState

    g = _grammar()
    mask_no_boundary = g.legal_token_mask(
        vocab_size=64,
        state=GenerationState.CHUNK_END,
        boundary_candidate=False,
    )
    assert bool(mask_no_boundary[15].item()) is True
    assert bool(mask_no_boundary[14].item()) is False
    assert bool(mask_no_boundary[10].item()) is False

    mask_boundary = g.legal_token_mask(
        vocab_size=64,
        state=GenerationState.CHUNK_END,
        boundary_candidate=True,
    )
    assert bool(mask_boundary[15].item()) is True
    assert bool(mask_boundary[14].item()) is True
    assert bool(mask_boundary[10].item()) is True


def test_generation_illegal_token_rate_scripted_rollout() -> None:
    from ehr_hier.transformer.generation import GenerationState

    g = _grammar()
    # content, content, continue, open-next-type
    token_ids = [5, 6, 15, 12]
    states = [
        GenerationState.INSIDE_CHUNK,
        GenerationState.INSIDE_CHUNK,
        GenerationState.CHUNK_END,
        GenerationState.WINDOW_END,
    ]
    boundary = [False, False, False, True]
    rate = g.illegal_token_rate(
        token_ids=token_ids,
        states=states,
        boundary_candidates=boundary,
        vocab_size=64,
    )
    assert rate == 0.0

    bad = [5, 14]  # WIN_END is illegal inside_chunk
    bad_states = [GenerationState.INSIDE_CHUNK, GenerationState.INSIDE_CHUNK]
    bad_rate = g.illegal_token_rate(token_ids=bad, states=bad_states, vocab_size=64)
    assert bad_rate > 0.0


def test_generation_apply_legal_mask_sets_illegal_logits_to_neg_inf() -> None:
    from ehr_hier.transformer.generation import GenerationState

    g = _grammar()
    logits = torch.zeros((64,), dtype=torch.float)
    masked = g.apply_legal_mask(
        logits,
        state=GenerationState.CHUNK_END,
        boundary_candidate=False,
    )
    assert torch.isfinite(masked[15])
    assert torch.isneginf(masked[14])
    assert torch.isneginf(masked[10])

