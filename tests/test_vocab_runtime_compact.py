import torch


def test_dense_id_remapper_lookup_mode_maps_sparse_ids() -> None:
    from ehr_hier.transformer.vocab_runtime import DenseIdBlock, DenseIdRemapper

    blocks = [
        DenseIdBlock(
            name="special",
            head="logits_struct",
            global_offset=100,
            source_size=11,
            dense_offset=0,
            dense_size=3,
            mode="lookup",
            sparse_global_ids=(100, 103, 110),
        ),
        DenseIdBlock(
            name="rvq",
            head="logits_rvq",
            global_offset=200,
            source_size=4,
            dense_offset=3,
            dense_size=4,
            mode="identity",
        ),
    ]
    remapper = DenseIdRemapper(blocks, unk_dense_id=99)
    ids = torch.tensor([[99, 100, 101, 103, 110, 111, 200, 203]], dtype=torch.long)
    valid = torch.ones_like(ids)
    out, stats = remapper.map_tensor(ids, valid_mask=valid)

    assert out.tolist()[0] == [99, 0, 99, 1, 2, 99, 3, 6]
    assert stats["mapped_tokens"] == 5
    assert stats["unmapped_tokens"] == 3
    assert stats["block_hits"]["special"] == 3
    assert stats["block_hits"]["rvq"] == 2

    payload = remapper.serialize()
    remapper2 = DenseIdRemapper.from_serialized(payload)
    out2, stats2 = remapper2.map_tensor(ids, valid_mask=valid)
    assert torch.equal(out, out2)
    assert stats2["mapped_tokens"] == stats["mapped_tokens"]


def test_compact_runtime_builder_reduces_total_size() -> None:
    from ehr_hier.transformer.vocab_runtime import (
        DenseIdBlock,
        DenseIdRemapper,
        build_compact_runtime_vocab_and_remapper,
    )

    base_blocks = [
        DenseIdBlock("special", "logits_struct", 0, 50, 0, 50, "identity"),
        DenseIdBlock("measurement_value", "logits_rvq", 200, 16, 50, 16, "identity"),
        DenseIdBlock("observation_code", "logits_meas", 300, 1000, 66, 1000, "identity"),
        DenseIdBlock("medication", "logits_medtok", 2000, 500, 1066, 500, "identity"),
    ]
    base_remapper = DenseIdRemapper(base_blocks, unk_dense_id=0)
    base_vocab = {
        "window_markers": {
            "type_token_offset": 10,
            "num_types": 4,
            "end_token_id": 14,
            "continue_token_id": 15,
            "unk_type_id": 0,
        }
    }
    observed = {
        "special": [0, 1, 2, 10, 11, 12, 13, 14, 15],
        "measurement_value": [200, 201, 202],
        "observation_code": [300, 350, 999],
        "medication": [2000, 2001, 2009],
    }

    compact_vocab, compact_remapper = build_compact_runtime_vocab_and_remapper(
        base_vocab_config=base_vocab,
        base_remapper=base_remapper,
        observed_ids_by_block=observed,
        preserve_full_blocks={"measurement_value"},
    )
    assert compact_vocab["total_size"] < (50 + 16 + 1000 + 500)
    assert compact_vocab["size_rvq"] == 16

    x = torch.tensor([[0, 10, 15, 300, 999, 2009, 2300, 202]], dtype=torch.long)
    out, stats = compact_remapper.map_tensor(x, valid_mask=torch.ones_like(x))
    assert stats["mapped_tokens"] == 7
    assert stats["unmapped_tokens"] == 1
    # unmapped token stays at UNK dense id (0)
    assert int(out[0, 6].item()) == 0
