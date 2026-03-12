from __future__ import annotations

import json

import yaml


def test_runtime_vocab_builder_and_dense_remapper(tmp_path) -> None:
    from ehr_hier.transformer.vocab_runtime import build_runtime_vocab_and_remapper

    contract_fp = tmp_path / "tokenization_v1.yaml"
    manifest_fp = tmp_path / "vocab_manifest.json"
    medtok_dir = tmp_path / "medtok"
    medtok_dir.mkdir(parents=True, exist_ok=True)

    contract_fp.write_text(
        yaml.safe_dump(
            {
                "frozen_ranges": {
                    "special": {"offset": 0, "reserved_max_id": 31},
                    "diagnosis": {"offset": 1000000},
                    "diagnosis_residual": {"offset": 1160000, "buckets": 99},
                    "procedure": {"offset": 1200000},
                    "procedure_residual": {"offset": 1360000, "buckets": 99},
                    "medication": {"offset": 1400000},
                    "medication_residual": {"offset": 1800000, "buckets": 99},
                    "measurement_code": {"offset": 2000000},
                    "measurement_value": {"offset": 2100000},
                    "structural": {"offset": 2200000},
                    "observation_code": {"offset": 2300000},
                    "observation_value": {"offset": 2320000},
                },
                "residual_fallback": {
                    "enabled": True,
                    "buckets": 99,
                    "offsets": {
                        "diagnosis": 1160000,
                        "procedure": 1360000,
                        "medication": 1800000,
                    },
                },
                "window_markers": {
                    "enabled": True,
                    "end_mode": "end_token",
                    "type_token_offset": 10,
                    "num_types": 7,
                    "unk_type_id": 0,
                    "end_token_id": 17,
                    "continue_token_id": 18,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_fp.write_text(
        json.dumps(
            {
                "special": {"offset": 0},
                "diagnosis": {"offset": 1000000},
                "diagnosis_residual": {"offset": 1160000, "buckets": 99},
                "procedure": {"offset": 1200000},
                "procedure_residual": {"offset": 1360000, "buckets": 99},
                "medication": {"offset": 1400000},
                "medication_residual": {"offset": 1800000, "buckets": 99},
                "measurement_code": {"offset": 2000000},
                "measurement_value": {"offset": 2100000},
                "structural": {"offset": 2200000},
                "observation_code": {"offset": 2300000},
                "observation_value": {"offset": 2320000},
            }
        ),
        encoding="utf-8",
    )
    structural_fp = tmp_path / "structural_codes.yaml"
    structural_fp.write_text(
        yaml.safe_dump(
            {
                "structural_map": {
                    "ADMISSION": "START_ADM",
                    "DISCHARGE": "END_ADM",
                },
                "transition_map": {"ADMISSION": "open_next"},
                "window_types": {"UNK": 0, "INPATIENT": 1},
                "window_type_map": {"ADMISSION": "INPATIENT"},
            }
        ),
        encoding="utf-8",
    )
    (medtok_dir / "diag_vocab.json").write_text(json.dumps({"<UNK>": 0, "I10": 1}), encoding="utf-8")
    (medtok_dir / "proc_vocab.json").write_text(json.dumps({"<UNK>": 0, "XYZ": 1}), encoding="utf-8")
    (medtok_dir / "med_vocab.json").write_text(json.dumps({"<UNK>": 0}), encoding="utf-8")

    vocab_config, remapper = build_runtime_vocab_and_remapper(
        tokenization_contract=contract_fp,
        vocab_manifest=manifest_fp,
        structural_yaml=structural_fp,
        medtok_vocab_dir=medtok_dir,
        measurement_code_size=128,
        rvq_size=32,
    )

    assert vocab_config["total_size"] > 0
    assert "routing" in vocab_config
    assert set(vocab_config["routing"].keys()) == {
        "logits_struct",
        "logits_rvq",
        "logits_meas",
        "logits_medtok",
    }
    assert vocab_config["window_markers"]["end_token_id"] == 17

    import torch

    ids = torch.tensor([[0, 17, 2000001, 2100010, 2200001, 9999999]], dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.long)
    mapped, stats = remapper.map_tensor(ids, valid_mask=valid)
    assert mapped.shape == ids.shape
    assert stats["total_tokens"] == 6
    assert stats["mapped_tokens"] >= 5
    assert stats["unmapped_tokens"] <= 1
    assert int(mapped[0, 0].item()) == 0

    from ehr_hier.transformer.vocab_runtime import DenseIdRemapper

    remapper_roundtrip = DenseIdRemapper.from_serialized(remapper.serialize())
    mapped2, stats2 = remapper_roundtrip.map_tensor(ids, valid_mask=valid)
    assert mapped2.tolist() == mapped.tolist()
    assert stats2["unmapped_tokens"] == stats["unmapped_tokens"]


def test_sparse_vocab_contract_becomes_runtime_source_of_truth(tmp_path) -> None:
    from ehr_hier.tokenizers.vocab_contract import (
        build_legacy_manifest_from_sparse_contract,
        build_sparse_vocab_contract,
    )
    from ehr_hier.transformer.vocab_runtime import build_runtime_vocab_and_remapper

    contract_fp = tmp_path / "tokenization_v1.yaml"
    manifest_fp = tmp_path / "vocab_manifest.json"
    structural_fp = tmp_path / "structural_codes.yaml"
    medtok_dir = tmp_path / "medtok"
    medtok_attr_dir = tmp_path / "medtok_attrs"
    medtok_dir.mkdir(parents=True, exist_ok=True)
    medtok_attr_dir.mkdir(parents=True, exist_ok=True)

    contract_fp.write_text(
        yaml.safe_dump(
            {
                "frozen_ranges": {
                    "special": {"offset": 0, "reserved_max_id": 31},
                    "diagnosis": {"offset": 1000000},
                    "diagnosis_residual": {"offset": 1160000, "buckets": 99},
                    "procedure": {"offset": 1200000},
                    "procedure_residual": {"offset": 1360000, "buckets": 99},
                    "medication": {"offset": 1400000},
                    "medication_residual": {"offset": 1800000, "buckets": 99},
                    "measurement_code": {"offset": 2000000},
                    "measurement_value": {"offset": 2100000},
                    "structural": {"offset": 2200000},
                    "observation_code": {"offset": 2300000},
                    "observation_value": {"offset": 2320000},
                    "med_route": {"offset": 1600000},
                    "med_form": {"offset": 1620000},
                    "med_freq": {"offset": 1640000},
                    "med_unit": {"offset": 1660000},
                    "med_dosage": {"offset": 1680000},
                    "med_rate": {"offset": 1700000},
                    "med_duration": {"offset": 1720000},
                },
                "residual_fallback": {
                    "enabled": True,
                    "buckets": 99,
                    "offsets": {
                        "diagnosis": 1160000,
                        "procedure": 1360000,
                        "medication": 1800000,
                    },
                },
                "window_markers": {
                    "enabled": True,
                    "end_mode": "end_token",
                    "type_token_offset": 10,
                    "num_types": 7,
                    "unk_type_id": 0,
                    "end_token_id": 17,
                    "continue_token_id": 18,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_fp.write_text(
        json.dumps(
            {
                "special": {"offset": 0},
                "diagnosis": {"offset": 1000000},
                "procedure": {"offset": 1200000},
                "medication": {"offset": 1400000},
                "measurement_code": {"offset": 2000000},
                "measurement_value": {"offset": 2100000},
                "structural": {"offset": 2200000},
                "observation_code": {"offset": 2300000},
                "observation_value": {"offset": 2320000},
            }
        ),
        encoding="utf-8",
    )
    structural_fp.write_text(
        yaml.safe_dump(
            {
                "structural_map": {
                    "ADMISSION": "START_ADM",
                    "DISCHARGE": "END_ADM",
                },
                "transition_map": {"ADMISSION": "open_next"},
                "window_types": {"UNK": 0, "INPATIENT": 1},
                "window_type_map": {"ADMISSION": "INPATIENT"},
            }
        ),
        encoding="utf-8",
    )
    (medtok_dir / "diag_vocab.json").write_text(json.dumps({"<UNK>": 0, "I10": 1}), encoding="utf-8")
    (medtok_dir / "proc_vocab.json").write_text(json.dumps({"<UNK>": 0, "XYZ": 1}), encoding="utf-8")
    (medtok_dir / "med_vocab.json").write_text(json.dumps({"<UNK>": 0, "RXNORM//1": 1}), encoding="utf-8")
    (medtok_dir / "med_fallback_vocab.json").write_text(
        json.dumps({"<UNK>": 0, "MEDICATION//ACETAMINOPHEN": 1, "MEDICATION//FUROSEMIDE": 2}),
        encoding="utf-8",
    )
    (medtok_dir / "obs_code_vocab.json").write_text(
        json.dumps({"<UNK>": 0, "BLOOD PRESSURE::Blood Pressure": 1}),
        encoding="utf-8",
    )
    (medtok_dir / "obs_value_vocab.json").write_text(
        json.dumps({"<UNK>": 0, "UNK": 1, "N/A": 2, "NONE": 3, "": 4, "120/80": 5}),
        encoding="utf-8",
    )
    (medtok_attr_dir / "route_vocab.json").write_text(json.dumps({"<UNK>": 0, "IV": 1}), encoding="utf-8")

    sparse_contract = build_sparse_vocab_contract(
        tokenization_contract=contract_fp,
        vocab_manifest=manifest_fp,
        structural_yaml=structural_fp,
        medtok_vocab_dir=medtok_dir,
        medtok_attr_dir=medtok_attr_dir,
        measurement_code_size=128,
        rvq_size=32,
    )
    assert sparse_contract["families"]["diagnosis"]["offset"] == 1000000
    assert sparse_contract["families"]["med_route"]["source_size"] == 2
    assert sparse_contract["families"]["structural"]["runtime_head"] == "logits_struct"
    assert sparse_contract["families"]["medication_residual"]["source_size"] == 3
    assert sparse_contract["families"]["observation_code"]["source_size"] == 2
    assert sparse_contract["families"]["observation_value"]["source_size"] == 6
    assert sparse_contract["families"]["medication_residual"]["family_type"] == "residual_exact"
    assert sparse_contract["residual_fallback"]["families"]["medication"]["mode"] == "exact_vocab"
    assert sparse_contract["residual_fallback"]["families"]["medication"]["tail_policy"] == "drop"
    assert sparse_contract["legacy_sources"]["vocab_manifest"] is None
    assert sparse_contract["structural_contract"]["transition_map"]["ADMISSION"] == "open_next"
    assert "ADMISSION" in sparse_contract["structural_contract"]["surface_vocab_codes"]
    assert (
        sparse_contract["structural_contract"]["builder_policy"]["transition_map_is_authoritative_when_codebook_present"]
        is True
    )

    legacy_manifest = build_legacy_manifest_from_sparse_contract(sparse_contract)
    assert legacy_manifest["diagnosis"]["offset"] == 1000000
    assert legacy_manifest["window_end"]["id"] == 17

    vocab_config, remapper = build_runtime_vocab_and_remapper(
        sparse_vocab_contract=sparse_contract,
    )
    assert vocab_config["sparse_vocab_contract"]["families"]["diagnosis"]["offset"] == 1000000
    assert vocab_config["window_markers"]["continue_token_id"] == 18

    import torch

    ids = torch.tensor([[0, 17, 2000001, 2100003, 2200001, 1400001]], dtype=torch.long)
    mapped, stats = remapper.map_tensor(ids, valid_mask=torch.ones_like(ids))
    assert mapped.shape == ids.shape
    assert stats["unmapped_tokens"] == 0


def test_sparse_vocab_contract_requires_explicit_medtok_source(tmp_path) -> None:
    from ehr_hier.tokenizers.vocab_contract import build_sparse_vocab_contract

    contract_fp = tmp_path / "tokenization_v1.yaml"
    contract_fp.write_text(
        yaml.safe_dump(
            {
                "frozen_ranges": {
                    "special": {"offset": 0, "reserved_max_id": 31},
                    "diagnosis": {"offset": 1000000},
                    "procedure": {"offset": 1200000},
                    "medication": {"offset": 1400000},
                    "measurement_code": {"offset": 2000000},
                    "measurement_value": {"offset": 2100000},
                    "structural": {"offset": 2200000},
                    "observation_code": {"offset": 2300000},
                    "observation_value": {"offset": 2320000},
                },
                "window_markers": {
                    "enabled": True,
                    "type_token_offset": 10,
                    "num_types": 7,
                    "end_token_id": 17,
                    "continue_token_id": 18,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        build_sparse_vocab_contract(
            tokenization_contract=contract_fp,
            measurement_code_size=128,
            rvq_size=32,
        )
        assert False, "Expected missing MedTok source to raise"
    except ValueError as exc:
        assert "MedTok source" in str(exc)


def test_collator_reports_overflow_and_remap_stats_and_next_type_fallback() -> None:
    from ehr_hier.data.token_types import EventToken, TokenCategory
    from ehr_hier.data.window_segmentation import SegmentedChunk
    from ehr_hier.transformer.collator import AETHierarchicalCollator, WindowMarkerConfig
    from ehr_hier.transformer.vocab_runtime import DenseIdBlock, DenseIdRemapper

    remapper = DenseIdRemapper(
        [
            DenseIdBlock(
                name="special",
                head="logits_struct",
                global_offset=0,
                source_size=64,
                dense_offset=0,
                dense_size=64,
                mode="identity",
            ),
            DenseIdBlock(
                name="measurement_code",
                head="logits_meas",
                global_offset=2000000,
                source_size=2048,
                dense_offset=64,
                dense_size=2048,
                mode="identity",
            ),
        ],
        unk_dense_id=0,
    )

    summary = EventToken(
        value_id=1,
        category_id=int(TokenCategory.SPECIAL),
        t_from_start_hours=0.0,
        dt_from_prev_hours=0.0,
        cat_attrs={},
        num_attrs={},
    )
    timeline = [summary]
    for idx in range(24):
        t = float(idx // 3)
        timeline.append(
            EventToken(
                value_id=2000000 + (idx % 10) + 1,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=t,
                dt_from_prev_hours=0.0 if idx > 0 else 1.0,
                cat_attrs={},
                num_attrs={"numeric_value": float(idx)},
            )
        )

    collator = AETHierarchicalCollator(
        max_windows=4,
        max_chunks_per_window=2,
        max_len_per_window=6,
        pad_id=0,
        window_markers=WindowMarkerConfig(enabled=True, end_mode="next_type", type_token_offset=10, num_types=4),
        id_remapper=remapper,
    )

    batch = collator([timeline])
    assert "overflow_stats" in batch
    assert "id_remap_stats" in batch
    assert int(batch["overflow_stats"]["chunks_dropped_by_max_chunks"]) > 0
    assert float(batch["id_remap_stats"]["unmapped_frac"]) == 0.0

    # Exercise the private next-type fallback path where next_start_abs is None.
    chunk = SegmentedChunk(
        tokens=[
            EventToken(
                value_id=2000001,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=12.0,
                dt_from_prev_hours=0.0,
                cat_attrs={},
                num_attrs={"numeric_value": 1.0},
            )
        ],
        start_time_hours=12.0,
        chunk_index=0,
        is_first_chunk=True,
        is_last_chunk=True,
    )
    ids, *_ = collator._process_chunk(
        chunk,
        special_tokens=[],
        w_type_id=1,
        w_start_abs=12.0,
        next_type_id=2,
        next_start_abs=None,
    )
    assert ids[-1] == 12  # type_token_offset(10) + next_type_id(2)


def test_runtime_bundle_loader_recovers_top_level_sparse_contract(tmp_path) -> None:
    from ehr_hier.transformer.vocab_runtime import load_runtime_vocab_bundle

    bundle_fp = tmp_path / "runtime_bundle.json"
    bundle_fp.write_text(
        json.dumps(
            {
                "sparse_vocab_contract": {
                    "families": {
                        "special": {"offset": 0, "source_size": 4},
                    }
                },
                "vocab_config": {
                    "total_size": 4,
                    "size_special": 4,
                    "size_rvq": 0,
                    "size_meas_labels": 0,
                    "size_meds": 0,
                    "routing": {"logits_struct": [{"offset": 0, "size": 4, "name": "special"}]},
                    "window_markers": {"type_token_offset": 10, "num_types": 1, "end_token_id": 11, "continue_token_id": 12},
                    "dense_blocks": [
                        {
                            "name": "special",
                            "head": "logits_struct",
                            "global_offset": 0,
                            "source_size": 4,
                            "global_max": 3,
                            "dense_offset": 0,
                            "dense_size": 4,
                            "mode": "identity",
                            "sparse_global_ids": None,
                        }
                    ],
                },
                "id_remapper": {
                    "unk_dense_id": 0,
                    "blocks": [
                        {
                            "name": "special",
                            "head": "logits_struct",
                            "global_offset": 0,
                            "source_size": 4,
                            "dense_offset": 0,
                            "dense_size": 4,
                            "mode": "identity",
                            "sparse_global_ids": None,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    vocab_config, _ = load_runtime_vocab_bundle(bundle_fp)
    assert vocab_config["sparse_vocab_contract"]["families"]["special"]["offset"] == 0
