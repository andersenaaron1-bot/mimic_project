from __future__ import annotations


def test_tokenization_freeze_check_passes_with_clean_payload() -> None:
    from scripts.check_tokenization_freeze_v1 import evaluate_tokenization_freeze

    audit_payload = {
        "timeline": {
            "semantic_effective_capture_by_category": {
                "DIAGNOSIS": {
                    "mapped_rate_over_semantic_total": 1.0,
                    "medtok_only_rate_over_semantic_total": 0.78,
                    "residual_hash": 0,
                },
                "PROCEDURE": {
                    "mapped_rate_over_semantic_total": 1.0,
                    "medtok_only_rate_over_semantic_total": 0.85,
                    "residual_hash": 0,
                },
                "MEDICATION": {
                    "mapped_rate_over_semantic_total": 1.0,
                    "medtok_only_rate_over_semantic_total": 0.22,
                    "residual_hash": 0,
                },
            }
        },
        "collation": {"window_type_unk_frac": 0.0},
    }
    runtime_payload = {
        "qual_obs_tail_policy": "drop",
        "sparse_vocab_contract": {
            "residual_fallback": {
                "tail_policy": "drop",
                "families": {
                    "diagnosis": {"tail_policy": "drop"},
                    "procedure": {"tail_policy": "drop"},
                    "medication": {"tail_policy": "drop"},
                },
            }
        },
        "summary": {
            "preserve_full_blocks": ["special", "structural"],
            "observed_ids_per_block": {"structural": 5},
        }
    }

    result = evaluate_tokenization_freeze(
        audit_payload,
        runtime_payload=runtime_payload,
        require_no_residual_hash=True,
        required_preserve_full_blocks={"special", "structural"},
    )

    assert result["passed"] is True
    assert all(item["passed"] for item in result["checks"])


def test_tokenization_freeze_check_fails_on_structural_and_hash_tail() -> None:
    from scripts.check_tokenization_freeze_v1 import evaluate_tokenization_freeze

    audit_payload = {
        "timeline": {
            "semantic_effective_capture_by_category": {
                "DIAGNOSIS": {
                    "mapped_rate_over_semantic_total": 1.0,
                    "medtok_only_rate_over_semantic_total": 0.78,
                    "residual_hash": 0,
                },
                "PROCEDURE": {
                    "mapped_rate_over_semantic_total": 1.0,
                    "medtok_only_rate_over_semantic_total": 0.85,
                    "residual_hash": 0,
                },
                "MEDICATION": {
                    "mapped_rate_over_semantic_total": 1.0,
                    "medtok_only_rate_over_semantic_total": 0.22,
                    "residual_hash": 4,
                },
            }
        },
        "collation": {"window_type_unk_frac": 0.0},
    }
    runtime_payload = {
        "qual_obs_tail_policy": "hash",
        "sparse_vocab_contract": {
            "residual_fallback": {
                "tail_policy": "drop",
                "families": {
                    "diagnosis": {"tail_policy": "drop"},
                    "procedure": {"tail_policy": "drop"},
                    "medication": {"tail_policy": "hash"},
                },
            }
        },
        "summary": {
            "preserve_full_blocks": ["special"],
            "observed_ids_per_block": {"structural": 1},
        }
    }

    result = evaluate_tokenization_freeze(
        audit_payload,
        runtime_payload=runtime_payload,
        require_no_residual_hash=True,
        required_preserve_full_blocks={"special", "structural"},
    )

    assert result["passed"] is False
    failed = {item["name"] for item in result["checks"] if not item["passed"]}
    assert "medication_residual_hash_zero" in failed
    assert "preserve_full_blocks" in failed
    assert "structural_observed_ids" in failed
    assert "medication_tail_policy_not_hash" in failed
    assert "observation_tail_policy_not_hash" in failed
