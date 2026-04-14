import argparse


def test_world_model_objective_preset_demotes_unified_token_and_delays_precedent() -> None:
    from scripts.train_transformer_v1 import (
        LOSS_WEIGHT_ARG_TO_KEY,
        precedent_loss_scale_for_step,
        resolve_objective_configuration,
        resolve_scheduled_loss_weights,
    )

    args_dict = {
        "objective_preset": "world_model_mttee",
        "prefer_unified_token_loss": None,
        "token_family_weight_preset": "none",
        "precedent_loss_start_step": None,
        "precedent_loss_ramp_steps": None,
    }
    for arg_name in LOSS_WEIGHT_ARG_TO_KEY:
        args_dict[arg_name] = 999.0
    args = argparse.Namespace(**args_dict)

    cfg = resolve_objective_configuration(args=args, raw_argv=())
    assert cfg["preset"] == "world_model_mttee"
    assert cfg["prefer_unified_token_loss"] is False
    assert cfg["loss_weights"]["event_token"] == 0.0
    assert cfg["loss_weights"]["event_family"] == 1.0
    assert cfg["loss_weights"]["token"] < cfg["loss_weights"]["event_family"]
    assert cfg["precedent_loss_start_step"] == 2000
    assert cfg["precedent_loss_ramp_steps"] == 2000

    warm_weights, warm_logs = resolve_scheduled_loss_weights(
        base_weights=cfg["loss_weights"],
        global_step=0,
        precedent_loss_start_step=cfg["precedent_loss_start_step"],
        precedent_loss_ramp_steps=cfg["precedent_loss_ramp_steps"],
    )
    full_weights, full_logs = resolve_scheduled_loss_weights(
        base_weights=cfg["loss_weights"],
        global_step=5000,
        precedent_loss_start_step=cfg["precedent_loss_start_step"],
        precedent_loss_ramp_steps=cfg["precedent_loss_ramp_steps"],
    )

    assert warm_weights["precedent_future"] == 0.0
    assert warm_weights["precedent_contrast"] == 0.0
    assert warm_weights["precedent_anchor"] == 0.0
    assert warm_logs["precedent_loss_scale"] == 0.0
    assert full_weights["precedent_future"] == cfg["loss_weights"]["precedent_future"]
    assert full_weights["precedent_contrast"] == cfg["loss_weights"]["precedent_contrast"]
    assert full_weights["precedent_anchor"] == cfg["loss_weights"]["precedent_anchor"]
    assert full_logs["precedent_loss_scale"] == 1.0

    assert precedent_loss_scale_for_step(global_step=1999, start_step=2000, ramp_steps=2000) == 0.0
    assert precedent_loss_scale_for_step(global_step=3999, start_step=2000, ramp_steps=2000) == 1.0
