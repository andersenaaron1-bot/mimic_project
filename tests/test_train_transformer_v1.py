from __future__ import annotations

import torch
import torch.nn as nn

from scripts.train_transformer_v1 import (
    build_dataloader_kwargs,
    build_lr_lambda,
    build_optimizer_param_groups,
    resolve_epoch_range,
    resolve_precompiled_num_workers,
    resolve_token_family_weights,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_build_optimizer_param_groups_splits_decay_and_no_decay() -> None:
    model = _TinyModel()
    groups = build_optimizer_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2
    decay_group = next(g for g in groups if float(g["weight_decay"]) > 0.0)
    no_decay_group = next(g for g in groups if float(g["weight_decay"]) == 0.0)

    decay_params = {id(p) for p in decay_group["params"]}
    no_decay_params = {id(p) for p in no_decay_group["params"]}

    assert id(model.linear.weight) in decay_params
    assert id(model.linear.bias) in no_decay_params
    assert id(model.norm.weight) in no_decay_params
    assert id(model.norm.bias) in no_decay_params
    assert decay_params.isdisjoint(no_decay_params)


def test_build_lr_lambda_has_warmup_then_decay() -> None:
    fn = build_lr_lambda(total_steps=10, warmup_steps=2, min_lr_scale=0.1)
    vals = [float(fn(step)) for step in range(10)]
    assert vals[0] < vals[1] <= 1.0
    assert vals[2] <= 1.0
    assert vals[-1] >= 0.1
    assert vals[-1] < vals[2]


def test_resolve_epoch_range_treats_epochs_as_epochs_to_run() -> None:
    assert list(resolve_epoch_range(start_epoch=1, epochs_to_run=1)) == [1]
    assert list(resolve_epoch_range(start_epoch=2, epochs_to_run=1)) == [2]
    assert list(resolve_epoch_range(start_epoch=2, epochs_to_run=3)) == [2, 3, 4]


def test_resolve_token_family_weights_merges_preset_and_overrides() -> None:
    weights = resolve_token_family_weights(
        preset="semantic_boost_v1",
        overrides=["measurement_value=0.25", "diagnosis=4.0"],
    )
    assert weights["diagnosis"] == 4.0
    assert weights["measurement_value"] == 0.25
    assert weights["procedure"] == 5.0
    assert weights["unk"] == 0.1


def test_resolve_token_family_weights_supports_semantic_boost_v2() -> None:
    weights = resolve_token_family_weights(
        preset="semantic_boost_v2",
        overrides=["unk=0.4"],
    )
    assert weights["diagnosis"] == 2.5
    assert weights["procedure"] == 4.0
    assert weights["medication"] == 1.75
    assert weights["structural"] == 0.85
    assert weights["unk"] == 0.4


def test_resolve_precompiled_num_workers_uses_safe_auto_default(monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_transformer_v1.os.cpu_count", lambda: 6)
    assert resolve_precompiled_num_workers(0) == 5
    assert resolve_precompiled_num_workers(3) == 3


def test_build_dataloader_kwargs_enables_prefetch_for_precompiled(monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_transformer_v1.os.cpu_count", lambda: 6)
    kwargs = build_dataloader_kwargs(
        device=torch.device("cpu"),
        num_workers=0,
        precompiled=True,
        prefetch_factor=5,
    )
    assert kwargs["num_workers"] == 5
    assert kwargs["pin_memory"] is False
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 5


def test_build_dataloader_kwargs_keeps_on_the_fly_zero_workers() -> None:
    kwargs = build_dataloader_kwargs(
        device=torch.device("cpu"),
        num_workers=0,
        precompiled=False,
        prefetch_factor=5,
    )
    assert kwargs == {"num_workers": 0, "pin_memory": False}
