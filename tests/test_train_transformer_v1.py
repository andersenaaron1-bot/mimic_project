from __future__ import annotations

import torch
import torch.nn as nn

from scripts.train_transformer_v1 import build_lr_lambda, build_optimizer_param_groups


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
