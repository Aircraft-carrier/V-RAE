from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found")
    return parameters


def build_optimizer(
    parameters: Iterable[nn.Parameter], config: Mapping[str, object]
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("Optimizer would have no trainable parameters")
    name = str(config.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    betas_value = config.get("betas", (0.9, 0.95))
    betas = tuple(float(value) for value in betas_value)
    if len(betas) != 2:
        raise ValueError("AdamW betas must contain two values")
    fused_value = config.get("fused")
    fused = (
        all(parameter.is_cuda for parameter in parameters)
        if fused_value is None
        else bool(fused_value)
    )
    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("lr", 1.0e-4)),
        betas=betas,
        eps=float(config.get("eps", 1.0e-8)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        fused=fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
    *,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    name = str(config.get("name", "constant")).lower()
    if name == "constant":
        warmup_steps = int(config.get("warmup_steps", 0))
        if warmup_steps != 0:
            raise ValueError("The formal constant recipe has no implicit warmup")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name not in {"linear", "cosine"}:
        raise ValueError(f"Unsupported scheduler: {name}")
    if steps_per_epoch is None or int(steps_per_epoch) <= 0:
        raise ValueError(f"{name} scheduler requires positive steps_per_epoch")

    base_lr = float(config.get("base_lr", optimizer.param_groups[0]["lr"]))
    final_lr = float(config.get("final_lr", base_lr))
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0
    warmup_steps_value = config.get("warmup_steps")
    warmup_steps = (
        int(warmup_steps_value)
        if warmup_steps_value is not None
        else int(float(config.get("warmup_epochs", 0)) * int(steps_per_epoch))
    )
    decay_end_steps_value = config.get("decay_end_steps")
    decay_end_steps = (
        int(decay_end_steps_value)
        if decay_end_steps_value is not None
        else int(float(config["decay_end_epoch"]) * int(steps_per_epoch))
    )
    warmup_steps = max(warmup_steps, 0)
    decay_end_steps = max(decay_end_steps, warmup_steps)
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)
    warmup_from_zero = bool(config.get("warmup_from_zero", False))
    for group in optimizer.param_groups:
        group["lr"] = base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps if warmup_from_zero else 1.0
        if step >= decay_end_steps:
            return final_ratio
        progress = (step - warmup_steps) / total_decay_steps
        if name == "linear":
            return 1.0 - (1.0 - final_ratio) * progress
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_ratio + (1.0 - final_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
