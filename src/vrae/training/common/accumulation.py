from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


class GradientAccumulator:
    """Tracks microsteps while keeping every checkpoint on an update boundary."""

    def __init__(self, steps: int) -> None:
        if int(steps) <= 0:
            raise ValueError("gradient accumulation steps must be positive")
        self.steps = int(steps)
        self.microstep = 0

    @property
    def is_first_microstep(self) -> bool:
        return self.microstep == 0

    @property
    def will_step(self) -> bool:
        return self.microstep + 1 == self.steps

    def sync_context(self, model: nn.Module):
        no_sync = getattr(model, "no_sync", None)
        if not self.will_step and callable(no_sync):
            return no_sync()
        return nullcontext()

    def backward(self, loss: torch.Tensor, scaler: Any = None) -> None:
        scaled_loss = loss / self.steps
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

    def advance(self) -> bool:
        boundary = self.will_step
        self.microstep = 0 if boundary else self.microstep + 1
        return boundary

    def load_data_state(self, data_state: dict[str, Any]) -> None:
        microstep = int(data_state.get("gradient_accumulation_microstep", 0))
        if microstep != 0:
            raise ValueError(
                "Checkpoint is inside a gradient accumulation window but contains no "
                "parameter-gradient state; exact resume requires a boundary checkpoint"
            )
        self.microstep = 0

    def require_boundary(self) -> None:
        if self.microstep != 0:
            raise RuntimeError("Checkpointing is only allowed at an optimizer-step boundary")


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any = None,
    max_grad_norm: float | None = None,
    parameters=None,
) -> bool:
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if scaler is not None:
            scaler.unscale_(optimizer)
        if parameters is None:
            parameters = [
                parameter for group in optimizer.param_groups for parameter in group["params"]
            ]
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if scaler is None:
        optimizer.step()
        return True
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    # GradScaler lowers the scale exactly when non-finite gradients caused the
    # optimizer update to be skipped.  Callers must not advance LR/EMA/step in
    # that case.
    return float(scaler.get_scale()) >= scale_before
