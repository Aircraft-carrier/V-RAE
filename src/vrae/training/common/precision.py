from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    device_type: str

    @property
    def dtype(self) -> torch.dtype | None:
        values = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
        if self.name not in values:
            raise ValueError(f"Unknown precision: {self.name}")
        return values[self.name]

    def autocast(self):
        dtype = self.dtype
        if dtype is None:
            return nullcontext()
        if self.device_type == "cpu" and dtype == torch.float16:
            raise ValueError("CPU fp16 autocast is unsupported; use bf16 or fp32")
        return torch.autocast(device_type=self.device_type, dtype=dtype)

    def make_scaler(self) -> torch.amp.GradScaler | None:
        if self.name != "fp16" or self.device_type != "cuda":
            return None
        return torch.amp.GradScaler("cuda")


def backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
    parameters=None,
) -> None:
    if scaler is None:
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        return
    scaler.scale(loss).backward()
    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
