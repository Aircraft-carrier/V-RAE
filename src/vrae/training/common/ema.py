from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        grouped: dict[
            tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]
        ] = {}
        for name, average in self.shadow.items():
            if name not in current:
                raise KeyError(f"EMA model is missing state key: {name}")
            value = current[name].detach().to(device=average.device, dtype=average.dtype)
            averages, values = grouped.setdefault((average.device, average.dtype), ([], []))
            averages.append(average)
            values.append(value)
        for averages, values in grouped.values():
            torch._foreach_lerp_(averages, values, 1.0 - self.decay)
        self.num_updates += 1

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, average in self.shadow.items():
            state[name].copy_(average.to(device=state[name].device, dtype=state[name].dtype))

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        state = model.state_dict()
        backup = {name: state[name].detach().clone() for name in self.shadow if name in state}
        if set(backup) != set(self.shadow):
            raise KeyError("EMA model keys do not match the target model")
        self.copy_to(model)
        try:
            yield
        finally:
            for name, value in backup.items():
                state[name].copy_(value)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError(f"EMA decay mismatch: {state['decay']} vs {self.decay}")
        shadow = state["shadow"]
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise ValueError("EMA state keys do not match the model")
        for name, value in shadow.items():
            if not torch.is_tensor(value) or value.shape != self.shadow[name].shape:
                raise ValueError(f"Invalid EMA tensor: {name}")
            self.shadow[name].copy_(value)
        self.num_updates = int(state["num_updates"])
