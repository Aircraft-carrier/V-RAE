from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GuidanceInterval:
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= float(self.minimum) <= float(self.maximum) <= 1:
            raise ValueError("guidance interval must satisfy 0 <= minimum <= maximum <= 1")


@dataclass(frozen=True)
class GuidanceConfig:
    cfg_scale: float = 1.0
    ig_scale: float = 1.0
    ig_intervals: tuple[GuidanceInterval, ...] = (GuidanceInterval(),)


def _scale_like(scale: float | torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale_tensor = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
    if scale_tensor.ndim == 1:
        if scale_tensor.shape[0] != value.shape[0]:
            raise ValueError("a per-sample guidance scale must have shape [B]")
        scale_tensor = scale_tensor.reshape(scale_tensor.shape[0], *([1] * (value.ndim - 1)))
    return scale_tensor


def classifier_free_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    if conditional.shape != unconditional.shape:
        raise ValueError("conditional and unconditional predictions must have matching shapes")
    return unconditional + _scale_like(scale, conditional) * (conditional - unconditional)


def _normalize_intervals(
    intervals: Sequence[GuidanceInterval | tuple[float, float]] | None,
) -> tuple[GuidanceInterval, ...]:
    if intervals is None:
        return (GuidanceInterval(),)
    normalized: list[GuidanceInterval] = []
    for interval in intervals:
        if isinstance(interval, GuidanceInterval):
            normalized.append(interval)
        else:
            if len(interval) != 2:
                raise ValueError("each guidance interval must contain two values")
            normalized.append(GuidanceInterval(float(interval[0]), float(interval[1])))
    return tuple(normalized)


def guidance_interval_mask(
    time: torch.Tensor,
    intervals: Sequence[GuidanceInterval | tuple[float, float]] | None = None,
) -> torch.Tensor:
    if time.ndim == 0:
        time = time[None]
    if time.ndim != 1:
        raise ValueError("time must be scalar or [B]")
    result = torch.zeros_like(time, dtype=torch.bool)
    for interval in _normalize_intervals(intervals):
        result |= (time >= interval.minimum) & (time <= interval.maximum)
    return result


def internal_guidance(
    full: torch.Tensor,
    base: torch.Tensor,
    scale: float | torch.Tensor,
    *,
    time: torch.Tensor | None = None,
    intervals: Sequence[GuidanceInterval | tuple[float, float]] | None = None,
) -> torch.Tensor:
    if full.shape != base.shape:
        raise ValueError("full and base predictions must have matching shapes")
    guided = base + _scale_like(scale, full) * (full - base)
    if time is None:
        return guided
    mask = guidance_interval_mask(time.to(full.device), intervals)
    if mask.shape[0] == 1 and full.shape[0] != 1:
        mask = mask.expand(full.shape[0])
    if mask.shape[0] != full.shape[0]:
        raise ValueError("time batch size must match the prediction batch size")
    mask = mask.reshape(mask.shape[0], *([1] * (full.ndim - 1)))
    return torch.where(mask, guided, full)


def combined_guidance(
    conditional_full: torch.Tensor,
    unconditional_full: torch.Tensor,
    *,
    cfg_scale: float | torch.Tensor,
    conditional_base: torch.Tensor | None = None,
    unconditional_base: torch.Tensor | None = None,
    ig_scale: float | torch.Tensor = 1.0,
    time: torch.Tensor | None = None,
    ig_intervals: Sequence[GuidanceInterval | tuple[float, float]] | None = None,
) -> torch.Tensor:
    conditional_base = conditional_full if conditional_base is None else conditional_base
    unconditional_base = unconditional_full if unconditional_base is None else unconditional_base
    conditional = internal_guidance(
        conditional_full,
        conditional_base,
        ig_scale,
        time=time,
        intervals=ig_intervals,
    )
    unconditional = internal_guidance(
        unconditional_full,
        unconditional_base,
        ig_scale,
        time=time,
        intervals=ig_intervals,
    )
    return classifier_free_guidance(conditional, unconditional, cfg_scale)


def split_full_base(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output, output
    if isinstance(output, tuple) and len(output) == 2:
        full, base = output
        if isinstance(full, torch.Tensor) and isinstance(base, torch.Tensor):
            return full, base
    raise TypeError("model output must be a tensor or a (full, base) tensor pair")


def guided_model_forward(
    model: nn.Module,
    value: torch.Tensor,
    time: torch.Tensor,
    *,
    condition_kwargs: Mapping[str, Any] | None = None,
    unconditional_kwargs: Mapping[str, Any] | None = None,
    cfg_scale: float | torch.Tensor = 1.0,
    ig_scale: float | torch.Tensor = 1.0,
    ig_intervals: Sequence[GuidanceInterval | tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Evaluate model branches and apply IG to each branch before CFG."""

    conditional_arguments = dict(condition_kwargs or {})
    request_base = bool(torch.as_tensor(ig_scale).ne(1).any())
    if request_base:
        conditional_arguments.setdefault("return_base", True)
    conditional_full, conditional_base = split_full_base(
        model(value, time, **conditional_arguments)
    )

    use_cfg = bool(torch.as_tensor(cfg_scale).ne(1).any())
    if not use_cfg:
        return internal_guidance(
            conditional_full,
            conditional_base,
            ig_scale,
            time=time,
            intervals=ig_intervals,
        )
    if unconditional_kwargs is None:
        raise ValueError("unconditional_kwargs are required when cfg_scale is not 1")
    unconditional_arguments = dict(unconditional_kwargs)
    if request_base:
        unconditional_arguments.setdefault("return_base", True)
    unconditional_full, unconditional_base = split_full_base(
        model(value, time, **unconditional_arguments)
    )
    return combined_guidance(
        conditional_full,
        unconditional_full,
        cfg_scale=cfg_scale,
        conditional_base=conditional_base,
        unconditional_base=unconditional_base,
        ig_scale=ig_scale,
        time=time,
        ig_intervals=ig_intervals,
    )


apply_cfg = classifier_free_guidance
apply_internal_guidance = internal_guidance


__all__ = [
    "GuidanceConfig",
    "GuidanceInterval",
    "apply_cfg",
    "apply_internal_guidance",
    "classifier_free_guidance",
    "combined_guidance",
    "guidance_interval_mask",
    "guided_model_forward",
    "internal_guidance",
    "split_full_base",
]
