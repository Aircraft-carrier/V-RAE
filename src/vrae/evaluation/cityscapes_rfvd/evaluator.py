from __future__ import annotations

from typing import Any

import torch


def _validate_rgb_pair(context: torch.Tensor, future: torch.Tensor) -> None:
    expected_tail = (12, 3)
    if context.ndim != 5 or tuple(context.shape[1:3]) != expected_tail:
        raise ValueError("context must have shape [B,12,3,H,W]")
    if future.shape != context.shape:
        raise ValueError("future must match context shape [B,12,3,H,W]")


def _validate_latents(context: torch.Tensor, future: torch.Tensor) -> None:
    if context.ndim != 5 or future.shape != context.shape:
        raise ValueError("context/future latents must have matching [B,T,C,H,W] shapes")
    if context.shape[1] != 3:
        raise ValueError("each independently encoded 12-frame clip must produce three chunks")


@torch.inference_mode()
def reconstruct_future_variants(
    stage1: Any,
    context: torch.Tensor,
    future: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reconstruct one future through the two locked Cityscapes decode protocols.

    Context and future are always encoded by two independent V-RAE calls. The
    primary result decodes only the three future chunks. The ablation concatenates
    the already independent latent grids, performs one six-chunk causal decode, and
    returns only its final 12 RGB frames.
    """

    _validate_rgb_pair(context, future)
    context_latents = stage1.encode(context)
    future_latents = stage1.encode(future)
    _validate_latents(context_latents, future_latents)

    future_only = stage1.decode(future_latents)
    joint_latents = torch.cat((context_latents, future_latents), dim=1)
    if joint_latents.shape[1] != 6:
        raise RuntimeError("context-conditioned decode requires exactly six latent chunks")
    joint = stage1.decode(joint_latents)
    if future_only.shape != future.shape:
        raise RuntimeError("future-only decode must reconstruct exactly 12 RGB frames")
    expected_joint = (context.shape[0], 24, *context.shape[2:])
    if tuple(joint.shape) != expected_joint:
        raise RuntimeError(f"joint decode must return {expected_joint}, got {tuple(joint.shape)}")
    return {
        "future_only": future_only,
        "context_conditioned": joint[:, -12:].contiguous(),
    }


__all__ = ["reconstruct_future_variants"]
