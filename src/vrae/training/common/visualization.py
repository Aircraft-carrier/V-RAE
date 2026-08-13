from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch


def video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("Expected video [B,T,C,H,W]")
    if video.dtype == torch.uint8:
        return video
    return video.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)


def comparison_video(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    if real.shape != reconstructed.shape:
        raise ValueError("Real and reconstructed videos must have equal shape")
    return torch.cat((video_to_uint8(real), video_to_uint8(reconstructed)), dim=-1)


def save_sample_tensors(tensors: Mapping[str, torch.Tensor], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save({name: value.detach().cpu() for name, value in tensors.items()}, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
