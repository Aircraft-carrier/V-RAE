from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from vrae.evaluation.models.checkpoint_path import local_checkpoint_entry


def _unwrap_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Inception checkpoint must contain a state mapping")
    for key in ("state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    state = {str(key): value for key, value in payload.items() if torch.is_tensor(value)}
    if not state:
        raise TypeError("Inception checkpoint contains no tensor state")
    return state


class TorchFidelityInceptionPool3(nn.Module):
    """Pool3 tower for the converted TensorFlow FID Inception-v3 weights."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.model = models.inception_v3(
            weights=None,
            aux_logits=False,
            num_classes=1008,
            init_weights=False,
        )
        state = _unwrap_state(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.load_state_dict(state, strict=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        model = self.model
        value = model.Conv2d_1a_3x3(value)
        value = model.Conv2d_2a_3x3(value)
        value = model.Conv2d_2b_3x3(value)
        value = F.max_pool2d(value, kernel_size=3, stride=2)
        value = model.Conv2d_3b_1x1(value)
        value = model.Conv2d_4a_3x3(value)
        value = F.max_pool2d(value, kernel_size=3, stride=2)
        value = model.Mixed_5b(value)
        value = model.Mixed_5c(value)
        value = model.Mixed_5d(value)
        value = model.Mixed_6a(value)
        value = model.Mixed_6b(value)
        value = model.Mixed_6c(value)
        value = model.Mixed_6d(value)
        value = model.Mixed_6e(value)
        value = model.Mixed_7a(value)
        value = model.Mixed_7b(value)
        value = model.Mixed_7c(value)
        return F.adaptive_avg_pool2d(value, output_size=(1, 1)).flatten(1)


class InceptionFeatureExtractor(nn.Module):
    """Loads a local TensorFlow-compatible FID Inception pool3 extractor."""

    def __init__(self, checkpoint: str | Path, *, checkpoint_root: str | Path) -> None:
        super().__init__()
        checkpoint = local_checkpoint_entry(checkpoint, checkpoint_root=checkpoint_root)
        if checkpoint.suffix.lower() == ".pth":
            self.model = TorchFidelityInceptionPool3(checkpoint).eval()
        else:
            self.model = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> InceptionFeatureExtractor:
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Inception expects RGB frames [N,3,H,W]")
        frames = frames.float()
        if frames.min() < 0:
            raise ValueError("Inception input must be non-negative RGB")
        if frames.max() <= 1:
            frames = frames * 255.0
        elif frames.max() > 255 or frames.min() < 0:
            raise ValueError("Inception input must be RGB in [0,1] or [0,255]")
        frames = F.interpolate(
            frames, (299, 299), mode="bilinear", align_corners=False, antialias=True
        )
        output = self.model((frames - 128.0) / 128.0)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("pool3", next(iter(output.values())))
        output = output.flatten(1)
        if output.shape[1] != 2048:
            raise ValueError(f"Expected Inception pool3 dimension 2048, got {output.shape[1]}")
        return output
