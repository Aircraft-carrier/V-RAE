from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


class VRAELatentAdapter(nn.Module):
    """Frozen V-RAE boundary and the sole grid-to-token layout conversion."""

    def __init__(
        self,
        autoencoder: nn.Module,
        metadata: Mapping[str, Any] | None = None,
        *,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        precision = str(precision).lower()
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("VRAELatentAdapter precision must be fp32, bf16, or fp16")
        self._autoencoder = autoencoder
        self._autoencoder.requires_grad_(False)
        self._autoencoder.eval()
        self.precision = precision
        self._metadata = dict(metadata or getattr(autoencoder, "metadata", lambda: {})())
        self._metadata["execution_precision"] = precision

    def _autocast(self, value: torch.Tensor):
        if value.device.type != "cuda" or self.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def train(self, mode: bool = True) -> VRAELatentAdapter:
        super().train(mode)
        self._autoencoder.eval()
        return self

    @torch.no_grad()
    def encode_grid(
        self,
        video: torch.Tensor,
        stream_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self._autocast(video):
            if stream_ids is None:
                latents = self._autoencoder.encode(video)
            else:
                latents = self._autoencoder.encode(video, stream_ids=stream_ids)
        return latents.float()

    @torch.no_grad()
    def decode_grid(
        self,
        latents: torch.Tensor,
        stream_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self._autocast(latents):
            if stream_ids is None:
                video = self._autoencoder.decode(latents)
            else:
                video = self._autoencoder.decode(latents, stream_ids=stream_ids)
        return video.float()

    def decoder_execution_metadata(self) -> dict[str, Any]:
        return dict(self._autoencoder.decoder.execution_metadata())

    @staticmethod
    def grid_to_tokens(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 6:
            return (
                latents.permute(0, 1, 2, 4, 5, 3)
                .flatten(3, 4)
                .contiguous()
            )
        if latents.ndim != 5:
            raise ValueError(
                "Expected [B,T,C,H,W] or [B,T,V,C,H,W], "
                f"got {tuple(latents.shape)}"
            )
        return (
            latents.permute(0, 1, 3, 4, 2)
            .flatten(2, 3)
            .contiguous()
        )

    @staticmethod
    def tokens_to_grid(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if tokens.ndim == 5:
            if tokens.shape[3] != height * width:
                raise ValueError(
                    f"Expected [B,T,V,{height * width},C], "
                    f"got {tuple(tokens.shape)}"
                )
            batch, time, views, _, channels = tokens.shape
            grid = tokens.reshape(
                batch,
                time,
                views,
                height,
                width,
                channels,
            )
            return grid.permute(0, 1, 2, 5, 3, 4).contiguous()
        if tokens.ndim != 4 or tokens.shape[2] != height * width:
            raise ValueError(
                f"Expected [B,T,{height * width},C] or [B,T,V,N,C], "
                f"got {tuple(tokens.shape)}"
            )
        batch, time, _, channels = tokens.shape
        grid = tokens.reshape(batch, time, height, width, channels)
        return grid.permute(0, 1, 4, 2, 3).contiguous()

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)
