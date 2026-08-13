"""Tensor-native spatial transforms for CPU NCHW video clips."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as functional
from torch import Tensor

SpatialSize: TypeAlias = int | tuple[int, int]
CropRounding: TypeAlias = Literal["floor", "round"]


def _pair(size: SpatialSize) -> tuple[int, int]:
    if isinstance(size, bool):
        raise ValueError("spatial size must be a positive integer or (height, width)")
    if isinstance(size, int):
        result = (size, size)
    elif isinstance(size, tuple) and len(size) == 2:
        result = (int(size[0]), int(size[1]))
    else:
        raise ValueError("spatial size must be a positive integer or (height, width)")
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("spatial dimensions must be positive")
    return result


def validate_video(video: Tensor) -> Tensor:
    if not isinstance(video, Tensor):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim != 4:
        raise ValueError(f"video must have shape [T,C,H,W], got {tuple(video.shape)}")
    if video.shape[0] <= 0 or video.shape[1] != 3 or video.shape[2] <= 0 or video.shape[3] <= 0:
        raise ValueError(f"video must have non-empty RGB shape [T,3,H,W], got {tuple(video.shape)}")
    return video


def _restore_dtype(video: Tensor, dtype: torch.dtype) -> Tensor:
    if dtype == torch.uint8:
        return video.round().clamp_(0, 255).to(torch.uint8)
    return video.to(dtype)


def resize_video(
    video: Tensor,
    size: SpatialSize,
    *,
    mode: str = "bilinear",
    antialias: bool = True,
) -> Tensor:
    """Resize every frame to an exact spatial size while preserving dtype."""

    source = validate_video(video)
    output_size = _pair(size)
    if tuple(source.shape[-2:]) == output_size:
        return source.contiguous()
    if mode not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("mode must be nearest, bilinear, or bicubic")

    original_dtype = source.dtype
    working = source.float() if not source.is_floating_point() else source
    kwargs: dict[str, object] = {"size": output_size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs.update(align_corners=False, antialias=antialias)
    resized = functional.interpolate(working, **kwargs)
    return _restore_dtype(resized, original_dtype).contiguous()


def resize_short_side(
    video: Tensor,
    size: int,
    *,
    mode: str = "bilinear",
    antialias: bool = True,
) -> Tensor:
    """Resize with preserved aspect ratio so the shorter side equals ``size``."""

    source = validate_video(video)
    target = _pair(size)[0]
    height, width = source.shape[-2:]
    if height <= width:
        output = (target, max(target, round(width * target / height)))
    else:
        output = (max(target, round(height * target / width)), target)
    return resize_video(source, output, mode=mode, antialias=antialias)


def center_crop_video(
    video: Tensor,
    size: SpatialSize,
    *,
    rounding: CropRounding = "floor",
) -> Tensor:
    source = validate_video(video)
    crop_height, crop_width = _pair(size)
    height, width = source.shape[-2:]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"crop {(crop_height, crop_width)} exceeds video spatial size {(height, width)}; "
            "implicit padding is disabled"
        )
    if rounding == "floor":
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
    elif rounding == "round":
        # Python round uses ties-to-even for half-pixel offsets.
        top = int(round((height - crop_height) / 2.0))
        left = int(round((width - crop_width) / 2.0))
    else:
        raise ValueError("rounding must be floor or round")
    return source[..., top : top + crop_height, left : left + crop_width].contiguous()


def random_crop_video(
    video: Tensor,
    size: SpatialSize,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    source = validate_video(video)
    crop_height, crop_width = _pair(size)
    height, width = source.shape[-2:]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"crop {(crop_height, crop_width)} exceeds video spatial size {(height, width)}; "
            "implicit padding is disabled"
        )
    top = int(torch.randint(height - crop_height + 1, (), generator=generator).item())
    left = int(torch.randint(width - crop_width + 1, (), generator=generator).item())
    return source[..., top : top + crop_height, left : left + crop_width].contiguous()


def uint8_to_float(video: Tensor) -> Tensor:
    source = validate_video(video)
    if source.dtype != torch.uint8:
        raise TypeError(f"uint8_to_float expects torch.uint8 input, got {source.dtype}")
    return source.to(torch.float32).div_(255.0)


def normalize_video(
    video: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    source = validate_video(video)
    if not source.is_floating_point():
        raise TypeError("normalize_video expects floating-point RGB input")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must contain three RGB values")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be positive")
    mean_tensor = source.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = source.new_tensor(std).view(1, 3, 1, 1)
    return (source - mean_tensor) / std_tensor


def resize_center_crop(
    video: Tensor,
    size: SpatialSize,
    *,
    resize_shorter_side: int | None = None,
    mode: str = "bilinear",
    antialias: bool = True,
    crop_rounding: CropRounding = "floor",
) -> Tensor:
    """Aspect-ratio resize followed by center crop, as used by rFVD."""

    crop_size = _pair(size)
    shorter_side = resize_shorter_side or max(crop_size)
    resized = resize_short_side(video, shorter_side, mode=mode, antialias=antialias)
    return center_crop_video(resized, crop_size, rounding=crop_rounding)


class Compose:
    def __init__(self, transforms: Iterable[Callable[[Tensor], Tensor]]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, video: Tensor) -> Tensor:
        for transform in self.transforms:
            video = transform(video)
        return video


@dataclass(frozen=True)
class Resize:
    size: SpatialSize
    mode: str = "bilinear"
    antialias: bool = True

    def __call__(self, video: Tensor) -> Tensor:
        return resize_video(video, self.size, mode=self.mode, antialias=self.antialias)


@dataclass(frozen=True)
class ResizeShortSide:
    size: int
    mode: str = "bilinear"
    antialias: bool = True

    def __call__(self, video: Tensor) -> Tensor:
        return resize_short_side(video, self.size, mode=self.mode, antialias=self.antialias)


@dataclass(frozen=True)
class CenterCrop:
    size: SpatialSize

    def __call__(self, video: Tensor) -> Tensor:
        return center_crop_video(video, self.size)


@dataclass(frozen=True)
class RandomCrop:
    size: SpatialSize

    def __call__(self, video: Tensor) -> Tensor:
        return random_crop_video(video, self.size)


class Uint8ToFloat:
    def __call__(self, video: Tensor) -> Tensor:
        return uint8_to_float(video)


@dataclass(frozen=True)
class Normalize:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __call__(self, video: Tensor) -> Tensor:
        return normalize_video(video, self.mean, self.std)


__all__ = [
    "CenterCrop",
    "Compose",
    "Normalize",
    "RandomCrop",
    "Resize",
    "ResizeShortSide",
    "SpatialSize",
    "Uint8ToFloat",
    "center_crop_video",
    "normalize_video",
    "random_crop_video",
    "resize_center_crop",
    "resize_short_side",
    "resize_video",
    "uint8_to_float",
    "validate_video",
]
