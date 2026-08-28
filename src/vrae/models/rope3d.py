from __future__ import annotations

import torch


def split_axis_dims(head_dim: int) -> tuple[int, int, int, int]:
    axis_dim = (int(head_dim) // 3) // 2 * 2
    if axis_dim <= 0:
        raise ValueError(f"head_dim={head_dim} is too small for 3D RoPE")
    return axis_dim, axis_dim, axis_dim, int(head_dim) - 3 * axis_dim


def build_3d_positions(
    num_chunks: int, height: int, width: int, *, device: torch.device | None = None
) -> torch.Tensor:
    t = torch.arange(num_chunks, device=device)
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack((tt.flatten(), yy.flatten(), xx.flatten()), dim=-1)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _axis_sincos(
    positions: torch.Tensor, dim: int, *, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim == 0:
        empty = positions.new_empty((*positions.shape, 0), dtype=dtype)
        return empty, empty
    inv_frequency = 1.0 / (
        float(theta)
        ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
    )
    phase = positions.float().unsqueeze(-1) * inv_frequency
    phase = torch.repeat_interleave(phase, 2, dim=-1)
    return phase.cos().to(dtype), phase.sin().to(dtype)


def build_3d_rope_cache_from_positions(
    positions: torch.Tensor,
    head_dim: int,
    *,
    theta: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions [tokens,3], got {tuple(positions.shape)}")
    dimensions = split_axis_dims(head_dim)[:3]
    cosine: list[torch.Tensor] = []
    sine: list[torch.Tensor] = []
    for dim, axis in zip(dimensions, positions.unbind(dim=-1), strict=True):
        axis_cosine, axis_sine = _axis_sincos(axis, dim, theta=theta, dtype=dtype)
        cosine.append(axis_cosine)
        sine.append(axis_sine)
    return torch.cat(cosine, dim=-1), torch.cat(sine, dim=-1)


def build_3d_rope_cache(
    num_chunks: int,
    height: int,
    width: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = build_3d_positions(num_chunks, height, width, device=device)
    return build_3d_rope_cache_from_positions(
        positions,
        head_dim,
        theta=theta,
        dtype=dtype,
    )


def apply_3d_rope_cache(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,heads,tokens,head_dim], got {tuple(tensor.shape)}")
    dim_t, dim_y, dim_x, tail = split_axis_dims(tensor.shape[-1])
    rotated_dim = dim_t + dim_y + dim_x
    expected = (tensor.shape[-2], rotated_dim)
    if cosine.shape != expected or sine.shape != expected:
        raise ValueError(
            f"Expected RoPE cosine/sine {expected}, got {tuple(cosine.shape)} and "
            f"{tuple(sine.shape)}"
        )
    if cosine.device != tensor.device or sine.device != tensor.device:
        raise ValueError("RoPE cosine/sine cache must be on the tensor device")
    if cosine.dtype != tensor.dtype or sine.dtype != tensor.dtype:
        raise ValueError("RoPE cosine/sine cache must match the tensor dtype")

    result: list[torch.Tensor] = []
    offset = 0
    for dim in (dim_t, dim_y, dim_x):
        part = tensor[..., offset : offset + dim]
        axis_cosine = cosine[..., offset : offset + dim]
        axis_sine = sine[..., offset : offset + dim]
        result.append(part * axis_cosine[None, None] + _rotate_half(part) * axis_sine[None, None])
        offset += dim
    if tail:
        result.append(tensor[..., offset:])
    return torch.cat(result, dim=-1)
