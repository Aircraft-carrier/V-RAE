from __future__ import annotations

import torch


def centered_six_to_four(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 5 or latents.shape[1] != 6:
        raise ValueError("centered_6_to_4 expects [B,6,C,H,W]")
    return torch.stack(
        (
            (latents[:, 0] + latents[:, 2]) / 2,
            (latents[:, 1] + latents[:, 3]) / 2,
            (latents[:, 2] + latents[:, 4]) / 2,
            (latents[:, 3] + latents[:, 5]) / 2,
        ),
        dim=1,
    )


@torch.no_grad()
def temporal_interpolation_pair(vrae, video_24: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if video_24.shape[1] != 24:
        raise ValueError("tFVD input must contain 24 frames")
    clean = vrae.encode(video_24)
    if clean.shape[1] != 6:
        raise ValueError("V-RAE must produce six chunks from 24 frames")
    interpolated = vrae.decode(centered_six_to_four(clean))
    reference = video_24[:, 4:20]
    if interpolated.shape[1] != 16 or reference.shape[1] != 16:
        raise AssertionError("tFVD sides must both contain 16 frames")
    return reference, interpolated
