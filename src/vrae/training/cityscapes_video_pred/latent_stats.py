from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from vrae.training.cityscapes_video_pred.data import (
    build_raw_cityscapes_dataset,
    collate_cityscapes,
)
from vrae.training.cityscapes_video_pred.latent_cache import (
    LatentCacheDataset,
    encode_context_future_separately,
)
from vrae.training.common.contracts import structural_stage1_metadata
from vrae.training.common.distributed import initialize_distributed
from vrae.training.common.engine import load_frozen_stage1
from vrae.training.common.latent_norm import DistributedLatentStats, LatentNormalizer
from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths


def accumulate_future_latents(
    batches: Iterable[Mapping[str, Any]],
    accumulator: DistributedLatentStats,
    *,
    stage1: Any | None = None,
    device: torch.device | str = "cpu",
) -> None:
    """Accumulate raw future latents; RGB batches preserve the two-call encode rule."""

    with torch.no_grad():
        for batch in batches:
            future = batch["future"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            if future.ndim == 5:
                if stage1 is None:
                    raise ValueError("stage1 is required when accumulating RGB batches")
                _, future = encode_context_future_separately(stage1, context, future)
            elif future.ndim != 4:
                raise ValueError("future must be RGB [B,12,3,H,W] or tokens [B,3,N,C]")
            accumulator.update(future)


def fit_future_latent_normalizer(
    batches: Iterable[Mapping[str, Any]],
    *,
    channels: int,
    stage1: Any | None = None,
    device: torch.device | str = "cpu",
    metadata: Mapping[str, Any] | None = None,
) -> LatentNormalizer:
    accumulator = DistributedLatentStats(channels, device=device)
    accumulate_future_latents(
        batches,
        accumulator,
        stage1=stage1,
        device=device,
    )
    readable_metadata = {
        "dataset": "cityscapes",
        "split": "train",
        "scope": "future",
        "future_relative_frames": [16, 27],
        "clean_latent": True,
        "normalized": False,
        **dict(metadata or {}),
    }
    return accumulator.finalize(metadata=readable_metadata)


def _resolve_cache_root(value: object, paths: ProjectPaths) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (paths.project_root / path).resolve()


def compute_cityscapes_latent_stats(
    config: Mapping[str, Any],
    paths: ProjectPaths,
) -> Path | None:
    distributed = initialize_distributed()
    stage1, _ = load_frozen_stage1(config, paths, distributed.device)
    stage1_metadata = stage1.metadata()
    cache_value = config["data"].get("latent_cache")
    if cache_value:
        dataset = LatentCacheDataset(
            _resolve_cache_root(cache_value, paths),
            "train",
            expected_stage1_metadata=stage1_metadata,
        )
        stage1_for_stats = None
    else:
        dataset = build_raw_cityscapes_dataset(config, paths, split="train")
        stage1_for_stats = stage1
    local_indices = range(distributed.rank, len(dataset), distributed.world_size)
    local_dataset = Subset(dataset, list(local_indices))
    global_batch = int(config["training"]["global_batch_size"])
    local_batch = max(1, global_batch // distributed.world_size)
    loader = DataLoader(
        local_dataset,
        batch_size=local_batch,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_cityscapes,
    )
    normalizer = fit_future_latent_normalizer(
        loader,
        channels=int(stage1_metadata["hidden_size"]),
        stage1=stage1_for_stats,
        device=distributed.device,
        metadata={
            "stage1": structural_stage1_metadata(stage1_metadata),
            "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        },
    )
    if not distributed.is_main:
        return None
    output = paths.checkpoint(
        config["latent_normalizer"]["path"],
        require_exists=False,
    )
    return normalizer.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Cityscapes train-future latent statistics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    output = compute_cityscapes_latent_stats(config, paths)
    if output is not None:
        print(f"saved Cityscapes train-future latent statistics to {output}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "accumulate_future_latents",
    "compute_cityscapes_latent_stats",
    "fit_future_latent_normalizer",
]
