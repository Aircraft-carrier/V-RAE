from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vrae.evaluation.common.population import sample_seed
from vrae.training.cityscapes_video_pred.data import (
    build_raw_cityscapes_dataset,
    collate_cityscapes,
)
from vrae.training.cityscapes_video_pred.latent_cache import LatentCacheDataset
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import build_flow_transport, load_frozen_stage1
from vrae.training.common.latent_norm import LatentNormalizer, validate_normalizer_compatibility
from vrae.checkpoint import compare_metadata, load_checkpoint
from vrae.config import load_config
from vrae.models.dit.transport import FlowMatchingTransport
from vrae.paths import ProjectPaths, find_project_root, load_project_paths


def deterministic_future_noise(
    sample_indices: Sequence[int],
    *,
    base_seed: int,
    latent_shape: Sequence[int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    shape = tuple(int(value) for value in latent_shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("latent_shape must be [chunks,tokens,channels]")
    samples: list[torch.Tensor] = []
    for sample_index in sample_indices:
        generator = torch.Generator(device=device).manual_seed(
            sample_seed(int(base_seed), int(sample_index))
        )
        samples.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
    if not samples:
        return torch.empty((0, *shape), device=device, dtype=dtype)
    return torch.stack(samples)


@torch.no_grad()
def predict_future_tokens(
    dit: nn.Module,
    raw_context: torch.Tensor,
    normalizer: LatentNormalizer,
    *,
    sample_indices: Sequence[int],
    base_seed: int,
    steps: int,
    transport: FlowMatchingTransport,
    cfg_scale: float = 1.0,
    internal_guidance_scale: float = 1.0,
) -> torch.Tensor:
    if raw_context.ndim != 4 or raw_context.shape[1] != 3:
        raise ValueError("raw_context must contain [B,3,tokens,channels]")
    if len(sample_indices) != raw_context.shape[0]:
        raise ValueError("sample_indices length must match the context batch")
    device = raw_context.device
    normalized_context = normalizer.normalize(raw_context)
    noise = deterministic_future_noise(
        sample_indices,
        base_seed=base_seed,
        latent_shape=raw_context.shape[1:],
        device=device,
        dtype=normalized_context.dtype,
    )
    conditional_mask = torch.zeros(raw_context.shape[0], dtype=torch.bool, device=device)
    model_kwargs = {
        "context": normalized_context,
        "context_drop_mask": conditional_mask,
    }
    unconditional_kwargs = None
    if float(cfg_scale) != 1:
        unconditional_kwargs = {
            "context": normalized_context,
            "context_drop_mask": torch.ones_like(conditional_mask),
        }
    normalized_future = transport.euler_sample(
        dit,
        noise,
        model_kwargs=model_kwargs,
        unconditional_kwargs=unconditional_kwargs,
        num_steps=int(steps),
        cfg_scale=float(cfg_scale),
        ig_scale=float(internal_guidance_scale),
    )
    return normalizer.denormalize(normalized_future)


@torch.no_grad()
def decode_future_tokens(
    stage1: Any,
    raw_future: torch.Tensor,
    *,
    grid_size: tuple[int, int] = (27, 48),
) -> torch.Tensor:
    grid = stage1.tokens_to_grid(
        raw_future,
        height=int(grid_size[0]),
        width=int(grid_size[1]),
    )
    return stage1.decode_grid(grid)


def _resolve_project_path(value: object, paths: ProjectPaths) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (paths.project_root / path).resolve()


def load_sampling_stack(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[Any, nn.Module, LatentNormalizer]:
    from vrae.training.cityscapes_video_pred.train import (
        build_cityscapes_dit,
        cityscapes_model_metadata,
    )

    stage1, _ = load_frozen_stage1(config, paths, device)
    dit = build_cityscapes_dit(config, stage1.metadata()).to(device)
    normalizer = LatentNormalizer.load(paths.checkpoint(config["latent_normalizer"]["path"])).to(
        device
    )
    validate_normalizer_compatibility(
        normalizer,
        stage1_metadata=stage1.metadata(),
        stage1_checkpoint=str(config["stage1"]["checkpoint"]),
        dataset="cityscapes",
        split="train",
        scope="future",
    )
    payload = load_checkpoint(paths.checkpoint(checkpoint))
    expected_metadata = cityscapes_model_metadata(config, stage1.metadata(), dit, normalizer)
    compare_metadata(
        expected_metadata,
        payload["model_metadata"],
        tuple(expected_metadata),
    )
    dit.load_state_dict(payload["model"], strict=True)
    if payload.get("ema") is not None:
        average = ExponentialMovingAverage(dit, decay=float(payload["ema"]["decay"]))
        average.load_state_dict(payload["ema"])
        average.copy_to(dit)
    return stage1, dit.eval(), normalizer


@torch.no_grad()
def generate_validation_predictions(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    checkpoint: str | Path,
    output: str | Path,
    *,
    count: int | None = None,
    batch_size: int = 4,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1, dit, normalizer = load_sampling_stack(config, paths, checkpoint, device)
    cache_value = config["data"].get("latent_cache")
    if cache_value:
        dataset = LatentCacheDataset(
            _resolve_project_path(cache_value, paths),
            "val",
            expected_stage1_metadata=stage1.metadata(),
        )
    else:
        dataset = build_raw_cityscapes_dataset(config, paths, split="val")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collate_cityscapes,
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    maximum = len(dataset) if count is None else min(int(count), len(dataset))
    completed = 0
    transport = build_flow_transport(config)
    for batch in loader:
        if completed >= maximum:
            break
        keep = min(len(batch["sample_id"]), maximum - completed)
        context = batch["context"][:keep].to(device, non_blocking=True)
        if context.ndim == 5:
            context = stage1.grid_to_tokens(stage1.encode_grid(context))
        indices = [int(value) for value in batch["index"][:keep]]
        future = predict_future_tokens(
            dit,
            context,
            normalizer,
            sample_indices=indices,
            base_seed=int(config["sampling"].get("base_seed", 3407)),
            steps=int(config["sampling"].get("steps", 100)),
            cfg_scale=float(config["sampling"].get("cfg_scale", 1)),
            internal_guidance_scale=float(config["sampling"].get("internal_guidance_scale", 1)),
            transport=transport,
        )
        video = decode_future_tokens(stage1, future, grid_size=tuple(dit.grid_size))
        for sample_id, sample_video in zip(batch["sample_id"][:keep], video.cpu(), strict=True):
            path = output_path / f"{sample_id}.pt"
            temporary = path.with_name(f".{path.name}.tmp")
            torch.save(sample_video, temporary)
            temporary.replace(path)
        completed += keep
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Cityscapes future videos")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    paths = load_project_paths(
        config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    generate_validation_predictions(
        config,
        paths,
        arguments.checkpoint,
        arguments.output,
        count=arguments.count,
        batch_size=arguments.batch_size,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "decode_future_tokens",
    "deterministic_future_noise",
    "generate_validation_predictions",
    "load_sampling_stack",
    "predict_future_tokens",
]
