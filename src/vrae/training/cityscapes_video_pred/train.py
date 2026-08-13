from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vrae.training.cityscapes_video_pred.data import (
    CITYSCAPES_IMAGE_SIZE,
    build_cityscapes_loader,
    build_raw_cityscapes_dataset,
    validate_frame_protocol,
)
from vrae.training.cityscapes_video_pred.latent_cache import (
    LatentCacheDataset,
    encode_context_future_separately,
)
from vrae.training.common.accumulation import GradientAccumulator, optimizer_step
from vrae.training.common.checkpoint import (
    build_training_checkpoint,
    load_model_init,
    resume_training,
    save_training_checkpoint,
)
from vrae.training.common.contracts import (
    load_run_metadata,
    resolve_batch_contract,
    update_decoder_execution_metadata,
    update_run_metadata,
)
from vrae.training.common.distributed import gather_rng_states, initialize_distributed, reduce_mean
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import (
    build_flow_transport,
    dit_architecture_metadata,
    flow_transport_metadata,
    load_frozen_stage1,
    parse_config_argument,
    prepare_run,
    seed_everything,
)
from vrae.training.common.latent_norm import LatentNormalizer, validate_normalizer_compatibility
from vrae.training.common.optim import build_optimizer, build_scheduler
from vrae.training.common.precision import PrecisionPolicy
from vrae.training.common.visualization import comparison_video, save_sample_tensors
from vrae.training.common.wandb import WandbLogger
from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.registry import DIT_MODELS, register_builtin_models


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrae.training.common.wandb import validate_wandb_config

    if "wandb" in config:
        validate_wandb_config(config["wandb"])
    if str(config.get("task")) != "cityscapes_video_pred":
        raise ValueError("Cityscapes entry requires task=cityscapes_video_pred")
    data = config["data"]
    if str(data.get("dataset")) != "cityscapes" or str(data.get("split")) != "train":
        raise ValueError("Cityscapes future prediction trains on the train split")
    image_size = tuple(int(value) for value in data.get("image_size", ()))
    if image_size != CITYSCAPES_IMAGE_SIZE:
        raise ValueError("formal Cityscapes training requires image_size=[432,768]")
    validate_frame_protocol(
        data.get("context_frames", ()),
        data.get("future_frames", ()),
    )
    dit = config["dit"]
    if str(dit.get("name")) != "vrae_video_prediction_dit":
        raise ValueError("Cityscapes must use the shared VRAEVideoPredictionDiT")
    if int(dit.get("context_chunks", 3)) != 3 or int(dit.get("future_chunks", 3)) != 3:
        raise ValueError("Cityscapes requires exactly three context and three future chunks")
    encoder = config["model"]["encoder"]
    hidden_size = int(encoder["hidden_size"])
    if int(dit.get("input_dim", hidden_size)) != hidden_size:
        raise ValueError("Cityscapes DiT input_dim must match the V-RAE hidden size")
    patch_size = int(encoder["patch_size"])
    if image_size[0] % patch_size or image_size[1] % patch_size:
        raise ValueError("Cityscapes image size must be divisible by the V-RAE patch size")
    latent_grid = (image_size[0] // patch_size, image_size[1] // patch_size)
    if latent_grid != (27, 48):
        raise ValueError(f"formal Cityscapes latent grid must be 27x48, got {latent_grid}")
    transport = flow_transport_metadata(config)
    future_chunks = int(dit.get("future_chunks", 3))
    expected_shift = math.sqrt(future_chunks * latent_grid[0] * latent_grid[1] * hidden_size / 4096)
    if transport["prediction"] != "x" or abs(transport["time_dist_shift"] - expected_shift) > 1e-9:
        raise ValueError("Cityscapes requires x-prediction and the future-latent time shift")
    return {
        "task": "cityscapes_video_pred",
        "dit": "vrae_video_prediction_dit",
        "image_size": [432, 768],
        "latent_grid": list(latent_grid),
        "context_relative_frames": [4, 15],
        "future_relative_frames": [16, 27],
        "context_chunks": 3,
        "future_chunks": 3,
        "video_backend": str(data.get("video_backend", "auto")),
        "latent_cache": data.get("latent_cache"),
        "transport": transport,
    }


def ensure_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    """Create the matching Cityscapes train-future normalizer when absent."""

    from vrae.training.cityscapes_video_pred.latent_stats import compute_cityscapes_latent_stats
    from vrae.training.common.distributed import barrier, broadcast_object, initialize_distributed

    distributed = initialize_distributed()
    output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
    missing = None
    if distributed.is_main:
        missing = not output.is_file() or output.stat().st_size == 0
    missing = bool(broadcast_object(missing))
    if not missing:
        if distributed.is_main:
            print(f"[latent-stats] using existing file: {output}", flush=True)
        return output

    if distributed.is_main:
        print(
            "[latent-stats] file missing; computing Cityscapes train-future statistics: "
            f"{output} (global_batch={int(config['training']['global_batch_size'])})",
            flush=True,
        )
    compute_cityscapes_latent_stats(config, paths)
    barrier()

    failure = None
    if distributed.is_main and (not output.is_file() or output.stat().st_size == 0):
        failure = f"Latent statistics were not created successfully: {output}"
    failure = broadcast_object(failure)
    if failure is not None:
        raise RuntimeError(str(failure))
    if distributed.is_main:
        print(f"[latent-stats] ready: {output}", flush=True)
    return output


def build_cityscapes_dit(
    config: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any],
) -> nn.Module:
    register_builtin_models()
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    patch_size = int(stage1_metadata["patch_size"])
    if image_size[0] % patch_size or image_size[1] % patch_size:
        raise ValueError("Cityscapes image size must be divisible by the V-RAE patch size")
    grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
    if grid_size != (27, 48):
        raise ValueError(f"formal Cityscapes latent grid must be 27x48, got {grid_size}")
    return DIT_MODELS.build(
        config["dit"],
        grid_size=grid_size,
        in_channels=int(stage1_metadata["hidden_size"]),
        context_chunks=3,
        future_chunks=3,
    )


def cityscapes_model_metadata(
    config: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any],
    dit: nn.Module,
    normalizer: LatentNormalizer | None = None,
) -> dict[str, Any]:
    from vrae.training.common.contracts import structural_stage1_metadata
    from vrae.training.common.latent_norm import normalizer_identity

    return {
        "stage1": structural_stage1_metadata(stage1_metadata),
        "stage1_checkpoint": str(config["stage1"]["checkpoint"]),
        "latent_normalizer_source": str(config["latent_normalizer"]["path"]),
        "dit_name": "vrae_video_prediction_dit",
        "dit_input_dim": int(dit.in_channels),
        "dit_grid_size": list(dit.grid_size),
        "context_relative_frames": [4, 15],
        "future_relative_frames": [16, 27],
        "context_chunks": int(dit.context_chunks),
        "future_chunks": int(dit.future_chunks),
        **dit_architecture_metadata(dit),
        "transport": flow_transport_metadata(config),
        "latent_normalizer_identity": (
            normalizer_identity(normalizer) if normalizer is not None else None
        ),
    }


def _resolve_project_path(value: object, paths: ProjectPaths) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (paths.project_root / path).resolve()


def build_training_dataset(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    stage1_metadata: Mapping[str, Any],
):
    cache_value = config["data"].get("latent_cache")
    if cache_value:
        return LatentCacheDataset(
            _resolve_project_path(cache_value, paths),
            "train",
            expected_stage1_metadata=stage1_metadata,
        )
    return build_raw_cityscapes_dataset(config, paths, split="train")


@torch.no_grad()
def prepare_cityscapes_latents(
    stage1: Any,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    if context.ndim == 5 and future.ndim == 5:
        return encode_context_future_separately(stage1, context, future)
    if context.ndim == 4 and future.ndim == 4:
        if context.shape != future.shape or context.shape[1] != 3:
            raise ValueError("cached context/future must be matching three-chunk tensors")
        return context, future
    raise ValueError("context/future must both be RGB clips or both be cached token latents")


def scalar_loss_metrics(losses: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only scalar optimization losses for reduction and logging."""

    return {
        name: value
        for name, value in losses.items()
        if name.startswith("loss") and value.numel() == 1
    }


def train(
    config: dict[str, Any],
    paths: ProjectPaths,
    run: Path,
    *,
    max_steps: int | None = None,
) -> None:
    distributed = initialize_distributed()
    seed_everything(int(config["data"].get("seed", 3407)), rank=distributed.rank)
    stage1, _ = load_frozen_stage1(config, paths, distributed.device)
    stage1_metadata = stage1.metadata()
    dit = build_cityscapes_dit(config, stage1_metadata).to(distributed.device)
    dit.requires_grad_(True)
    model: nn.Module = dit
    if distributed.world_size > 1:
        model = DistributedDataParallel(
            dit,
            device_ids=[distributed.local_rank] if distributed.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    normalizer = LatentNormalizer.load(paths.checkpoint(config["latent_normalizer"]["path"])).to(
        distributed.device
    )
    validate_normalizer_compatibility(
        normalizer,
        stage1_metadata=stage1_metadata,
        stage1_checkpoint=str(config["stage1"]["checkpoint"]),
        dataset="cityscapes",
        split="train",
        scope="future",
    )
    model_metadata = cityscapes_model_metadata(config, stage1_metadata, dit, normalizer)
    optimizer = build_optimizer(dit.parameters(), config["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, config["training"]["scheduler"])
    ema = ExponentialMovingAverage(dit, decay=float(config["training"]["ema"]["decay"]))
    precision = PrecisionPolicy(
        str(config["training"].get("precision", "bf16")),
        distributed.device.type,
    )
    scaler = precision.make_scaler()
    transport = build_flow_transport(config)
    dataset = build_training_dataset(
        config,
        paths,
        stage1_metadata=stage1_metadata,
    )
    loader, sampler = build_cityscapes_loader(
        dataset,
        config,
        rank=distributed.rank,
        world_size=distributed.world_size,
        shuffle=True,
        drop_last=True,
    )
    accumulation = GradientAccumulator(
        int(config["training"].get("gradient_accumulation_steps", 1))
    )
    batch_contract = resolve_batch_contract(config["training"], world_size=distributed.world_size)
    if distributed.is_main:
        update_run_metadata(
            run,
            {
                "world_size": distributed.world_size,
                "local_micro_batch_size": batch_contract.local_micro_batch_size,
                "global_micro_batch_size": batch_contract.global_micro_batch_size,
            },
        )
        wandb_run_id = str(load_run_metadata(run)["wandb_run_id"])
    else:
        wandb_run_id = None
    logger = WandbLogger(
        config["wandb"],
        enabled=distributed.is_main,
        run_dir=str(run / "wandb"),
        run_id=wandb_run_id,
        exact_resume=bool(config["training"].get("resume")),
    )

    start_epoch = 0
    step = 0
    metadata_fields = tuple(model_metadata)
    init_path = config["training"].get("init_from")
    resume_path = config["training"].get("resume")
    if init_path:
        load_model_init(
            paths.checkpoint(init_path),
            dit,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
        )
    if resume_path:
        resume_checkpoint = paths.checkpoint(resume_path)
        payload = resume_training(
            resume_checkpoint,
            model=dit,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
            current_config=config,
            run_directory=run,
        )
        accumulation.load_data_state(payload["data_state"])
        start_epoch = sampler.epoch
        step = int(payload["step"])
    if max_steps is not None and step >= int(max_steps):
        logger.finish()
        return

    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    log_interval = int(config["wandb"]["log_interval"])
    last_saved_step = step
    stop = False
    for epoch in range(start_epoch, int(config["training"].get("epochs", 1))):
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        for batch in loader:
            if accumulation.is_first_microstep:
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                raw_context, raw_future = prepare_cityscapes_latents(
                    stage1,
                    batch,
                    device=distributed.device,
                )
                context = normalizer.normalize(raw_context)
                future = normalizer.normalize(raw_future)
            with accumulation.sync_context(model):
                with precision.autocast():
                    losses = transport.training_losses(model, future, context)
                    loss = losses["loss"]
                accumulation.backward(loss, scaler)
            sampler.commit_batch()
            if not accumulation.advance():
                continue
            if not optimizer_step(optimizer, scaler=scaler):
                continue
            scheduler.step()
            ema.update(dit)
            step += 1
            if step % log_interval == 0:
                reduced_losses = {
                    name: reduce_mean(value.detach())
                    for name, value in scalar_loss_metrics(losses).items()
                }
            else:
                reduced_losses = None
            if distributed.is_main and reduced_losses is not None:
                print(
                    json.dumps(
                        {
                            "task": "cityscapes_video_pred",
                            "epoch": epoch,
                            "step": step,
                            **{name: float(value.item()) for name, value in reduced_losses.items()},
                        }
                    ),
                    flush=True,
                )
                logger.log(
                    {
                        **{
                            f"train/{name}": float(value.item())
                            for name, value in reduced_losses.items()
                        },
                        "train/epoch": epoch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_batch_size": batch_contract.global_batch_size,
                    },
                    step=step,
                )
            sample_interval = int(config["wandb"]["sample_interval"])
            if distributed.is_main and step % sample_interval == 0:
                from vrae.training.cityscapes_video_pred.sample import (
                    decode_future_tokens,
                    predict_future_tokens,
                )

                sample_count = min(
                    int(config["wandb"].get("sample_count", 2)),
                    raw_context.shape[0],
                )
                sample_indices = [int(value) for value in batch["index"][:sample_count].tolist()]
                was_training = dit.training
                dit.eval()
                with ema.average_parameters(dit), precision.autocast():
                    predicted_tokens = predict_future_tokens(
                        dit,
                        raw_context[:sample_count],
                        normalizer,
                        sample_indices=sample_indices,
                        base_seed=int(config["sampling"]["base_seed"]),
                        steps=int(config["sampling"]["steps"]),
                        cfg_scale=float(config["sampling"]["cfg_scale"]),
                        internal_guidance_scale=float(
                            config["sampling"]["internal_guidance_scale"]
                        ),
                        transport=transport,
                    )
                    predicted_video = decode_future_tokens(stage1, predicted_tokens)
                    real_video = decode_future_tokens(
                        stage1,
                        raw_future[:sample_count],
                    )
                dit.train(was_training)
                save_sample_tensors(
                    {"real": real_video, "predicted": predicted_video},
                    run / "samples" / f"step-{step:08d}.pt",
                )
                logger.log_video(
                    "samples/future_prediction",
                    comparison_video(real_video, predicted_video),
                    step=step,
                    fps=int(config["data"].get("fps", 16)),
                )
                update_decoder_execution_metadata(
                    run,
                    stage1.decoder_execution_metadata(),
                    training_dtype=precision.name,
                )
            if step % checkpoint_interval == 0:
                rng_state = gather_rng_states()
                if distributed.is_main:
                    payload = build_training_checkpoint(
                        task="cityscapes_video_pred",
                        epoch=sampler.epoch,
                        step=step,
                        model=dit,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        data_state=sampler.state_dict(
                            gradient_accumulation_microstep=accumulation.microstep
                        ),
                        resolved_config=config,
                        model_metadata=model_metadata,
                        rng_state=rng_state,
                    )
                    save_training_checkpoint(payload, run / "checkpoints")
                last_saved_step = step
            if max_steps is not None and step >= int(max_steps):
                stop = True
                break
        if stop:
            break
    if step > 0 and step != last_saved_step:
        accumulation.require_boundary()
        rng_state = gather_rng_states()
        if distributed.is_main:
            payload = build_training_checkpoint(
                task="cityscapes_video_pred",
                epoch=sampler.epoch,
                step=step,
                model=dit,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_state=sampler.state_dict(
                    gradient_accumulation_microstep=accumulation.microstep
                ),
                resolved_config=config,
                model_metadata=model_metadata,
                rng_state=rng_state,
            )
            save_training_checkpoint(payload, run / "checkpoints")
    logger.finish()


def main() -> None:
    arguments = parse_config_argument("Train Cityscapes V-RAE future prediction")
    initial_config = load_config(arguments.config)
    if arguments.build_only:
        print(json.dumps(validate_build(initial_config), indent=2))
        return
    validate_build(initial_config)
    initial_paths = load_project_paths(
        initial_config, override=arguments.paths, project_root=find_project_root(arguments.config)
    )
    ensure_latent_stats(initial_config, initial_paths)
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train(config, paths, run, max_steps=arguments.max_steps)


if __name__ == "__main__":
    main()


__all__ = [
    "build_cityscapes_dit",
    "build_training_dataset",
    "cityscapes_model_metadata",
    "ensure_latent_stats",
    "prepare_cityscapes_latents",
    "scalar_loss_metrics",
    "train",
    "validate_build",
]
