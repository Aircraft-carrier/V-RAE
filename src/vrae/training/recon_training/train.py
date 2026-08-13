from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vrae.training.common.accumulation import GradientAccumulator, optimizer_step
from vrae.training.common.checkpoint import (
    build_training_checkpoint,
    load_model_init,
    resume_training,
    save_training_checkpoint,
)
from vrae.training.common.contracts import (
    STAGE1_STRUCTURE_FIELDS,
    STAGE1_WEIGHT_COMPATIBILITY_FIELDS,
    load_run_metadata,
    resolve_batch_contract,
    update_decoder_execution_metadata,
    update_run_metadata,
    validate_checkpoint_identity,
)
from vrae.training.common.distributed import (
    configure_ddp_gradient_compression,
    gather_rng_states,
    initialize_distributed,
    reduce_mean,
    resolve_ddp_gradient_compression,
)
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.training.common.engine import parse_config_argument, prepare_run, seed_everything
from vrae.training.common.optim import build_optimizer, build_scheduler
from vrae.training.common.precision import PrecisionPolicy
from vrae.training.common.visualization import comparison_video
from vrae.training.common.wandb import WandbLogger
from vrae.training.recon_training.data import CudaVideoPrefetchIterator, build_reconstruction_loader
from vrae.training.recon_training.gan import GANController, VideoMAEDiscriminator
from vrae.training.recon_training.losses import ReconstructionLoss
from vrae.training.recon_training.noise import add_reconstruction_noise
from vrae.training.recon_training.sample import save_reconstruction_sample
from vrae.checkpoint import CheckpointError, compare_metadata, load_checkpoint
from vrae.config import load_config
from vrae.models.autoencoder import VRAE


class ReconstructionGraph(nn.Module):
    def __init__(self, vrae: VRAE, noise_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.vrae = vrae
        self.noise_config = dict(noise_config)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        clean = self.vrae.encode(video)
        train_latents = clean
        sigma = clean.new_zeros((clean.shape[0], clean.shape[1], 1, 1, 1))
        if self.training and bool(self.noise_config.get("enabled", False)):
            train_latents, sigma = add_reconstruction_noise(
                clean, float(self.noise_config.get("tau", 0.0))
            )
        return {"recon": self.vrae.decode(train_latents), "latents": clean, "sigma": sigma}


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrae.training.common.wandb import validate_wandb_config

    if "wandb" in config:
        validate_wandb_config(config["wandb"])
    training = config["training"]
    if str(training["scheduler"]["name"]) != "constant":
        raise ValueError("The formal reconstruction scheduler must be constant")
    if float(training["optimizer"]["lr"]) <= 0:
        raise ValueError("The reconstruction learning rate must be positive")
    gradient_compression = resolve_ddp_gradient_compression(
        training.get("ddp_gradient_compression", "none")
    )
    init_weights = str(training.get("init_weights", "model"))
    if init_weights not in {"model", "ema"}:
        raise ValueError("training.init_weights must be model or ema")
    if init_weights == "ema" and not training.get("init_from") and not training.get("resume"):
        raise ValueError("training.init_weights=ema requires init_from or resume")
    gan = config["gan"]
    generator_start = int(gan["generator_start_epoch"])
    discriminator_start = int(gan["discriminator_start_epoch"])
    if generator_start != discriminator_start:
        raise ValueError("Generator and discriminator updates must start in the same epoch")
    if not 0 <= generator_start < int(training["epochs"]):
        raise ValueError("The adversarial start epoch must be inside the training schedule")
    if str(gan.get("input_range")) != "zero_one":
        raise ValueError("The formal VideoMAE discriminator input range must be zero_one")
    if tuple(int(value) for value in gan["key_depths"]) != (2, 5, 8, 11):
        raise ValueError("The formal VideoMAE feature depths must be [2,5,8,11]")
    if int(gan["head_kernel_size"]) != 9 or str(gan["head_norm_type"]) != "bn":
        raise ValueError("The formal VideoMAE discriminator uses 9-wide locally normalized heads")
    if not bool(gan["spectral_norm"]):
        raise ValueError("The formal VideoMAE discriminator heads require spectral normalization")
    if float(gan["diff_aug_prob"]) != 1.0 or float(gan["diff_aug_cutout"]) != 0.2:
        raise ValueError("The formal VideoMAE DiffAug recipe is probability=1.0, cutout=0.2")
    if str(gan["gen_loss_type"]) != "ns_g_loss":
        raise ValueError("The formal VideoMAE generator uses ns_g_loss")
    if str(gan["disc_loss_type"]) != "ns_smooth":
        raise ValueError("The formal VideoMAE discriminator uses ns_smooth")
    if not bool(gan["lecam"]) or float(gan["lecam_weight"]) != 0.001:
        raise ValueError("The formal VideoMAE discriminator uses LeCam weight 0.001")
    if float(gan["discriminator_weight"]) != 0.3:
        raise ValueError("The formal VideoMAE discriminator loss weight is 0.3")
    return {
        "task": config["task"],
        "encoder": config["model"]["encoder"]["name"],
        "pool_group": config["model"]["pooling"]["group_size"],
        "attention_mode": config["model"]["decoder"]["attention_mode"],
        "video_backend": config["data"].get("video_backend", "auto"),
        "ddp_gradient_compression": gradient_compression,
    }


def _resolve_loss_config(config: Mapping[str, Any], paths) -> dict[str, Any]:
    value = dict(config["loss"])
    for key in ("backbone_checkpoint", "calibration_checkpoint"):
        if key in value:
            value[key] = str(paths.checkpoint(value[key]))
    return value


@torch.no_grad()
def _copy_checkpoint_ema_to_trainable(
    trainable: nn.Module,
    ema_state: object,
) -> None:
    if not isinstance(ema_state, Mapping):
        raise CheckpointError("Initial checkpoint does not contain EMA state")
    shadow = ema_state.get("shadow")
    if not isinstance(shadow, Mapping):
        raise CheckpointError("Initial checkpoint EMA state does not contain a shadow mapping")
    target = trainable.state_dict()
    expected = {name for name, value in target.items() if torch.is_floating_point(value)}
    if set(shadow) != expected:
        missing = sorted(expected.difference(shadow))
        unexpected = sorted(set(shadow).difference(expected))
        raise CheckpointError(
            "Initial checkpoint EMA keys do not match trainable V-RAE modules: "
            f"missing={missing} unexpected={unexpected}"
        )
    for name, average in shadow.items():
        if not torch.is_tensor(average) or average.shape != target[name].shape:
            raise CheckpointError(f"Invalid initial checkpoint EMA tensor: {name}")
        target[name].copy_(average.to(device=target[name].device, dtype=target[name].dtype))


def _initialize_model(
    config: Mapping[str, Any],
    paths,
    model: VRAE,
    trainable: nn.Module,
    *,
    model_metadata: Mapping[str, Any],
    weight_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    init_value = config["training"].get("init_from")
    if not init_value:
        return None
    init_path = paths.checkpoint(init_value)
    payload = load_model_init(
        init_path,
        model,
        expected_metadata=model_metadata,
        metadata_fields=weight_fields,
    )
    if str(payload.get("task")) != "recon_training":
        raise CheckpointError(
            f"V-RAE initialization requires task='recon_training', got {payload.get('task')!r}"
        )
    weight_source = str(config["training"].get("init_weights", "model"))
    if weight_source not in {"model", "ema"}:
        raise ValueError("training.init_weights must be model or ema")
    if weight_source == "ema":
        _copy_checkpoint_ema_to_trainable(trainable, payload.get("ema"))
    summary = {
        "checkpoint": str(init_path.resolve()),
        "weights": weight_source,
        "source_epoch": int(payload["epoch"]),
        "source_step": int(payload["step"]),
    }
    del payload
    gc.collect()
    return summary


def train(
    config: dict[str, Any],
    paths,
    run: Path,
    *,
    max_steps: int | None = None,
    loader_builder=build_reconstruction_loader,
    batch_preparer: Callable[[Mapping[str, Any], torch.device, Mapping[str, Any]], torch.Tensor]
    | None = None,
) -> None:
    context = initialize_distributed()
    seed_everything(int(config["data"].get("seed", 3407)), rank=context.rank)
    if context.device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", False))
    model = VRAE.from_config(config, project_paths=paths).to(context.device)
    if bool(config["training"].get("compile_encoder", False)):
        model.encoder.enable_compile(
            mode=str(config["training"].get("compile_encoder_mode", "default")),
            fullgraph=bool(config["training"].get("compile_encoder_fullgraph", False)),
            dynamic=False,
        )
    if bool(config["training"].get("compile_decoder", False)):
        compile_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[str(config["training"].get("precision", "bf16")).lower()]
        model.decoder.prime_rope_cache(
            num_frames=int(config["data"]["num_frames"]),
            device=context.device,
            dtype=compile_dtype,
        )
        model.decoder.enable_compile(
            mode=str(config["training"].get("compile_decoder_mode", "default")),
            fullgraph=bool(config["training"].get("compile_decoder_fullgraph", False)),
            dynamic=False,
        )
    model_metadata = model.metadata()
    metadata_fields = tuple(field for field in STAGE1_STRUCTURE_FIELDS if field in model_metadata)
    weight_fields = tuple(
        field for field in STAGE1_WEIGHT_COMPATIBILITY_FIELDS if field in model_metadata
    )
    trainable = nn.ModuleDict(model.trainable_groups())
    initialization = _initialize_model(
        config,
        paths,
        model,
        trainable,
        model_metadata=model_metadata,
        weight_fields=weight_fields,
    )
    graph: nn.Module = ReconstructionGraph(model, config["training"]["latent_noise"]).to(
        context.device
    )
    if context.world_size > 1:
        graph = DistributedDataParallel(
            graph,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            bucket_cap_mb=config["training"].get("ddp_bucket_cap_mb"),
            bucket_cap_mb_list=config["training"].get("ddp_bucket_cap_mb_list"),
            gradient_as_bucket_view=bool(
                config["training"].get("ddp_gradient_as_bucket_view", False)
            ),
            static_graph=bool(config["training"].get("ddp_static_graph", False)),
            batched_grad_copy=bool(config["training"].get("ddp_batched_grad_copy", False)),
        )
    ddp_gradient_compression = configure_ddp_gradient_compression(
        graph,
        config["training"].get("ddp_gradient_compression", "none"),
    )
    optimizer = build_optimizer(trainable.parameters(), config["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, config["training"]["scheduler"])
    ema = ExponentialMovingAverage(trainable, decay=float(config["training"]["ema"]["decay"]))
    precision = PrecisionPolicy(
        str(config["training"].get("precision", "bf16")), context.device.type
    )
    scaler = precision.make_scaler()
    loss_module = ReconstructionLoss(_resolve_loss_config(config, paths)).to(context.device)
    if bool(config["training"].get("compile_reconstruction_loss", False)):
        loss_module.enable_compile(
            mode=str(config["training"].get("compile_reconstruction_loss_mode", "default")),
            fullgraph=bool(config["training"].get("compile_reconstruction_loss_fullgraph", False)),
            dynamic=False,
        )
    loader, sampler = loader_builder(
        config, paths, rank=context.rank, world_size=context.world_size
    )
    prepare_batch = batch_preparer or (
        lambda batch, device, _config: batch["video"].to(device, non_blocking=True)
    )
    gan_config = dict(config["gan"])
    gan_checkpoint = paths.checkpoint(gan_config["discriminator_checkpoint"], require_exists=False)

    def build_discriminator() -> nn.Module:
        if str(gan_config.get("input_range")) != "zero_one":
            raise ValueError("VideoMAE discriminator input_range must be zero_one")
        discriminator: nn.Module = VideoMAEDiscriminator(
            gan_checkpoint,
            taps=tuple(int(value) for value in gan_config["key_depths"]),
            input_clamp=bool(gan_config["input_clamp"]),
            head_kernel_size=int(gan_config["head_kernel_size"]),
            head_norm_type=str(gan_config["head_norm_type"]),
            spectral_norm_heads=bool(gan_config["spectral_norm"]),
            norm_eps=float(gan_config["norm_eps"]),
            diff_aug_prob=float(gan_config["diff_aug_prob"]),
            diff_aug_cutout=float(gan_config["diff_aug_cutout"]),
            temporal_sampling=str(gan_config.get("temporal_sampling", "strict")),
        ).to(context.device)
        if context.world_size > 1:
            discriminator = DistributedDataParallel(
                discriminator,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                broadcast_buffers=False,
                bucket_cap_mb=config["training"].get("ddp_bucket_cap_mb"),
                bucket_cap_mb_list=config["training"].get("ddp_bucket_cap_mb_list"),
                gradient_as_bucket_view=bool(
                    config["training"].get("ddp_gradient_as_bucket_view", False)
                ),
                static_graph=bool(config["training"].get("ddp_static_graph", False)),
                batched_grad_copy=bool(config["training"].get("ddp_batched_grad_copy", False)),
            )
            configure_ddp_gradient_compression(
                discriminator,
                config["training"].get("ddp_gradient_compression", "none"),
            )
        return discriminator

    gan = GANController(
        gan_config,
        build_discriminator,
        device=context.device,
    )
    accumulation = GradientAccumulator(
        int(config["training"].get("gradient_accumulation_steps", 1))
    )
    batch_contract = resolve_batch_contract(config["training"], world_size=context.world_size)
    if context.is_main:
        run_updates: dict[str, Any] = {
            "world_size": context.world_size,
            "local_micro_batch_size": batch_contract.local_micro_batch_size,
            "global_micro_batch_size": batch_contract.global_micro_batch_size,
            "ddp_static_graph": bool(config["training"].get("ddp_static_graph", False)),
            "ddp_gradient_as_bucket_view": bool(
                config["training"].get("ddp_gradient_as_bucket_view", False)
            ),
            "ddp_gradient_compression": ddp_gradient_compression,
            "ddp_batched_grad_copy": bool(config["training"].get("ddp_batched_grad_copy", False)),
            "ddp_bucket_cap_mb": config["training"].get("ddp_bucket_cap_mb"),
            "ddp_bucket_cap_mb_list": config["training"].get("ddp_bucket_cap_mb_list"),
            "optimizer_fused": bool(optimizer.defaults.get("fused", False)),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "distributed_timeout_seconds": context.timeout_seconds,
            "dataloader_num_workers": int(config["training"].get("num_workers", 4)),
            "dataloader_prefetch_factor": loader.prefetch_factor,
            "dataloader_persistent_workers": bool(loader.persistent_workers),
        }
        if initialization is not None:
            run_updates["initialization"] = initialization
        update_run_metadata(
            run,
            run_updates,
        )
        wandb_run_id = str(load_run_metadata(run)["wandb_run_id"])
    else:
        wandb_run_id = None
    logger = WandbLogger(
        config["wandb"],
        enabled=context.is_main,
        run_dir=str(run / "wandb"),
        run_id=wandb_run_id,
        exact_resume=bool(config["training"].get("resume")),
    )

    start_epoch = 0
    step = 0
    resume_path = config["training"].get("resume")
    if resume_path:
        resume_checkpoint = paths.checkpoint(resume_path)
        preview = load_checkpoint(resume_checkpoint)
        validate_checkpoint_identity(
            preview,
            config,
            checkpoint_path=resume_checkpoint,
            run_directory=run,
        )
        compare_metadata(
            model_metadata,
            preview["model_metadata"],
            metadata_fields,
        )
        checkpoint_epoch = int(preview["data_state"]["sampler_epoch"])
        gan_runtime = preview.get("gan_runtime")
        if not isinstance(gan_runtime, Mapping):
            raise CheckpointError("Reconstruction checkpoint is missing gan_runtime state")
        gan_initialized = bool(gan_runtime.get("initialized", False))
        if (preview.get("gan") is not None) != gan_initialized:
            raise CheckpointError("Checkpoint GAN parameters disagree with gan_runtime state")
        loss_state = gan_runtime.get("loss_state")
        if not isinstance(loss_state, Mapping):
            raise CheckpointError("Reconstruction checkpoint is missing GAN loss state")
        try:
            gan.load_loss_state_dict(loss_state)
        except (TypeError, ValueError) as error:
            raise CheckpointError(f"Incompatible GAN loss state: {error}") from error
        if gan_initialized:
            gan.ensure_initialized(checkpoint_epoch)
            constructed_epoch = gan_runtime.get("constructed_epoch")
            if constructed_epoch is None or int(constructed_epoch) > checkpoint_epoch:
                raise CheckpointError("Checkpoint GAN constructed_epoch is invalid")
            gan.constructed_epoch = int(constructed_epoch)
        optimizer_objects: dict[str, Any] = {"generator": optimizer}
        scheduler_objects: dict[str, Any] = {"generator": scheduler}
        if gan.optimizer is not None:
            optimizer_objects["discriminator"] = gan.optimizer
            scheduler_objects["discriminator"] = gan.scheduler
        payload = resume_training(
            resume_checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer_objects,
            scheduler=scheduler_objects,
            scaler=scaler,
            sampler=sampler,
            expected_metadata=model_metadata,
            metadata_fields=metadata_fields,
            current_config=config,
            run_directory=run,
        )
        accumulation.load_data_state(payload["data_state"])
        start_epoch, step = sampler.epoch, int(payload["step"])
        if gan_initialized:
            gan.discriminator.load_state_dict(payload["gan"], strict=True)

    if max_steps is not None and step >= int(max_steps):
        logger.finish()
        return

    checkpoint_interval = int(config["training"].get("checkpoint_interval", 1000))
    log_interval = int(config["wandb"]["log_interval"])
    sample_interval = int(config["wandb"]["sample_interval"])
    fixed_input_path = run / "samples" / "fixed_input.pt"
    fixed_video: torch.Tensor | None = None
    if context.is_main and fixed_input_path.is_file():
        fixed_payload = torch.load(fixed_input_path, map_location="cpu", weights_only=True)
        fixed_video = fixed_payload["video"]
    last_saved_step = step
    throughput_started = time.perf_counter()
    throughput_start_step = step
    execution_metadata_recorded = False
    stop = False
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        loader.dataset.set_epoch(epoch)
        batches = iter(loader)
        if (
            batch_preparer is None
            and context.device.type == "cuda"
            and bool(config["training"].get("prefetch_to_device", False))
        ):
            batches = CudaVideoPrefetchIterator(batches, context.device)
        for batch in batches:
            if accumulation.is_first_microstep:
                optimizer.zero_grad(set_to_none=True)
            video = prepare_batch(batch, context.device, config)
            if context.is_main and fixed_video is None:
                sample_count = min(int(config["wandb"].get("sample_count", 4)), video.shape[0])
                fixed_video = video[:sample_count].detach().cpu()
                save_reconstruction_sample({"video": fixed_video}, fixed_input_path)
            if any(
                bool(config["training"].get(enabled, False))
                and str(config["training"].get(mode, "default")) == "reduce-overhead"
                for enabled, mode in (
                    ("compile_encoder", "compile_encoder_mode"),
                    ("compile_decoder", "compile_decoder_mode"),
                    ("compile_reconstruction_loss", "compile_reconstruction_loss_mode"),
                )
            ):
                torch.compiler.cudagraph_mark_step_begin()
            with accumulation.sync_context(graph):
                with precision.autocast():
                    result = graph(video)
                    total, metrics = loss_module(result["recon"], video)
                    adversarial = (
                        gan.generator_loss(result["recon"], epoch) if gan.active(epoch) else None
                    )
                    if adversarial is not None:
                        total = total + float(gan_config.get("generator_weight", 0.0)) * adversarial
                accumulation.backward(total, scaler)
            if gan.active(epoch):
                with precision.autocast():
                    gan.discriminator_step(
                        video,
                        result["recon"],
                        epoch=epoch,
                        step=step,
                        microstep=accumulation.microstep,
                        accumulation_steps=accumulation.steps,
                    )
            sampler.commit_batch()
            if not accumulation.advance():
                continue
            if not optimizer_step(optimizer, scaler=scaler):
                continue
            scheduler.step()
            ema.update(trainable)
            step += 1
            if context.is_main and not execution_metadata_recorded:
                update_decoder_execution_metadata(
                    run,
                    model.decoder.execution_metadata(),
                    training_dtype=precision.name,
                )
                execution_metadata_recorded = True
            if step % log_interval == 0:
                metric_tensors = {**metrics, "total": total.detach()}
                if adversarial is not None:
                    metric_tensors["generator_adversarial"] = adversarial.detach()
                if gan.active(epoch):
                    metric_tensors.update(gan.monitoring_metrics())
                reduced_metrics = {
                    name: reduce_mean(value.detach()) for name, value in metric_tensors.items()
                }
                throughput_finished = time.perf_counter()
                measured_steps = step - throughput_start_step
                samples_per_second = (
                    measured_steps
                    * batch_contract.global_batch_size
                    / max(throughput_finished - throughput_started, 1.0e-9)
                )
                throughput_started = throughput_finished
                throughput_start_step = step
            else:
                reduced_metrics = None
                samples_per_second = None
            if context.is_main and reduced_metrics is not None:
                values = {name: float(value.item()) for name, value in reduced_metrics.items()}
                print(json.dumps({"epoch": epoch, "step": step, **values}), flush=True)
                logger.log(
                    {
                        **{f"train/{name}": value for name, value in values.items()},
                        "train/epoch": epoch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_batch_size": batch_contract.global_batch_size,
                        "train/samples_per_second": samples_per_second,
                    },
                    step=step,
                )
            if context.is_main and step % sample_interval == 0:
                if fixed_video is None:
                    raise RuntimeError("Fixed reconstruction sample was not initialized")
                was_training = model.training
                model.eval()
                with ema.average_parameters(trainable), torch.no_grad(), precision.autocast():
                    fixed_on_device = fixed_video.to(context.device)
                    if any(
                        bool(config["training"].get(enabled, False))
                        and str(config["training"].get(mode, "default")) == "reduce-overhead"
                        for enabled, mode in (
                            ("compile_encoder", "compile_encoder_mode"),
                            ("compile_decoder", "compile_decoder_mode"),
                        )
                    ):
                        torch.compiler.cudagraph_mark_step_begin()
                    reconstructed = model(fixed_on_device)["recon"]
                model.train(was_training)
                sample = {
                    "real": fixed_video,
                    "recon": reconstructed.detach().cpu(),
                }
                save_reconstruction_sample(
                    sample,
                    run / "samples" / f"step-{step:08d}.pt",
                )
                logger.log_video(
                    "samples/reconstruction",
                    comparison_video(sample["real"], sample["recon"]),
                    step=step,
                    fps=int(config["data"].get("fps", 8)),
                )
                update_decoder_execution_metadata(
                    run,
                    model.decoder.execution_metadata(),
                    training_dtype=precision.name,
                )
            if step % checkpoint_interval == 0:
                rng_state = gather_rng_states()
                if context.is_main:
                    optimizer_objects: dict[str, Any] = {"generator": optimizer}
                    scheduler_objects: dict[str, Any] = {"generator": scheduler}
                    if gan.optimizer is not None:
                        optimizer_objects["discriminator"] = gan.optimizer
                        scheduler_objects["discriminator"] = gan.scheduler
                    payload = build_training_checkpoint(
                        task="recon_training",
                        epoch=sampler.epoch,
                        step=step,
                        model=model,
                        ema=ema,
                        optimizer=optimizer_objects,
                        scheduler=scheduler_objects,
                        scaler=scaler,
                        data_state=sampler.state_dict(
                            gradient_accumulation_microstep=accumulation.microstep
                        ),
                        resolved_config=config,
                        model_metadata=model.metadata(),
                        rng_state=rng_state,
                    )
                    payload["gan"] = (
                        gan.discriminator.state_dict() if gan.discriminator is not None else None
                    )
                    payload["gan_runtime"] = {
                        "initialized": gan.discriminator is not None,
                        "constructed_epoch": gan.constructed_epoch,
                        "loss_state": gan.loss_state_dict(),
                    }
                    save_training_checkpoint(payload, run / "checkpoints")
                last_saved_step = step
            if max_steps is not None and step >= max_steps:
                stop = True
                break
        if stop:
            break
    if step > 0 and step != last_saved_step:
        accumulation.require_boundary()
        rng_state = gather_rng_states()
        if context.is_main:
            optimizer_objects = {"generator": optimizer}
            scheduler_objects = {"generator": scheduler}
            if gan.optimizer is not None:
                optimizer_objects["discriminator"] = gan.optimizer
                scheduler_objects["discriminator"] = gan.scheduler
            payload = build_training_checkpoint(
                task="recon_training",
                epoch=sampler.epoch,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer_objects,
                scheduler=scheduler_objects,
                scaler=scaler,
                data_state=sampler.state_dict(
                    gradient_accumulation_microstep=accumulation.microstep
                ),
                resolved_config=config,
                model_metadata=model.metadata(),
                rng_state=rng_state,
            )
            payload["gan"] = (
                gan.discriminator.state_dict() if gan.discriminator is not None else None
            )
            payload["gan_runtime"] = {
                "initialized": gan.discriminator is not None,
                "constructed_epoch": gan.constructed_epoch,
                "loss_state": gan.loss_state_dict(),
            }
            save_training_checkpoint(payload, run / "checkpoints")
    if context.is_main:
        update_decoder_execution_metadata(
            run,
            model.decoder.execution_metadata(),
            training_dtype=precision.name,
        )
    logger.finish()


def main() -> None:
    arguments = parse_config_argument("Train V-RAE reconstruction")
    if arguments.build_only:
        config = load_config(arguments.config)
        print(json.dumps(validate_build(config), indent=2))
        return
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train(config, paths, run, max_steps=arguments.max_steps)


if __name__ == "__main__":
    main()
