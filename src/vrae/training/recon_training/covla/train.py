"""Fine-tune the EUPE V-RAE reconstruction model on 432x768 CoVLA clips."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrae.training.recon_training.covla.data import (
    COVLA_IMAGE_SIZE,
    build_covla_reconstruction_loader,
    build_covla_visualization_batch,
    prepare_covla_reconstruction_batch,
)
from vrae.training.common.contracts import update_run_metadata
from vrae.training.common.distributed import broadcast_object, initialize_distributed
from vrae.training.common.engine import parse_config_argument, prepare_run
from vrae.training.recon_training.sample import save_reconstruction_sample
from vrae.training.recon_training.train import (
    train as train_reconstruction,
)
from vrae.training.recon_training.train import (
    validate_build as validate_reconstruction_build,
)
from vrae.config import load_config


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_reconstruction_build(config)
    if str(config.get("task")) != "recon_training":
        raise ValueError("CoVLA reconstruction requires task=recon_training")
    if str(config["model"]["encoder"]["name"]) != "eupe":
        raise ValueError("This CoVLA reconstruction recipe requires the EUPE encoder")

    data = config["data"]
    if str(data.get("dataset", "covla")).lower() != "covla":
        raise ValueError("CoVLA reconstruction requires data.dataset=covla")
    image_size = tuple(int(value) for value in data["image_size"])
    if image_size != COVLA_IMAGE_SIZE:
        raise ValueError(f"Formal CoVLA reconstruction image_size must be {COVLA_IMAGE_SIZE}")
    num_frames = int(data.get("num_frames", 24))
    if num_frames != 24:
        raise ValueError("Formal CoVLA reconstruction requires num_frames=24")
    if str(config["gan"].get("temporal_sampling", "strict")) != "uniform":
        raise ValueError(
            "24-frame CoVLA reconstruction requires gan.temporal_sampling=uniform "
            "for the frozen 16-frame VideoMAE discriminator"
        )
    if str(data.get("crop_mode", "center")) not in {"center", "random"}:
        raise ValueError("CoVLA crop_mode must be center or random")
    if bool(data.get("spatial_on_gpu", False)) and (
        str(data.get("crop_mode", "center")) != "center" or bool(data.get("random_flip", False))
    ):
        raise ValueError("data.spatial_on_gpu requires crop_mode=center and random_flip=false")
    validation_fraction = float(data.get("validation_fraction", 0.05))
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("CoVLA validation_fraction must be in [0,1)")
    validation_count = int(data.get("validation_count", -1))
    if validation_count != 4:
        raise ValueError("Formal CoVLA reconstruction requires validation_count=4")
    if str(data.get("validation_strategy", "hash")) != "tail":
        raise ValueError("Formal CoVLA reconstruction requires validation_strategy=tail")
    expected_total = int(data.get("expected_total", 0))
    if expected_total != 10_000:
        raise ValueError("Formal CoVLA reconstruction requires expected_total=10000")
    if round(expected_total * validation_fraction) != validation_count:
        raise ValueError("CoVLA validation_fraction must resolve to exactly four videos")

    training = config["training"]
    if not training.get("init_from") and not training.get("resume"):
        raise ValueError("CoVLA high-resolution fine-tuning requires init_from or resume")
    if training.get("init_from") and str(training.get("init_weights", "model")) != "ema":
        raise ValueError("The formal CoVLA warm start uses the source checkpoint EMA weights")
    decoder = config["model"]["decoder"]
    if tuple(int(value) for value in decoder["image_size"]) != (256, 256):
        raise ValueError(
            "Keep model.decoder.image_size=[256,256] when transferring the EUPE checkpoint"
        )
    return {
        **result,
        "dataset": "covla",
        "train_samples": expected_total - validation_count,
        "validation_samples": validation_count,
        "validation_strategy": str(data["validation_strategy"]),
        "num_frames": num_frames,
        "image_size": list(image_size),
        "crop_mode": str(data.get("crop_mode", "center")),
        "spatial_on_gpu": bool(data.get("spatial_on_gpu", False)),
        "init_weights": str(training.get("init_weights", "model")),
        "gradient_checkpointing": bool(decoder.get("gradient_checkpointing", False)),
    }


def _prepare_validation_preview(config: Mapping[str, Any], paths, run: Path) -> None:
    context = initialize_distributed()
    fixed_input_path = run / "samples" / "fixed_input.pt"
    root_error: Exception | None = None
    failure: dict[str, str] | None = None
    if context.is_main:
        try:
            expected_count = int(config["data"]["validation_count"])
            if fixed_input_path.is_file():
                payload = torch.load(fixed_input_path, map_location="cpu", weights_only=True)
                video = payload.get("video") if isinstance(payload, Mapping) else None
                sample_ids = payload.get("sample_ids") if isinstance(payload, Mapping) else None
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("split") != "val"
                    or not torch.is_tensor(video)
                    or video.ndim != 5
                    or int(video.shape[0]) != expected_count
                    or not isinstance(sample_ids, list)
                    or len(sample_ids) != expected_count
                ):
                    raise RuntimeError("Existing fixed_input.pt is not the configured val preview")
            else:
                batch = build_covla_visualization_batch(config, paths)
                video = prepare_covla_reconstruction_batch(batch, context.device, config)
                sample_ids = [str(value) for value in batch["sample_id"]]
                if int(video.shape[0]) != expected_count or len(set(sample_ids)) != expected_count:
                    raise RuntimeError("CoVLA val preview must contain every held-out video once")
                save_reconstruction_sample(
                    {
                        "video": video.detach().cpu(),
                        "split": "val",
                        "sample_ids": sample_ids,
                        "frame_indices": batch["frame_indices"].clone(),
                        "sampling": "center",
                    },
                    fixed_input_path,
                )
            update_run_metadata(
                run,
                {
                    "wandb_sample_split": "val",
                    "wandb_sample_ids": list(sample_ids),
                    "wandb_sample_count": expected_count,
                    "wandb_sample_sampling": "center",
                },
            )
        except Exception as error:
            root_error = error
            failure = {"type": type(error).__name__, "message": str(error)}
    failure = broadcast_object(failure)
    if failure is not None:
        if root_error is not None:
            raise root_error
        raise RuntimeError(
            "Rank 0 failed to prepare the fixed validation preview: "
            f"{failure['type']}: {failure['message']}"
        )


def train(config: dict[str, Any], paths, run: Path, *, max_steps: int | None = None) -> None:
    _prepare_validation_preview(config, paths, run)
    train_reconstruction(
        config,
        paths,
        run,
        max_steps=max_steps,
        loader_builder=build_covla_reconstruction_loader,
        batch_preparer=prepare_covla_reconstruction_batch,
    )


def main() -> None:
    arguments = parse_config_argument("Fine-tune EUPE V-RAE reconstruction on CoVLA at 432x768")
    if arguments.build_only:
        config = load_config(arguments.config)
        print(json.dumps(validate_build(config), indent=2))
        return
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train(config, paths, run, max_steps=arguments.max_steps)


if __name__ == "__main__":
    main()
