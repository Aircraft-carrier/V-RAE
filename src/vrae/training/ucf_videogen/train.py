from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrae.training.common.engine import (
    compute_class_conditional_latent_stats,
    flow_transport_metadata,
    parse_config_argument,
    prepare_run,
    train_class_conditional,
)
from vrae.config import load_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths


def ensure_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    """Create the configured UCF101 latent normalizer before training when absent."""

    from vrae.training.common.distributed import broadcast_object, initialize_distributed

    context = initialize_distributed()
    output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
    missing = None
    if context.is_main:
        missing = not output.is_file() or output.stat().st_size == 0
    missing = bool(broadcast_object(missing))
    if not missing:
        if context.is_main:
            print(f"[latent-stats] using existing file: {output}", flush=True)
        return output

    if context.is_main:
        print(
            "[latent-stats] file missing; computing before training: "
            f"{output} (global_batch={int(config['training']['global_batch_size'])})",
            flush=True,
        )
    compute_class_conditional_latent_stats(config, paths, dataset_name="ucf101")

    failure = None
    if context.is_main and (not output.is_file() or output.stat().st_size == 0):
        failure = f"Latent statistics were not created successfully: {output}"
    failure = broadcast_object(failure)
    if failure is not None:
        raise RuntimeError(str(failure))
    if context.is_main:
        print(f"[latent-stats] ready: {output}", flush=True)
    return output


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrae.training.common.wandb import validate_wandb_config

    if "wandb" in config:
        validate_wandb_config(config["wandb"])
    if config["task"] != "ucf_videogen":
        raise ValueError("UCF101 entry requires task=ucf_videogen")
    if config["data"]["dataset"] != "ucf101" or config["data"]["split"] != "train":
        raise ValueError("UCF101 generation trains on the UCF101 train split")
    if int(config["dit"]["num_classes"]) != 101:
        raise ValueError("UCF101 generation requires 101 classes")
    if int(config["data"]["num_frames"]) not in {16, 20}:
        raise ValueError("Class-conditional video training supports 16 or 20 RGB frames")
    runtime = config.get("runtime", {})
    pipeline = runtime.get("data_pipeline", {})
    if pipeline.get("kind", "torchcodec_cpu_bounded") != "torchcodec_cpu_bounded":
        raise ValueError("UCF101 training requires the bounded TorchCodec CPU pipeline")
    transport = flow_transport_metadata(config)
    num_chunks = int(config["data"]["num_frames"]) // 4
    hidden_size = int(config["model"]["encoder"]["hidden_size"])
    image_size = int(config["data"]["image_size"])
    patch_size = int(config["model"]["encoder"]["patch_size"])
    spatial_tokens = (image_size // patch_size) ** 2
    expected_shift = math.sqrt(num_chunks * spatial_tokens * hidden_size / 4096)
    if transport["prediction"] != "x" or abs(transport["time_dist_shift"] - expected_shift) > 1e-9:
        raise ValueError("UCF101 requires x-prediction and the latent-dimension time shift")
    return {
        "task": "ucf_videogen",
        "dit": "vrae_video_dit",
        "num_classes": 101,
        "num_chunks": num_chunks,
        "video_backend": "torchcodec",
        "data_pipeline": "torchcodec_cpu_bounded",
        "decode_threads_per_rank": int(pipeline.get("torchcodec_cpu_decode_threads", 8)),
        "max_inflight_per_rank": int(pipeline.get("torchcodec_cpu_max_inflight", 32)),
        "transport": transport,
    }


def main() -> None:
    arguments = parse_config_argument("Train UCF101 V-RAE VideoDiT")
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
    train_class_conditional(
        config, paths, run, dataset_name="ucf101", max_steps=arguments.max_steps
    )


if __name__ == "__main__":
    main()
