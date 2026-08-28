from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vrae.config import load_config
from vrae.libero import LiberoClassMap
from vrae.paths import ProjectPaths, find_project_root, load_project_paths
from vrae.training.common.engine import (
    compute_class_conditional_latent_stats,
    flow_transport_metadata,
    parse_config_argument,
    prepare_run,
    train_class_conditional,
)


EXPECTED_SUITES = (
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)
EXPECTED_TASK_INDICES = tuple(range(40))


def build_class_map(config: Mapping[str, Any]) -> LiberoClassMap:
    suites = config["data"].get("class_suites")
    if not isinstance(suites, Sequence) or isinstance(suites, (str, bytes)):
        raise ValueError("LIBERO generation requires data.class_suites")
    class_map = LiberoClassMap.from_config(
        suites,
        available_task_indices=EXPECTED_TASK_INDICES,
    )
    suite_names = tuple(str(item["name"]) for item in suites)
    if suite_names != EXPECTED_SUITES:
        raise ValueError(
            "data.class_suites must be ordered as "
            f"{EXPECTED_SUITES}, got {suite_names}"
        )
    if any(len(item["task_indices"]) != 10 for item in suites):
        raise ValueError("each standard LIBERO suite must contain exactly 10 tasks")
    return class_map


def validate_build(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrae.training.common.wandb import validate_wandb_config

    validate_wandb_config(config["wandb"])
    if config.get("task") != "libero_videogen":
        raise ValueError("LIBERO generation requires task=libero_videogen")
    if config["data"].get("dataset") != "lerobot":
        raise ValueError("LIBERO generation requires data.dataset=lerobot")

    class_map = build_class_map(config)
    if int(config["dit"]["num_classes"]) != len(class_map):
        raise ValueError("dit.num_classes must equal the 40 LIBERO suite/task classes")
    sample_class_ids = [int(value) for value in config["sampling"].get("class_ids", [])]
    if any(value < 0 or value >= len(class_map) for value in sample_class_ids):
        raise ValueError("sampling.class_ids must reference valid LIBERO classes")

    num_frames = int(config["data"]["num_frames"])
    if num_frames <= 0 or num_frames % 4:
        raise ValueError("data.num_frames must be a positive multiple of four")
    cameras = config["data"]["camera_keys"]
    multiview = config["model"].get("multiview", {})
    num_views = int(multiview.get("num_views", 1)) if multiview.get("enabled") else 1
    if num_views != len(cameras):
        raise ValueError("model.multiview.num_views must match data.camera_keys")

    transport = flow_transport_metadata(config)
    num_chunks = num_frames // 4
    image_size = int(config["data"]["image_size"])
    patch_size = int(config["model"]["encoder"]["patch_size"])
    hidden_size = int(config["model"]["encoder"]["hidden_size"])
    spatial_tokens = (image_size // patch_size) ** 2
    expected_shift = math.sqrt(
        num_chunks * num_views * spatial_tokens * hidden_size / 4096
    )
    if transport["prediction"] != "x":
        raise ValueError("LIBERO VideoDiT uses x-prediction")
    if abs(transport["time_dist_shift"] - expected_shift) > 1.0e-9:
        raise ValueError(
            "transport.time_dist_shift must match the multiview latent dimension: "
            f"expected {expected_shift}"
        )
    pipeline = config.get("runtime", {}).get("data_pipeline", {})
    if pipeline.get("kind") != "lerobot_bounded":
        raise ValueError("LIBERO generation requires runtime.data_pipeline.kind=lerobot_bounded")
    return {
        "task": config["task"],
        "dataset": "lerobot",
        "num_classes": len(class_map),
        "suites": list(EXPECTED_SUITES),
        "num_chunks": num_chunks,
        "num_views": num_views,
        "transport": transport,
    }


def ensure_latent_stats(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    from vrae.training.common.distributed import broadcast_object, initialize_distributed

    context = initialize_distributed()
    output = paths.checkpoint(config["latent_normalizer"]["path"], require_exists=False)
    missing: bool | None = None
    if context.is_main:
        missing = not output.is_file() or output.stat().st_size == 0
    missing = bool(broadcast_object(missing))
    if not missing:
        if context.is_main:
            print(f"[latent-stats] using existing file: {output}", flush=True)
        return output

    if context.is_main:
        print(f"[latent-stats] computing LIBERO statistics: {output}", flush=True)
    compute_class_conditional_latent_stats(config, paths, dataset_name="lerobot")

    failure: str | None = None
    if context.is_main and (not output.is_file() or output.stat().st_size == 0):
        failure = f"latent statistics were not created: {output}"
    failure = broadcast_object(failure)
    if failure is not None:
        raise RuntimeError(str(failure))
    return output


def main() -> None:
    arguments = parse_config_argument("Train LIBERO class-conditional V-RAE VideoDiT")
    initial_config = load_config(arguments.config)
    validation = validate_build(initial_config)
    if arguments.build_only:
        print(json.dumps(validation, indent=2))
        return

    initial_paths = load_project_paths(
        initial_config,
        override=arguments.paths,
        project_root=find_project_root(arguments.config),
    )
    ensure_latent_stats(initial_config, initial_paths)
    config, paths, run = prepare_run(arguments.config, arguments.paths)
    validate_build(config)
    train_class_conditional(
        config,
        paths,
        run,
        dataset_name="lerobot",
        max_steps=arguments.max_steps,
    )


if __name__ == "__main__":
    main()
