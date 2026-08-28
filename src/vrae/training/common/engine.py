"""Small shared runtime helpers for V-JEPA 2.1 reconstruction training."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vrae.config import load_config, save_resolved_config
from vrae.paths import ProjectPaths, find_project_root, load_project_paths


def seed_everything(seed: int, *, rank: int = 0) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual % (2**32))
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual)


def parse_config_argument(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def prepare_run(
    config_path: str | Path, paths_path: str | Path | None = None
) -> tuple[dict[str, Any], ProjectPaths, Path]:
    """Resolve paths and create an immutable run directory on rank zero."""

    from vrae.training.common.distributed import broadcast_object, initialize_distributed
    from vrae.training.common.wandb import bind_wandb_run_name, validate_wandb_config

    distributed = initialize_distributed()
    config = load_config(config_path)
    config["wandb"] = bind_wandb_run_name(config.get("wandb", {}), str(config["run_name"]))
    validate_wandb_config(config.get("wandb", {}))
    paths = load_project_paths(
        config,
        override=paths_path,
        project_root=find_project_root(config_path),
    )
    run = paths.training_run(str(config["task"]), str(config["run_name"]), create=False)
    root_error: Exception | None = None
    failure: dict[str, str] | None = None
    if distributed.is_main:
        try:
            from vrae.checkpoint import CheckpointError, load_checkpoint
            from vrae.training.common.contracts import (
                compare_resolved_configs,
                create_run_metadata,
                load_run_metadata,
                run_identity,
                update_run_metadata,
                validate_checkpoint_identity,
            )

            resume_value = config.get("training", {}).get("resume")
            if resume_value:
                if not run.is_dir():
                    raise CheckpointError(f"Exact-resume run directory does not exist: {run}")
                checkpoint_path = paths.checkpoint(resume_value)
                payload = load_checkpoint(checkpoint_path)
                validate_checkpoint_identity(
                    payload,
                    config,
                    checkpoint_path=checkpoint_path,
                    run_directory=run,
                )
                resolved_path = run / "resolved_config.yaml"
                if not resolved_path.is_file():
                    raise CheckpointError("Exact-resume run is missing resolved_config.yaml")
                previous_config = load_config(resolved_path)
                compare_resolved_configs(previous_config, payload["resolved_config"])
                compare_resolved_configs(previous_config, config)
                metadata = load_run_metadata(run)
                if metadata.get("run_identity") != run_identity(config):
                    raise CheckpointError("run_metadata.yaml has a different run identity")
                update_run_metadata(
                    run,
                    {
                        "last_resume_pid": os.getpid(),
                        "resume_count": int(metadata.get("resume_count", 0)) + 1,
                    },
                )
            else:
                if run.exists() and any(run.iterdir()):
                    raise FileExistsError(
                        f"Fresh run would overwrite existing artifacts: {run}. "
                        "Choose a new run_name or use training.resume."
                    )
                run.mkdir(parents=True, exist_ok=True)
                save_resolved_config(config, run / "resolved_config.yaml")
                metadata = create_run_metadata(config)
                metadata["torch_version"] = str(torch.__version__)
                save_resolved_config(metadata, run / "run_metadata.yaml")
            for directory in ("checkpoints", "samples", "wandb"):
                (run / directory).mkdir(exist_ok=True)
        except Exception as error:
            root_error = error
            failure = {"type": type(error).__name__, "message": str(error)}
    failure = broadcast_object(failure)
    if failure is not None:
        if root_error is not None:
            raise root_error
        raise RuntimeError(
            f"Rank 0 failed while preparing the run: {failure['type']}: {failure['message']}"
        )
    return config, paths, run
