from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from vrae.training.common.contracts import (
    canonical_resume_config,
    run_identity,
    validate_checkpoint_identity,
)
from vrae.training.common.ema import ExponentialMovingAverage
from vrae.checkpoint import (
    CheckpointError,
    FORMAT_VERSION,
    atomic_torch_save,
    capture_rng_state,
    compare_metadata,
    load_checkpoint,
    restore_rng_state,
    update_latest_pointer,
)


def _state(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {name: _state(item) for name, item in value.items()}
    return value.state_dict()


def build_training_checkpoint(
    *,
    task: str,
    epoch: int,
    step: int,
    model: nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    data_state: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_config = canonical_resume_config(resolved_config)
    return {
        "format_version": FORMAT_VERSION,
        "task": task,
        "epoch": int(epoch),
        "step": int(step),
        "model": model.state_dict(),
        "ema": _state(ema),
        "optimizer": _state(optimizer),
        "scheduler": _state(scheduler),
        "scaler": _state(scaler),
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(),
        "data_state": dict(data_state),
        "resolved_config": checkpoint_config,
        "model_metadata": dict(model_metadata),
        "run_identity": run_identity(checkpoint_config),
    }


def save_training_checkpoint(
    payload: Mapping[str, Any],
    checkpoint_dir: str | Path,
    *,
    drop_page_cache: bool = False,
) -> Path:
    directory = Path(checkpoint_dir)
    path = directory / f"step-{int(payload['step']):08d}.pt"
    atomic_torch_save(payload, path, drop_page_cache=drop_page_cache)
    update_latest_pointer(path, directory / "latest.pt")
    return path


def load_model_init(
    path: str | Path,
    model: nn.Module,
    *,
    expected_metadata: Mapping[str, Any],
    metadata_fields: tuple[str, ...],
) -> dict[str, Any]:
    payload = load_checkpoint(path)
    compare_metadata(expected_metadata, payload["model_metadata"], metadata_fields)
    result = model.load_state_dict(payload["model"], strict=False)
    allowed_missing = {key for key in result.missing_keys if "view_embedding" in key}
    if result.unexpected_keys or set(result.missing_keys) != allowed_missing:
        raise CheckpointError(
            "Initial checkpoint model keys are incompatible: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    return payload


def _load_nested(target: Any, state: Any, name: str) -> None:
    if target is None and state is None:
        return
    if isinstance(target, Mapping) and isinstance(state, Mapping):
        if set(target) != set(state):
            raise ValueError(f"Resume {name} keys differ: {set(state)} vs {set(target)}")
        for key in target:
            _load_nested(target[key], state[key], f"{name}.{key}")
        return
    if target is None or state is None:
        raise ValueError(f"Resume {name} presence differs")
    target.load_state_dict(state)


def _validate_nested_presence(target: Any, state: Any, name: str) -> None:
    if target is None and state is None:
        return
    if isinstance(target, Mapping) and isinstance(state, Mapping):
        if set(target) != set(state):
            raise ValueError(f"Resume {name} keys differ: {set(state)} vs {set(target)}")
        for key in target:
            _validate_nested_presence(target[key], state[key], f"{name}.{key}")
        return
    if target is None or state is None:
        raise ValueError(f"Resume {name} presence differs")


def resume_training(
    path: str | Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    sampler: Any,
    expected_metadata: Mapping[str, Any],
    metadata_fields: tuple[str, ...],
    current_config: Mapping[str, Any] | None = None,
    run_directory: str | Path | None = None,
    allow_world_size_change: bool = False,
) -> dict[str, Any]:
    payload = load_checkpoint(path)
    if current_config is not None:
        validate_checkpoint_identity(
            payload,
            current_config,
            checkpoint_path=path,
            run_directory=run_directory,
        )
    compare_metadata(expected_metadata, payload["model_metadata"], metadata_fields)
    for name, target in (
        ("ema", ema),
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        _validate_nested_presence(target, payload[name], name)
    if int(payload["epoch"]) != int(payload["data_state"].get("sampler_epoch", -1)):
        raise ValueError("Checkpoint epoch does not match the sampler epoch")
    if int(payload["data_state"].get("gradient_accumulation_microstep", 0)) != 0:
        raise ValueError("Exact resume requires a gradient accumulation boundary checkpoint")
    validate_sampler_state = getattr(sampler, "validate_state_dict", None)
    if not callable(validate_sampler_state):
        raise TypeError("Exact-resume sampler must expose validate_state_dict")
    validate_sampler_state(
        payload["data_state"],
        allow_world_size_change=allow_world_size_change,
    )
    model.load_state_dict(payload["model"], strict=True)
    _load_nested(ema, payload["ema"], "ema")
    _load_nested(optimizer, payload["optimizer"], "optimizer")
    _load_nested(scheduler, payload["scheduler"], "scheduler")
    _load_nested(scaler, payload["scaler"], "scaler")
    sampler.load_state_dict(
        payload["data_state"],
        allow_world_size_change=allow_world_size_change,
    )
    restore_rng_state(
        payload["rng_state"],
        allow_world_size_mismatch=allow_world_size_change,
    )
    return payload
