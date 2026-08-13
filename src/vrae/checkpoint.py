from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

FORMAT_VERSION = 1
REQUIRED_FIELDS = {
    "format_version",
    "task",
    "epoch",
    "step",
    "model",
    "ema",
    "optimizer",
    "scheduler",
    "scaler",
    "rng_state",
    "data_state",
    "resolved_config",
    "model_metadata",
}
FORBIDDEN_AFFINE_TERMS = ("latent_affine_scale", "latent_affine_bias")


class CheckpointError(RuntimeError):
    pass


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], *, allow_world_size_mismatch: bool = False) -> None:
    if "per_rank" in state:
        states = state["per_rank"]
        if not isinstance(states, Sequence):
            raise CheckpointError("Distributed RNG state must contain a per-rank sequence")
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        saved_world_size = int(state.get("world_size", len(states)))
        if saved_world_size != world_size or len(states) != world_size:
            if allow_world_size_mismatch:
                # The caller seeded every current rank before loading. There is
                # no one-to-one RNG mapping across different process counts, so
                # retain those fresh rank-local states for the topology change.
                return
            raise CheckpointError("Checkpoint RNG world size does not match the current job")
        selected = states[rank]
        if not isinstance(selected, Mapping):
            raise CheckpointError(f"Checkpoint RNG state for rank {rank} is invalid")
        state = selected
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise CheckpointError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_state_dict_keys(keys: Sequence[str]) -> None:
    rejected = [key for key in keys if any(term in key for term in FORBIDDEN_AFFINE_TERMS)]
    if rejected:
        raise CheckpointError(
            f"Legacy pooling-after-affine checkpoint keys are forbidden: {rejected}"
        )


def compare_metadata(
    expected: Mapping[str, Any], actual: Mapping[str, Any], fields: Sequence[str]
) -> None:
    mismatches = []
    for field in fields:
        if expected.get(field) != actual.get(field):
            mismatches.append(
                f"{field}: expected={expected.get(field)!r}, actual={actual.get(field)!r}"
            )
    if mismatches:
        raise CheckpointError("Incompatible model metadata:\n" + "\n".join(mismatches))


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise CheckpointError(f"Checkpoint is missing fields: {sorted(missing)}")
    if int(payload["format_version"]) != FORMAT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint format {payload['format_version']}; expected {FORMAT_VERSION}"
        )
    model_state = payload["model"]
    if not isinstance(model_state, Mapping):
        raise CheckpointError("Checkpoint model field must be a state dict")
    validate_state_dict_keys(list(model_state))


def _drop_file_page_cache(path: Path) -> None:
    fadvise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if not callable(fadvise) or advice is None:
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
        fadvise(descriptor, 0, 0, advice)
    finally:
        os.close(descriptor)


def atomic_torch_save(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    drop_page_cache: bool = False,
) -> Path:
    validate_checkpoint(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(output)
        if drop_page_cache:
            _drop_file_page_cache(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def update_latest_pointer(checkpoint: str | Path, latest: str | Path) -> Path:
    source = Path(checkpoint).resolve()
    target = Path(latest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.parent != target.parent.resolve():
        raise CheckpointError("latest pointer and checkpoint must share a directory")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.symlink_to(source.name)
        temporary.replace(target)
    finally:
        if temporary.is_symlink() or temporary.exists():
            temporary.unlink()
    return target


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
            mmap=bool(mmap),
        )
    except TypeError:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint root must be a mapping")
    result = dict(payload)
    validate_checkpoint(result)
    return result


def strict_load_model(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    metadata_fields: Sequence[str] = (),
) -> None:
    validate_checkpoint(payload)
    if expected_metadata is not None:
        compare_metadata(expected_metadata, payload["model_metadata"], metadata_fields)
    model.load_state_dict(payload["model"], strict=True)
