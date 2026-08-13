from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

# Use one OpenMP thread per process unless the launcher explicitly overrides it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

DDP_GRADIENT_COMPRESSION_MODES = ("none", "bf16")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    timeout_seconds: float

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def resolve_ddp_gradient_compression(value: object = "none") -> str:
    selected = str(value or "none").strip().lower()
    if selected not in DDP_GRADIENT_COMPRESSION_MODES:
        raise ValueError(
            "training.ddp_gradient_compression must be one of "
            f"{DDP_GRADIENT_COMPRESSION_MODES}, got {value!r}"
        )
    return selected


def configure_ddp_gradient_compression(
    module: torch.nn.Module,
    value: object = "none",
) -> str:
    """Optionally halve DDP gradient traffic with a BF16 all-reduce hook."""

    selected = resolve_ddp_gradient_compression(value)
    if selected == "none" or not isinstance(module, DistributedDataParallel):
        return selected
    if not is_distributed():
        raise RuntimeError("DDP gradient compression requires an initialized process group")
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    module.register_comm_hook(dist.group.WORLD, default_hooks.bf16_compress_hook)
    return selected


def _resolve_timeout_seconds(timeout_minutes: float | None) -> float:
    if timeout_minutes is not None:
        timeout_seconds = float(timeout_minutes) * 60.0
    else:
        timeout_seconds = float(os.environ.get("VRAE_DISTRIBUTED_TIMEOUT_SECONDS", "600"))
    if timeout_seconds <= 0:
        raise ValueError("distributed timeout must be positive")
    return timeout_seconds


def initialize_distributed(
    *, backend: str | None = None, timeout_minutes: float | None = None
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    timeout_seconds = _resolve_timeout_seconds(timeout_minutes)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not is_distributed():
        selected_backend = backend or ("nccl" if device.type == "cuda" else "gloo")
        init_options: dict[str, Any] = {
            "backend": selected_backend,
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(seconds=timeout_seconds),
        }
        if device.type == "cuda":
            init_options["device_id"] = device
        dist.init_process_group(
            **init_options,
        )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        timeout_seconds=timeout_seconds,
    )


def barrier() -> None:
    if is_distributed():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def shutdown_distributed() -> None:
    """Release the default process group before a torchrun worker exits."""

    if is_distributed():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result /= dist.get_world_size()
    return result


def all_gather_variable(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather an uneven first dimension without padding rows into the result."""
    if not is_distributed():
        return [tensor]
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    maximum = max(int(size.item()) for size in sizes)
    padded_shape = (maximum, *tensor.shape[1:])
    padded = tensor.new_zeros(padded_shape)
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return [item[: int(size.item())] for item, size in zip(gathered, sizes, strict=True)]


def broadcast_object(value: Any, *, source: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=source)
    return objects[0]


def gather_rng_states() -> dict[str, Any]:
    """Collect the independent RNG state of every rank for exact resume."""

    from vrae.checkpoint import capture_rng_state

    local_state = capture_rng_state()
    if not is_distributed():
        return {"world_size": 1, "per_rank": [local_state]}
    world_size = dist.get_world_size()
    states: list[Any] = [None] * world_size
    dist.all_gather_object(states, local_state)
    return {"world_size": world_size, "per_rank": states}
