from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping

import torch
from torch.utils.data import Sampler


class StatefulDistributedBatchSampler(Sampler[list[int]]):
    """Deterministic distributed batches whose global cursor is directly restorable."""

    policy = "global_permutation_strided_v1"

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 3407,
        shuffle: bool = True,
        drop_last: bool = True,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid rank/world_size")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.epoch = 0
        self.cursor = 0
        self.next_batch_index = 0
        self._pending_stops: deque[int] = deque()
        self.loader_generator = torch.Generator().manual_seed(self.seed + 0x5EED)

    @property
    def global_batch_size(self) -> int:
        return self.global_micro_batch_size * self.gradient_accumulation_steps

    @property
    def global_micro_batch_size(self) -> int:
        return self.batch_size * self.world_size

    def _order(self) -> list[int]:
        if not self.shuffle:
            return list(range(self.dataset_size))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        if self._pending_stops:
            raise RuntimeError("Cannot create a new sampler iterator with uncommitted batches")
        order = self._order()
        global_batch = self.global_batch_size
        global_micro_batch = self.global_micro_batch_size
        limit = (
            (self.dataset_size // global_batch) * global_batch
            if self.drop_last
            else self.dataset_size
        )
        dispatch_cursor = self.cursor
        while dispatch_cursor < limit:
            start = dispatch_cursor
            stop = min(start + global_micro_batch, self.dataset_size)
            global_indices = order[start:stop]
            local = global_indices[self.rank :: self.world_size]
            if self.drop_last and len(local) != self.batch_size:
                break
            dispatch_cursor = stop
            self._pending_stops.append(stop)
            yield local

    def commit_batch(self) -> None:
        """Mark one yielded batch consumed after its training update has completed."""

        if not self._pending_stops:
            raise RuntimeError("No dispatched sampler batch is waiting to be committed")
        self.cursor = self._pending_stops.popleft()
        self.next_batch_index += 1
        global_batch = self.global_batch_size
        limit = (
            (self.dataset_size // global_batch) * global_batch
            if self.drop_last
            else self.dataset_size
        )
        if self.cursor >= limit:
            if self._pending_stops:
                raise RuntimeError("Final sampler batch committed before earlier batches")
            self.epoch += 1
            self.cursor = 0
            self.next_batch_index = 0

    def __len__(self) -> int:
        if self.drop_last:
            total = (self.dataset_size // self.global_batch_size) * self.gradient_accumulation_steps
        else:
            total = (
                math.ceil(self.dataset_size / self.global_batch_size)
                * self.gradient_accumulation_steps
            )
        consumed = self.cursor // self.global_micro_batch_size
        return max(0, total - consumed)

    def state_dict(
        self, *, gradient_accumulation_microstep: int | None = None
    ) -> dict[str, object]:
        if gradient_accumulation_microstep is None:
            gradient_accumulation_microstep = (
                self.next_batch_index % self.gradient_accumulation_steps
            )
        return {
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
            "sampler_epoch": self.epoch,
            "sampler_cursor": self.cursor,
            "sampler_seed": self.seed,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "global_micro_batch_size": self.global_micro_batch_size,
            "local_micro_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "dataset_population": self.dataset_size,
            "sampler_policy": self.policy,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "gradient_accumulation_microstep": int(gradient_accumulation_microstep),
            "loader_generator_state": self.loader_generator.get_state(),
        }

    def validate_state_dict(
        self,
        state: Mapping[str, object],
        *,
        allow_world_size_change: bool = False,
    ) -> None:
        expected = {
            "sampler_seed": self.seed,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "global_micro_batch_size": self.global_micro_batch_size,
            "local_micro_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "dataset_population": self.dataset_size,
            "sampler_policy": self.policy,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }
        topology_fields = {"world_size", "local_micro_batch_size"}
        mismatches = [
            f"{key}: checkpoint={state.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if state.get(key) != value and not (allow_world_size_change and key in topology_fields)
        ]
        if mismatches:
            raise ValueError("Exact-resume sampler mismatch:\n" + "\n".join(mismatches))
        topology_changed = any(state.get(key) != expected[key] for key in topology_fields)
        cursor = int(state["sampler_cursor"])
        if cursor < 0 or cursor > self.dataset_size:
            raise ValueError(f"Invalid sampler cursor: {cursor}")
        if topology_changed and cursor != 0:
            raise ValueError(
                "World-size resume is only supported at an epoch boundary "
                "(sampler_cursor must be 0)"
            )
        if cursor % self.global_micro_batch_size:
            raise ValueError("Sampler cursor is not aligned to a global microbatch")
        if int(state.get("epoch", -1)) != int(state["sampler_epoch"]):
            raise ValueError("Checkpoint epoch aliases disagree")
        expected_batch_index = cursor // self.global_micro_batch_size
        if int(state["next_batch_index"]) != expected_batch_index:
            raise ValueError("Sampler next_batch_index does not match its cursor")
        generator_state = state.get("loader_generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("Exact-resume state is missing loader_generator_state")
        microstep = int(state.get("gradient_accumulation_microstep", 0))
        expected_microstep = int(state["next_batch_index"]) % self.gradient_accumulation_steps
        if microstep != expected_microstep:
            raise ValueError(
                "Exact-resume gradient accumulation microstep does not match the sampler cursor"
            )

    def load_state_dict(
        self,
        state: Mapping[str, object],
        *,
        allow_world_size_change: bool = False,
    ) -> None:
        self.validate_state_dict(state, allow_world_size_change=allow_world_size_change)
        self.epoch = int(state["sampler_epoch"])
        self.cursor = int(state["sampler_cursor"])
        self.next_batch_index = int(state["next_batch_index"])
        self._pending_stops.clear()
        self.loader_generator.set_state(state["loader_generator_state"])

    def set_epoch(self, epoch: int) -> None:
        if self.cursor or self._pending_stops:
            raise RuntimeError("Cannot change epoch while a sampler cursor is active")
        self.epoch = int(epoch)
        self.next_batch_index = 0
