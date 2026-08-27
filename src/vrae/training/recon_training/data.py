from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vrae.training.common.contracts import resolve_batch_contract
from vrae.training.common.sampler import StatefulDistributedBatchSampler
from vrae.data.datasets import Kinetics600Dataset, UCF101Dataset, VideoDataset, VideoRecord
from vrae.data.lerobot import LeRobotVideoDataset
from vrae.data.transforms import random_crop_video, resize_short_side, uint8_to_float
from vrae.paths import ProjectPaths


class CudaVideoPrefetchIterator:
    """Overlap the next pinned-memory video transfer with the current training step."""

    def __init__(self, iterator: Iterator[dict[str, Any]], device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA video prefetch requires a CUDA device")
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._next_batch: dict[str, Any] | None = None
        self._preload()

    def __iter__(self) -> CudaVideoPrefetchIterator:
        return self

    def _preload(self) -> None:
        try:
            batch = dict(next(self.iterator))
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            batch["video"] = batch["video"].to(self.device, non_blocking=True)
            if "stream_ids" in batch:
                batch["stream_ids"] = batch["stream_ids"].to(self.device, non_blocking=True)
        self._next_batch = batch

    def __next__(self) -> dict[str, Any]:
        batch = self._next_batch
        if batch is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        batch["video"].record_stream(current_stream)
        if "stream_ids" in batch:
            batch["stream_ids"].record_stream(current_stream)
        self._preload()
        return batch


def _validate_cpu_reconstruction_video(video: torch.Tensor) -> torch.Tensor:
    """Validate the post-transform unit-range video before it is copied to CUDA."""

    if video.device.type != "cpu":
        raise ValueError("Reconstruction data validation must run on CPU")
    if not torch.is_floating_point(video):
        raise TypeError("Reconstruction video must be floating-point RGB in [0,1]")

    minimum_tensor, maximum_tensor = torch.aminmax(video)
    minimum = float(minimum_tensor.item())
    maximum = float(maximum_tensor.item())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("Reconstruction video contains non-finite RGB values")

    tolerance = 1.0e-6
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"Reconstruction video RGB must be in [0,1], got min={minimum} max={maximum}"
        )
    if minimum < 0.0 or maximum > 1.0:
        return video.clamp(0.0, 1.0)
    return video


class ReconstructionDataset(VideoDataset):
    def __init__(
        self,
        records: Sequence[VideoRecord],
        *,
        image_size: int,
        random_flip: bool,
        max_decode_attempts: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(records, transform=None, **kwargs)
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        self.max_decode_attempts = int(max_decode_attempts)
        if self.max_decode_attempts <= 0:
            raise ValueError("max_decode_attempts must be positive")
        # Each DataLoader worker owns its own dataset copy.  Remembering rejected
        # paths avoids repeatedly opening a known-short or corrupt video while
        # retaining deterministic index+attempt fallback semantics.
        self._bad_video_paths: set[str] = set()

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != self.epoch:
            self._bad_video_paths.clear()
        super().set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested_index = int(index)
        if requested_index < 0 or requested_index >= len(self.records):
            raise IndexError(requested_index)

        attempts = min(len(self.records), self.max_decode_attempts)
        last_error: Exception | None = None
        for attempt in range(attempts):
            candidate_index = (requested_index + attempt) % len(self.records)
            candidate_path = str(self.records[candidate_index].path)
            if candidate_path in self._bad_video_paths:
                continue
            try:
                # Seed by logical sample and fallback attempt, not by the
                # candidate record.  The same stream drives temporal sampling
                # and spatial augmentation, matching the public reference semantics and
                # preventing a fallback from duplicating the next logical item.
                generator = self._generator(requested_index, attempt=attempt)
                item = super().__getitem__(candidate_index, generator=generator)
                video = uint8_to_float(item["video"])
                if self.random_flip and bool(torch.rand((), generator=generator) < 0.5):
                    video = video.flip(-1)
                video = resize_short_side(
                    video,
                    self.image_size,
                    mode="bilinear",
                    antialias=False,
                )
                item["video"] = _validate_cpu_reconstruction_video(
                    random_crop_video(
                        video,
                        self.image_size,
                        generator=generator,
                    )
                )
            except Exception as error:
                self._bad_video_paths.add(candidate_path)
                last_error = error
                continue
            item["decode_attempts"] = attempt + 1
            return item

        message = (
            f"No video could be decoded for logical sample {requested_index} after "
            f"{attempts} attempts"
        )
        if last_error is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from last_error


def _resolve_manifest_path(
    value: Mapping[str, Any],
    *,
    dataset_root: Path,
    paths: ProjectPaths,
) -> Path | None:
    manifest_value = value.get("manifest")
    scope = str(value.get("manifest_scope", "dataset"))
    if scope not in {"dataset", "project"}:
        raise ValueError("manifest_scope must be dataset or project")
    if not manifest_value:
        if "manifest_scope" in value:
            raise ValueError("manifest_scope requires a non-empty manifest")
        return None

    manifest = Path(str(manifest_value))
    if manifest.is_absolute():
        return manifest
    base = dataset_root if scope == "dataset" else paths.project_root
    return base / manifest


def _source_records(
    name: str,
    value: Mapping[str, Any],
    paths: ProjectPaths,
) -> list[VideoRecord]:
    root = paths.dataset(name)
    manifest = _resolve_manifest_path(value, dataset_root=root, paths=paths)
    common = {"root": root, "manifest": manifest, "split": value.get("split", "train")}
    if name == "ucf101":
        dataset = UCF101Dataset(
            **common,
            clip_length=4,
            require_temporal_multiple_of_four=True,
        )
    elif name == "k600":
        dataset = Kinetics600Dataset(
            **common,
            clip_length=4,
            require_temporal_multiple_of_four=True,
        )
    elif name == "lerobot":
        raise ValueError("Use build_reconstruction_dataset for LeRobot sources")
    else:
        raise ValueError(f"Reconstruction source must be ucf101 or k600, got {name!r}")
    return [
        VideoRecord(
            path=record.path,
            label=record.label,
            sample_id=f"{name}:{record.sample_id}",
            split=record.split,
            extra={**record.extra, "source": name},
        )
        for record in dataset.records
    ]


def build_reconstruction_dataset(
    config: Mapping[str, Any], paths: ProjectPaths
) -> ReconstructionDataset:
    data = config["data"]
    if str(data.get("dataset", "")) == "lerobot":
        root_value = data.get("root")
        root = Path(str(root_value)).expanduser() if root_value else paths.dataset("lerobot")
        camera_keys = data.get("camera_keys")
        multiview = config.get("model", {}).get("multiview", {})
        return LeRobotVideoDataset(
            root,
            repo_id=str(data.get("repo_id", "libero")),
            clip_length=int(data.get("num_frames", 16)),
            frame_interval=int(data.get("frame_interval", 1)),
            sampling=str(data.get("sampling", "random")),
            base_seed=int(data.get("seed", 3407)),
            camera_keys=camera_keys,
            image_size=int(data["image_size"]) if data.get("image_size") is not None else None,
            random_flip=bool(data.get("random_flip", False)),
            multiview_enabled=bool(multiview.get("enabled", bool(camera_keys and len(camera_keys) > 1))),
        )
    sources = data.get("sources", ("ucf101", "k600"))
    records: list[VideoRecord] = []
    for source in sources:
        if isinstance(source, str):
            name, value = source, {}
        else:
            name, value = str(source["name"]), source
        records.extend(_source_records(name, value, paths))
    if not records:
        raise ValueError("Reconstruction dataset contains no records")
    return ReconstructionDataset(
        records,
        clip_length=int(data.get("num_frames", 16)),
        frame_interval=int(data.get("frame_interval", 3)),
        sampling=str(data.get("sampling", "random")),
        backend=str(data.get("video_backend", "auto")),
        base_seed=int(data.get("seed", 3407)),
        num_threads=int(data.get("decode_threads", 1)),
        image_size=int(data.get("image_size", 256)),
        random_flip=bool(data.get("random_flip", True)),
        max_decode_attempts=int(data.get("max_decode_attempts", 128)),
    )


def collate_reconstruction(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "video": torch.stack([item["video"] for item in items]),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "sample_id": [item["sample_id"] for item in items],
        "frame_indices": torch.stack([item["frame_indices"] for item in items]),
        "decode_attempts": torch.tensor(
            [int(item.get("decode_attempts", 1)) for item in items], dtype=torch.long
        ),
    }
    if "stream_ids" in items[0]:
        stream_ids = torch.stack([item["stream_ids"] for item in items])
        if any(item["video"].ndim == 5 for item in items):
            batch["stream_ids"] = stream_ids
    for key in ("state", "action"):
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    if "task" in items[0]:
        batch["task"] = [item["task"] for item in items]
    return batch


def build_reconstruction_loader(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, StatefulDistributedBatchSampler]:
    dataset = build_reconstruction_dataset(config, paths)
    batch = resolve_batch_contract(config["training"], world_size=world_size)
    sampler = StatefulDistributedBatchSampler(
        len(dataset),
        batch.local_micro_batch_size,
        rank=rank,
        world_size=world_size,
        seed=int(config["data"].get("seed", 3407)),
        shuffle=True,
        drop_last=True,
        gradient_accumulation_steps=batch.gradient_accumulation_steps,
    )
    num_workers = int(config["training"].get("num_workers", 4))
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))
        loader_options["persistent_workers"] = bool(
            config["training"].get("persistent_workers", False)
        )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_reconstruction,
        generator=sampler.loader_generator,
        **loader_options,
    )
    return loader, sampler
