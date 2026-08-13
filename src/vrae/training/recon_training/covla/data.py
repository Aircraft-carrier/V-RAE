"""CoVLA input pipeline for rectangular V-RAE reconstruction training."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vrae.training.common.contracts import resolve_batch_contract
from vrae.training.common.sampler import StatefulDistributedBatchSampler
from vrae.data.datasets import VideoDataset, VideoRecord
from vrae.data.transforms import (
    center_crop_video,
    random_crop_video,
    resize_video,
    uint8_to_float,
)
from vrae.paths import ProjectPaths

COVLA_IMAGE_SIZE = (432, 768)
COVLA_SPLITS = ("train", "val", "all")


def _parse_image_size(value: object) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("data.image_size must be [height, width]")
    result = (int(value[0]), int(value[1]))
    if min(result) <= 0:
        raise ValueError("data.image_size dimensions must be positive")
    return result


def _validate_unit_video(video: torch.Tensor) -> torch.Tensor:
    if video.device.type != "cpu" or not torch.is_floating_point(video):
        raise TypeError("CoVLA reconstruction video must be floating-point CPU RGB")
    minimum_tensor, maximum_tensor = torch.aminmax(video)
    minimum = float(minimum_tensor.item())
    maximum = float(maximum_tensor.item())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("CoVLA reconstruction video contains non-finite RGB values")
    tolerance = 1.0e-6
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"CoVLA reconstruction RGB must be in [0,1], got min={minimum} max={maximum}"
        )
    return video.clamp(0.0, 1.0) if minimum < 0.0 or maximum > 1.0 else video


def resize_cover_video(
    video: torch.Tensor,
    image_size: Sequence[int],
    *,
    mode: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize with preserved aspect ratio until both target dimensions are covered."""

    target_height, target_width = _parse_image_size(image_size)
    source_height, source_width = (int(value) for value in video.shape[-2:])
    scale = max(target_height / source_height, target_width / source_width)
    output_height = max(target_height, round(source_height * scale))
    output_width = max(target_width, round(source_width * scale))
    return resize_video(
        video,
        (output_height, output_width),
        mode=mode,
        antialias=antialias,
    )


def transfer_covla_video_batch(
    video: torch.Tensor | Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | list[torch.Tensor]:
    """Move either a uniform tensor batch or a heterogeneous raw-video list to ``device``."""

    if torch.is_tensor(video):
        return video.to(device, non_blocking=True)
    if not isinstance(video, Sequence) or isinstance(video, (str, bytes)) or not video:
        raise TypeError("CoVLA batch video must be a tensor or a non-empty tensor sequence")
    if not all(torch.is_tensor(item) for item in video):
        raise TypeError("Every heterogeneous CoVLA video must be a tensor")
    return [item.to(device, non_blocking=True) for item in video]


def _device_spatial_transform(
    video: torch.Tensor,
    *,
    image_size: Sequence[int],
    resize_mode: str,
    resize_antialias: bool,
) -> torch.Tensor:
    if video.dtype != torch.uint8 or video.ndim not in {4, 5}:
        raise TypeError("Raw CoVLA device transform expects uint8 [T,C,H,W] or [B,T,C,H,W]")
    target_height, target_width = _parse_image_size(image_size)
    source_height, source_width = (int(value) for value in video.shape[-2:])
    scale = max(target_height / source_height, target_width / source_width)
    output_height = max(target_height, round(source_height * scale))
    output_width = max(target_width, round(source_width * scale))
    leading_shape = video.shape[:-3]
    frames = video.reshape(-1, *video.shape[-3:]).float().div_(255.0)
    frames = F.interpolate(
        frames,
        size=(output_height, output_width),
        mode=resize_mode,
        align_corners=False,
        antialias=resize_antialias,
    ).clamp_(0.0, 1.0)
    top = (output_height - target_height) // 2
    left = (output_width - target_width) // 2
    frames = frames[..., top : top + target_height, left : left + target_width]
    return frames.reshape(*leading_shape, 3, target_height, target_width).contiguous()


def spatial_transform_covla_video_batch(
    video: torch.Tensor | Sequence[torch.Tensor],
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Finish the optional speed-first raw-video path on the training device."""

    if torch.is_tensor(video) and torch.is_floating_point(video):
        return video
    data = config["data"]
    if not bool(data.get("spatial_on_gpu", False)):
        raise TypeError("The CPU spatial path must produce a floating-point tensor batch")
    if str(data.get("crop_mode", "center")) != "center" or bool(data.get("random_flip", False)):
        raise ValueError("data.spatial_on_gpu currently requires center crop and random_flip=false")
    options = {
        "image_size": data.get("image_size", COVLA_IMAGE_SIZE),
        "resize_mode": str(data.get("resize_mode", "bicubic")),
        "resize_antialias": bool(data.get("resize_antialias", True)),
    }
    if torch.is_tensor(video):
        return _device_spatial_transform(video, **options)
    transformed = [_device_spatial_transform(item, **options) for item in video]
    return torch.stack(transformed)


def prepare_covla_reconstruction_batch(
    batch: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
) -> torch.Tensor:
    transferred = transfer_covla_video_batch(batch["video"], device)
    return spatial_transform_covla_video_batch(transferred, config)


def _split_rank(video_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{int(seed)}:{video_id}".encode()).digest()


def split_covla_records(
    records: Sequence[VideoRecord],
    split: str,
    *,
    validation_fraction: float,
    seed: int,
    validation_count: int | None = None,
    strategy: str = "hash",
) -> list[VideoRecord]:
    """Create an exact hash-ranked or manifest-tail validation split."""

    selected_split = str(split).lower()
    if selected_split not in COVLA_SPLITS:
        raise ValueError(f"CoVLA split must be one of {COVLA_SPLITS}, got {split!r}")
    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("data.validation_fraction must be in [0,1)")
    selected_strategy = str(strategy).strip().lower()
    if selected_strategy not in {"hash", "tail"}:
        raise ValueError("data.validation_strategy must be hash or tail")
    if selected_split == "all":
        return list(records)

    computed_count = round(len(records) * fraction)
    has_explicit_count = validation_count is not None
    if validation_count is None:
        validation_count = computed_count
    elif isinstance(validation_count, bool) or int(validation_count) != validation_count:
        raise ValueError("data.validation_count must be an integer")
    validation_count = int(validation_count)
    if not 0 <= validation_count < len(records):
        raise ValueError("data.validation_count must be in [0, len(records))")
    if has_explicit_count and computed_count != validation_count:
        raise ValueError(
            "data.validation_fraction and data.validation_count disagree: "
            f"fraction implies {computed_count}, explicit count is {validation_count}"
        )
    if validation_count == 0:
        return [] if selected_split == "val" else list(records)

    if selected_strategy == "tail":
        validation_ids = {record.sample_id for record in records[-validation_count:]}
    else:
        ranked = sorted(
            records,
            key=lambda record: (_split_rank(record.sample_id, seed), record.sample_id),
        )
        validation_ids = {record.sample_id for record in ranked[:validation_count]}
    if selected_split == "val":
        return [record for record in records if record.sample_id in validation_ids]
    return [record for record in records if record.sample_id not in validation_ids]


def load_covla_records(
    root: str | Path,
    manifest: str | Path = "metadata.jsonl",
    *,
    expected_total: int | None = None,
    verify_files: bool = False,
) -> list[VideoRecord]:
    """Load the native CoVLA ``file_name``/``video_id`` JSONL schema."""

    dataset_root = Path(root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    records: list[VideoRecord] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, Mapping):
                raise ValueError(f"CoVLA manifest line {line_number} must be a JSON object")
            file_name = value.get("file_name")
            video_id = value.get("video_id")
            if not file_name or not video_id:
                raise ValueError(
                    f"CoVLA manifest line {line_number} requires file_name and video_id"
                )
            sample_id = str(video_id)
            raw_path = Path(str(file_name))
            path = (raw_path if raw_path.is_absolute() else dataset_root / raw_path).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(
                    f"CoVLA video escapes dataset root on line {line_number}: {file_name}"
                ) from error
            if sample_id in seen_ids:
                raise ValueError(f"duplicate CoVLA video_id on line {line_number}: {sample_id}")
            if path in seen_paths:
                raise ValueError(f"duplicate CoVLA file_name on line {line_number}: {file_name}")
            if verify_files and not path.is_file():
                raise FileNotFoundError(path)
            seen_ids.add(sample_id)
            seen_paths.add(path)
            records.append(
                VideoRecord(
                    path=path,
                    label=-1,
                    sample_id=sample_id,
                    extra={
                        **{
                            str(key): item
                            for key, item in value.items()
                            if key not in {"file_name", "video_id"}
                        },
                        "file_name": str(file_name),
                        "source": "covla",
                    },
                )
            )

    if expected_total is not None and len(records) != int(expected_total):
        raise ValueError(
            f"CoVLA manifest contains {len(records)} records, expected {int(expected_total)}"
        )
    if not records:
        raise ValueError("CoVLA manifest contains no records")
    return records


class CoVLAReconstructionDataset(VideoDataset):
    def __init__(
        self,
        records: Sequence[VideoRecord],
        *,
        image_size: Sequence[int] = COVLA_IMAGE_SIZE,
        crop_mode: str = "center",
        resize_mode: str = "bicubic",
        resize_antialias: bool = True,
        random_flip: bool = False,
        max_decode_attempts: int = 128,
        profile_timing: bool = False,
        spatial_on_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(records, transform=None, **kwargs)
        self.image_size = _parse_image_size(image_size)
        if any(size % 16 for size in self.image_size):
            raise ValueError("CoVLA image_size dimensions must be divisible by patch size 16")
        self.crop_mode = str(crop_mode)
        if self.crop_mode not in {"center", "random"}:
            raise ValueError("data.crop_mode must be center or random")
        self.resize_mode = str(resize_mode)
        if self.resize_mode not in {"bilinear", "bicubic"}:
            raise ValueError("data.resize_mode must be bilinear or bicubic")
        self.resize_antialias = bool(resize_antialias)
        self.random_flip = bool(random_flip)
        self.max_decode_attempts = int(max_decode_attempts)
        if self.max_decode_attempts <= 0:
            raise ValueError("max_decode_attempts must be positive")
        self._bad_video_paths: set[str] = set()
        self.profile_timing = bool(profile_timing)
        self.spatial_on_gpu = bool(spatial_on_gpu)
        if self.spatial_on_gpu and (self.crop_mode != "center" or self.random_flip):
            raise ValueError("spatial_on_gpu currently requires center crop and random_flip=false")

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != self.epoch:
            self._bad_video_paths.clear()
        super().set_epoch(epoch)

    def _spatial_transform(
        self,
        video: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        video = uint8_to_float(video)
        video = resize_cover_video(
            video,
            self.image_size,
            mode=self.resize_mode,
            antialias=self.resize_antialias,
        )
        # Bicubic interpolation may overshoot the source range around sharp
        # road/sky boundaries. Clamp that expected interpolation artifact before
        # the strict post-transform validation below.
        video = video.clamp(0.0, 1.0)
        if self.crop_mode == "random":
            video = random_crop_video(video, self.image_size, generator=generator)
        else:
            video = center_crop_video(video, self.image_size)
        if self.random_flip and bool(torch.rand((), generator=generator) < 0.5):
            video = video.flip(-1)
        return _validate_unit_video(video)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_started = time.perf_counter() if self.profile_timing else 0.0
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
                generator = self._generator(requested_index, attempt=attempt)
                decode_started = time.perf_counter() if self.profile_timing else 0.0
                item = super().__getitem__(candidate_index, generator=generator)
                decode_finished = time.perf_counter() if self.profile_timing else 0.0
                if not self.spatial_on_gpu:
                    item["video"] = self._spatial_transform(item["video"], generator=generator)
                spatial_finished = time.perf_counter() if self.profile_timing else 0.0
            except Exception as error:
                self._bad_video_paths.add(candidate_path)
                last_error = error
                continue
            item["decode_attempts"] = attempt + 1
            if self.profile_timing:
                item["data_timing"] = torch.tensor(
                    (
                        decode_finished - decode_started,
                        spatial_finished - decode_finished,
                        spatial_finished - sample_started,
                    ),
                    dtype=torch.float64,
                )
            return item

        message = (
            f"No CoVLA video could be decoded for logical sample {requested_index} "
            f"after {attempts} attempts"
        )
        if last_error is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from last_error


def _resolve_dataset_root(data: Mapping[str, Any], paths: ProjectPaths) -> Path:
    root_value = data.get("root")
    if root_value:
        root = Path(str(root_value)).expanduser()
        if not root.is_absolute():
            root = paths.project_root / root
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    return paths.dataset(str(data.get("dataset", "covla")))


def build_covla_reconstruction_dataset(
    config: Mapping[str, Any],
    paths: ProjectPaths,
) -> CoVLAReconstructionDataset:
    data = config["data"]
    root = _resolve_dataset_root(data, paths)
    records = load_covla_records(
        root,
        data.get("manifest", "metadata.jsonl"),
        expected_total=(
            None if data.get("expected_total") is None else int(data["expected_total"])
        ),
        verify_files=bool(data.get("verify_files", False)),
    )
    records = split_covla_records(
        records,
        str(data.get("split", "train")),
        validation_fraction=float(data.get("validation_fraction", 0.05)),
        seed=int(data.get("split_seed", 3407)),
        validation_count=(
            None if data.get("validation_count") is None else int(data["validation_count"])
        ),
        strategy=str(data.get("validation_strategy", "hash")),
    )
    max_samples = data.get("max_samples")
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("data.max_samples must be positive")
        records = records[: int(max_samples)]
    if not records:
        raise ValueError("Selected CoVLA split contains no records")
    return CoVLAReconstructionDataset(
        records,
        clip_length=int(data.get("num_frames", 16)),
        frame_interval=int(data.get("frame_interval", 3)),
        sampling=str(data.get("sampling", "random")),
        backend=str(data.get("video_backend", "auto")),
        base_seed=int(data.get("seed", 3407)),
        num_threads=int(data.get("decode_threads", 1)),
        seek_mode=str(data.get("torchcodec_seek_mode", "approximate")),
        image_size=data.get("image_size", COVLA_IMAGE_SIZE),
        crop_mode=str(data.get("crop_mode", "center")),
        resize_mode=str(data.get("resize_mode", "bicubic")),
        resize_antialias=bool(data.get("resize_antialias", True)),
        random_flip=bool(data.get("random_flip", False)),
        max_decode_attempts=int(data.get("max_decode_attempts", 128)),
        profile_timing=bool(data.get("profile_timing", False)),
        spatial_on_gpu=bool(data.get("spatial_on_gpu", False)),
    )


def collate_covla_reconstruction(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty CoVLA reconstruction batch")
    videos = [item["video"] for item in items]
    uniform_shape = len({tuple(video.shape) for video in videos}) == 1
    batch = {
        "video": torch.stack(videos) if uniform_shape else videos,
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "sample_id": [str(item["sample_id"]) for item in items],
        "frame_indices": torch.stack([item["frame_indices"] for item in items]),
        "decode_attempts": torch.tensor(
            [int(item.get("decode_attempts", 1)) for item in items], dtype=torch.long
        ),
    }
    if all("data_timing" in item for item in items):
        batch["data_timing"] = torch.stack([item["data_timing"] for item in items])
    return batch


def build_covla_reconstruction_loader(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, StatefulDistributedBatchSampler]:
    dataset = build_covla_reconstruction_dataset(config, paths)
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
        collate_fn=collate_covla_reconstruction,
        generator=sampler.loader_generator,
        **loader_options,
    )
    return loader, sampler


def build_covla_visualization_batch(
    config: Mapping[str, Any],
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Decode the complete held-out split at a fixed center clip for W&B previews."""

    visualization_data = dict(config["data"])
    visualization_data.update(
        {
            "split": "val",
            "sampling": "center",
            "max_decode_attempts": 1,
        }
    )
    visualization_config = {**config, "data": visualization_data}
    dataset = build_covla_reconstruction_dataset(visualization_config, paths)
    expected = int(visualization_data["validation_count"])
    if len(dataset) != expected:
        raise RuntimeError(
            f"CoVLA visualization split has {len(dataset)} records, expected {expected}"
        )
    return collate_covla_reconstruction([dataset[index] for index in range(len(dataset))])


__all__ = [
    "COVLA_IMAGE_SIZE",
    "COVLA_SPLITS",
    "CoVLAReconstructionDataset",
    "build_covla_reconstruction_dataset",
    "build_covla_reconstruction_loader",
    "build_covla_visualization_batch",
    "collate_covla_reconstruction",
    "load_covla_records",
    "prepare_covla_reconstruction_batch",
    "resize_cover_video",
    "spatial_transform_covla_video_batch",
    "split_covla_records",
    "transfer_covla_video_batch",
]
