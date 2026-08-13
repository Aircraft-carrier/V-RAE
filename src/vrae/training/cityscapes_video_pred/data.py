from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from vrae.training.common.contracts import resolve_batch_contract
from vrae.training.common.sampler import StatefulDistributedBatchSampler
from vrae.data import VideoReader, resize_video, uint8_to_float
from vrae.paths import ProjectPaths

FRAMES_PER_SEQUENCE = 30
CONTEXT_RELATIVE_INDICES = tuple(range(4, 16))
FUTURE_RELATIVE_INDICES = tuple(range(16, 28))
CONTEXT_FRAMES = len(CONTEXT_RELATIVE_INDICES)
FUTURE_FRAMES = len(FUTURE_RELATIVE_INDICES)
CITYSCAPES_IMAGE_SIZE = (432, 768)
EXPECTED_SPLIT_COUNTS = {"train": 2_975, "val": 500}
CITYSCAPES_SPLITS = ("train", "val", "test")
FRAME_PATTERN = re.compile(r"^(?P<city>.+)_(?P<sequence>\d+)_(?P<frame>\d+)_leftImg8bit\.png$")


def validate_split(split: str) -> str:
    value = str(split).lower()
    if value not in CITYSCAPES_SPLITS:
        raise ValueError(f"split must be one of {CITYSCAPES_SPLITS}, got {split!r}")
    return value


def validate_frame_protocol(
    context_frames: Sequence[int] = (4, 15),
    future_frames: Sequence[int] = (16, 27),
) -> None:
    if tuple(int(value) for value in context_frames) != (4, 15):
        raise ValueError("Cityscapes context must use relative frames 4..15 inclusive")
    if tuple(int(value) for value in future_frames) != (16, 27):
        raise ValueError("Cityscapes future must use relative frames 16..27 inclusive")


def _sample_id(
    split: str,
    city: str,
    sequence: str,
    first_frame: int,
    last_frame: int,
) -> str:
    return f"{split}-{city}-{sequence}-f{int(first_frame):06d}-f{int(last_frame):06d}"


def discover_cityscapes_sequences(
    data_root: str | Path,
    split: str,
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Discover strictly contiguous 30-frame Cityscapes PNG sequences."""

    split = validate_split(split)
    root = Path(data_root).expanduser().resolve()
    split_root = root / "leftImg8bit_sequence" / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    records: list[dict[str, Any]] = []
    for city_directory in sorted(path for path in split_root.iterdir() if path.is_dir()):
        paths = sorted(city_directory.glob("*_leftImg8bit.png"))
        if len(paths) % FRAMES_PER_SEQUENCE:
            raise ValueError(
                f"{city_directory} contains {len(paths)} frames; expected a multiple of 30"
            )
        for offset in range(0, len(paths), FRAMES_PER_SEQUENCE):
            group = paths[offset : offset + FRAMES_PER_SEQUENCE]
            matches = [FRAME_PATTERN.fullmatch(path.name) for path in group]
            if any(match is None for match in matches):
                raise ValueError(f"invalid Cityscapes frame name near {group[0]}")
            parsed = [match for match in matches if match is not None]
            identities = {(match.group("city"), match.group("sequence")) for match in parsed}
            frame_numbers = [int(match.group("frame")) for match in parsed]
            if len(identities) != 1 or any(
                right != left + 1 for left, right in pairwise(frame_numbers)
            ):
                raise ValueError(f"mixed or non-contiguous sequence near {group[0]}")
            city, sequence = next(iter(identities))
            if city != city_directory.name:
                raise ValueError(
                    f"frame city {city!r} differs from directory {city_directory.name!r}"
                )
            record = {
                "index": len(records),
                "sample_id": _sample_id(
                    split,
                    city,
                    sequence,
                    frame_numbers[0],
                    frame_numbers[-1],
                ),
                "split": split,
                "city": city,
                "sequence": sequence,
                "first_frame": frame_numbers[0],
                "last_frame": frame_numbers[-1],
                "frame_numbers": frame_numbers,
                "frame_paths": [path.relative_to(root).as_posix() for path in group],
                "context_relative_indices": list(CONTEXT_RELATIVE_INDICES),
                "future_relative_indices": list(FUTURE_RELATIVE_INDICES),
            }
            records.append(record)
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(
            f"Cityscapes {split} has {len(records)} sequences, expected {expected_count}"
        )
    return records


def build_cityscapes_manifest(
    data_root: str | Path,
    split: str,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    split = validate_split(split)
    root = Path(data_root).expanduser().resolve()
    records = discover_cityscapes_sequences(root, split, expected_count=expected_count)
    return {
        "schema_version": 1,
        "dataset": "cityscapes",
        "data_root": str(root),
        "split": split,
        "num_sequences": len(records),
        "frames_per_sequence": FRAMES_PER_SEQUENCE,
        "context_relative_indices": list(CONTEXT_RELATIVE_INDICES),
        "future_relative_indices": list(FUTURE_RELATIVE_INDICES),
        "image_size": list(CITYSCAPES_IMAGE_SIZE),
        "records": records,
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if str(manifest.get("dataset", "")).lower() != "cityscapes":
        raise ValueError("manifest dataset must be cityscapes")
    validate_split(str(manifest.get("split", "")))
    if tuple(manifest.get("context_relative_indices", ())) != CONTEXT_RELATIVE_INDICES:
        raise ValueError("manifest context indices differ from relative frames 4..15")
    if tuple(manifest.get("future_relative_indices", ())) != FUTURE_RELATIVE_INDICES:
        raise ValueError("manifest future indices differ from relative frames 16..27")
    records = manifest.get("records")
    if not isinstance(records, list) or not all(isinstance(item, Mapping) for item in records):
        raise TypeError("manifest records must be a list of mappings")
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("every Cityscapes record requires a sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Cityscapes sample_ids must be unique")
    count = int(manifest.get("num_sequences", len(records)))
    if count != len(records):
        raise ValueError("manifest num_sequences does not match its records")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    _validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("Cityscapes manifest must contain a JSON object")
    result = dict(value)
    _validate_manifest(result)
    return result


def _record_frame_paths(
    record: Mapping[str, Any],
    data_root: str | Path | None,
) -> list[Path]:
    values = record.get("frame_paths")
    if not isinstance(values, list) or len(values) != FRAMES_PER_SEQUENCE:
        raise ValueError("a PNG sequence record must contain exactly 30 frame_paths")
    root = None if data_root is None else Path(data_root).expanduser().resolve()
    result: list[Path] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            if root is None:
                raise ValueError("data_root is required for relative frame paths")
            path = root / path
        result.append(path)
    return result


def _read_one_png(path: Path, *, backend: str, num_threads: int) -> torch.Tensor:
    reader = VideoReader(path, backend=backend, num_threads=num_threads)
    if len(reader) != 1:
        raise ValueError(f"Cityscapes PNG must decode as one frame, got {len(reader)}: {path}")
    return reader.get_frames([0])[0]


def load_cityscapes_rgb(
    record: Mapping[str, Any],
    *,
    data_root: str | Path | None = None,
    image_size: Sequence[int] = CITYSCAPES_IMAGE_SIZE,
    backend: str = "auto",
    num_threads: int = 1,
) -> dict[str, torch.Tensor]:
    """Decode only relative frames 4..27 through the unified VideoReader."""

    target_size = tuple(int(value) for value in image_size)
    if len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError("image_size must contain positive height and width")
    selected = CONTEXT_RELATIVE_INDICES + FUTURE_RELATIVE_INDICES
    video_path_value = record.get("video_path")
    if video_path_value is not None:
        video_path = Path(str(video_path_value))
        if not video_path.is_absolute():
            if data_root is None:
                raise ValueError("data_root is required for a relative video_path")
            video_path = Path(data_root).expanduser().resolve() / video_path
        reader = VideoReader(video_path, backend=backend, num_threads=num_threads)
        frames = reader.get_frames(selected)
    else:
        paths = _record_frame_paths(record, data_root)
        frames = torch.stack(
            [
                _read_one_png(paths[index], backend=backend, num_threads=num_threads)
                for index in selected
            ]
        )
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise TypeError("VideoReader must return CPU uint8 RGB frames in [T,C,H,W]")
    frames = resize_video(frames, target_size, mode="bicubic")
    frames = uint8_to_float(frames)
    context = frames[:CONTEXT_FRAMES].contiguous()
    future = frames[CONTEXT_FRAMES:].contiguous()
    expected = (12, 3, target_size[0], target_size[1])
    if context.shape != expected or future.shape != expected:
        raise RuntimeError("Cityscapes context/future decoding produced an invalid shape")
    return {"context": context, "future": future}


class CityscapesSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: Mapping[str, Any] | str | Path,
        *,
        data_root: str | Path | None = None,
        image_size: Sequence[int] = CITYSCAPES_IMAGE_SIZE,
        backend: str = "auto",
        num_threads: int = 1,
    ) -> None:
        if isinstance(manifest, Mapping):
            self.manifest = dict(manifest)
            _validate_manifest(self.manifest)
        else:
            self.manifest = load_manifest(manifest)
        root_value = data_root if data_root is not None else self.manifest.get("data_root")
        if root_value is None:
            raise ValueError("data_root is absent from arguments and manifest")
        self.data_root = Path(str(root_value)).expanduser().resolve()
        self.records = list(self.manifest["records"])
        self.split = validate_split(str(self.manifest["split"]))
        self.image_size = tuple(int(value) for value in image_size)
        self.backend = str(backend)
        self.num_threads = int(num_threads)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        rgb = load_cityscapes_rgb(
            record,
            data_root=self.data_root,
            image_size=self.image_size,
            backend=self.backend,
            num_threads=self.num_threads,
        )
        return {
            **rgb,
            "sample_id": str(record["sample_id"]),
            "index": int(record.get("index", index)),
            "metadata": dict(record),
        }


def collate_cityscapes(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty Cityscapes batch")
    return {
        "context": torch.stack([item["context"] for item in items]),
        "future": torch.stack([item["future"] for item in items]),
        "sample_id": [str(item["sample_id"]) for item in items],
        "index": torch.tensor([int(item["index"]) for item in items], dtype=torch.long),
        "metadata": [item["metadata"] for item in items],
    }


def build_raw_cityscapes_dataset(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    split: str | None = None,
) -> CityscapesSequenceDataset:
    data = config["data"]
    selected_split = validate_split(split or str(data.get("split", "train")))
    validate_frame_protocol(
        data.get("context_frames", (4, 15)),
        data.get("future_frames", (16, 27)),
    )
    root = paths.dataset("cityscapes")
    manifest_value = data.get("manifest")
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        manifest: Mapping[str, Any] | Path = manifest_path
    else:
        expected = data.get("expected_count")
        manifest = build_cityscapes_manifest(
            root,
            selected_split,
            expected_count=None if expected is None else int(expected),
        )
    return CityscapesSequenceDataset(
        manifest,
        data_root=root,
        image_size=data.get("image_size", CITYSCAPES_IMAGE_SIZE),
        backend=str(data.get("video_backend", "auto")),
        num_threads=int(data.get("decode_threads", 1)),
    )


def build_cityscapes_loader(
    dataset: Dataset[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader[dict[str, Any]], StatefulDistributedBatchSampler]:
    batch = resolve_batch_contract(config["training"], world_size=int(world_size))
    sampler = StatefulDistributedBatchSampler(
        len(dataset),
        batch.local_micro_batch_size,
        rank=int(rank),
        world_size=int(world_size),
        seed=int(config["data"].get("seed", 3407)),
        shuffle=shuffle,
        drop_last=drop_last,
        gradient_accumulation_steps=batch.gradient_accumulation_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_cityscapes,
        persistent_workers=False,
        generator=sampler.loader_generator,
    )
    return loader, sampler


__all__ = [
    "CITYSCAPES_IMAGE_SIZE",
    "CITYSCAPES_SPLITS",
    "CONTEXT_FRAMES",
    "CONTEXT_RELATIVE_INDICES",
    "CityscapesSequenceDataset",
    "EXPECTED_SPLIT_COUNTS",
    "FRAMES_PER_SEQUENCE",
    "FUTURE_FRAMES",
    "FUTURE_RELATIVE_INDICES",
    "build_cityscapes_loader",
    "build_cityscapes_manifest",
    "build_raw_cityscapes_dataset",
    "collate_cityscapes",
    "discover_cityscapes_sequences",
    "load_cityscapes_rgb",
    "load_manifest",
    "validate_frame_protocol",
    "validate_split",
    "write_manifest",
]
