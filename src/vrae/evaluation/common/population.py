from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def sample_seed(base_seed: int, sample_id: int) -> int:
    return splitmix64((int(base_seed) << 32) ^ int(sample_id)) & 0x7FFFFFFFFFFFFFFF


def ucf101_sample_seed(base_seed: int, sample_id: int, stream: int = 0) -> int:
    """Match the uni-vug UCF101 global-sample-v1 seed derivation."""

    payload = f"{int(base_seed)}:{int(sample_id)}:{int(stream)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def deterministic_random_offset(
    seed: int,
    raw_index: int,
    *,
    stream: str,
    max_offset: int,
) -> int:
    """Choose the same item-local temporal offset as uni-vug UCF101 gFVD."""

    maximum = int(max_offset)
    if maximum <= 0:
        return 0
    payload = f"{int(seed)}:{int(raw_index)}:{stream}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (maximum + 1)


def exact_shard(population_size: int, rank: int, world_size: int) -> range:
    if population_size < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid population shard arguments")
    return range(rank, population_size, world_size)


def ucf101_generation_population(
    *, count: int = 2048, base_seed: int = 3407, num_classes: int = 101
) -> list[dict[str, Any]]:
    if count <= 0 or num_classes != 101:
        raise ValueError("Formal UCF101 generation uses positive count and exactly 101 classes")
    return [
        {
            "sample_id": sample_id,
            "label": sample_id % num_classes,
            "seed": ucf101_sample_seed(base_seed, sample_id),
            "filename": f"sample-{sample_id:06d}.mp4",
        }
        for sample_id in range(count)
    ]


def ucf101_gfvd_real_population(
    records: Sequence[Mapping[str, Any]],
    video_lengths: Sequence[int],
    *,
    count: int = 2048,
    base_seed: int = 3407,
    num_frames: int = 17,
    frame_interval: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the exact uni-vug real UCF101 fvd2048_17f population."""

    if len(records) != len(video_lengths):
        raise ValueError("UCF101 records and video lengths must have equal length")
    if count <= 0 or num_frames <= 0 or frame_interval <= 0:
        raise ValueError("UCF101 gFVD count and frame settings must be positive")

    minimum_length = num_frames * frame_interval
    valid = [
        (dict(record), int(length))
        for record, length in zip(records, video_lengths, strict=True)
        if int(length) >= minimum_length
    ]
    if len(valid) < count:
        raise ValueError(
            f"UCF101 gFVD needs {count} videos with at least {minimum_length} frames, "
            f"found {len(valid)}"
        )

    # This intentionally uses random.Random.sample, matching the reference evaluator.
    selected_indices = random.Random(int(base_seed)).sample(range(len(valid)), int(count))
    population: list[dict[str, Any]] = []
    for population_index, dataset_index in enumerate(selected_indices):
        source, video_length = valid[dataset_index]
        maximum_offset = video_length - minimum_length + frame_interval - 1
        offset = deterministic_random_offset(
            base_seed,
            dataset_index,
            stream="consecutive",
            max_offset=maximum_offset,
        )
        population.append(
            {
                **source,
                "sample_id": f"ucf101-train-{dataset_index:05d}",
                "population_index": population_index,
                "dataset_index": dataset_index,
                "split": "train",
                "video_length": video_length,
                "frame_indices": [offset + index * frame_interval for index in range(num_frames)],
                "preprocessing": {
                    "mode": "center_square_lanczos",
                    "crop_size": 256,
                },
            }
        )
    return population, {
        "source_count": len(records),
        "valid_count": len(valid),
        "short_video_count": len(records) - len(valid),
    }


def k600_balanced_population(*, base_seed: int = 3407) -> list[dict[str, Any]]:
    labels = [label for label in range(600) for _ in range(84 if label < 200 else 83)]
    if len(labels) != 50_000:
        raise AssertionError("K600 formal quota must total 50,000")
    return [
        {
            "sample_id": sample_id,
            "label": label,
            "seed": sample_seed(base_seed, sample_id),
            "filename": f"sample-{sample_id:06d}.pt",
        }
        for sample_id, label in enumerate(labels)
    ]


def validate_population(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_labels: Sequence[int] | None = None,
) -> None:
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} population records, got {len(records)}")
    ids = [int(record["sample_id"]) for record in records]
    if sorted(ids) != list(range(expected_count)):
        raise ValueError("Population sample IDs must be unique and contiguous from zero")
    if expected_labels is not None:
        labels = [
            int(record["label"])
            for record in sorted(records, key=lambda item: int(item["sample_id"]))
        ]
        if labels != list(expected_labels):
            raise ValueError("Population label assignment differs from the protocol")


def label_histogram(records: Iterable[Mapping[str, Any]]) -> Counter[int]:
    return Counter(int(record["label"]) for record in records)


def write_population(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def read_population(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
