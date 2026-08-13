"""Manifest and class-directory video datasets shared by all V-RAE tasks."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .sampling import ClipSampler, ClipSamplingMode
from .video_reader import VideoBackend, VideoReader, VideoSeekMode

DEFAULT_VIDEO_EXTENSIONS = (
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
)


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int
    sample_id: str
    split: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)


def load_class_index(path: str | Path) -> dict[str, int]:
    """Load ``class_name index`` or UCF-style ``index class_name`` text."""

    entries: list[tuple[str, int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"invalid class index line {line_number}: {line!r}")
            if parts[0].lstrip("+-").isdigit():
                index, name = int(parts[0]), parts[1]
            elif parts[1].lstrip("+-").isdigit():
                name, index = parts[0], int(parts[1])
            else:
                raise ValueError(f"class index line {line_number} has no integer index")
            entries.append((name, index))
    if not entries:
        return {}
    indices = [index for _, index in entries]
    offset = 1 if min(indices) == 1 and 0 not in indices else 0
    result = {name: index - offset for name, index in entries}
    if len(result) != len(entries):
        raise ValueError("class index contains duplicate class names")
    if len(set(result.values())) != len(result):
        raise ValueError("class index contains duplicate numeric labels")
    return result


def _read_manifest_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"manifest line {line_number} must be a JSON object")
                rows.append(dict(row))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("records", payload.get("videos"))
        if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
            raise ValueError("JSON manifest must be a list, or contain a records/videos list")
        return [dict(row) for row in payload]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2 and parts[1].lstrip("+-").isdigit():
                rows.append({"path": parts[0], "label": int(parts[1])})
            elif len(parts) == 1:
                rows.append({"path": parts[0]})
            else:
                raise ValueError(f"invalid text manifest line {line_number}: {line!r}")
    return rows


def _first_present(row: Mapping[str, object], names: Sequence[str]) -> object | None:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def load_video_manifest(
    manifest: str | Path,
    *,
    root: str | Path | None = None,
    split: str | None = None,
    class_to_idx: Mapping[str, int] | None = None,
) -> list[VideoRecord]:
    """Load JSON(L), CSV/TSV, or ``path [label]`` text manifests."""

    manifest_path = Path(manifest)
    rows = _read_manifest_rows(manifest_path)
    root_path = Path(root) if root is not None else manifest_path.parent
    selected = [
        row
        for row in rows
        if split is None or str(_first_present(row, ("split", "subset")) or "") == split
    ]

    label_values = [_first_present(row, ("label", "class_id", "class", "target")) for row in rows]
    label_names = sorted(
        {
            str(value)
            for value in label_values
            if value is not None and not str(value).lstrip("+-").isdigit()
        }
    )
    resolved_classes = dict(class_to_idx or {name: index for index, name in enumerate(label_names)})

    records: list[VideoRecord] = []
    for row_number, row in enumerate(selected, 1):
        path_value = _first_present(row, ("path", "video", "video_path", "filepath", "file"))
        if path_value is None:
            raise ValueError(f"manifest row {row_number} has no video path")
        raw_path = Path(str(path_value))
        path = raw_path if raw_path.is_absolute() else root_path / raw_path

        label_value = _first_present(row, ("label", "class_id", "class", "target"))
        if label_value is None:
            inferred_name = raw_path.parts[0] if len(raw_path.parts) > 1 else raw_path.parent.name
            label = resolved_classes.get(inferred_name, -1)
        elif str(label_value).lstrip("+-").isdigit():
            inferred_name = raw_path.parts[0] if len(raw_path.parts) > 1 else raw_path.parent.name
            label = resolved_classes.get(inferred_name, int(str(label_value)))
        else:
            label_name = str(label_value)
            if label_name not in resolved_classes:
                raise ValueError(f"manifest label {label_name!r} is absent from class_to_idx")
            label = resolved_classes[label_name]

        sample_value = _first_present(row, ("sample_id", "id", "uid"))
        sample_id = (
            str(sample_value) if sample_value is not None else raw_path.with_suffix("").as_posix()
        )
        row_split_value = _first_present(row, ("split", "subset"))
        row_split = str(row_split_value) if row_split_value is not None else split
        known = {
            "path",
            "video",
            "video_path",
            "filepath",
            "file",
            "label",
            "class_id",
            "class",
            "target",
            "sample_id",
            "id",
            "uid",
            "split",
            "subset",
        }
        extra = {key: value for key, value in row.items() if key not in known}
        records.append(VideoRecord(path, label, sample_id, row_split, extra))
    return records


def scan_class_directories(
    root: str | Path,
    *,
    class_to_idx: Mapping[str, int] | None = None,
    extensions: Iterable[str] = DEFAULT_VIDEO_EXTENSIONS,
) -> tuple[list[VideoRecord], dict[str, int]]:
    """Scan ``root/class_name/**/video`` deterministically."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    extension_set = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    class_directories = sorted(path for path in root_path.iterdir() if path.is_dir())
    if class_to_idx is None:
        resolved_classes = {path.name: index for index, path in enumerate(class_directories)}
    else:
        resolved_classes = dict(class_to_idx)

    records: list[VideoRecord] = []
    for class_directory in class_directories:
        if class_directory.name not in resolved_classes:
            continue
        label = resolved_classes[class_directory.name]
        video_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in extension_set
        )
        for path in video_paths:
            relative = path.relative_to(root_path)
            records.append(
                VideoRecord(
                    path=path,
                    label=label,
                    sample_id=relative.with_suffix("").as_posix(),
                )
            )
    return records, resolved_classes


class VideoDataset(Dataset[dict[str, Any]]):
    """Decode strict clips from pre-resolved records through ``VideoReader``."""

    def __init__(
        self,
        records: Sequence[VideoRecord],
        *,
        clip_length: int,
        frame_interval: int = 1,
        sampling: ClipSamplingMode = "random",
        backend: VideoBackend = "auto",
        transform: Callable[[Tensor], Tensor] | None = None,
        base_seed: int = 0,
        num_threads: int = 1,
        seek_mode: VideoSeekMode | None = None,
        require_temporal_multiple_of_four: bool = True,
    ) -> None:
        if require_temporal_multiple_of_four and clip_length % 4 != 0:
            raise ValueError("formal V-RAE clip_length must be divisible by four")
        self.records = tuple(records)
        self.sampler = ClipSampler(clip_length, frame_interval, sampling)
        self.backend = backend
        self.transform = transform
        self.base_seed = int(base_seed)
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
        self.num_threads = num_threads
        if seek_mode not in {None, "exact", "approximate"}:
            raise ValueError("seek_mode must be exact or approximate")
        self.seek_mode = seek_mode
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def epoch(self) -> int:
        return int(self._shared_epoch.item())

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._shared_epoch.fill_(epoch)

    def _generator(self, index: int, *, attempt: int = 0) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        seed = (
            self.base_seed + self.epoch * 1_000_003 + int(index) * 10_007 + int(attempt) * 97
        ) % (2**63 - 1)
        return generator.manual_seed(seed)

    def _reader_for(self, path: str | Path) -> VideoReader:
        reader_options: dict[str, Any] = {
            "backend": self.backend,
            "num_threads": self.num_threads,
        }
        if self.seek_mode is not None:
            reader_options["seek_mode"] = self.seek_mode
        return VideoReader(path, **reader_options)

    def __getitem__(
        self,
        index: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        record = self.records[index]
        reader = self._reader_for(record.path)
        generator = generator or self._generator(index)
        indices = self.sampler(len(reader), generator=generator)
        start = int(indices[0].item())
        step = int(self.sampler.frame_interval)
        video = reader.get_range(start, start + len(indices) * step, step)
        if self.transform is not None:
            video = self.transform(video)
        if not isinstance(video, Tensor) or video.ndim != 4:
            raise TypeError("video transform must return a [T,C,H,W] torch.Tensor")
        return {
            "video": video,
            "label": record.label,
            "sample_id": record.sample_id,
            "path": str(record.path),
            "frame_indices": indices,
            "video_metadata": reader.metadata.to_dict(),
            "extra": dict(record.extra),
        }


class ManifestVideoDataset(VideoDataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        root: str | Path | None = None,
        split: str | None = None,
        class_to_idx: Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        records = load_video_manifest(
            manifest,
            root=root,
            split=split,
            class_to_idx=class_to_idx,
        )
        self.manifest = Path(manifest)
        self.class_to_idx = dict(class_to_idx or {})
        super().__init__(records, **kwargs)


class DirectoryVideoDataset(VideoDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        class_to_idx: Mapping[str, int] | None = None,
        extensions: Iterable[str] = DEFAULT_VIDEO_EXTENSIONS,
        **kwargs: Any,
    ) -> None:
        records, resolved_classes = scan_class_directories(
            root,
            class_to_idx=class_to_idx,
            extensions=extensions,
        )
        self.root = Path(root)
        self.class_to_idx = resolved_classes
        super().__init__(records, **kwargs)


class UCF101Dataset(VideoDataset):
    """UCF101 dataset backed by either an official-style manifest or class folders."""

    num_classes = 101

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: str | Path | None = None,
        split: str | None = None,
        class_index: str | Path | Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(class_index, Mapping):
            class_to_idx = dict(class_index)
        elif class_index is not None:
            class_to_idx = load_class_index(class_index)
        else:
            class_to_idx = None
        if manifest is None:
            records, resolved_classes = scan_class_directories(root, class_to_idx=class_to_idx)
        else:
            records = load_video_manifest(
                manifest,
                root=root,
                split=split,
                class_to_idx=class_to_idx,
            )
            resolved_classes = dict(class_to_idx or {})
        self.root = Path(root)
        self.manifest = Path(manifest) if manifest is not None else None
        self.class_to_idx = resolved_classes
        super().__init__(records, **kwargs)


class Kinetics600Dataset(VideoDataset):
    """Kinetics-600 dataset backed by a path manifest or class folders."""

    num_classes = 600

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: str | Path | None = None,
        split: str | None = None,
        class_to_idx: Mapping[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        if manifest is None:
            records, resolved_classes = scan_class_directories(root, class_to_idx=class_to_idx)
        else:
            records = load_video_manifest(
                manifest,
                root=root,
                split=split,
                class_to_idx=class_to_idx,
            )
            resolved_classes = dict(class_to_idx or {})
        self.root = Path(root)
        self.manifest = Path(manifest) if manifest is not None else None
        self.class_to_idx = resolved_classes
        super().__init__(records, **kwargs)


K600Dataset = Kinetics600Dataset


__all__ = [
    "DEFAULT_VIDEO_EXTENSIONS",
    "DirectoryVideoDataset",
    "K600Dataset",
    "Kinetics600Dataset",
    "ManifestVideoDataset",
    "UCF101Dataset",
    "VideoDataset",
    "VideoRecord",
    "load_class_index",
    "load_video_manifest",
    "scan_class_directories",
]
