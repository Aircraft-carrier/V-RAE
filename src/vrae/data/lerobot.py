"""LeRobot v3 dataset with V-RAE clip sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from vrae.libero import LiberoClass, LiberoClassMap

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .sampling import ClipSampler, ClipSamplingMode


class LeRobotVideoDataset(LeRobotDataset):
    """Sample synchronized episodes while retaining native LeRobot interfaces."""

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str = "libero",
        clip_length: int = 16,
        frame_interval: int = 1,
        sampling: ClipSamplingMode = "random",
        base_seed: int = 0,
        camera_keys: Sequence[str | Mapping[str, Any]] | None = None,
        image_size: int | None = None,
        random_flip: bool = False,
        multiview_enabled: bool | None = None,
        class_suites: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(repo_id, root=Path(root), download_videos=False)
        self.sampler = ClipSampler(clip_length, frame_interval, sampling)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.image_size = None if image_size is None else int(image_size)
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.random_flip = bool(random_flip)
        self.camera_keys = self._resolve_camera_keys(camera_keys)
        self.multiview_enabled = (
            len(self.camera_keys) > 1 if multiview_enabled is None else bool(multiview_enabled)
        )
        self.stream_ids = torch.tensor(
            [item["stream_id"] for item in self.camera_keys], dtype=torch.long
        )
        self.num_views = len(self.camera_keys)
        self.num_streams = max(item["stream_id"] for item in self.camera_keys) + 1
        task_names = {
            int(row.task_index): str(task) for task, row in self.meta.tasks.iterrows()
        }
        self.class_map = LiberoClassMap.from_config(
            class_suites,
            available_task_indices=tuple(task_names),
        )
        self.num_classes = len(self.class_map)
        self.class_names = tuple(
            f"{entry.suite}/{task_names[entry.task_index]}"
            for entry in self.class_map.entries
        )
        self._episodes: list[tuple[int, int, int, LiberoClass, str]] = []
        columns = self.meta.episodes.select_columns(
            ["episode_index", "dataset_from_index", "dataset_to_index", "tasks"]
        )
        for row in columns:
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index"])
            if stop - start < self.sampler.required_frames:
                continue
            task = str(row["tasks"][0])
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                raise ValueError(f"episode references an unknown task: {task!r}")
            self._episodes.append(
                (
                    int(row["episode_index"]),
                    start,
                    stop,
                    self.class_map.for_task_index(task_index),
                    task,
                )
            )
        self.clip_fps = float(self.meta.fps)

    def _resolve_camera_keys(
        self, camera_keys: Sequence[str | Mapping[str, Any]] | None
    ) -> tuple[dict[str, Any], ...]:
        values = camera_keys or ("observation.images.image",)
        features = self.meta.features
        resolved: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        seen_ids: set[int] = set()
        for position, value in enumerate(values):
            if isinstance(value, str):
                key, name, stream_id = value, value, position
            elif isinstance(value, Mapping) and "key" in value:
                key = str(value["key"])
                name = str(value.get("name", key))
                stream_id = int(value.get("stream_id", position))
            else:
                raise ValueError("camera_keys entries must be strings or mappings with key")
            if key in seen_keys or stream_id in seen_ids or stream_id < 0:
                raise ValueError("camera keys and stream IDs must be unique")
            feature = features.get(key)
            shape = (
                feature.get("shape")
                if isinstance(feature, Mapping)
                else getattr(feature, "shape", None)
            )
            dtype = (
                feature.get("dtype")
                if isinstance(feature, Mapping)
                else getattr(feature, "dtype", None)
            )
            if (
                feature is None
                or str(dtype) != "image"
                or shape is None
                or len(shape) != 3
                or int(shape[-1]) != 3
            ):
                raise ValueError(f"camera key {key!r} must be an RGB image feature")
            resolved.append({"key": key, "name": name, "stream_id": stream_id})
            seen_keys.add(key)
            seen_ids.add(stream_id)
        return tuple(resolved)

    def __len__(self) -> int:
        return len(self._episodes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        seed = self.base_seed + self.epoch * 1_000_003 + int(index) * 10_007
        return torch.Generator(device="cpu").manual_seed(seed % (2**63 - 1))

    def get_frame(self, index: int) -> dict[str, Any]:
        """Return a native LeRobot frame, including action and state fields."""

        return LeRobotDataset.__getitem__(self, int(index))

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start, stop, class_entry, task = self._episodes[index]
        global_indices = self.sampler(stop - start, generator=self._generator(index)) + start
        rows = [self.get_frame(row_index) for row_index in global_indices.tolist()]
        views: list[torch.Tensor] = []
        reference_shape: tuple[int, int, int] | None = None
        for camera in self.camera_keys:
            frames = []
            for row in rows:
                frame = row[camera["key"]]
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    frame = frame.permute(2, 0, 1)
                if frame.ndim != 3 or frame.shape[0] != 3:
                    raise ValueError(f"camera {camera['key']!r} must decode to [3,H,W]")
                frame = frame.float()
                if float(frame.max()) > 1.0:
                    frame = frame / 255.0
                frames.append(frame)
            view = torch.stack(frames)
            shape = tuple(int(value) for value in view.shape[1:])
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise ValueError("all camera views must share channels and resolution")
            views.append(view)
        video = torch.stack(views, dim=1)
        if self.image_size is not None and tuple(video.shape[-2:]) != (
            self.image_size,
            self.image_size,
        ):
            video = torch.nn.functional.interpolate(
                video.flatten(0, 1),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(
                video.shape[0],
                video.shape[1],
                3,
                self.image_size,
                self.image_size,
            )
        if self.random_flip and bool(torch.rand((), generator=self._generator(index)) < 0.5):
            video = video.flip(-1)
        output_video = video if self.multiview_enabled else video[:, 0]
        result: dict[str, Any] = {
            "video": output_video,
            "label": class_entry.class_id,
            "class_name": self.class_names[class_entry.class_id],
            "suite": class_entry.suite,
            "suite_task_index": class_entry.suite_task_index,
            "source_task_index": class_entry.task_index,
            "sample_id": f"episode-{episode:06d}",
            "path": str(Path(self.meta.root) / self.meta.get_data_file_path(episode)),
            "frame_indices": global_indices,
            "video_metadata": {
                "fps": self.clip_fps,
                "num_frames": stop - start,
                "height": int(output_video.shape[-2]),
                "width": int(output_video.shape[-1]),
                "channels": int(output_video.shape[-3]),
                "num_views": self.num_views,
            },
            "task": task,
            "extra": {
                "episode_index": episode,
                "task": task,
                "suite": class_entry.suite,
                "suite_task_index": class_entry.suite_task_index,
                "source_task_index": class_entry.task_index,
            },
        }
        if all("observation.state" in row for row in rows):
            state = torch.stack([row["observation.state"] for row in rows])
            result["state"] = state
            result["extra"]["state"] = state
        if all("action" in row for row in rows):
            action = torch.stack([row["action"] for row in rows])
            result["action"] = action
            result["extra"]["action"] = action
        if self.multiview_enabled:
            result["stream_ids"] = self.stream_ids.clone()
        return result


__all__ = ["LeRobotVideoDataset"]
