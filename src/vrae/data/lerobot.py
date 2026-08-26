"""LeRobot v3.0 adapter with the same clip-shaped output as ``VideoDataset``.

The adapter is intentionally small: LeRobot stores one row per frame, while
V-RAE consumes one sampled clip.  Rows are grouped by episode and decoded
through :class:`lerobot.datasets.lerobot_dataset.LeRobotDataset`.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from .sampling import ClipSampler, ClipSamplingMode


class LeRobotVideoDataset(Dataset[dict[str, Any]]):
    """Expose LeRobot v3.0 episodes using the V-RAE ``VideoDataset`` schema.

    Each item contains ``video``, ``label``, ``sample_id``, ``path``,
    ``frame_indices``, ``video_metadata`` and ``extra``.  ``label`` is the
    episode's task index; state, actions and task text are retained in
    ``extra``.
    """

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
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - exercised in the wrong environment
            raise ImportError("LeRobotVideoDataset requires the lerobot v3.0 environment") from exc

        self.dataset = LeRobotDataset(repo_id, root=Path(root), download_videos=False)
        self.sampler = ClipSampler(clip_length, frame_interval, sampling)
        self.base_seed = int(base_seed)
        self.image_size = None if image_size is None else int(image_size)
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.random_flip = bool(random_flip)
        self.camera_keys = self._resolve_camera_keys(camera_keys)
        self.multiview_enabled = len(self.camera_keys) > 1 if multiview_enabled is None else bool(multiview_enabled)
        self.stream_ids = torch.tensor([item["stream_id"] for item in self.camera_keys], dtype=torch.long)
        self.num_views = len(self.camera_keys)
        self.num_streams = max(item["stream_id"] for item in self.camera_keys) + 1
        total_tasks = int(getattr(self.dataset.meta, "total_tasks", 0) or 0)
        if not total_tasks:
            total_tasks = int(getattr(self.dataset.meta.info, "total_tasks", 0) or 0)
        self.null_task_id = total_tasks
        self._episodes: list[tuple[int, int, int, int]] = []
        columns = self.dataset.meta.episodes.select_columns(
            ["episode_index", "dataset_from_index", "dataset_to_index", "tasks"]
        )
        for row in columns:
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index"])
            if stop - start >= self.sampler.required_frames:
                task_index = self.dataset.meta.get_task_index(row["tasks"][0])
                self._episodes.append((int(row["episode_index"]), start, stop, self.null_task_id if task_index is None else int(task_index)))
        self.fps = float(self.dataset.meta.fps)

    def _resolve_camera_keys(
        self, camera_keys: Sequence[str | Mapping[str, Any]] | None
    ) -> tuple[dict[str, Any], ...]:
        if camera_keys is None:
            camera_keys = ("observation.images.image",)
        if not camera_keys:
            raise ValueError("camera_keys must contain at least one image feature")
        features = getattr(self.dataset.meta, "features", {})
        resolved: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        seen_ids: set[int] = set()
        for position, value in enumerate(camera_keys):
            if isinstance(value, str):
                key, name, stream_id = value, value, position
            elif isinstance(value, Mapping):
                if "key" not in value:
                    raise ValueError("each camera_keys mapping requires key")
                key = str(value["key"])
                name = str(value.get("name", key))
                stream_id = int(value.get("stream_id", position))
            else:
                raise TypeError("camera_keys entries must be strings or mappings")
            if key in seen_keys:
                raise ValueError(f"duplicate camera key: {key}")
            if stream_id in seen_ids or stream_id < 0:
                raise ValueError(f"duplicate or invalid stream_id: {stream_id}")
            feature = features.get(key)
            if feature is None:
                raise ValueError(f"camera key {key!r} is missing from LeRobot metadata")
            dtype = str(feature.get("dtype", "")) if isinstance(feature, Mapping) else str(getattr(feature, "dtype", ""))
            shape = feature.get("shape") if isinstance(feature, Mapping) else getattr(feature, "shape", None)
            if dtype != "image" or shape is None or len(shape) != 3 or int(shape[-1]) != 3:
                raise ValueError(f"camera key {key!r} must be an RGB image feature")
            seen_keys.add(key)
            seen_ids.add(stream_id)
            resolved.append({"key": key, "name": name, "stream_id": stream_id})
        return tuple(resolved)

    def __len__(self) -> int:
        return len(self._episodes)

    def _generator(self, index: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(
            (self.base_seed + int(index) * 10_007) % (2**63 - 1)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start, stop, task_index = self._episodes[index]
        relative = self.sampler(stop - start, generator=self._generator(index))
        global_indices = relative + start
        rows = [self.dataset[int(row_index)] for row_index in global_indices.tolist()]
        views = []
        reference_shape: tuple[int, int, int] | None = None
        for camera in self.camera_keys:
            frames = []
            for row in rows:
                frame = row[camera["key"]]
                if not torch.is_tensor(frame):
                    frame = torch.as_tensor(frame)
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    frame = frame.permute(2, 0, 1)
                if frame.ndim != 3 or frame.shape[0] != 3:
                    raise ValueError(f"camera {camera['key']!r} must decode to [3,H,W]")
                frame = frame.float()
                if frame.max() > 1.0:
                    frame = frame / 255.0
                frames.append(frame)
            view = torch.stack(frames)
            shape = (int(view.shape[1]), int(view.shape[2]), int(view.shape[3]))
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise ValueError("all camera views must share channels and resolution")
            views.append(view)
        video = torch.stack(views, dim=1)
        if self.image_size is not None and tuple(video.shape[-2:]) != (self.image_size, self.image_size):
            video = torch.nn.functional.interpolate(
                video.flatten(0, 1), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            ).reshape(video.shape[0], video.shape[1], 3, self.image_size, self.image_size)
        if self.random_flip and bool(torch.rand((), generator=self._generator(index)) < 0.5):
            video = video.flip(-1)
        states = torch.stack([row["observation.state"] for row in rows])
        actions = torch.stack([row["action"] for row in rows])
        task = rows[0].get("task", "")
        parquet = self.dataset.meta.get_data_file_path(episode)
        output_video = video if self.multiview_enabled else video[:, 0]
        result = {
            "video": output_video,
            "label": task_index,
            "sample_id": f"episode-{episode:06d}",
            "path": str(Path(self.dataset.meta.root) / parquet),
            "frame_indices": global_indices,
            "video_metadata": {
                "fps": self.fps,
                "num_frames": stop - start,
                "height": int(output_video.shape[-2]),
                "width": int(output_video.shape[-1]),
                "channels": int(output_video.shape[-3]),
                "num_views": self.num_views,
            },
            "extra": {"episode_index": episode, "task": task, "state": states, "action": actions},
        }
        if self.multiview_enabled:
            result["stream_ids"] = self.stream_ids.clone()
        return result


__all__ = ["LeRobotVideoDataset"]
