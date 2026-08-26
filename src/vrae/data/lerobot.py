"""LeRobot v3.0 adapter with the same clip-shaped output as ``VideoDataset``.

The adapter is intentionally small: LeRobot stores one row per frame, while
V-RAE consumes one sampled clip.  Rows are grouped by episode and decoded
through :class:`lerobot.datasets.lerobot_dataset.LeRobotDataset`.
"""

from __future__ import annotations

from pathlib import Path
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
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - exercised in the wrong environment
            raise ImportError("LeRobotVideoDataset requires the lerobot v3.0 environment") from exc

        self.dataset = LeRobotDataset(repo_id, root=Path(root), download_videos=False)
        self.sampler = ClipSampler(clip_length, frame_interval, sampling)
        self.base_seed = int(base_seed)
        self._episodes: list[tuple[int, int, int, int]] = []
        columns = self.dataset.meta.episodes.select_columns(
            ["episode_index", "dataset_from_index", "dataset_to_index", "tasks"]
        )
        for row in columns:
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index"])
            if stop - start >= self.sampler.required_frames:
                task_index = self.dataset.meta.get_task_index(row["tasks"][0])
                self._episodes.append(
                    (int(row["episode_index"]), start, stop, -1 if task_index is None else task_index)
                )
        self.fps = float(self.dataset.meta.fps)

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
        video = torch.stack([row["observation.images.image"] for row in rows])
        states = torch.stack([row["observation.state"] for row in rows])
        actions = torch.stack([row["action"] for row in rows])
        task = rows[0].get("task", "")
        parquet = self.dataset.meta.get_data_file_path(episode)
        return {
            "video": video,
            "label": task_index,
            "sample_id": f"episode-{episode:06d}",
            "path": str(Path(self.dataset.meta.root) / parquet),
            "frame_indices": global_indices,
            "video_metadata": {
                "fps": self.fps,
                "num_frames": stop - start,
                "height": int(video.shape[-2]),
                "width": int(video.shape[-1]),
                "channels": int(video.shape[-3]),
            },
            "extra": {"episode_index": episode, "task": task, "state": states, "action": actions},
        }


__all__ = ["LeRobotVideoDataset"]
