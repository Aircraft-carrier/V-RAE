"""Smoke test for the LeRobot v3.0 LIBERO cache.

Run with the ``lerobotv3`` environment (or the existing ``lerobot`` env):
``PYTHONPATH=src python test/data/load_libero.py``
"""

from __future__ import annotations

import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vrae.data import LeRobotVideoDataset


ROOT = Path(os.environ.get("LIBERO_ROOT", "/zsh/cache/data/Lerobot/libero"))


def main() -> None:
    native = LeRobotDataset("libero", root=ROOT, download_videos=False)
    item = native[0]
    expected_native = {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "task",
    }
    assert set(item) == expected_native
    adapted = LeRobotVideoDataset(ROOT, clip_length=8, sampling="start")
    clip = adapted[0]
    expected_vrae = {
        "video",
        "label",
        "sample_id",
        "path",
        "frame_indices",
        "video_metadata",
        "extra",
    }
    assert set(clip) == expected_vrae
    print(f"native: {len(native)} frames, keys={sorted(item)}")
    print(f"adapted: {len(adapted)} episodes, video={tuple(clip['video'].shape)}")
    print(f"sample_id={clip['sample_id']} label={clip['label']} task={clip['extra']['task']}")


if __name__ == "__main__":
    main()
