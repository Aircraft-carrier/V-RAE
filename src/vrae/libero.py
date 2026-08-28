from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LiberoClass:
    class_id: int
    suite: str
    suite_task_index: int
    task_index: int


class LiberoClassMap:
    """Stable class IDs built from an explicit LIBERO suite/task layout."""

    def __init__(self, entries: Sequence[LiberoClass]) -> None:
        self.entries = tuple(entries)
        self._by_task_index = {entry.task_index: entry for entry in self.entries}

    @classmethod
    def from_config(
        cls,
        suites: Sequence[Mapping[str, Any]] | None,
        *,
        available_task_indices: Sequence[int],
    ) -> LiberoClassMap:
        available = {int(value) for value in available_task_indices}
        if suites is None:
            suites = (
                {
                    "name": "lerobot",
                    "task_indices": sorted(available),
                },
            )
        if isinstance(suites, (str, bytes)) or not suites:
            raise ValueError("data.class_suites must be a non-empty list")

        entries: list[LiberoClass] = []
        suite_names: set[str] = set()
        task_indices: set[int] = set()
        for suite in suites:
            if not isinstance(suite, Mapping):
                raise ValueError("each data.class_suites entry must be a mapping")
            suite_name = str(suite.get("name", "")).strip()
            if not suite_name or suite_name in suite_names:
                raise ValueError("data.class_suites names must be non-empty and unique")
            suite_names.add(suite_name)

            values = suite.get("task_indices")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
                raise ValueError(f"suite {suite_name!r} requires non-empty task_indices")
            for suite_task_index, value in enumerate(values):
                task_index = int(value)
                if task_index in task_indices:
                    raise ValueError(f"duplicate LIBERO task_index: {task_index}")
                task_indices.add(task_index)
                entries.append(
                    LiberoClass(
                        class_id=len(entries),
                        suite=suite_name,
                        suite_task_index=suite_task_index,
                        task_index=task_index,
                    )
                )

        missing = sorted(available.difference(task_indices))
        unknown = sorted(task_indices.difference(available))
        if missing or unknown:
            raise ValueError(
                "data.class_suites must cover every dataset task exactly once: "
                f"missing={missing}, unknown={unknown}"
            )
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def for_task_index(self, task_index: int) -> LiberoClass:
        try:
            return self._by_task_index[int(task_index)]
        except KeyError as error:
            raise ValueError(f"unmapped LIBERO task_index: {task_index}") from error

    def metadata(self) -> list[dict[str, int | str]]:
        return [
            {
                "class_id": entry.class_id,
                "suite": entry.suite,
                "suite_task_index": entry.suite_task_index,
                "task_index": entry.task_index,
            }
            for entry in self.entries
        ]


__all__ = ["LiberoClass", "LiberoClassMap"]
