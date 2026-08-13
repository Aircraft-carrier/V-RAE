from __future__ import annotations

from collections.abc import Sequence

from vrae.evaluation.common.protocol import run_task_cli

TASK = "cityscapes_gfid_gfvd"


def main(argv: Sequence[str] | None = None) -> int:
    return run_task_cli(TASK, argv)


if __name__ == "__main__":
    raise SystemExit(main())
