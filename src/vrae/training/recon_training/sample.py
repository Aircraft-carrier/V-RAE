from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def save_reconstruction_sample(sample: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(sample, temporary)
    temporary.replace(output)
    return output
