from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrae.evaluation.common.distributed import gather_indexed_features


@torch.no_grad()
def extract_features(extractor, inputs: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        feature = extractor(inputs[start : start + batch_size])
        if feature.ndim != 2 or not torch.isfinite(feature).all():
            raise ValueError("Feature extractor must return finite [N,D] values")
        outputs.append(feature)
    return torch.cat(outputs)


def gather_features(
    sample_ids: torch.Tensor, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return gather_indexed_features(sample_ids, features)


def save_feature_cache(
    directory: str | Path,
    *,
    sample_ids: torch.Tensor,
    features: torch.Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / "features.pt"
    temporary = directory / f".features.{os.getpid()}.tmp"
    torch.save({"sample_ids": sample_ids.cpu(), "features": features.cpu()}, temporary)
    temporary.replace(payload_path)
    metadata_value = {
        **dict(metadata),
        "count": int(features.shape[0]),
        "dimension": int(features.shape[1]),
    }
    metadata_path = directory / "metadata.json"
    temporary_metadata = directory / f".metadata.{os.getpid()}.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata_value, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return payload_path


def load_feature_cache(
    directory: str | Path, *, expected_metadata: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Feature cache metadata mismatch for {key}")
    payload = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)
    ids, features = payload["sample_ids"], payload["features"]
    if ids.shape[0] != metadata["count"] or features.shape != (
        metadata["count"],
        metadata["dimension"],
    ):
        raise ValueError("Feature cache shape/count does not match readable metadata")
    return ids, features
