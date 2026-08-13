from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vrae.training.cityscapes_video_pred.data import validate_split

CACHE_SCHEMA_VERSION = 1


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"unsupported latent cache dtype: {name}")
    return values[name]


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tensor(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_component(sample_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._")
    if not value:
        raise ValueError("sample_id does not contain a usable filename component")
    return value


def encode_context_future_separately(
    stage1: Any,
    context_rgb: torch.Tensor,
    future_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the two 12-frame clips with exactly two independent V-RAE calls."""

    for name, value in (("context", context_rgb), ("future", future_rgb)):
        if value.ndim != 5 or value.shape[1] != 12 or value.shape[2] != 3:
            raise ValueError(f"{name} RGB must have shape [B,12,3,H,W]")
    if context_rgb.shape[0] != future_rgb.shape[0]:
        raise ValueError("context and future batch sizes must match")
    context_grid = stage1.encode_grid(context_rgb)
    future_grid = stage1.encode_grid(future_rgb)
    context_tokens = stage1.grid_to_tokens(context_grid)
    future_tokens = stage1.grid_to_tokens(future_grid)
    if context_tokens.ndim != 4 or future_tokens.ndim != 4:
        raise ValueError("V-RAE token latents must be [B,chunks,tokens,channels]")
    if context_tokens.shape != future_tokens.shape:
        raise ValueError("context and future V-RAE latent shapes must match")
    if context_tokens.shape[1] != 3:
        raise ValueError("each 12-frame Cityscapes clip must produce exactly three chunks")
    return context_tokens, future_tokens


class LatentCacheWriter:
    """Write raw, unnormalized clean latent pairs under one split directory."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        stage1_metadata: Mapping[str, Any],
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.split = validate_split(split)
        self.split_root = self.cache_root / self.split
        self.tensor_root = self.split_root / "latents"
        self.metadata_root = self.split_root / "metadata"
        self.stage1_metadata = dict(stage1_metadata)
        self.dtype = _dtype_from_name(_dtype_name(dtype))
        self.records: list[dict[str, Any]] = []
        self._sample_ids: set[str] = set()

    def write(
        self,
        sample_id: str,
        context: torch.Tensor,
        future: torch.Tensor,
        *,
        source_metadata: Mapping[str, Any] | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        sample_id = str(sample_id)
        if not sample_id or sample_id in self._sample_ids:
            raise ValueError(f"empty or duplicate cache sample_id: {sample_id!r}")
        if context.ndim != 3 or future.ndim != 3 or context.shape != future.shape:
            raise ValueError("cached context/future must be matching [chunks,tokens,channels]")
        if context.shape[0] != 3:
            raise ValueError("Cityscapes cache tensors must contain three latent chunks")
        if not context.is_floating_point() or not future.is_floating_point():
            raise TypeError("latent cache tensors must be floating point")
        if not torch.isfinite(context).all() or not torch.isfinite(future).all():
            raise ValueError("latent cache tensors must be finite")
        context = context.detach().to(device="cpu", dtype=self.dtype).contiguous()
        future = future.detach().to(device="cpu", dtype=self.dtype).contiguous()
        record_index = len(self.records) if index is None else int(index)
        stem = f"{record_index:06d}-{_safe_component(sample_id)}"
        tensor_path = self.tensor_root / f"{stem}.pt"
        metadata_path = self.metadata_root / f"{stem}.json"
        tensors = {
            "context": {"shape": list(context.shape), "dtype": _dtype_name(context.dtype)},
            "future": {"shape": list(future.shape), "dtype": _dtype_name(future.dtype)},
        }
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset": "cityscapes",
            "split": self.split,
            "sample_id": sample_id,
            "index": record_index,
            "clean_latent": True,
            "normalized": False,
            "stage1": self.stage1_metadata,
            "tensors": tensors,
            "source": dict(source_metadata or {}),
        }
        _atomic_tensor(
            {"sample_id": sample_id, "context": context, "future": future},
            tensor_path,
        )
        _atomic_json(metadata, metadata_path)
        record = {
            "sample_id": sample_id,
            "index": record_index,
            "tensor_path": tensor_path.relative_to(self.split_root).as_posix(),
            "metadata_path": metadata_path.relative_to(self.split_root).as_posix(),
            "tensors": tensors,
        }
        self.records.append(record)
        self._sample_ids.add(sample_id)
        return record

    def finalize(self) -> Path:
        ordered = sorted(self.records, key=lambda item: int(item["index"]))
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset": "cityscapes",
            "split": self.split,
            "num_samples": len(ordered),
            "clean_latent": True,
            "normalized": False,
            "stage1": self.stage1_metadata,
            "records": ordered,
        }
        path = self.split_root / "manifest.json"
        _atomic_json(manifest, path)
        return path


def load_cache_manifest(cache_root: str | Path, split: str) -> tuple[Path, dict[str, Any]]:
    split = validate_split(split)
    path = Path(cache_root).expanduser().resolve() / split / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("latent cache manifest must contain a JSON object")
    manifest = dict(payload)
    if str(manifest.get("dataset", "")).lower() != "cityscapes":
        raise ValueError("latent cache dataset must be cityscapes")
    if str(manifest.get("split")) != split:
        raise ValueError("latent cache split differs from the requested split")
    if manifest.get("clean_latent") is not True or manifest.get("normalized") is not False:
        raise ValueError("latent cache must contain raw, unnormalized clean latents")
    records = manifest.get("records")
    if not isinstance(records, list) or int(manifest.get("num_samples", -1)) != len(records):
        raise ValueError("latent cache manifest has an invalid record count")
    return path, manifest


class LatentCacheDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        expected_stage1_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.manifest_path, self.manifest = load_cache_manifest(cache_root, split)
        self.split_root = self.manifest_path.parent
        self.split = validate_split(split)
        self.records = list(self.manifest["records"])
        self.stage1_metadata = dict(self.manifest.get("stage1", {}))
        if expected_stage1_metadata is not None and self.stage1_metadata != dict(
            expected_stage1_metadata
        ):
            raise ValueError("latent cache V-RAE metadata differs from the loaded V-RAE")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve(self, value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.split_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        tensor_path = self._resolve(record["tensor_path"])
        metadata_path = self._resolve(record["metadata_path"])
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError(f"latent cache tensor payload must be a mapping: {tensor_path}")
        sample_id = str(record["sample_id"])
        payload_id = str(payload.get("sample_id"))
        metadata_id = str(metadata.get("sample_id"))
        if payload_id != sample_id or metadata_id != sample_id:
            raise ValueError(f"latent cache sample_id mismatch: {tensor_path}")
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "index": int(record.get("index", index)),
            "metadata": metadata,
        }
        for name in ("context", "future"):
            tensor = payload.get(name)
            specification = record["tensors"][name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} is not a tensor in {tensor_path}")
            if list(tensor.shape) != list(specification["shape"]):
                raise ValueError(f"{name} shape differs from readable cache metadata")
            if _dtype_name(tensor.dtype) != str(specification["dtype"]):
                raise TypeError(f"{name} dtype differs from readable cache metadata")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite latent values")
            result[name] = tensor
        return result


@torch.no_grad()
def extract_latent_cache(
    stage1: Any,
    batches: Iterable[Mapping[str, Any]],
    cache_root: str | Path,
    split: str,
    *,
    stage1_metadata: Mapping[str, Any],
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
) -> Path:
    writer = LatentCacheWriter(
        cache_root,
        split,
        stage1_metadata=stage1_metadata,
        dtype=dtype,
    )
    next_index = 0
    for batch in batches:
        context_rgb = batch["context"].to(device, non_blocking=True)
        future_rgb = batch["future"].to(device, non_blocking=True)
        context, future = encode_context_future_separately(
            stage1,
            context_rgb,
            future_rgb,
        )
        sample_ids = [str(value) for value in batch["sample_id"]]
        indices = batch.get("index", torch.arange(next_index, next_index + len(sample_ids)))
        metadata_values = batch.get("metadata", [{} for _ in sample_ids])
        for offset, sample_id in enumerate(sample_ids):
            writer.write(
                sample_id,
                context[offset],
                future[offset],
                source_metadata=metadata_values[offset],
                index=int(indices[offset]),
            )
        next_index += len(sample_ids)
    return writer.finalize()


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "LatentCacheDataset",
    "LatentCacheWriter",
    "encode_context_future_separately",
    "extract_latent_cache",
    "load_cache_manifest",
]
