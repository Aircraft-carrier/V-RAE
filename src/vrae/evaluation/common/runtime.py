"""End-to-end execution for the locked evaluation protocols."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, default_collate

from vrae.evaluation.common.distributed import gather_indexed_features
from vrae.evaluation.common.features import load_feature_cache, save_feature_cache
from vrae.evaluation.common.frechet import frechet_from_features
from vrae.evaluation.common.paired import PairedMetricAccumulator
from vrae.evaluation.common.population import (
    exact_shard,
    k600_balanced_population,
    read_population,
    ucf101_generation_population,
    ucf101_gfvd_real_population,
    ucf101_sample_seed,
)
from vrae.evaluation.common.protocol import DEFAULT_GLOBAL_BATCH_SIZE
from vrae.evaluation.models.i3d import I3DFeatureExtractor
from vrae.evaluation.models.inception import InceptionFeatureExtractor
from vrae.evaluation.models.lpips import LPIPSMetric
from vrae.evaluation.models.temporal import centered_six_to_four
from vrae.training.common.distributed import (
    barrier,
    broadcast_object,
    initialize_distributed,
)
from vrae.training.common.precision import PrecisionPolicy
from vrae.checkpoint import load_checkpoint
from vrae.data import VideoReader, resize_center_crop, resize_short_side, uint8_to_float
from vrae.paths import ProjectPaths

_UNI_VUG_K600_SOURCE_COUNT = 27_910
_UNI_VUG_K600_VALID_COUNT = 27_874
_UNI_VUG_K600_SHORT_COUNT = 36


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _model_project_paths(plan: Any, model_config: Mapping[str, Any]) -> ProjectPaths:
    """Resolve released models against the current project's standard layout."""

    project_root = Path(plan.project_root).resolve()
    configured = model_config.get("paths")
    if isinstance(configured, Mapping):
        value = dict(configured)
        value.setdefault("project_root", str(project_root))
        paths = ProjectPaths.from_mapping(value)
    else:
        # A checkpoint may retain a legacy paths.local.yaml filename from the
        # machine that trained it.  It must not impose that machine's layout on
        # users loading a released model.
        paths = ProjectPaths(project_root=project_root)
    if paths.project_root != project_root:
        raise ValueError(
            "model paths project_root differs from the evaluation project root: "
            f"{paths.project_root} != {project_root}"
        )
    return paths


def _project_file(plan: Any, value: object, name: str) -> Path:
    if value is None or value == "":
        raise ValueError(f"inputs.{name} is required for end-to-end evaluation")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = plan.project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _ucf101_gfvd_csv_population(
    plan: Any,
    annotation: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce the uni-vug fvd2048_17f real-data population from its CSV."""

    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    root_value = inputs.get("data_root")
    if root_value in {None, ""}:
        raise ValueError("inputs.data_root is required for UCF101 gFVD CSV populations")
    data_root = Path(str(root_value)).expanduser()
    if not data_root.is_absolute():
        data_root = plan.project_root / data_root
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    path_column = str(inputs.get("population_path_column", "path"))
    source_records: list[dict[str, Any]] = []
    absolute_paths: list[Path] = []
    seen: set[Path] = set()
    with annotation.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or path_column not in reader.fieldnames:
            raise ValueError(f"UCF101 annotation must contain column {path_column!r}: {annotation}")
        for row in reader:
            value = str(row.get(path_column, "")).strip()
            if not value:
                continue
            raw_path = Path(value).expanduser()
            resolved = (raw_path if raw_path.is_absolute() else data_root / raw_path).resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            source_records.append(
                {
                    "path": value,
                    "label": int(row["label"]) if str(row.get("label", "")).strip() else -1,
                    "action": str(row.get("action", "")),
                }
            )
            absolute_paths.append(resolved)
    if not source_records:
        raise ValueError(f"UCF101 annotation contains no videos: {annotation}")

    cache_value = inputs.get("video_length_cache")
    if cache_value not in {None, ""}:
        cache_path = _project_file(plan, cache_value, "video_length_cache")
        cache = _mapping(
            json.loads(cache_path.read_text(encoding="utf-8")), "UCF101 video length cache"
        )
        cached_paths = [
            Path(str(value)).expanduser().resolve() for value in cache.get("video_paths", [])
        ]
        if cached_paths != absolute_paths:
            raise ValueError(
                "UCF101 video length cache paths differ from the annotation population"
            )
        video_lengths = [int(value) for value in cache.get("video_lengths", [])]
        if len(video_lengths) != len(source_records):
            raise ValueError("UCF101 video length cache count differs from the annotation")
        cache_identity: str | None = str(cache_path)
    else:
        video_lengths = []
        for video_path in absolute_paths:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            reader = VideoReader(
                video_path,
                backend=_video_backend(plan),
                num_threads=int(evaluation.get("decode_threads", 1)),
            )
            video_lengths.append(len(reader))
        cache_identity = None

    records, counts = ucf101_gfvd_real_population(
        source_records,
        video_lengths,
        count=2048,
        base_seed=int(evaluation.get("base_seed", 3407)),
        num_frames=17,
        frame_interval=3,
    )
    for record in records:
        preprocessing = dict(_mapping(record["preprocessing"], "preprocessing"))
        preprocessing["crop_size"] = int(plan.protocol.image_size[0])
        record["preprocessing"] = preprocessing
    return records, {
        "dataset": "ucf101",
        "split": "train",
        "protocol_complete": True,
        "record_count": 2048,
        "annotation_file": str(annotation),
        "annotation_sha256": _sha256(annotation),
        "video_length_cache": cache_identity,
        "reference_protocol": "uni-vug-fvd2048-17f",
        "real_frame_interval": 3,
        "real_preprocessing": "center_square_lanczos",
        **counts,
    }


def _population_document(plan: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    population_value = inputs.get("population")
    if plan.task == "cityscapes_rfvd" and population_value in {None, ""}:
        from vrae.training.cityscapes_video_pred.data import build_cityscapes_manifest

        data_root = inputs.get("data_root")
        if data_root in {None, ""}:
            raise ValueError("inputs.data_root is required when Cityscapes population is omitted")
        data_root_path = Path(str(data_root)).expanduser()
        if not data_root_path.is_absolute():
            data_root_path = plan.project_root / data_root_path
        manifest = build_cityscapes_manifest(data_root_path, "val", expected_count=500)
        records = [dict(_mapping(item, "population record")) for item in manifest["records"]]
        metadata = {
            **{key: value for key, value in manifest.items() if key != "records"},
            "record_count": len(records),
            "protocol_complete": True,
        }
        return records, metadata

    path = _project_file(plan, population_value, "population")
    if plan.task == "ucf101_gfvd" and path.suffix.lower() == ".csv":
        records, metadata = _ucf101_gfvd_csv_population(plan, path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            records_value = payload.get("records")
            if not isinstance(records_value, list):
                raise ValueError("population JSON object must contain a records list")
            metadata = {str(key): value for key, value in payload.items() if key != "records"}
            records = []
            for index, item in enumerate(records_value):
                record = dict(_mapping(item, "population record"))
                # Preserve the checked-in index ordering and decode metadata.
                if "rel_path" in record and "path" not in record and "video_path" not in record:
                    record["path"] = record.pop("rel_path")
                    record.setdefault("sample_id", index)
                    record.setdefault("split", metadata.get("split"))
                records.append(record)
        elif isinstance(payload, list):
            records = [dict(_mapping(item, "population record")) for item in payload]
            metadata = {}
        else:
            raise ValueError("population JSON must contain an object or list")
    else:
        records = read_population(path)
        metadata = {}

    metadata_value = inputs.get("population_metadata")
    if metadata_value not in {None, ""}:
        metadata_path = _project_file(plan, metadata_value, "population_metadata")
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(dict(_mapping(sidecar, "population metadata")))
    return records, metadata


def _dataset_name(task: str) -> str:
    if task.startswith("ucf101"):
        return "ucf101"
    if task.startswith("k600"):
        return "k600"
    return "cityscapes"


def _expected_count(plan: Any, available: int) -> int:
    if plan.mode == "smoke":
        expected = int(plan.protocol.population_size)
        if available < expected:
            raise ValueError(f"smoke protocol needs {expected} records, got {available}")
        return expected
    if plan.protocol.population_size > 0:
        return int(plan.protocol.population_size)
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    value = evaluation.get("population_size")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
        raise ValueError("formal full-split evaluation requires evaluation.population_size > 1")
    return value


def _validate_population_source(
    plan: Any,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    count = _expected_count(plan, len(records))
    if plan.mode == "formal" and len(records) != count:
        raise ValueError(f"formal population requires exactly {count} records, got {len(records)}")
    selected = [dict(record) for record in records[:count]]
    ids = [str(record.get("sample_id", "")) for record in selected]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("population sample_id values must be non-empty and unique")
    if any(record.get("split") != plan.protocol.split for record in selected):
        raise ValueError(f"all population records must declare split={plan.protocol.split!r}")

    if plan.mode == "formal":
        expected_dataset = _dataset_name(plan.task)
        if metadata.get("dataset") != expected_dataset:
            raise ValueError(f"population metadata must declare dataset={expected_dataset!r}")
        if metadata.get("split") != plan.protocol.split:
            raise ValueError(f"population metadata must declare split={plan.protocol.split!r}")
        if metadata.get("protocol_complete") is not True:
            raise ValueError("formal population metadata must set protocol_complete=true")
        declared = metadata.get("record_count", metadata.get("num_sequences"))
        if int(declared) != count:
            raise ValueError("population metadata record_count does not match the protocol")
        if plan.protocol.population_size < 0:
            if metadata.get("complete_split") is not True:
                raise ValueError("formal rFVD/tFVD metadata must set complete_split=true")
            if int(metadata.get("valid_count", -1)) != count:
                raise ValueError("population metadata valid_count does not match records")
            source_count = int(metadata.get("source_count", -1))
            short_count = int(metadata.get("short_video_count", -1))
            if source_count != count + short_count or short_count < 0:
                raise ValueError(
                    "population metadata must account for source_count as valid plus short videos"
                )
        if plan.task == "k600_rfvd":
            expected = {
                "source_count": _UNI_VUG_K600_SOURCE_COUNT,
                "valid_count": _UNI_VUG_K600_VALID_COUNT,
                "short_video_count": _UNI_VUG_K600_SHORT_COUNT,
                "video_backend": "torchcodec",
                "seek_mode": "approximate",
                "num_frames": 16,
                "frame_interval": 3,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"formal k600_rfvd metadata {key}={metadata.get(key)!r}; "
                        f"the locked K600 protocol requires {value!r}"
                    )
            if any(int(record.get("total_frames", -1)) < 46 for record in selected):
                raise ValueError(
                    "formal k600_rfvd valid-index contains a video shorter than 46 frames"
                )
            evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
            required_runtime = {
                "precision": "bf16",
                "attention_backend": "sdpa",
                "video_backend": "torchcodec",
                "seek_mode": "approximate",
                "decode_threads": 1,
                "num_workers": 4,
                "prefetch_factor": 1,
                "i3d_batch_size": 32,
                "perceptual_batch_size": 32,
                "seed": 42,
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
            }
            for key, value in required_runtime.items():
                if evaluation.get(key) != value:
                    raise ValueError(
                        f"formal k600_rfvd evaluation.{key}={evaluation.get(key)!r}; "
                        f"the locked all9 protocol requires {value!r}"
                    )
    if plan.task in {"ucf101_gfvd", "k600_gfvd"}:
        for record in selected:
            preprocessing = record.get("preprocessing", record)
            if not isinstance(record.get("frame_indices"), list):
                raise ValueError("gFVD real records must store deterministic frame_indices")
            if plan.task == "ucf101_gfvd":
                valid_preprocessing = (
                    isinstance(preprocessing, Mapping)
                    and preprocessing.get("mode") == "center_square_lanczos"
                    and int(preprocessing.get("crop_size", -1)) == int(plan.protocol.image_size[0])
                )
            else:
                valid_preprocessing = isinstance(preprocessing, Mapping) and all(
                    key in preprocessing
                    for key in (
                        "resize_height",
                        "resize_width",
                        "crop_top",
                        "crop_left",
                        "crop_size",
                        "horizontal_flip",
                    )
                )
            if not valid_preprocessing:
                raise ValueError(
                    "gFVD real records do not match the task's deterministic preprocessing"
                )
    return selected


def _record_path(plan: Any, record: Mapping[str, Any]) -> Path:
    # Prefer the portable entry so the current project's data root controls
    # resolution even when a legacy manifest also provides ``video_path``.
    value = record.get("relative_path", record.get("path", record.get("video_path")))
    if value in {None, ""}:
        raise ValueError("video population record must contain path or video_path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        inputs = _mapping(plan.config.get("inputs", {}), "inputs")
        root_value = inputs.get("data_root")
        root = plan.project_root if root_value in {None, ""} else Path(str(root_value)).expanduser()
        if not root.is_absolute():
            root = plan.project_root / root
        path = root / path
    return path.resolve()


def _indices_for_record(
    record: Mapping[str, Any],
    *,
    count: int,
    interval: int,
    require_explicit: bool,
) -> list[int]:
    supplied = record.get("frame_indices")
    if supplied is not None:
        indices = [int(value) for value in supplied]
        if len(indices) != count:
            raise ValueError(f"frame_indices must contain exactly {count} values")
        if any(right <= left for left, right in zip(indices, indices[1:], strict=False)):
            raise ValueError("frame_indices must be strictly increasing")
        return indices
    if require_explicit:
        raise ValueError("generation real-population records require deterministic frame_indices")
    return [index * interval for index in range(count)]


def _video_backend(plan: Any) -> str:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    backend = str(evaluation.get("video_backend", "auto"))
    if backend not in {"auto", "torchcodec"}:
        raise ValueError("evaluation.video_backend must be auto or torchcodec")
    return backend


def _video_seek_mode(plan: Any) -> str | None:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    value = evaluation.get("seek_mode")
    if value in {None, ""}:
        return None
    seek_mode = str(value)
    if seek_mode not in {"exact", "approximate"}:
        raise ValueError("evaluation.seek_mode must be exact or approximate")
    return seek_mode


def _clamp_rgb_roundoff(video: Tensor, *, tolerance: float = 1.0e-5) -> Tensor:
    """Clamp interpolation roundoff without accepting genuinely invalid RGB."""

    minimum = float(video.amin().item())
    maximum = float(video.amax().item())
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            "resized RGB values must be in [0,1] up to interpolation roundoff; "
            f"got [{minimum}, {maximum}]"
        )
    return video.clamp_(0.0, 1.0)


def _load_center_square_lanczos_clip(
    path: Path,
    indices: Sequence[int],
    *,
    image_size: tuple[int, int],
) -> Tensor:
    """Decode and preprocess exactly like uni-vug's VideoFilesFolderDataset."""

    if image_size[0] != image_size[1]:
        raise ValueError("center-square preprocessing requires a square output")
    try:
        import imageio.v2 as imageio
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("uni-vug UCF101 gFVD requires imageio and Pillow") from error

    reader = imageio.get_reader(str(path))
    frames: list[np.ndarray] = []
    try:
        for frame_index in indices:
            frame = reader.get_data(int(frame_index))
            image = Image.fromarray(frame).convert("RGB")
            width, height = image.size
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
            if image.size != image_size:
                image = image.resize(image_size, Image.Resampling.LANCZOS)
            frames.append(np.array(image, dtype=np.uint8, copy=True))
    finally:
        reader.close()
    value = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    return value


def decode_record_clip(
    plan: Any,
    record: Mapping[str, Any],
    *,
    frames: int,
    interval: int = 1,
    require_explicit_indices: bool = False,
) -> Tensor:
    """Decode one record through the sole torchcodec-CPU/decord boundary."""

    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    indices = _indices_for_record(
        record,
        count=frames,
        interval=interval,
        require_explicit=require_explicit_indices,
    )
    if getattr(plan, "task", None) == "ucf101_gfvd" and require_explicit_indices:
        preprocessing = _mapping(record.get("preprocessing", record), "preprocessing")
        if preprocessing.get("mode") != "center_square_lanczos":
            raise ValueError("UCF101 gFVD requires center-square Lanczos preprocessing")
        video_length = int(record.get("video_length", -1))
        if video_length > 0 and indices[-1] >= video_length:
            raise ValueError(
                f"short video {record.get('sample_id')!r}: needs frame {indices[-1]}, "
                f"only {video_length} frames"
            )
        return uint8_to_float(
            _load_center_square_lanczos_clip(
                _record_path(plan, record),
                indices,
                image_size=plan.protocol.image_size,
            )
        )
    reader_options: dict[str, Any] = {
        "backend": _video_backend(plan),
        "num_threads": int(evaluation.get("decode_threads", 1)),
    }
    seek_mode = _video_seek_mode(plan)
    if seek_mode is not None:
        reader_options["seek_mode"] = seek_mode
    reader = VideoReader(_record_path(plan, record), **reader_options)
    if indices[-1] >= len(reader):
        raise ValueError(
            f"short video {record.get('sample_id')!r}: needs frame {indices[-1]}, "
            f"only {len(reader)} frames"
        )
    if getattr(plan, "task", None) in {"ucf101_tfvd", "k600_rfvd", "k600_tfvd"}:
        # Preserve the locked range-read path for fixed interval-3 FVD inputs.
        clip = reader.get_range(indices[0], indices[-1] + 1, interval)
    else:
        clip = reader.get_frames(indices)
    if require_explicit_indices:
        if plan.protocol.image_size[0] != plan.protocol.image_size[1]:
            raise ValueError("class-conditional generation evaluation requires a square crop")
        preprocessing = _mapping(record.get("preprocessing", record), "preprocessing")
        crop_size = int(plan.protocol.image_size[0])
        clip = resize_short_side(clip, crop_size, mode="bilinear")
        if (
            int(preprocessing["resize_height"]) != clip.shape[-2]
            or int(preprocessing["resize_width"]) != clip.shape[-1]
            or int(preprocessing["crop_size"]) != crop_size
        ):
            raise ValueError("recorded resize/crop geometry differs from the decoded video")
        top = int(preprocessing["crop_top"])
        left = int(preprocessing["crop_left"])
        height, width = clip.shape[-2:]
        if top < 0 or left < 0 or top + crop_size > height or left + crop_size > width:
            raise ValueError("recorded deterministic crop lies outside the resized video")
        clip = clip[..., top : top + crop_size, left : left + crop_size].contiguous()
        flip = preprocessing["horizontal_flip"]
        if not isinstance(flip, bool):
            raise ValueError("horizontal_flip must be a boolean")
        if flip:
            clip = clip.flip(-1)
        return uint8_to_float(clip)

    # Convert before interpolation to avoid an intermediate uint8 requantization.
    clip = uint8_to_float(clip)
    resize_options: dict[str, Any] = {"mode": "bilinear", "antialias": True}
    if getattr(plan, "task", None) in {"ucf101_tfvd", "k600_rfvd", "k600_tfvd"}:
        resize_options["crop_rounding"] = "round"
    resized = resize_center_crop(clip, plan.protocol.image_size, **resize_options)
    return _clamp_rgb_roundoff(resized)


def _decode_city_record(plan: Any, record: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    from vrae.training.cityscapes_video_pred.data import load_cityscapes_rgb

    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    value = load_cityscapes_rgb(
        record,
        data_root=inputs.get("data_root") or record.get("data_root"),
        image_size=plan.protocol.image_size,
        backend=_video_backend(plan),
        num_threads=int(evaluation.get("decode_threads", 1)),
    )
    return value["context"], value["future"]


def _ema_to_model(model: nn.Module, payload: Mapping[str, Any]) -> None:
    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise ValueError("EMA evaluation requested but checkpoint has no EMA state")
    shadow = _mapping(ema.get("shadow"), "checkpoint EMA shadow")
    state = model.state_dict()
    if set(shadow) != {name for name, value in state.items() if torch.is_floating_point(value)}:
        raise ValueError("checkpoint EMA keys do not match the evaluated model")
    for name, average in shadow.items():
        state[name].copy_(torch.as_tensor(average).to(state[name]))


def _weight_selection(plan: Any) -> dict[str, str | None]:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    fallback = str(evaluation.get("weights", "ema"))
    stage1 = str(evaluation.get("stage1_weights", fallback))
    if plan.task in {"ucf101_gfvd", "k600_gfvd"}:
        downstream: str | None = str(evaluation.get("dit_weights", fallback))
    elif plan.task == "cityscapes_gfid_gfvd":
        downstream = str(evaluation.get("prediction_weights", fallback))
    else:
        downstream = None
    if stage1 not in {"ema", "model"} or downstream not in {None, "ema", "model"}:
        raise ValueError("evaluation weight selections must be ema or model")
    return {"stage1": stage1, "downstream": downstream}


def _evaluation_precision_name(plan: Any) -> str:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    precision = str(evaluation.get("precision", "bf16")).lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("evaluation.precision must be bf16, fp16, or fp32")
    return precision


def _evaluation_autocast_policy(plan: Any, device: torch.device) -> PrecisionPolicy:
    # CPU evaluation remains float32 when the recorded precision is bf16.
    precision = _evaluation_precision_name(plan) if device.type == "cuda" else "fp32"
    return PrecisionPolicy(precision, device.type)


def _evaluation_attention_backend(plan: Any) -> str:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    backend = str(evaluation.get("attention_backend", "sdpa")).lower()
    allowed = {"auto", "sdpa", "fa3", "fa3_fwd", "fa4_cute"}
    if backend not in allowed:
        raise ValueError(f"evaluation.attention_backend must be one of {sorted(allowed)}")
    return backend


def _evaluation_microbatch_size(plan: Any, key: str, *, default: int = 32) -> int:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    value = evaluation.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"evaluation.{key} must be a positive integer")
    return value


def _stage1_evaluation_config(
    plan: Any,
    raw_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply evaluation-only execution choices without mutating checkpoint config."""

    config = copy.deepcopy(dict(raw_config))
    has_model_section = "model" in config
    model = dict(_mapping(config["model"] if has_model_section else config, "V-RAE model"))
    decoder = dict(_mapping(model.get("decoder"), "V-RAE decoder"))
    parameters = decoder.get("parameters")
    if isinstance(parameters, Mapping):
        parameters = dict(parameters)
        parameters["attention_backend"] = _evaluation_attention_backend(plan)
        decoder["parameters"] = parameters
    else:
        decoder["attention_backend"] = _evaluation_attention_backend(plan)
    model["decoder"] = decoder
    if has_model_section:
        config["model"] = model
    else:
        config.update(model)
    return config


def load_stage1(plan: Any, device: torch.device) -> nn.Module:
    from vrae.models.autoencoder import VRAE

    payload = load_checkpoint(plan.model_checkpoints["stage1"], map_location="cpu")
    raw_config = _mapping(payload.get("resolved_config"), "V-RAE resolved_config")
    config = _stage1_evaluation_config(plan, raw_config)
    stage1 = VRAE.from_config(
        config,
        project_paths=_model_project_paths(plan, raw_config),
    ).to(device)
    stage1.load_state_dict(payload["model"], strict=True)
    stage1_weights = str(_weight_selection(plan)["stage1"])
    if stage1_weights == "ema":
        trainable = nn.ModuleDict(stage1.trainable_groups())
        _ema_to_model(trainable, payload)
    return stage1.requires_grad_(False).eval()


def _stage1_identity(plan: Any, payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    metadata = _mapping(payload.get("model_metadata"), "downstream model_metadata")
    configured = _mapping(config.get("stage1"), "downstream stage1 config")
    identity = str(metadata.get("stage1_checkpoint", configured.get("checkpoint", "")))
    if not identity:
        raise ValueError("downstream checkpoint does not record its V-RAE checkpoint identity")
    if Path(identity).is_absolute():
        raise ValueError("downstream V-RAE identity must be canonical and project-relative")
    # The selected file may be a renamed release artifact (for example
    # ckpts/vrae/vrae_eupe.pt).  The sampling loader validates its architecture
    # and weights against the downstream metadata; retain the original identity
    # here for latent-normalizer and DiT metadata comparisons.
    return identity


def _sampling_overrides(
    plan: Any, config: dict[str, Any], *, stage1_identity: str
) -> dict[str, Any]:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    result = copy.deepcopy(config)
    result.setdefault("sampling", {})
    result["sampling"].update(
        base_seed=int(evaluation.get("base_seed", 3407)),
        steps=int(evaluation.get("sampler_steps", 50)),
        cfg_scale=float(evaluation.get("guidance", 1.0)),
        internal_guidance_scale=float(evaluation.get("internal_guidance_scale", 1.0)),
        internal_guidance_t_min=float(
            evaluation.get(
                "internal_guidance_t_min",
                result["sampling"].get("internal_guidance_t_min", 0.0),
            )
        ),
        internal_guidance_t_max=float(
            evaluation.get(
                "internal_guidance_t_max",
                result["sampling"].get("internal_guidance_t_max", 1.0),
            )
        ),
    )
    result.setdefault("stage1", {})
    result["stage1"]["checkpoint"] = stage1_identity
    result["stage1"]["weights"] = str(_weight_selection(plan)["stage1"])
    return result


def load_generation_stack(
    plan: Any, device: torch.device
) -> tuple[Any, nn.Module, Any, dict[str, Any]]:
    from vrae.training.common.engine import load_class_conditional_sampling_stack

    payload = load_checkpoint(plan.model_checkpoints["dit"], map_location="cpu")
    raw_config = dict(_mapping(payload.get("resolved_config"), "DiT resolved_config"))
    config = _sampling_overrides(
        plan,
        raw_config,
        stage1_identity=_stage1_identity(plan, payload, raw_config),
    )
    stack = load_class_conditional_sampling_stack(
        config,
        _model_project_paths(plan, raw_config),
        plan.model_checkpoints["dit"],
        device,
        stage1_checkpoint_override=plan.model_checkpoints["stage1"],
        latent_normalizer_override=plan.model_checkpoints.get("latent_normalizer"),
    )
    downstream_weights = str(_weight_selection(plan)["downstream"])
    if downstream_weights == "model":
        stack[1].load_state_dict(payload["model"], strict=True)
    return (*stack, config)


def load_prediction_stack(
    plan: Any, device: torch.device
) -> tuple[Any, nn.Module, Any, dict[str, Any]]:
    from vrae.training.cityscapes_video_pred.sample import load_sampling_stack

    payload = load_checkpoint(plan.model_checkpoints["prediction"], map_location="cpu")
    raw_config = dict(_mapping(payload.get("resolved_config"), "prediction resolved_config"))
    config = _sampling_overrides(
        plan,
        raw_config,
        stage1_identity=_stage1_identity(plan, payload, raw_config),
    )
    stack = load_sampling_stack(
        config,
        _model_project_paths(plan, raw_config),
        plan.model_checkpoints["prediction"],
        device,
    )
    downstream_weights = str(_weight_selection(plan)["downstream"])
    if downstream_weights == "model":
        stack[1].load_state_dict(payload["model"], strict=True)
    return (*stack, config)


def load_i3d(plan: Any, device: torch.device) -> nn.Module:
    checkpoint_root = plan.project_root / "ckpts" / "eval_models"
    return (
        I3DFeatureExtractor(
            plan.extractor_checkpoints["i3d"],
            checkpoint_root=checkpoint_root,
            batch_size=_evaluation_microbatch_size(plan, "i3d_batch_size"),
        )
        .to(device)
        .eval()
    )


def load_inception(plan: Any, device: torch.device) -> nn.Module:
    checkpoint_root = plan.project_root / "ckpts" / "eval_models"
    return (
        InceptionFeatureExtractor(
            plan.extractor_checkpoints["inception"],
            checkpoint_root=checkpoint_root,
        )
        .to(device)
        .eval()
    )


def load_lpips(plan: Any, device: torch.device) -> nn.Module:
    checkpoint_root = plan.project_root / "ckpts" / "eval_models"
    return (
        LPIPSMetric(
            plan.extractor_checkpoints["lpips"],
            checkpoint_root=checkpoint_root,
        )
        .to(device)
        .eval()
    )


def _global_batch_size(plan: Any) -> int:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    if "batch_size" in evaluation:
        raise ValueError(
            "evaluation.batch_size is no longer supported; use evaluation.global_batch_size"
        )
    value = evaluation.get("global_batch_size", DEFAULT_GLOBAL_BATCH_SIZE)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("evaluation.global_batch_size must be a positive integer")
    return value


def _local_batch_size(plan: Any, world_size: int) -> int:
    global_batch_size = _global_batch_size(plan)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if global_batch_size % world_size:
        raise ValueError(
            "evaluation.global_batch_size must be divisible by world_size; got "
            f"{global_batch_size} and {world_size}"
        )
    return global_batch_size // world_size


def fixed_global_batches(population_size: int, batch_size: int) -> list[list[int]]:
    """Define paired sample-ID batches before sharding each batch across ranks."""

    if population_size <= 0 or batch_size <= 0:
        raise ValueError("population_size and batch_size must be positive")
    return [
        list(range(start, min(start + batch_size, population_size)))
        for start in range(0, population_size, batch_size)
    ]


def _atomic_tensor(path: Path, value: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value.detach().cpu(), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_mp4(path: Path, video: Tensor, *, fps: int) -> None:
    """Write the same imageio quality-9 MP4 used by the uni-vug sampler."""

    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError("uni-vug UCF101 gFVD requires imageio") from error
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError("generated MP4 video must have shape [T,3,H,W]")
    frames = (
        video.detach()
        .cpu()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
    try:
        imageio.mimwrite(
            str(temporary),
            frames,
            fps=int(fps),
            quality=9,
            macro_block_size=1,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_ucf101_generated_samples(
    plan: Any,
    samples: Sequence[tuple[int, Tensor, Mapping[str, Any] | None]],
) -> None:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    fps = int(evaluation.get("fps", 8))
    for sample_id, video, record in samples:
        _atomic_mp4(
            plan.output_directory / "samples" / f"sample-{int(sample_id):06d}.mp4",
            video,
            fps=fps,
        )
        if record is not None:
            _atomic_json(
                plan.output_directory / "samples" / f"sample-{int(sample_id):06d}.json",
                record,
            )


def _rank_zero_write_samples(
    plan: Any,
    samples: Sequence[tuple[int, Tensor, Mapping[str, Any] | None]],
    *,
    rank: int,
    world_size: int,
) -> None:
    local = [
        (int(sample_id), video.detach().cpu(), None if record is None else dict(record))
        for sample_id, video, record in samples
    ]
    if world_size > 1:
        gathered: list[Any] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank != 0:
            return
        local = [item for shard in gathered or [] for item in shard]
    if rank == 0:
        for sample_id, video, record in local:
            _atomic_tensor(
                plan.output_directory / "samples" / f"sample-{sample_id:06d}.pt",
                video,
            )
            if record is not None:
                _atomic_json(
                    plan.output_directory / "samples" / f"sample-{sample_id:06d}.json",
                    record,
                )


def _feature_rows(extractor: nn.Module, video: Tensor, device: torch.device) -> Tensor:
    result = extractor(video.to(device, non_blocking=True)).detach()
    if result.ndim != 2 or not torch.isfinite(result).all():
        raise ValueError("feature extractor must return finite [N,D] values")
    return result


def _gather(
    ids: list[int], rows: list[Tensor], dimension: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    feature_tensor = (
        torch.cat(rows) if rows else torch.empty((0, dimension), dtype=torch.float32, device=device)
    )
    return gather_indexed_features(id_tensor, feature_tensor)


def _require_exact_ids(ids: Tensor, count: int, name: str) -> None:
    expected = torch.arange(count, dtype=torch.long, device=ids.device)
    if not torch.equal(ids, expected):
        raise ValueError(f"{name} gathered IDs are not exactly [0, {count})")


def _score_pair(
    plan: Any,
    name: str,
    ids_a: Tensor,
    features_a: Tensor,
    ids_b: Tensor,
    features_b: Tensor,
) -> float:
    count = len(ids_a)
    _require_exact_ids(ids_a, count, f"{name} real")
    _require_exact_ids(ids_b, count, f"{name} fake")
    if not torch.equal(ids_a, ids_b) or features_a.shape != features_b.shape:
        raise ValueError(f"{name} real/fake gathered populations differ")
    if plan.task in {"ucf101_tfvd", "k600_rfvd", "k600_tfvd"}:
        # Keep the locked fake-first operand and solver order.
        return frechet_from_features(
            features_b,
            features_a,
            covariance=plan.protocol.covariance,
            implementation="torch_svd",
        )
    if plan.task == "ucf101_gfvd":
        return frechet_from_features(
            features_a,
            features_b,
            covariance=plan.protocol.covariance,
            implementation="scipy_sqrtm",
        )
    return frechet_from_features(features_a, features_b, covariance=plan.protocol.covariance)


def _save_features(
    plan: Any,
    name: str,
    ids: Tensor,
    features: Tensor,
    *,
    rank: int,
) -> None:
    if rank != 0:
        return
    save_feature_cache(
        plan.output_directory / "features" / name,
        sample_ids=ids,
        features=features,
        metadata=_feature_cache_metadata(plan, name),
    )


def _feature_cache_metadata(plan: Any, name: str) -> dict[str, Any]:
    def relative(path: Path) -> str:
        return path.relative_to(plan.project_root).as_posix()

    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    metadata = {
        "task": plan.task,
        "mode": plan.mode,
        "feature_name": name,
        "covariance": plan.protocol.covariance,
        "source": "end_to_end",
        "protocol": plan.protocol.metadata(),
        "population_source": inputs.get("population"),
        "evaluation": dict(evaluation),
        "weights": _weight_selection(plan),
        "model_checkpoints": {key: relative(path) for key, path in plan.model_checkpoints.items()},
        "extractor_checkpoints": {
            key: relative(path) for key, path in plan.extractor_checkpoints.items()
        },
    }
    return json.loads(json.dumps(metadata))


def _load_feature_pair_cache(
    plan: Any,
    real_name: str,
    fake_name: str,
    *,
    count: int,
    dimension: int,
) -> tuple[Tensor, Tensor, Tensor] | None:
    real_directory = plan.output_directory / "features" / real_name
    fake_directory = plan.output_directory / "features" / fake_name
    real_exists = (real_directory / "features.pt").is_file()
    fake_exists = (fake_directory / "features.pt").is_file()
    if real_exists != fake_exists:
        raise ValueError("resume found only one side of an end-to-end feature cache pair")
    if not real_exists:
        return None
    real_ids, real = load_feature_cache(
        real_directory,
        expected_metadata=_feature_cache_metadata(plan, real_name),
    )
    fake_ids, fake = load_feature_cache(
        fake_directory,
        expected_metadata=_feature_cache_metadata(plan, fake_name),
    )
    _require_exact_ids(real_ids, count, real_name)
    _require_exact_ids(fake_ids, count, fake_name)
    if not torch.equal(real_ids, fake_ids):
        raise ValueError("cached real/fake feature IDs differ")
    expected_shape = (count, dimension)
    if tuple(real.shape) != expected_shape or tuple(fake.shape) != expected_shape:
        raise ValueError(f"cached feature pair must have shape {expected_shape}")
    return real_ids, real, fake


class _RFVDRecordDataset(Dataset[dict[str, Any]]):
    """Decode one fixed-protocol clip per item for DataLoader worker streaming."""

    def __init__(self, plan: Any, records: Sequence[Mapping[str, Any]]) -> None:
        self.plan = plan
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "sample_id": int(index),
            "video": decode_record_clip(
                self.plan,
                self.records[index],
                frames=int(self.plan.protocol.num_frames),
                interval=int(self.plan.protocol.frame_interval),
            ),
        }


class _TFVDRecordDataset(_RFVDRecordDataset):
    """Decode one formal tFVD clip using the protocol's frame interval."""


class _CityscapesRFVDRecordDataset(Dataset[dict[str, Any]]):
    """Decode one fixed Cityscapes context/future pair per worker item."""

    def __init__(self, plan: Any, records: Sequence[Mapping[str, Any]]) -> None:
        self.plan = plan
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        context, future = _decode_city_record(self.plan, self.records[index])
        return {
            "sample_id": int(index),
            "context": context,
            "future": future,
        }


def _collate_item_stream(loader: DataLoader, batch_size: int):
    pending: list[dict[str, Any]] = []
    for item in loader:
        pending.append(item)
        if len(pending) == batch_size:
            yield default_collate(pending)
            pending.clear()
    if pending:
        yield default_collate(pending)


def _progress_batches(iterator: Any, *, total: int, rank: int, task: str) -> Any:
    if rank != 0:
        return iterator
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterator
    description = (
        "rFVD rank0" if task in {"ucf101_rfvd", "k600_rfvd", "cityscapes_rfvd"} else f"{task} rank0"
    )
    return tqdm(iterator, total=total, desc=description, dynamic_ncols=True)


def _progress_fvd_batches(
    iterator: Any,
    *,
    total_records: int,
    global_batch_size: int,
    rank: int,
    task: str,
) -> Any:
    """Show rank-zero progress in global videos rather than local batches."""

    if rank != 0:
        yield from iterator
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterator
        return
    descriptions = {
        "ucf101_rfvd": "UCF101 rFVD",
        "k600_rfvd": "K600 rFVD",
        "cityscapes_rfvd": "Cityscapes rFVD",
        "ucf101_tfvd": "UCF101 tFVD",
        "k600_tfvd": "K600 tFVD",
    }
    progress = tqdm(
        total=total_records,
        desc=descriptions.get(task, "FVD"),
        unit="video",
        dynamic_ncols=True,
    )
    try:
        for batch in iterator:
            yield batch
            progress.update(min(global_batch_size, total_records - progress.n))
    finally:
        progress.close()


def _rfvd_worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _configure_rfvd_runtime(plan: Any) -> None:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    seed = int(evaluation.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(evaluation.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(evaluation.get("cudnn_deterministic", False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_real_feature_cache_directory(plan: Any) -> Path | None:
    if plan.task != "k600_rfvd" or plan.mode != "formal":
        return None
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    configured = evaluation.get("real_feature_cache")
    if configured is False:
        return None
    if configured not in {None, ""}:
        path = Path(str(configured)).expanduser()
        return path.resolve() if path.is_absolute() else (plan.project_root / path).resolve()
    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    population = _project_file(plan, inputs.get("population"), "population")
    tag = _sha256(population)[:12]
    return plan.project_root / "outputs" / plan.task / "_cache" / f"real_i3d_{tag}"


def _shared_real_feature_cache_metadata(plan: Any) -> dict[str, Any]:
    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    population = _project_file(plan, inputs.get("population"), "population")

    def readable(path: Path) -> str:
        try:
            return path.relative_to(plan.project_root).as_posix()
        except ValueError:
            return str(path)

    return {
        "version": 1,
        "task": plan.task,
        "mode": plan.mode,
        "feature_name": "real_i3d",
        "source": "shared_end_to_end_real_population",
        "protocol": plan.protocol.metadata(),
        "population_source": readable(population),
        "population_sha256": _sha256(population),
        "data_root": str(inputs.get("data_root")),
        "video_backend": _video_backend(plan),
        "seek_mode": _video_seek_mode(plan),
        "decode_threads": int(evaluation.get("decode_threads", 1)),
        "resize": "short_side_256_bilinear_antialias_true",
        "center_crop_rounding": "round",
        "extractor_checkpoint": readable(plan.extractor_checkpoints["i3d"]),
    }


def _load_shared_real_feature_cache(
    plan: Any,
    *,
    count: int,
    rank: int,
) -> tuple[bool, Path | None, Tensor | None, Tensor | None]:
    directory = _shared_real_feature_cache_directory(plan) if rank == 0 else None
    ids: Tensor | None = None
    features: Tensor | None = None
    state: dict[str, str]
    if rank == 0 and directory is not None:
        payload_exists = (directory / "features.pt").is_file()
        metadata_exists = (directory / "metadata.json").is_file()
        if payload_exists != metadata_exists:
            state = {"status": "error", "message": f"partial real feature cache: {directory}"}
        elif not payload_exists:
            state = {"status": "missing", "message": str(directory)}
        else:
            try:
                ids, features = load_feature_cache(
                    directory,
                    expected_metadata=_shared_real_feature_cache_metadata(plan),
                )
                _require_exact_ids(ids, count, "shared real_i3d")
                if tuple(features.shape) != (count, 400):
                    raise ValueError(f"shared real_i3d must have shape {(count, 400)}")
            except Exception as error:
                state = {"status": "error", "message": str(error)}
            else:
                state = {"status": "loaded", "message": str(directory)}
    elif rank == 0:
        state = {"status": "disabled", "message": ""}
    else:
        state = {"status": "pending", "message": ""}

    state = broadcast_object(state if rank == 0 else None, source=0)
    if state["status"] == "error":
        raise ValueError(f"unable to reuse K600 real feature cache: {state['message']}")
    return state["status"] == "loaded", directory, ids, features


@torch.inference_mode()
def _run_rfvd(
    plan: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[dict[str, float], bool]:
    _configure_rfvd_runtime(plan)
    stage1 = load_stage1(plan, device)
    i3d = load_i3d(plan, device)
    lpips = load_lpips(plan, device)
    autocast = _evaluation_autocast_policy(plan, device)
    paired = PairedMetricAccumulator(
        device=device,
        batch_size=_evaluation_microbatch_size(plan, "perceptual_batch_size"),
        lpips_model=lpips,
    )
    ids: list[int] = []
    real_features: list[Tensor] = []
    fake_features: list[Tensor] = []
    indices = list(exact_shard(len(records), rank, world_size))
    batch_size = _local_batch_size(plan, world_size)
    real_cache_reused, real_cache_directory, cached_real_ids, cached_real_values = (
        _load_shared_real_feature_cache(plan, count=len(records), rank=rank)
    )
    local_records = _RFVDRecordDataset(plan, [records[index] for index in indices])
    workers = int(_mapping(plan.config.get("evaluation", {}), "evaluation").get("num_workers", 0))
    prefetch = int(
        _mapping(plan.config.get("evaluation", {}), "evaluation").get("prefetch_factor", 1)
    )
    loader_options: dict[str, Any] = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": _rfvd_worker_init_fn,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = prefetch
    loader = DataLoader(local_records, **loader_options)
    batches = _collate_item_stream(loader, batch_size)
    for batch in _progress_fvd_batches(
        batches,
        total_records=len(records),
        global_batch_size=_global_batch_size(plan),
        rank=rank,
        task=plan.task,
    ):
        local_ids = [int(value) for value in batch["sample_id"].tolist()]
        batch_ids = [indices[index] for index in local_ids]
        real = batch["video"].to(device, non_blocking=True)
        with autocast.autocast():
            reconstructed = stage1(real)["recon"]
        reconstructed = reconstructed.clamp(0, 1)
        if reconstructed.shape != real.shape:
            raise ValueError("V-RAE reconstruction shape differs from its real input")
        if not real_cache_reused:
            real_features.append(_feature_rows(i3d, real, device))
        fake_features.append(_feature_rows(i3d, reconstructed, device))
        paired.update(real, reconstructed)
        ids.extend(batch_ids)
        del real, reconstructed
    fake_ids, fake_values = _gather(ids, fake_features, 400, device)
    if real_cache_reused:
        if rank == 0:
            if cached_real_ids is None or cached_real_values is None:
                raise RuntimeError("rank 0 did not retain the loaded real feature cache")
            real_ids = cached_real_ids.to(device=device)
            real_values = cached_real_values
        else:
            real_ids = torch.empty(0, dtype=torch.long, device=device)
            real_values = torch.empty((0, 400), dtype=torch.float32, device=device)
    else:
        real_ids, real_values = _gather(ids, real_features, 400, device)
    paired_metrics = paired.compute(synchronize=True)
    if rank != 0:
        return {}, real_cache_reused
    if not real_cache_reused and real_cache_directory is not None:
        save_feature_cache(
            real_cache_directory,
            sample_ids=real_ids,
            features=real_values,
            metadata=_shared_real_feature_cache_metadata(plan),
        )
    if plan.task not in {"ucf101_rfvd", "k600_rfvd"}:
        _save_features(plan, "real_i3d", real_ids, real_values, rank=rank)
        _save_features(plan, "reconstructed_i3d", fake_ids, fake_values, rank=rank)
    return (
        {
            "rFVD": _score_pair(plan, "rFVD", real_ids, real_values, fake_ids, fake_values),
            **paired_metrics,
        },
        real_cache_reused,
    )


@torch.inference_mode()
def _run_cityscapes_rfvd(
    plan: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    from vrae.evaluation.cityscapes_rfvd.evaluator import reconstruct_future_variants

    _configure_rfvd_runtime(plan)
    stage1 = load_stage1(plan, device)
    i3d = load_i3d(plan, device)
    autocast = _evaluation_autocast_policy(plan, device)
    ids: list[int] = []
    real_features: list[Tensor] = []
    future_only_features: list[Tensor] = []
    context_conditioned_features: list[Tensor] = []
    indices = list(exact_shard(len(records), rank, world_size))
    batch_size = _local_batch_size(plan, world_size)
    local_records = _CityscapesRFVDRecordDataset(
        plan,
        [records[index] for index in indices],
    )
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    workers = int(evaluation.get("num_workers", 0))
    prefetch = int(evaluation.get("prefetch_factor", 1))
    loader_options: dict[str, Any] = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": _rfvd_worker_init_fn,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = prefetch
    loader = DataLoader(local_records, **loader_options)
    batches = _collate_item_stream(loader, batch_size)
    total_batches = (len(indices) + batch_size - 1) // batch_size
    for batch in _progress_batches(batches, total=total_batches, rank=rank, task=plan.task):
        local_ids = [int(value) for value in batch["sample_id"].tolist()]
        batch_ids = [indices[index] for index in local_ids]
        context = batch["context"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        with autocast.autocast():
            reconstructions = reconstruct_future_variants(stage1, context, future)
        future_only = reconstructions["future_only"].float().clamp(0, 1)
        context_conditioned = reconstructions["context_conditioned"].float().clamp(0, 1)
        if future_only.shape != future.shape or context_conditioned.shape != future.shape:
            raise ValueError("Cityscapes V-RAE reconstruction shape differs from future input")
        real_features.append(_feature_rows(i3d, future, device))
        future_only_features.append(_feature_rows(i3d, future_only, device))
        context_conditioned_features.append(_feature_rows(i3d, context_conditioned, device))
        ids.extend(batch_ids)
        del context, future, future_only, context_conditioned, reconstructions

    real_ids, real_values = _gather(ids, real_features, 400, device)
    future_only_ids, future_only_values = _gather(ids, future_only_features, 400, device)
    conditioned_ids, conditioned_values = _gather(
        ids,
        context_conditioned_features,
        400,
        device,
    )
    if rank != 0:
        return {}
    _save_features(plan, "real_future_i3d", real_ids, real_values, rank=rank)
    _save_features(
        plan,
        "reconstructed_future_only_i3d",
        future_only_ids,
        future_only_values,
        rank=rank,
    )
    _save_features(
        plan,
        "reconstructed_context_conditioned_i3d",
        conditioned_ids,
        conditioned_values,
        rank=rank,
    )
    return {
        "rFVD_future_only": _score_pair(
            plan,
            "rFVD_future_only",
            real_ids,
            real_values,
            future_only_ids,
            future_only_values,
        ),
        "rFVD_context_conditioned": _score_pair(
            plan,
            "rFVD_context_conditioned",
            real_ids,
            real_values,
            conditioned_ids,
            conditioned_values,
        ),
    }


@torch.no_grad()
def _run_tfvd(
    plan: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    _configure_rfvd_runtime(plan)
    stage1 = load_stage1(plan, device)
    i3d = load_i3d(plan, device)
    autocast = _evaluation_autocast_policy(plan, device)
    ids: list[int] = []
    reference_features: list[Tensor] = []
    interpolated_features: list[Tensor] = []
    indices = list(exact_shard(len(records), rank, world_size))
    batch_size = _local_batch_size(plan, world_size)
    local_records = _TFVDRecordDataset(plan, [records[index] for index in indices])
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    workers = int(evaluation.get("num_workers", 0))
    prefetch = int(evaluation.get("prefetch_factor", 1))
    loader_options: dict[str, Any] = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": _rfvd_worker_init_fn,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = prefetch
    loader = DataLoader(local_records, **loader_options)
    batches = _collate_item_stream(loader, batch_size)
    for batch in _progress_fvd_batches(
        batches,
        total_records=len(records),
        global_batch_size=_global_batch_size(plan),
        rank=rank,
        task=plan.task,
    ):
        local_ids = [int(value) for value in batch["sample_id"].tolist()]
        batch_ids = [indices[index] for index in local_ids]
        source = batch["video"].to(device, non_blocking=True)
        with autocast.autocast():
            clean = stage1.encode(source)
        if clean.shape[1] != 6:
            raise ValueError("V-RAE must produce six chunks from 24 tFVD frames")

        # The four interpolated chunks decode to the temporal span covered by
        # source frames 4..19. Compare them with those aligned ground-truth
        # frames, not with a clean encode/decode reconstruction.
        reference = source[:, 4:20]
        expected_shape = (source.shape[0], 16, 3, *plan.protocol.image_size)
        if tuple(reference.shape) != expected_shape:
            raise ValueError(
                f"temporal reference must have shape {expected_shape}, got {tuple(reference.shape)}"
            )
        reference_features.append(_feature_rows(i3d, reference, device))
        del reference

        with autocast.autocast():
            interpolated = stage1.decode(centered_six_to_four(clean))
        interpolated = interpolated.clamp_(0.0, 1.0)
        if tuple(interpolated.shape) != expected_shape:
            raise ValueError(
                f"temporal interpolation must have shape {expected_shape}, "
                f"got {tuple(interpolated.shape)}"
            )
        interpolated_features.append(_feature_rows(i3d, interpolated, device))
        ids.extend(batch_ids)
        del source, clean, interpolated
    reference_ids, reference_values = _gather(ids, reference_features, 400, device)
    interpolated_ids, interpolated_values = _gather(ids, interpolated_features, 400, device)
    if rank != 0:
        return {}
    _save_features(plan, "reference_i3d", reference_ids, reference_values, rank=rank)
    _save_features(plan, "interpolated_i3d", interpolated_ids, interpolated_values, rank=rank)
    return {
        "tFVD": _score_pair(
            plan,
            "tFVD",
            reference_ids,
            reference_values,
            interpolated_ids,
            interpolated_values,
        )
    }


def _generation_records(plan: Any, count: int) -> list[dict[str, Any]]:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    base_seed = int(evaluation.get("base_seed", 3407))
    if plan.task == "ucf101_gfvd":
        return ucf101_generation_population(count=count, base_seed=base_seed)
    return k600_balanced_population(base_seed=base_seed)[:count]


def _annotate_generation_records(
    plan: Any,
    generated: list[dict[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    model_frames: int,
) -> None:
    evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
    generation_batch_size = (
        int(evaluation.get("generation_batch_size", 64))
        if plan.task == "ucf101_gfvd"
        else _global_batch_size(plan)
    )
    for record, source in zip(generated, sources, strict=True):
        record.update(
            split=plan.protocol.split,
            source_sample_id=str(source["sample_id"]),
            sampler_steps=int(evaluation.get("sampler_steps", 50)),
            guidance=float(evaluation.get("guidance", 1.0)),
            internal_guidance_scale=float(evaluation.get("internal_guidance_scale", 1.0)),
            generation_batch_size=generation_batch_size,
            model_frames=model_frames,
            i3d_frames=int(plan.protocol.num_frames),
        )


def _load_generated_sample(plan: Any, sample_id: int, *, model_frames: int) -> Tensor:
    if plan.task == "ucf101_gfvd":
        path = plan.output_directory / "samples" / f"sample-{sample_id:06d}.mp4"
        value = uint8_to_float(
            _load_center_square_lanczos_clip(
                path,
                list(range(plan.protocol.num_frames)),
                image_size=plan.protocol.image_size,
            )
        )
        expected = (plan.protocol.num_frames, 3, *plan.protocol.image_size)
        if tuple(value.shape) != expected:
            raise ValueError(f"existing generated sample {path} must have shape {expected}")
        return value.contiguous()
    path = plan.output_directory / "samples" / f"sample-{sample_id:06d}.pt"
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected = (model_frames, 3, *plan.protocol.image_size)
    if not isinstance(value, Tensor) or tuple(value.shape) != expected:
        raise ValueError(f"existing generated sample {path} must have shape {expected}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"existing generated sample {path} must be a finite floating tensor")
    if value.min().item() < 0 or value.max().item() > 1:
        raise ValueError(f"existing generated sample {path} must be in RGB range [0,1]")
    return value.contiguous()


def _validate_generation_resume(
    plan: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    model_frames: int,
) -> set[int]:
    sample_directory = plan.output_directory / "samples"
    existing_ids: set[int] = set()
    metadata_ids: set[int] = set()
    sample_suffix = ".mp4" if plan.task == "ucf101_gfvd" else ".pt"
    if sample_directory.is_dir():
        for path in sample_directory.iterdir():
            if path.name.startswith("."):
                raise ValueError(f"unfinished temporary sample exists: {path}")
            if (
                not path.is_file()
                or path.suffix not in {sample_suffix, ".json"}
                or not path.stem.startswith("sample-")
            ):
                raise ValueError(f"unexpected generated sample artifact: {path}")
            suffix = path.stem.removeprefix("sample-")
            if len(suffix) != 6 or not suffix.isdigit():
                raise ValueError(f"invalid generated sample filename: {path.name}")
            sample_id = int(suffix)
            if sample_id >= len(expected):
                raise ValueError(f"generated sample ID is outside the protocol: {sample_id}")
            target = existing_ids if path.suffix == sample_suffix else metadata_ids
            if sample_id in target:
                raise ValueError(f"duplicated generated sample artifact ID: {sample_id}")
            target.add(sample_id)
    if existing_ids != metadata_ids:
        raise ValueError("every existing generated sample must have one metadata sidecar")

    required = (
        "sample_id",
        "label",
        "seed",
        "filename",
        "split",
        "source_sample_id",
        "sampler_steps",
        "guidance",
        "internal_guidance_scale",
        "generation_batch_size",
        "model_frames",
        "i3d_frames",
    )
    for sample_id in sorted(existing_ids):
        _load_generated_sample(plan, sample_id, model_frames=model_frames)
        metadata_path = sample_directory / f"sample-{sample_id:06d}.json"
        metadata = _mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")), "sample metadata"
        )
        reference = expected[sample_id]
        if any(metadata.get(key) != reference.get(key) for key in required):
            raise ValueError(
                f"existing sample metadata {sample_id} differs from deterministic protocol"
            )

    population_path = plan.output_directory / "population.jsonl"
    if population_path.is_file() and population_path.read_text(encoding="utf-8").strip():
        records = read_population(population_path)
        ids = [int(record.get("sample_id", -1)) for record in records]
        if len(ids) != len(set(ids)) or any(sample_id < 0 for sample_id in ids):
            raise ValueError("existing generation population has duplicate or invalid IDs")
        if set(ids) != existing_ids:
            raise ValueError("existing population IDs must exactly match saved sample IDs")
        for record in records:
            sample_id = int(record["sample_id"])
            if sample_id >= len(expected):
                raise ValueError("existing population sample ID is outside the protocol")
            reference = expected[sample_id]
            if any(record.get(key) != reference.get(key) for key in required):
                raise ValueError(
                    f"existing population record {sample_id} differs from deterministic protocol"
                )
            if sample_id not in existing_ids:
                raise ValueError(f"population record {sample_id} has no validated sample file")
    return existing_ids


@torch.no_grad()
def _run_gfvd(
    plan: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    resume: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], bool]:
    from vrae.training.common.engine import generate_class_conditional
    from vrae.models.dit.transport import FlowMatchingTransport

    stage1, dit, normalizer, config = load_generation_stack(plan, device)
    transport = FlowMatchingTransport(**dict(config.get("transport", {})))
    i3d = load_i3d(plan, device)
    autocast = _evaluation_autocast_policy(plan, device)
    generated_records = _generation_records(plan, len(records))
    if plan.task == "k600_gfvd":
        real_labels = [int(record.get("label", -1)) for record in records]
        target_labels = [int(record["label"]) for record in generated_records]
        if real_labels != target_labels:
            raise ValueError("real population label assignment differs from deterministic protocol")
    model_frames = int(config["data"]["num_frames"])
    expected_model_frames = int(
        _mapping(plan.config.get("evaluation", {}), "evaluation").get("model_frames", 20)
    )
    if model_frames != expected_model_frames:
        raise ValueError("DiT checkpoint frame count differs from evaluation protocol")
    _annotate_generation_records(plan, generated_records, records, model_frames=model_frames)
    existing_ids = (
        _validate_generation_resume(plan, generated_records, model_frames=model_frames)
        if resume
        else set()
    )
    if resume and existing_ids == set(range(len(records))):
        cached = _load_feature_pair_cache(
            plan,
            "real_i3d",
            "generated_i3d",
            count=len(records),
            dimension=400,
        )
        if cached is not None:
            cached_ids, cached_real, cached_generated = cached
            return (
                {
                    "gFVD": _score_pair(
                        plan,
                        "gFVD",
                        cached_ids,
                        cached_real,
                        cached_ids,
                        cached_generated,
                    )
                },
                generated_records,
                True,
            )
    ids: list[int] = []
    real_features: list[Tensor] = []
    generated_features: list[Tensor] = []
    if plan.task == "ucf101_gfvd":
        evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
        generation_batch_size = int(evaluation.get("generation_batch_size", 64))
        if generation_batch_size <= 0:
            raise ValueError("evaluation.generation_batch_size must be positive")
        local_batch_size = generation_batch_size
        local_batches = fixed_global_batches(len(records), generation_batch_size)[rank::world_size]
    else:
        global_batch_size = _global_batch_size(plan)
        local_batch_size = _local_batch_size(plan, world_size)
        local_batches = [
            batch[rank::world_size]
            for batch in fixed_global_batches(len(records), global_batch_size)
        ]
    for batch_ids in _progress_batches(
        local_batches,
        total=len(local_batches),
        rank=rank,
        task=plan.task,
    ):
        if len(batch_ids) > local_batch_size:
            raise AssertionError("rank shard exceeds the resolved local batch size")
        if not batch_ids:
            _rank_zero_write_samples(plan, (), rank=rank, world_size=world_size)
            continue
        real = torch.stack(
            [
                decode_record_clip(
                    plan,
                    records[index],
                    frames=plan.protocol.num_frames if plan.task == "ucf101_gfvd" else model_frames,
                    require_explicit_indices=True,
                )
                for index in batch_ids
            ]
        ).to(device)
        slots: list[Tensor | None] = [None] * len(batch_ids)
        missing_positions = [
            position
            for position, sample_id in enumerate(batch_ids)
            if sample_id not in existing_ids
        ]
        for position, sample_id in enumerate(batch_ids):
            if sample_id in existing_ids:
                slots[position] = _load_generated_sample(plan, sample_id, model_frames=model_frames)
        new_samples: list[tuple[int, Tensor, Mapping[str, Any] | None]] = []
        if missing_positions:
            batch_population = [generated_records[sample_id] for sample_id in batch_ids]
            if plan.task == "ucf101_gfvd":
                batch_seed = ucf101_sample_seed(
                    int(config["sampling"]["base_seed"]),
                    int(batch_ids[0]),
                    stream=2,
                )
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(batch_seed)
            with autocast.autocast():
                generated_full_batch = generate_class_conditional(
                    dit,
                    stage1,
                    normalizer,
                    transport=transport,
                    sample_ids=batch_ids,
                    labels=torch.tensor(
                        [record["label"] for record in batch_population], dtype=torch.long
                    ),
                    base_seed=int(config["sampling"]["base_seed"]),
                    num_chunks=int(dit.num_chunks),
                    grid_size=tuple(dit.grid_size),
                    channels=int(dit.in_channels),
                    steps=int(config["sampling"]["steps"]),
                    cfg_scale=float(config["sampling"]["cfg_scale"]),
                    internal_guidance_scale=float(config["sampling"]["internal_guidance_scale"]),
                    seed_protocol=(
                        "ucf101-gfvd-global-sample-v1"
                        if plan.task == "ucf101_gfvd"
                        else "splitmix64"
                    ),
                    internal_guidance_t_min=float(
                        config["sampling"].get("internal_guidance_t_min", 0.0)
                    ),
                    internal_guidance_t_max=float(
                        config["sampling"].get("internal_guidance_t_max", 1.0)
                    ),
                    device=device,
                )
            generated_full_batch = generated_full_batch.float().clamp(0, 1)
            for position in missing_positions:
                sample_id = batch_ids[position]
                video = generated_full_batch[position]
                if plan.task == "ucf101_gfvd":
                    if video.shape[0] < plan.protocol.num_frames:
                        raise ValueError("generated UCF101 video is shorter than 17 frames")
                    video = video[: plan.protocol.num_frames]
                slots[position] = video
                new_samples.append((sample_id, video, generated_records[sample_id]))
        if plan.task == "ucf101_gfvd":
            _write_ucf101_generated_samples(plan, new_samples)
            for position in missing_positions:
                slots[position] = _load_generated_sample(
                    plan,
                    batch_ids[position],
                    model_frames=model_frames,
                )
        else:
            _rank_zero_write_samples(
                plan,
                new_samples,
                rank=rank,
                world_size=world_size,
            )
        if any(video is None for video in slots):
            raise AssertionError("generation resume left an unfilled sample slot")
        generated = torch.stack([video for video in slots if video is not None]).to(device)
        if generated.shape != real.shape:
            raise ValueError("generated and real video populations have different shapes")
        if not torch.isfinite(generated).all():
            raise ValueError("generated video population contains non-finite values")
        i3d_real = real[:, :17] if plan.task == "k600_gfvd" else real
        i3d_generated = generated[:, :17] if plan.task == "k600_gfvd" else generated
        real_features.append(_feature_rows(i3d, i3d_real, device))
        generated_features.append(_feature_rows(i3d, i3d_generated, device))
        ids.extend(batch_ids)
    real_ids, real_values = _gather(ids, real_features, 400, device)
    generated_ids, generated_values = _gather(ids, generated_features, 400, device)
    if rank != 0:
        return {}, generated_records, False
    _save_features(plan, "real_i3d", real_ids, real_values, rank=rank)
    _save_features(plan, "generated_i3d", generated_ids, generated_values, rank=rank)
    return (
        {
            "gFVD": _score_pair(
                plan,
                "gFVD",
                real_ids,
                real_values,
                generated_ids,
                generated_values,
            )
        },
        generated_records,
        False,
    )


@torch.no_grad()
def _run_city(
    plan: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    from vrae.training.cityscapes_video_pred.sample import (
        decode_future_tokens,
        predict_future_tokens,
    )
    from vrae.models.dit.transport import FlowMatchingTransport

    stage1, dit, normalizer, config = load_prediction_stack(plan, device)
    transport = FlowMatchingTransport(**dict(config.get("transport", {})))
    i3d = load_i3d(plan, device)
    inception = load_inception(plan, device)
    ids: list[int] = []
    real_i3d: list[Tensor] = []
    fake_i3d: list[Tensor] = []
    frame_ids: list[int] = []
    real_inception: list[Tensor] = []
    fake_inception: list[Tensor] = []
    indices = list(exact_shard(len(records), rank, world_size))
    batch_size = _local_batch_size(plan, world_size)
    rounds = (max((len(records) + world_size - 1) // world_size, 1) + batch_size - 1) // batch_size
    for round_index in range(rounds):
        start = round_index * batch_size
        batch_ids = indices[start : start + batch_size]
        if not batch_ids:
            _rank_zero_write_samples(plan, (), rank=rank, world_size=world_size)
            continue
        decoded = [_decode_city_record(plan, records[index]) for index in batch_ids]
        context = torch.stack([value[0] for value in decoded]).to(device)
        future = torch.stack([value[1] for value in decoded]).to(device)
        raw_context = stage1.grid_to_tokens(stage1.encode_grid(context))
        raw_future = predict_future_tokens(
            dit,
            raw_context,
            normalizer,
            sample_indices=batch_ids,
            base_seed=int(config["sampling"]["base_seed"]),
            steps=int(config["sampling"]["steps"]),
            transport=transport,
            cfg_scale=float(config["sampling"]["cfg_scale"]),
            internal_guidance_scale=float(config["sampling"]["internal_guidance_scale"]),
        )
        predicted = decode_future_tokens(stage1, raw_future, grid_size=tuple(dit.grid_size))
        predicted = predicted.float().clamp(0, 1)
        if predicted.shape != future.shape:
            raise ValueError("Cityscapes prediction shape differs from real future")
        real_i3d.append(_feature_rows(i3d, future, device))
        fake_i3d.append(_feature_rows(i3d, predicted, device))
        real_frames = future.flatten(0, 1)
        fake_frames = predicted.flatten(0, 1)
        real_inception.append(_feature_rows(inception, real_frames, device))
        fake_inception.append(_feature_rows(inception, fake_frames, device))
        ids.extend(batch_ids)
        frame_ids.extend(sample_id * 12 + frame for sample_id in batch_ids for frame in range(12))
        _rank_zero_write_samples(
            plan,
            [
                (sample_id, video, None)
                for sample_id, video in zip(batch_ids, predicted.cpu(), strict=True)
            ],
            rank=rank,
            world_size=world_size,
        )
    real_ids, real_i3d_values = _gather(ids, real_i3d, 400, device)
    fake_ids, fake_i3d_values = _gather(ids, fake_i3d, 400, device)
    real_frame_ids, real_frame_values = _gather(frame_ids, real_inception, 2048, device)
    fake_frame_ids, fake_frame_values = _gather(frame_ids, fake_inception, 2048, device)
    if rank != 0:
        return {}
    _save_features(plan, "real_i3d", real_ids, real_i3d_values, rank=rank)
    _save_features(plan, "predicted_i3d", fake_ids, fake_i3d_values, rank=rank)
    _save_features(plan, "real_inception", real_frame_ids, real_frame_values, rank=rank)
    _save_features(plan, "predicted_inception", fake_frame_ids, fake_frame_values, rank=rank)
    return {
        "gFID": _score_pair(
            plan,
            "gFID",
            real_frame_ids,
            real_frame_values,
            fake_frame_ids,
            fake_frame_values,
        ),
        "gFVD": _score_pair(
            plan,
            "gFVD",
            real_ids,
            real_i3d_values,
            fake_ids,
            fake_i3d_values,
        ),
    }


def run_end_to_end(
    plan: Any, *, resume: bool = False
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Execute media decoding, model inference, feature extraction, and scoring."""

    context = initialize_distributed()
    records_value, population_metadata = _population_document(plan)
    records = _validate_population_source(plan, records_value, population_metadata)
    if plan.task == "k600_rfvd" and plan.mode == "formal" and context.world_size != 8:
        raise ValueError(f"formal k600_rfvd requires 8 processes, got {context.world_size}")
    if plan.task in {"cityscapes_gfid_gfvd", "cityscapes_rfvd"} and population_metadata.get(
        "data_root"
    ):
        for record in records:
            record.setdefault("data_root", population_metadata["data_root"])
    if context.is_main:
        plan.output_directory.mkdir(parents=True, exist_ok=True)
        if plan.task not in {"ucf101_rfvd", "k600_rfvd"}:
            for directory in ("features", "samples", "logs"):
                (plan.output_directory / directory).mkdir(exist_ok=True)
    barrier()
    feature_cache_reused = False
    if plan.task in {"ucf101_rfvd", "k600_rfvd"}:
        metrics, feature_cache_reused = _run_rfvd(
            plan,
            records,
            context.device,
            context.rank,
            context.world_size,
        )
        output_population = records
    elif plan.task == "cityscapes_rfvd":
        metrics = _run_cityscapes_rfvd(
            plan,
            records,
            context.device,
            context.rank,
            context.world_size,
        )
        output_population = records
    elif plan.task in {"ucf101_tfvd", "k600_tfvd"}:
        metrics = _run_tfvd(plan, records, context.device, context.rank, context.world_size)
        output_population = records
    elif plan.task in {"ucf101_gfvd", "k600_gfvd"}:
        metrics, output_population, feature_cache_reused = _run_gfvd(
            plan,
            records,
            context.device,
            context.rank,
            context.world_size,
            resume=resume,
        )
    else:
        metrics = _run_city(plan, records, context.device, context.rank, context.world_size)
        output_population = records
    barrier()
    if plan.task == "ucf101_gfvd":
        evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
        local_batch_size = int(evaluation.get("generation_batch_size", 64))
        runtime_global_batch_size = local_batch_size * context.world_size
        batch_assignment = "canonical_batches_round_robin"
    else:
        local_batch_size = _local_batch_size(plan, context.world_size)
        runtime_global_batch_size = _global_batch_size(plan)
        batch_assignment = "global_batch_rank_strided"
    metadata = {
        "rank": context.rank,
        "world_size": context.world_size,
        "global_batch_size": runtime_global_batch_size,
        "local_batch_size": local_batch_size,
        "batch_assignment": batch_assignment,
        "i3d_batch_size": _evaluation_microbatch_size(plan, "i3d_batch_size"),
        "population_metadata": population_metadata,
        "execution": "end_to_end",
        "resume": resume,
        "weights": _weight_selection(plan),
        "feature_cache_reused": feature_cache_reused,
    }
    if plan.task in {"ucf101_gfvd", "k600_gfvd"}:
        metadata["precision"] = _evaluation_precision_name(plan)
    if plan.task in {"ucf101_tfvd", "k600_tfvd"}:
        evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
        metadata.update(
            precision=_evaluation_precision_name(plan),
            attention_backend=_evaluation_attention_backend(plan),
            video_backend=_video_backend(plan),
            seek_mode=_video_seek_mode(plan),
            decode_threads=int(evaluation.get("decode_threads", 1)),
            num_workers=int(evaluation.get("num_workers", 0)),
            prefetch_factor=int(evaluation.get("prefetch_factor", 1)),
            seed=int(evaluation.get("seed", 42)),
            cudnn_benchmark=bool(evaluation.get("cudnn_benchmark", False)),
            cudnn_deterministic=bool(evaluation.get("cudnn_deterministic", False)),
            clamp_decode=True,
            center_crop_rounding="round",
            frechet_implementation="torch_svd",
        )
    if plan.task in {"ucf101_rfvd", "k600_rfvd", "cityscapes_rfvd"}:
        evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
        metadata.update(
            precision=_evaluation_precision_name(plan),
            attention_backend=_evaluation_attention_backend(plan),
            perceptual_batch_size=_evaluation_microbatch_size(plan, "perceptual_batch_size"),
            video_backend=_video_backend(plan),
            seek_mode=_video_seek_mode(plan),
            decode_threads=int(evaluation.get("decode_threads", 1)),
            num_workers=int(evaluation.get("num_workers", 0)),
            prefetch_factor=int(evaluation.get("prefetch_factor", 1)),
            seed=int(evaluation.get("seed", 42)),
            cudnn_benchmark=bool(evaluation.get("cudnn_benchmark", False)),
            cudnn_deterministic=bool(evaluation.get("cudnn_deterministic", False)),
            center_crop_rounding=(
                "round"
                if plan.task == "k600_rfvd"
                else ("none" if plan.task == "cityscapes_rfvd" else "floor")
            ),
            frechet_implementation=("torch_svd" if plan.task == "k600_rfvd" else "scipy_eigh"),
        )
    value = (metrics, output_population, metadata) if context.is_main else None
    return broadcast_object(value, source=0)


__all__ = [
    "decode_record_clip",
    "fixed_global_batches",
    "load_generation_stack",
    "load_i3d",
    "load_inception",
    "load_lpips",
    "load_prediction_stack",
    "load_stage1",
    "run_end_to_end",
]
