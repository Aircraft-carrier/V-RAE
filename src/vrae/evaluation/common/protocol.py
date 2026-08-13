"""Protocol-locked orchestration shared by all evaluation entrypoints."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from vrae.evaluation.common.frechet import frechet_from_features
from vrae.evaluation.common.population import (
    k600_balanced_population,
    read_population,
    ucf101_generation_population,
    write_population,
)
from vrae.config import load_config, save_resolved_config
from vrae.paths import ProjectPaths, find_project_root


class EvaluationConfigError(ValueError):
    """Raised when an evaluation config violates its locked protocol."""


DEFAULT_GLOBAL_BATCH_SIZE = 512
COMPACT_RESULT_TASKS = frozenset(
    {
        "ucf101_rfvd",
        "ucf101_gfvd",
        "ucf101_tfvd",
        "k600_rfvd",
        "k600_gfvd",
        "k600_tfvd",
    }
)
CHECKPOINT_SWEEP_CHECKPOINTS = {
    "ucf101_rfvd": "stage1",
    "ucf101_gfvd": "dit",
    "ucf101_tfvd": "stage1",
    "k600_rfvd": "stage1",
    "k600_tfvd": "stage1",
}
CHECKPOINT_SWEEP_TASKS = frozenset(CHECKPOINT_SWEEP_CHECKPOINTS)
_STEP_CHECKPOINT_PATTERN = re.compile(r"^step-(\d{8})\.pt$")
_CHECKPOINT_SWEEP_ALIASES = ("last.pt", "latest.pt")


@dataclass(frozen=True)
class EvaluationProtocol:
    task: str
    split: str
    population_size: int
    num_frames: int
    frame_interval: int
    image_size: tuple[int, int]
    covariance: str
    metrics: tuple[str, ...] = ()
    population: str = ""
    input_contract: str = ""
    complete_protocol: bool = True

    def metadata(self) -> dict[str, object]:
        return asdict(self)


PROTOCOLS = {
    "ucf101_rfvd": EvaluationProtocol(
        "ucf101_rfvd",
        "test",
        -1,
        16,
        3,
        (256, 256),
        "sample",
        ("rFVD", "LPIPS", "PSNR", "SSIM", "PSNR_global"),
        "complete UCF101 test split after explicit short-video filtering",
        "frame 0, interval 3, float bilinear antialias resize and center crop",
    ),
    "ucf101_gfvd": EvaluationProtocol(
        "ucf101_gfvd",
        "train",
        2048,
        17,
        3,
        (256, 256),
        "population",
        ("gFVD",),
        "uni-vug fvd2048_17f: 2,048 seeded UCF101-train clips and generated samples",
        "real stride 3; fake stride 1; center-square Lanczos; global-sample-v1 IDs",
    ),
    "ucf101_tfvd": EvaluationProtocol(
        "ucf101_tfvd",
        "test",
        -1,
        24,
        3,
        (256, 256),
        "sample",
        ("tFVD",),
        "complete UCF101 test split after explicit 24f/dt3 filtering",
        "frame 0, interval 3; ground-truth frames 4-19 vs 16 frames decoded from "
        "centered_6_to_4 interpolated chunks",
    ),
    "k600_rfvd": EvaluationProtocol(
        "k600_rfvd",
        "val",
        -1,
        16,
        3,
        (256, 256),
        "sample",
        ("rFVD", "LPIPS", "PSNR", "SSIM", "PSNR_global"),
        "complete Kinetics-600 validation split after explicit short-video filtering",
        "frame 0, interval 3, float bilinear antialias resize and center crop",
    ),
    "k600_gfvd": EvaluationProtocol(
        "k600_gfvd",
        "train",
        50_000,
        17,
        1,
        (256, 256),
        "population",
        ("gFVD",),
        "50,000 samples with exact 200x84 + 400x83 class quota",
        "real/fake label histograms match; 20-frame sources use [0:17] for I3D",
    ),
    "k600_tfvd": EvaluationProtocol(
        "k600_tfvd",
        "val",
        -1,
        24,
        3,
        (256, 256),
        "sample",
        ("tFVD",),
        "complete Kinetics-600 validation split after explicit 24f/dt3 filtering",
        "frame 0, interval 3; ground-truth frames 4-19 vs 16 frames decoded from "
        "centered_6_to_4 interpolated chunks",
    ),
    "cityscapes_gfid_gfvd": EvaluationProtocol(
        "cityscapes_gfid_gfvd",
        "val",
        500,
        12,
        1,
        (432, 768),
        "sample",
        ("gFID", "gFVD"),
        "exactly 500 Cityscapes validation clips",
        "frames 4-15 context and frames 16-27 real/predicted future at 16 fps",
    ),
    "cityscapes_rfvd": EvaluationProtocol(
        "cityscapes_rfvd",
        "val",
        500,
        12,
        1,
        (432, 768),
        "sample",
        ("rFVD_future_only", "rFVD_context_conditioned"),
        "exactly 500 Cityscapes validation future clips",
        (
            "frames 4-15 and 16-27 are encoded in separate 12-frame calls; "
            "score future-only and context-conditioned causal decode variants"
        ),
    ),
}

MODEL_CHECKPOINTS = {
    "ucf101_rfvd": ("stage1",),
    "ucf101_gfvd": ("stage1", "dit"),
    "ucf101_tfvd": ("stage1",),
    "k600_rfvd": ("stage1",),
    "k600_gfvd": ("stage1", "dit"),
    "k600_tfvd": ("stage1",),
    "cityscapes_gfid_gfvd": ("stage1", "prediction"),
    "cityscapes_rfvd": ("stage1",),
}

EXTRACTOR_CHECKPOINTS = {
    "ucf101_rfvd": ("i3d", "lpips"),
    "ucf101_gfvd": ("i3d",),
    "ucf101_tfvd": ("i3d",),
    "k600_rfvd": ("i3d", "lpips"),
    "k600_gfvd": ("i3d",),
    "k600_tfvd": ("i3d",),
    "cityscapes_gfid_gfvd": ("i3d", "inception"),
    "cityscapes_rfvd": ("i3d",),
}


@dataclass(frozen=True)
class EvaluationRunPlan:
    task: str
    config_path: Path
    config: Mapping[str, Any]
    protocol: EvaluationProtocol
    mode: str
    build_only: bool
    project_root: Path
    run_name: str
    output_directory: Path
    model_checkpoints: Mapping[str, Path]
    extractor_checkpoints: Mapping[str, Path]

    def metadata(self) -> dict[str, object]:
        return {
            "task": self.task,
            "run_name": self.run_name,
            "mode": self.mode,
            "formal_protocol": self.mode == "formal",
            "complete_protocol": False if self.build_only else self.protocol.complete_protocol,
            "build_only": self.build_only,
            "protocol": self.protocol.metadata(),
            "model_checkpoints": {
                key: path.relative_to(self.project_root).as_posix()
                for key, path in self.model_checkpoints.items()
            },
            "extractor_checkpoints": {
                key: path.relative_to(self.project_root).as_posix()
                for key, path in self.extractor_checkpoints.items()
            },
        }


@dataclass(frozen=True)
class FeatureMatrix:
    values: Tensor
    sample_ids: Tensor | None = None


def protocol_for(task: str, *, smoke: bool = False) -> EvaluationProtocol:
    if task not in PROTOCOLS:
        raise KeyError(f"Unknown evaluation protocol: {task}")
    formal = PROTOCOLS[task]
    if not smoke:
        return formal
    return EvaluationProtocol(
        task=formal.task,
        split=formal.split,
        population_size=min(8, formal.population_size) if formal.population_size > 0 else 8,
        num_frames=formal.num_frames,
        frame_interval=formal.frame_interval,
        image_size=formal.image_size,
        covariance=formal.covariance,
        metrics=formal.metrics,
        population=f"smoke subset of: {formal.population}",
        input_contract=formal.input_contract,
        complete_protocol=False,
    )


def _protocol_for_config(
    task: str,
    config: Mapping[str, Any],
    *,
    smoke: bool,
) -> EvaluationProtocol:
    """Resolve supported non-reference protocol variants from explicit config values."""

    protocol = protocol_for(task, smoke=smoke)
    evaluation = _mapping(config.get("evaluation", {}), "evaluation")
    if task != "ucf101_rfvd" or evaluation.get("frame_interval", 3) != 1:
        return protocol
    return replace(
        protocol,
        frame_interval=1,
        population="complete UCF101 Split-1 test set at adjacent-frame sampling",
        input_contract=(
            "frame 0, interval 1, float bilinear antialias resize and center crop; "
            "custom non-reference temporal protocol"
        ),
        complete_protocol=False,
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationConfigError(f"{name} must be a mapping")
    return value


def _project_root(config: Mapping[str, Any], config_path: Path) -> Path:
    paths = config.get("paths", {})
    if paths is None:
        paths = {}
    paths = _mapping(paths, "paths")
    configured = paths.get("project_root")
    if configured is not None:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        return candidate.resolve()
    return find_project_root(config_path)


def _relative_local_file(
    project_root: Path,
    value: object,
    *,
    required_root: Path,
    description: str,
) -> Path:
    if value is None or value == "":
        raise EvaluationConfigError(f"missing required local {description}")
    raw = Path(str(value))
    if raw.is_absolute():
        raise EvaluationConfigError(f"{description} must be project-relative, got {value!r}")
    candidate = Path(os.path.abspath(project_root / raw))
    required = Path(os.path.abspath(required_root))
    try:
        candidate.relative_to(required)
    except ValueError as error:
        raise EvaluationConfigError(f"{description} must be inside {required}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _validate_checkpoint_config(
    task: str,
    config: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    checkpoint_config = _mapping(config.get("checkpoints", {}), "checkpoints")
    extractor_config = _mapping(config.get("extractors", {}), "extractors")
    model_root = project_root / "ckpts"
    extractor_root = model_root / "eval_models"
    model_paths = {
        name: _relative_local_file(
            project_root,
            checkpoint_config.get(name),
            required_root=model_root,
            description=f"{name} checkpoint",
        )
        for name in MODEL_CHECKPOINTS[task]
    }
    # Generation checkpoints keep the latent-normalizer identity in their
    # training config, but released bundles may relocate the tiny statistics
    # file beside the DiT checkpoint.  Treat an explicit override as a model
    # artifact so it receives the same project-local path validation.
    latent_normalizer = checkpoint_config.get("latent_normalizer")
    if task in {"ucf101_gfvd", "k600_gfvd"} and latent_normalizer not in {None, ""}:
        model_paths["latent_normalizer"] = _relative_local_file(
            project_root,
            latent_normalizer,
            required_root=model_root,
            description="latent-normalizer checkpoint",
        )
    extractor_paths = {
        name: _relative_local_file(
            project_root,
            extractor_config.get(name),
            required_root=extractor_root,
            description=f"{name} extractor checkpoint",
        )
        for name in EXTRACTOR_CHECKPOINTS[task]
    }
    return model_paths, extractor_paths


def _checkpoint_override_entry(
    project_root: Path,
    value: str | Path,
    *,
    checkpoint_key: str,
) -> str:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else project_root / raw
    candidate = Path(os.path.abspath(candidate))
    checkpoint_root = Path(os.path.abspath(project_root / "ckpts"))
    try:
        relative_to_checkpoints = candidate.relative_to(checkpoint_root)
    except ValueError as error:
        raise EvaluationConfigError(
            f"{checkpoint_key} checkpoint override must be inside {checkpoint_root}"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return (Path("ckpts") / relative_to_checkpoints).as_posix()


def discover_checkpoint_sweep(
    start_checkpoint: str | Path,
    *,
    project_root: str | Path,
) -> tuple[Path, ...]:
    """List numeric step checkpoints from a selected step, then local final aliases."""

    root = Path(project_root).expanduser().resolve()
    start_entry = _checkpoint_override_entry(root, start_checkpoint, checkpoint_key="sweep")
    start = Path(os.path.abspath(root / start_entry))
    match = _STEP_CHECKPOINT_PATTERN.fullmatch(start.name)
    if match is None:
        raise EvaluationConfigError(
            "--checkpoint-sweep-from must select a step-XXXXXXXX.pt checkpoint"
        )
    start_step = int(match.group(1))
    step_checkpoints: list[tuple[int, Path]] = []
    for candidate in start.parent.iterdir():
        candidate_match = _STEP_CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if candidate_match is None or not candidate.is_file():
            continue
        step = int(candidate_match.group(1))
        if step >= start_step:
            step_checkpoints.append((step, candidate))
    step_checkpoints.sort(key=lambda item: (item[0], item[1].name))
    if not step_checkpoints or step_checkpoints[0][1] != start:
        raise EvaluationConfigError("selected checkpoint is absent from its checkpoint directory")
    aliases = [
        start.parent / name for name in _CHECKPOINT_SWEEP_ALIASES if (start.parent / name).is_file()
    ]
    return tuple(path for _, path in step_checkpoints) + tuple(aliases)


def _reject_protocol_override(
    evaluation: Mapping[str, Any],
    key: str,
    expected: object,
) -> None:
    if key not in evaluation:
        return
    actual = evaluation[key]
    if isinstance(expected, tuple) and isinstance(actual, list):
        actual = tuple(actual)
    if actual != expected:
        raise EvaluationConfigError(
            f"evaluation.{key}={actual!r} conflicts with locked value {expected!r}"
        )


def _validate_protocol_config(task: str, config: Mapping[str, Any]) -> None:
    evaluation = _mapping(config.get("evaluation", {}), "evaluation")
    if "batch_size" in evaluation:
        raise EvaluationConfigError(
            "evaluation.batch_size is no longer supported; use evaluation.global_batch_size"
        )
    global_batch_size = evaluation.get("global_batch_size", DEFAULT_GLOBAL_BATCH_SIZE)
    if (
        isinstance(global_batch_size, bool)
        or not isinstance(global_batch_size, int)
        or global_batch_size <= 0
    ):
        raise EvaluationConfigError("evaluation.global_batch_size must be a positive integer")
    video_backend = str(evaluation.get("video_backend", "auto"))
    if video_backend not in {"auto", "torchcodec"}:
        raise EvaluationConfigError("evaluation.video_backend must be auto or torchcodec")
    seek_mode = evaluation.get("seek_mode")
    if seek_mode not in {None, "exact", "approximate"}:
        raise EvaluationConfigError("evaluation.seek_mode must be exact or approximate")
    decode_threads = evaluation.get("decode_threads", 1)
    if (
        isinstance(decode_threads, bool)
        or not isinstance(decode_threads, int)
        or decode_threads <= 0
    ):
        raise EvaluationConfigError("evaluation.decode_threads must be a positive integer")
    num_workers = evaluation.get("num_workers", 0)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise EvaluationConfigError("evaluation.num_workers must be a non-negative integer")
    prefetch_factor = evaluation.get("prefetch_factor", 1)
    if (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor <= 0
    ):
        raise EvaluationConfigError("evaluation.prefetch_factor must be a positive integer")
    seed = evaluation.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationConfigError("evaluation.seed must be a non-negative integer")
    for key in ("cudnn_benchmark", "cudnn_deterministic"):
        if key in evaluation and not isinstance(evaluation[key], bool):
            raise EvaluationConfigError(f"evaluation.{key} must be a boolean")
    real_feature_cache = evaluation.get("real_feature_cache")
    if (
        real_feature_cache is not None
        and real_feature_cache is not False
        and not isinstance(real_feature_cache, str)
    ):
        raise EvaluationConfigError("evaluation.real_feature_cache must be a path or false")
    precision = str(evaluation.get("precision", "bf16")).lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise EvaluationConfigError("evaluation.precision must be bf16, fp16, or fp32")
    attention_backend = str(evaluation.get("attention_backend", "sdpa")).lower()
    attention_backends = {"auto", "sdpa", "fa3", "fa3_fwd", "fa4_cute"}
    if attention_backend not in attention_backends:
        raise EvaluationConfigError(
            "evaluation.attention_backend must be auto, sdpa, fa3, fa3_fwd, or fa4_cute"
        )
    for key in ("i3d_batch_size", "perceptual_batch_size"):
        value = evaluation.get(key, 32)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvaluationConfigError(f"evaluation.{key} must be a positive integer")
    for key in ("weights", "stage1_weights", "dit_weights", "prediction_weights"):
        if key in evaluation and evaluation[key] not in {"ema", "model"}:
            raise EvaluationConfigError(f"evaluation.{key} must be ema or model")
    formal = PROTOCOLS[task]
    _reject_protocol_override(evaluation, "split", formal.split)
    _reject_protocol_override(evaluation, "image_size", formal.image_size)
    _reject_protocol_override(evaluation, "covariance", formal.covariance)
    if task == "ucf101_rfvd":
        frame_interval = evaluation.get("frame_interval", formal.frame_interval)
        if (
            isinstance(frame_interval, bool)
            or not isinstance(frame_interval, int)
            or frame_interval not in {1, formal.frame_interval}
        ):
            raise EvaluationConfigError(
                "evaluation.frame_interval must be 1 or the locked UCF101 rFVD value 3"
            )
        if frame_interval == 1:
            _reject_protocol_override(evaluation, "population_size", 3_783)
    else:
        _reject_protocol_override(evaluation, "frame_interval", formal.frame_interval)

    if task in {"ucf101_rfvd", "k600_rfvd"}:
        _reject_protocol_override(evaluation, "frame_start", 0)
        _reject_protocol_override(evaluation, "num_frames", 16)
        if task == "k600_rfvd":
            _reject_protocol_override(evaluation, "population_size", 27_874)
    elif task == "ucf101_gfvd":
        _reject_protocol_override(evaluation, "population_size", 2048)
        _reject_protocol_override(evaluation, "num_frames", 17)
        model_frames = evaluation.get("model_frames")
        if model_frames is not None and model_frames != 20:
            raise EvaluationConfigError(
                "uni-vug-aligned UCF101 gFVD evaluation.model_frames must be 20"
            )
        _reject_protocol_override(evaluation, "real_frame_interval", 3)
        _reject_protocol_override(evaluation, "generated_frame_interval", 1)
        _reject_protocol_override(evaluation, "real_preprocessing", "center_square_lanczos")
        _reject_protocol_override(evaluation, "generated_video_format", "imageio_mp4")
    elif task == "k600_gfvd":
        _reject_protocol_override(evaluation, "population_size", 50_000)
        _reject_protocol_override(evaluation, "class_quota", "200x84_plus_400x83")
        _reject_protocol_override(evaluation, "i3d_frame_slice", (0, 17))
        model_frames = evaluation.get("model_frames")
        if model_frames is not None and model_frames not in {16, 20}:
            raise EvaluationConfigError("evaluation.model_frames must be 16 or 20")
    elif task in {"ucf101_tfvd", "k600_tfvd"}:
        _reject_protocol_override(evaluation, "source_frames", 24)
        _reject_protocol_override(evaluation, "output_frames", 16)
        _reject_protocol_override(evaluation, "interpolation", "centered_6_to_4_linear")
        _reject_protocol_override(evaluation, "reference_crop", (4, 20))
        _reject_protocol_override(evaluation, "frame_start", 0)
        _reject_protocol_override(
            evaluation,
            "population_size",
            3_620 if task == "ucf101_tfvd" else 27_814,
        )
    elif task in {"cityscapes_gfid_gfvd", "cityscapes_rfvd"}:
        _reject_protocol_override(evaluation, "population_size", 500)
        _reject_protocol_override(evaluation, "context_frames", (4, 15))
        _reject_protocol_override(evaluation, "future_frames", (16, 27))
        _reject_protocol_override(evaluation, "num_future_frames", 12)
        _reject_protocol_override(evaluation, "fps", 16)
    else:
        raise EvaluationConfigError(f"task {task!r} has no locked protocol validation")


def _run_name_for_mode(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    features_only: bool,
) -> str:
    run_config = _mapping(config.get("run", {}), "run")
    base_name = str(run_config.get("name", "")).strip()
    if not base_name:
        raise EvaluationConfigError("run.name is required")
    if smoke:
        return f"{base_name}-smoke"
    if features_only:
        return str(run_config.get("features_name", f"{base_name}-features"))
    return base_name


def build_evaluation_plan(
    task: str,
    config_path: str | Path,
    *,
    smoke: bool = False,
    build_only: bool = False,
    features_only: bool = False,
    stage1_checkpoint_override: str | Path | None = None,
    dit_checkpoint_override: str | Path | None = None,
    run_name_override: str | None = None,
) -> EvaluationRunPlan:
    """Resolve and validate a task without loading model checkpoint contents."""

    if task not in PROTOCOLS:
        raise EvaluationConfigError(f"unknown evaluation task {task!r}")
    resolved_path = Path(config_path).expanduser().resolve()
    config = copy.deepcopy(load_config(resolved_path))
    configured_task = config.get("task")
    if configured_task != task:
        raise EvaluationConfigError(
            f"entrypoint task {task!r} does not match config task {configured_task!r}"
        )
    project_root = _project_root(config, resolved_path)
    checkpoint_overrides = {
        "stage1": stage1_checkpoint_override,
        "dit": dit_checkpoint_override,
    }
    if any(value is not None for value in checkpoint_overrides.values()):
        checkpoint_config = dict(_mapping(config.get("checkpoints", {}), "checkpoints"))
        for checkpoint_key, override in checkpoint_overrides.items():
            if override is None:
                continue
            if checkpoint_key not in MODEL_CHECKPOINTS[task]:
                raise EvaluationConfigError(
                    f"task {task!r} does not define a {checkpoint_key} checkpoint"
                )
            checkpoint_config[checkpoint_key] = _checkpoint_override_entry(
                project_root,
                override,
                checkpoint_key=checkpoint_key,
            )
        config["checkpoints"] = checkpoint_config
    _validate_protocol_config(task, config)
    run_name = _run_name_for_mode(config, smoke=smoke, features_only=features_only)
    if run_name_override is not None:
        run_name = str(run_name_override).strip()
        if not smoke:
            run_config = dict(_mapping(config.get("run", {}), "run"))
            if features_only:
                run_config["features_name"] = run_name
            else:
                run_config["name"] = run_name
            config["run"] = run_config
    paths = ProjectPaths(project_root=project_root)
    output_directory = paths.eval_run(task, run_name, create=False)
    model_paths, extractor_paths = _validate_checkpoint_config(task, config, project_root)
    return EvaluationRunPlan(
        task=task,
        config_path=resolved_path,
        config=config,
        protocol=_protocol_for_config(task, config, smoke=smoke),
        mode="smoke" if smoke else ("features_only" if features_only else "formal"),
        build_only=build_only,
        project_root=project_root,
        run_name=run_name,
        output_directory=output_directory,
        model_checkpoints=model_paths,
        extractor_checkpoints=extractor_paths,
    )


def _input_path(plan: EvaluationRunPlan, key: str, *, required: bool = True) -> Path | None:
    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    value = inputs.get(key)
    if value is None or value == "":
        if required:
            raise EvaluationConfigError(f"inputs.{key} is required for evaluation")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = plan.project_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def load_feature_matrix(path: str | Path) -> FeatureMatrix:
    """Load readable precomputed features without loading executable model objects."""

    feature_path = Path(path)
    if feature_path.suffix.lower() == ".npy":
        values = torch.from_numpy(np.load(feature_path, allow_pickle=False))
        sample_ids = None
    else:
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        if isinstance(payload, Tensor):
            values, sample_ids = payload, None
        elif isinstance(payload, Mapping) and "features" in payload:
            values = torch.as_tensor(payload["features"])
            ids = payload.get("sample_ids")
            sample_ids = None if ids is None else torch.as_tensor(ids, dtype=torch.long)
        else:
            raise EvaluationConfigError(
                f"feature file {feature_path} must contain a tensor or features mapping"
            )
    if values.ndim != 2 or values.shape[0] < 2 or not torch.isfinite(values).all():
        raise EvaluationConfigError(f"features in {feature_path} must be finite [N,D] with N>=2")
    if sample_ids is not None:
        if sample_ids.ndim != 1 or sample_ids.shape[0] != values.shape[0]:
            raise EvaluationConfigError("feature sample_ids must have shape [N]")
        if sample_ids.numel() > 1 and torch.unique(sample_ids).numel() != sample_ids.numel():
            raise EvaluationConfigError("feature sample_ids contain duplicates")
    return FeatureMatrix(values.detach().cpu(), sample_ids)


def validate_video_batch(
    video: Tensor,
    *,
    frames: int,
    image_size: tuple[int, int],
    count: int | None = None,
    name: str = "video",
) -> None:
    """Validate an optional uncompressed population tensor against a protocol."""

    expected_tail = (frames, 3, *image_size)
    if video.ndim != 5 or tuple(video.shape[1:]) != expected_tail:
        raise EvaluationConfigError(
            f"{name} must have shape [N,{frames},3,{image_size[0]},{image_size[1]}]"
        )
    if count is not None and video.shape[0] != count:
        raise EvaluationConfigError(f"{name} expected {count} clips, got {video.shape[0]}")


def _load_tensor(path: Path) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping):
        for key in ("video", "videos", "clips"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, Tensor):
        raise EvaluationConfigError(f"tensor file {path} does not contain a video tensor")
    return payload


def _validate_optional_media(plan: EvaluationRunPlan, count: int) -> list[str]:
    contracts: dict[str, tuple[tuple[str, int], ...]] = {
        "ucf101_rfvd": (("real_clips", 16), ("reconstructed_clips", 16)),
        "k600_rfvd": (("real_clips", 16), ("reconstructed_clips", 16)),
        "ucf101_gfvd": (("real_clips", 17), ("generated_clips", 17)),
        "ucf101_tfvd": (
            ("source_clips", 24),
            ("reference_clips", 16),
            ("interpolated_clips", 16),
        ),
        "k600_tfvd": (
            ("source_clips", 24),
            ("reference_clips", 16),
            ("interpolated_clips", 16),
        ),
        "cityscapes_gfid_gfvd": (
            ("context_clips", 12),
            ("real_future_clips", 12),
            ("predicted_future_clips", 12),
        ),
        "cityscapes_rfvd": (
            ("context_clips", 12),
            ("real_future_clips", 12),
            ("future_only_clips", 12),
            ("context_conditioned_clips", 12),
        ),
    }
    validated: list[str] = []
    for key, frames in contracts.get(plan.task, ()):
        path = _input_path(plan, key, required=False)
        if path is None:
            continue
        validate_video_batch(
            _load_tensor(path),
            frames=frames,
            image_size=plan.protocol.image_size,
            count=count,
            name=key,
        )
        validated.append(key)
    return validated


def _feature_pair(
    plan: EvaluationRunPlan,
    real_key: str,
    fake_key: str,
    *,
    expected_count: int | None,
) -> tuple[FeatureMatrix, FeatureMatrix, float]:
    real = load_feature_matrix(_input_path(plan, real_key))
    fake = load_feature_matrix(_input_path(plan, fake_key))
    if real.values.shape != fake.values.shape:
        raise EvaluationConfigError(
            f"{real_key} and {fake_key} must have identical [N,D] shapes, got "
            f"{tuple(real.values.shape)} and {tuple(fake.values.shape)}"
        )
    if expected_count is not None and real.values.shape[0] != expected_count:
        raise EvaluationConfigError(
            f"{plan.task} requires {expected_count} samples, got {real.values.shape[0]}"
        )
    if real.sample_ids is not None and fake.sample_ids is not None:
        if not torch.equal(torch.sort(real.sample_ids).values, torch.sort(fake.sample_ids).values):
            raise EvaluationConfigError("real/fake feature sample IDs do not match")
    score = frechet_from_features(
        real.values,
        fake.values,
        covariance=plan.protocol.covariance,
    )
    return real, fake, score


def _require_contiguous_feature_ids(
    feature: FeatureMatrix,
    *,
    count: int,
    name: str,
) -> None:
    if feature.sample_ids is None:
        raise EvaluationConfigError(f"{name} must store explicit sample_ids")
    expected = torch.arange(count, dtype=torch.long)
    if not torch.equal(torch.sort(feature.sample_ids).values, expected):
        raise EvaluationConfigError(f"{name} sample_ids must be exactly [0, {count})")


def _paired_metrics(plan: EvaluationRunPlan) -> dict[str, float]:
    path = _input_path(plan, "paired_metrics")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
        payload = payload["metrics"]
    if not isinstance(payload, Mapping):
        raise EvaluationConfigError("paired_metrics must contain a JSON object")
    required = ("LPIPS", "PSNR", "SSIM", "PSNR_global")
    missing = [name for name in required if name not in payload]
    if missing:
        raise EvaluationConfigError(f"paired_metrics is missing {missing}")
    result = {name: float(payload[name]) for name in required}
    if not all(np.isfinite(value) for value in result.values()):
        raise EvaluationConfigError("paired metrics must be finite")
    return result


def _generic_population(count: int, *, split: str) -> list[dict[str, object]]:
    return [{"sample_id": index, "split": split} for index in range(count)]


def _validate_external_population(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    expected_split: str,
) -> None:
    if len(records) != count:
        raise EvaluationConfigError(
            f"population contains {len(records)} records but features contain {count}"
        )
    ids = [str(record.get("sample_id", "")) for record in records]
    if any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise EvaluationConfigError("population sample_id values must be non-empty and unique")
    splits = [record.get("split") for record in records]
    if any(value != expected_split for value in splits):
        raise EvaluationConfigError(f"population records must all declare split={expected_split!r}")


def _population_for(plan: EvaluationRunPlan, count: int) -> list[dict[str, Any]]:
    inputs = _mapping(plan.config.get("inputs", {}), "inputs")
    configured_population = inputs.get("population")
    if configured_population is not None and configured_population != "":
        path = _input_path(plan, "population")
        records = read_population(path)
        _validate_external_population(records, count, expected_split=plan.protocol.split)
        return records
    if plan.task == "ucf101_gfvd":
        evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
        return ucf101_generation_population(
            count=count,
            base_seed=int(evaluation.get("base_seed", 3407)),
        )
    if plan.task == "k600_gfvd":
        evaluation = _mapping(plan.config.get("evaluation", {}), "evaluation")
        formal = k600_balanced_population(base_seed=int(evaluation.get("base_seed", 3407)))
        return formal[:count]
    if plan.task in {"cityscapes_gfid_gfvd", "cityscapes_rfvd"}:
        return [
            {
                "sample_id": index,
                "split": "val",
                "context_frames": [4, 15],
                "future_frames": [16, 27],
            }
            for index in range(count)
        ]
    if plan.mode == "formal":
        raise EvaluationConfigError(
            f"inputs.population is required for formal {plan.task} split completeness"
        )
    return _generic_population(count, split=plan.protocol.split)


def _evaluate(plan: EvaluationRunPlan) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    expected = plan.protocol.population_size if plan.protocol.population_size > 0 else None
    if plan.task in {"ucf101_rfvd", "k600_rfvd"}:
        real, _, score = _feature_pair(
            plan,
            "real_features",
            "reconstructed_features",
            expected_count=expected,
        )
        count = real.values.shape[0]
        metrics = {"rFVD": score, **_paired_metrics(plan)}
    elif plan.task == "cityscapes_rfvd":
        real, future_only, future_only_score = _feature_pair(
            plan,
            "real_features",
            "future_only_features",
            expected_count=expected,
        )
        _, context_conditioned, context_conditioned_score = _feature_pair(
            plan,
            "real_features",
            "context_conditioned_features",
            expected_count=expected,
        )
        count = real.values.shape[0]
        for feature, name in (
            (real, "real_features"),
            (future_only, "future_only_features"),
            (context_conditioned, "context_conditioned_features"),
        ):
            _require_contiguous_feature_ids(feature, count=count, name=name)
        metrics = {
            "rFVD_future_only": future_only_score,
            "rFVD_context_conditioned": context_conditioned_score,
        }
    elif plan.task in {"ucf101_gfvd", "k600_gfvd"}:
        real, generated, score = _feature_pair(
            plan,
            "real_features",
            "generated_features",
            expected_count=expected,
        )
        count = real.values.shape[0]
        _require_contiguous_feature_ids(real, count=count, name="real_features")
        _require_contiguous_feature_ids(generated, count=count, name="generated_features")
        metrics = {"gFVD": score}
    elif plan.task in {"ucf101_tfvd", "k600_tfvd"}:
        reference, _, score = _feature_pair(
            plan,
            "reference_features",
            "interpolated_features",
            expected_count=expected,
        )
        count = reference.values.shape[0]
        metrics = {"tFVD": score}
    else:
        real_i3d, predicted_i3d, gfvd = _feature_pair(
            plan,
            "real_i3d_features",
            "predicted_i3d_features",
            expected_count=expected,
        )
        frame_count = None if expected is None else expected * 12
        real_frames, predicted_frames, gfid = _feature_pair(
            plan,
            "real_inception_features",
            "predicted_inception_features",
            expected_count=frame_count,
        )
        count = real_i3d.values.shape[0]
        _require_contiguous_feature_ids(real_i3d, count=count, name="real_i3d_features")
        _require_contiguous_feature_ids(
            predicted_i3d,
            count=count,
            name="predicted_i3d_features",
        )
        _require_contiguous_feature_ids(
            real_frames,
            count=count * 12,
            name="real_inception_features",
        )
        _require_contiguous_feature_ids(
            predicted_frames,
            count=count * 12,
            name="predicted_inception_features",
        )
        metrics = {"gFID": gfid, "gFVD": gfvd}
    population = _population_for(plan, count)
    validated_media = _validate_optional_media(plan, count)
    return metrics, population, validated_media


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _stored_result(task: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
    if task not in COMPACT_RESULT_TASKS:
        return result
    return {
        "task": result["task"],
        "run_name": result["run_name"],
        "status": result["status"],
        "complete_protocol": result["complete_protocol"],
        "population_count": result["population_count"],
        "metrics": result["metrics"],
    }


def run_evaluation(
    task: str,
    config_path: str | Path,
    *,
    smoke: bool = False,
    build_only: bool = False,
    features_only: bool = False,
    resume: bool = False,
    stage1_checkpoint_override: str | Path | None = None,
    dit_checkpoint_override: str | Path | None = None,
    run_name_override: str | None = None,
) -> dict[str, Any]:
    """Validate a run or execute its end-to-end media/model evaluation path."""

    if build_only and features_only:
        raise EvaluationConfigError("--build-only and --features-only are mutually exclusive")
    if resume and (build_only or features_only):
        raise EvaluationConfigError("--resume is only valid for end-to-end generation evaluation")
    plan = build_evaluation_plan(
        task,
        config_path,
        smoke=smoke,
        build_only=build_only,
        features_only=features_only,
        stage1_checkpoint_override=stage1_checkpoint_override,
        dit_checkpoint_override=dit_checkpoint_override,
        run_name_override=run_name_override,
    )
    resolved = dict(plan.config)
    resolved["execution"] = {
        "mode": plan.mode,
        "build_only": build_only,
        "features_only": features_only,
        "resume": resume,
        "protocol": plan.protocol.metadata(),
    }
    existing_metrics_path = plan.output_directory / "metrics.json"
    existing_result: Mapping[str, Any] | None = None
    if existing_metrics_path.is_file():
        existing_value = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
        existing_result = _mapping(existing_value, "existing metrics.json")
    existing_config_path = plan.output_directory / "resolved_config.yaml"
    existing_config: dict[str, Any] | None = None
    if existing_config_path.is_file():
        existing_config = load_config(existing_config_path)
        existing_config.pop("execution", None)
        if existing_config != dict(plan.config):
            raise EvaluationConfigError(
                "existing evaluation directory belongs to a different readable config"
            )
    if existing_result is not None and existing_result.get("status") == "built":
        if existing_config is None:
            raise EvaluationConfigError(
                "existing build-only skeleton belongs to a different readable config"
            )
    if resume:
        if task not in {"ucf101_gfvd", "k600_gfvd"}:
            raise EvaluationConfigError("resume is implemented only for deterministic gFVD runs")
        if (
            existing_config is None
            and plan.output_directory.exists()
            and any(plan.output_directory.rglob("*.pt"))
        ):
            raise EvaluationConfigError(
                "cannot resume sample artifacts without the existing readable config"
            )
        if existing_result is not None and existing_result.get("complete_protocol") is True:
            raise EvaluationConfigError("evaluation run is already complete")
    elif existing_result is not None and existing_result.get("status") != "built":
        raise FileExistsError(
            "evaluation output already contains a result; use a new run name: "
            f"{plan.output_directory}"
        )
    if not resume and plan.output_directory.exists():
        population_path = plan.output_directory / "population.jsonl"
        has_population = population_path.is_file() and bool(
            population_path.read_text(encoding="utf-8").strip()
        )
        has_samples = any((plan.output_directory / "samples").glob("*"))
        has_features = any((plan.output_directory / "features").glob("**/features.pt"))
        if has_population or has_samples or has_features:
            raise FileExistsError(
                "evaluation output contains partial artifacts; pass --resume for gFVD or "
                "choose a new run name"
            )
    metadata = plan.metadata()
    environment_rank = int(os.environ.get("RANK", "0"))
    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if build_only:
        if environment_rank == 0:
            plan.output_directory.mkdir(parents=True, exist_ok=True)
            if task not in {"ucf101_rfvd", "k600_rfvd"}:
                for directory in ("features", "samples", "logs"):
                    (plan.output_directory / directory).mkdir(exist_ok=True)
        metrics: dict[str, float] = {}
        population: list[dict[str, Any]] = []
        validated_media: list[str] = []
        runtime_metadata: dict[str, Any] = {"execution": "validation_only"}
        status = "built"
    elif features_only:
        if environment_world_size != 1:
            raise EvaluationConfigError("--features-only is a single-process offline scorer")
        plan.output_directory.mkdir(parents=True, exist_ok=True)
        if task not in {"ucf101_rfvd", "k600_rfvd"}:
            for directory in ("features", "samples", "logs"):
                (plan.output_directory / directory).mkdir(exist_ok=True)
        metrics, population, validated_media = _evaluate(plan)
        runtime_metadata = {"execution": "features_only"}
        status = "features_only"
    else:
        from vrae.evaluation.common.runtime import run_end_to_end

        if environment_rank == 0:
            plan.output_directory.mkdir(parents=True, exist_ok=True)
            if task not in {"ucf101_rfvd", "k600_rfvd"}:
                for directory in ("features", "samples", "logs"):
                    (plan.output_directory / directory).mkdir(exist_ok=True)
            save_resolved_config(resolved, plan.output_directory / "resolved_config.yaml")
        metrics, population, runtime_metadata = run_end_to_end(plan, resume=resume)
        validated_media = ["decoded_through_vrae_video_reader"]
        status = "evaluated"

    is_main = environment_rank == 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main = torch.distributed.get_rank() == 0
    result = {
        **metadata,
        "status": status,
        "complete_protocol": bool(
            not build_only
            and not features_only
            and plan.mode == "formal"
            and plan.protocol.complete_protocol
        ),
        "population_count": len(population),
        "validated_media_inputs": validated_media,
        "runtime": runtime_metadata,
        "resumed": resume,
        "metrics": metrics,
    }
    if is_main:
        save_resolved_config(resolved, plan.output_directory / "resolved_config.yaml")
        if task not in {"ucf101_rfvd", "k600_rfvd"}:
            write_population(plan.output_directory / "population.jsonl", population)
        _write_json(plan.output_directory / "metrics.json", _stored_result(task, result))
    return {**result, "output_directory": str(plan.output_directory)}


def run_checkpoint_sweep(
    task: str,
    config_path: str | Path,
    start_checkpoint: str | Path,
    *,
    smoke: bool = False,
    build_only: bool = False,
    checkpoint_stride: int = 1,
    include_aliases: bool = True,
    watch: bool = False,
    poll_seconds: float = 30.0,
    idle_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Evaluate selected task-specific checkpoints from a starting numeric step."""

    if task not in CHECKPOINT_SWEEP_TASKS:
        raise EvaluationConfigError("checkpoint sweeps are not supported for this evaluation task")
    if (
        not isinstance(checkpoint_stride, int)
        or isinstance(checkpoint_stride, bool)
        or checkpoint_stride <= 0
    ):
        raise EvaluationConfigError("checkpoint sweep stride must be a positive integer")
    if watch and (checkpoint_stride != 1 or not include_aliases):
        raise EvaluationConfigError(
            "checkpoint sweep stride/alias filtering cannot be combined with watch mode"
        )
    if poll_seconds <= 0:
        raise EvaluationConfigError("checkpoint sweep poll seconds must be positive")
    if idle_timeout_seconds is not None and idle_timeout_seconds < 0:
        raise EvaluationConfigError("checkpoint sweep idle timeout must be non-negative")
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_config(resolved_config_path)
    if config.get("task") != task:
        raise EvaluationConfigError(
            f"entrypoint task {task!r} does not match config task {config.get('task')!r}"
        )
    project_root = _project_root(config, resolved_config_path)
    discovered_checkpoints = discover_checkpoint_sweep(
        start_checkpoint,
        project_root=project_root,
    )
    sweep_start = discovered_checkpoints[0]
    numeric_checkpoints = tuple(
        checkpoint
        for checkpoint in discovered_checkpoints
        if _STEP_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    )
    aliases = tuple(
        checkpoint
        for checkpoint in discovered_checkpoints
        if checkpoint.name in _CHECKPOINT_SWEEP_ALIASES
    )
    checkpoints = numeric_checkpoints[::checkpoint_stride]
    if include_aliases:
        checkpoints += aliases
    checkpoint_key = CHECKPOINT_SWEEP_CHECKPOINTS[task]
    base_run_name = _run_name_for_mode(config, smoke=smoke, features_only=False)
    training_run_name = sweep_start.parent.parent.name
    is_main = int(os.environ.get("RANK", "0")) == 0
    runs: list[dict[str, Any]] = []
    evaluated_checkpoints: list[Path] = []
    evaluated_paths: set[Path] = set()

    def evaluate_checkpoint(checkpoint: Path, *, total: int | None) -> None:
        index = len(evaluated_checkpoints) + 1
        relative_checkpoint = checkpoint.relative_to(project_root).as_posix()
        run_name = f"{base_run_name}-{training_run_name}-{checkpoint.stem}"
        if is_main:
            progress = f"{index}/{total}" if total is not None else f"{index}/dynamic"
            print(
                f"[checkpoint-sweep {progress}] {relative_checkpoint} -> {run_name}",
                file=sys.stderr,
                flush=True,
            )
        checkpoint_override = {
            f"{checkpoint_key}_checkpoint_override": checkpoint,
        }
        result = run_evaluation(
            task,
            resolved_config_path,
            smoke=smoke,
            build_only=build_only,
            run_name_override=run_name,
            **checkpoint_override,
        )
        runs.append({"checkpoint": relative_checkpoint, **result})
        evaluated_checkpoints.append(checkpoint)
        evaluated_paths.add(checkpoint)
        if is_main:
            print(
                f"[checkpoint-sweep {progress}] complete",
                file=sys.stderr,
                flush=True,
            )

    if not watch:
        for checkpoint in checkpoints:
            evaluate_checkpoint(checkpoint, total=len(checkpoints))
    else:
        last_new_checkpoint_at = time.monotonic()
        while True:
            discovered = discover_checkpoint_sweep(sweep_start, project_root=project_root)
            pending_steps = [
                checkpoint
                for checkpoint in discovered
                if _STEP_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
                and checkpoint not in evaluated_paths
            ]
            if pending_steps:
                for checkpoint in pending_steps:
                    evaluate_checkpoint(checkpoint, total=None)
                last_new_checkpoint_at = time.monotonic()
                continue

            terminal_alias_present = (sweep_start.parent / "last.pt").is_file()
            idle_timeout_reached = (
                idle_timeout_seconds is not None
                and time.monotonic() - last_new_checkpoint_at >= idle_timeout_seconds
            )
            if terminal_alias_present or idle_timeout_reached:
                final_discovered = discover_checkpoint_sweep(
                    sweep_start,
                    project_root=project_root,
                )
                final_steps = [
                    checkpoint
                    for checkpoint in final_discovered
                    if _STEP_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
                    and checkpoint not in evaluated_paths
                ]
                if final_steps:
                    for checkpoint in final_steps:
                        evaluate_checkpoint(checkpoint, total=None)
                    last_new_checkpoint_at = time.monotonic()
                    continue
                for checkpoint in final_discovered:
                    if (
                        checkpoint.name in _CHECKPOINT_SWEEP_ALIASES
                        and checkpoint not in evaluated_paths
                    ):
                        evaluate_checkpoint(checkpoint, total=None)
                break

            if is_main:
                print(
                    f"[checkpoint-sweep watch] no new step checkpoint; "
                    f"polling again in {poll_seconds:g}s",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(poll_seconds)

    return {
        "task": task,
        "mode": "smoke" if smoke else "formal",
        "build_only": build_only,
        "watch": watch,
        "checkpoint_stride": checkpoint_stride,
        "include_aliases": include_aliases,
        "poll_seconds": poll_seconds if watch else None,
        "idle_timeout_seconds": idle_timeout_seconds if watch else None,
        "checkpoint_key": checkpoint_key,
        "checkpoint_sweep_from": sweep_start.relative_to(project_root).as_posix(),
        "checkpoint_count": len(evaluated_checkpoints),
        "checkpoints": [
            path.relative_to(project_root).as_posix() for path in evaluated_checkpoints
        ],
        "runs": runs,
    }


def build_argument_parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the locked {task} protocol")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="run an 8-sample non-formal protocol")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="validate config/checkpoint placement and write a non-complete run skeleton",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="score explicit precomputed features; this mode is always non-complete",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and continue an interrupted deterministic gFVD sample population",
    )
    parser.add_argument(
        "--checkpoint-sweep-from",
        type=Path,
        help=(
            "evaluate this step checkpoint and every later step checkpoint in its directory, "
            "then append last.pt/latest.pt when present"
        ),
    )
    parser.add_argument(
        "--checkpoint-sweep-watch",
        action="store_true",
        help="rescan after each checkpoint and keep evaluating newly written numeric steps",
    )
    parser.add_argument(
        "--checkpoint-sweep-stride",
        type=int,
        default=1,
        help="evaluate every Nth numeric checkpoint, starting with --checkpoint-sweep-from",
    )
    parser.add_argument(
        "--checkpoint-sweep-skip-aliases",
        action="store_true",
        help="do not append last.pt/latest.pt after numeric checkpoints",
    )
    parser.add_argument(
        "--checkpoint-sweep-poll-seconds",
        type=float,
        default=30.0,
        help="seconds between directory rescans while checkpoint sweep watch is idle",
    )
    parser.add_argument(
        "--checkpoint-sweep-idle-seconds",
        type=float,
        default=None,
        help=(
            "stop a watch after this many seconds without a new step and evaluate final aliases; "
            "omit to watch until last.pt appears or the process is interrupted"
        ),
    )
    return parser


def _rfvd_result_table(result: Mapping[str, Any]) -> str:
    metrics = _mapping(result.get("metrics", {}), "metrics")

    def metric_value(name: str) -> str:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):.6f}"
        return "-"

    headers = ("rFVD", "LPIPS", "PSNR", "SSIM", "Result path")
    values = (
        metric_value("rFVD"),
        metric_value("LPIPS"),
        metric_value("PSNR"),
        metric_value("SSIM"),
        str(result["output_directory"]),
    )
    widths = tuple(
        max(len(header), len(value)) for header, value in zip(headers, values, strict=True)
    )
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def row(items: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(item.ljust(width) for item, width in zip(items, widths, strict=True))
            + " |"
        )

    return "\n".join((border, row(headers), border, row(values), border))


def _single_fvd_result_table(result: Mapping[str, Any], metric_name: str) -> str:
    metrics = _mapping(result.get("metrics", {}), "metrics")
    value = metrics.get(metric_name)
    metric = (
        f"{float(value):.6f}"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else "-"
    )
    headers = (metric_name, "Result path")
    values = (metric, str(result["output_directory"]))
    widths = tuple(
        max(len(header), len(item)) for header, item in zip(headers, values, strict=True)
    )
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def row(items: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(item.ljust(width) for item, width in zip(items, widths, strict=True))
            + " |"
        )

    return "\n".join((border, row(headers), border, row(values), border))


def _gfvd_result_table(result: Mapping[str, Any]) -> str:
    return _single_fvd_result_table(result, "gFVD")


def _tfvd_result_table(result: Mapping[str, Any]) -> str:
    return _single_fvd_result_table(result, "tFVD")


def _run_task_cli(task: str, argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser(task).parse_args(argv)
    if arguments.checkpoint_sweep_watch and arguments.checkpoint_sweep_from is None:
        raise EvaluationConfigError("--checkpoint-sweep-watch requires --checkpoint-sweep-from")
    if arguments.checkpoint_sweep_from is None and (
        arguments.checkpoint_sweep_stride != 1 or arguments.checkpoint_sweep_skip_aliases
    ):
        raise EvaluationConfigError(
            "--checkpoint-sweep-stride/--checkpoint-sweep-skip-aliases require "
            "--checkpoint-sweep-from"
        )
    if arguments.checkpoint_sweep_idle_seconds is not None and not arguments.checkpoint_sweep_watch:
        raise EvaluationConfigError(
            "--checkpoint-sweep-idle-seconds requires --checkpoint-sweep-watch"
        )
    if arguments.checkpoint_sweep_from is not None:
        if arguments.features_only or arguments.resume:
            raise EvaluationConfigError(
                "--checkpoint-sweep-from cannot be combined with --features-only or --resume"
            )
        result = run_checkpoint_sweep(
            task,
            arguments.config,
            arguments.checkpoint_sweep_from,
            smoke=arguments.smoke,
            build_only=arguments.build_only,
            checkpoint_stride=arguments.checkpoint_sweep_stride,
            include_aliases=not arguments.checkpoint_sweep_skip_aliases,
            watch=arguments.checkpoint_sweep_watch,
            poll_seconds=arguments.checkpoint_sweep_poll_seconds,
            idle_timeout_seconds=arguments.checkpoint_sweep_idle_seconds,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = run_evaluation(
            task,
            arguments.config,
            smoke=arguments.smoke,
            build_only=arguments.build_only,
            features_only=arguments.features_only,
            resume=arguments.resume,
        )
        is_main = int(os.environ.get("RANK", "0")) == 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = torch.distributed.get_rank() == 0
        if is_main:
            if task in {"ucf101_rfvd", "k600_rfvd"}:
                print(_rfvd_result_table(result))
            elif task in {"ucf101_gfvd", "k600_gfvd"}:
                print(_gfvd_result_table(result))
            elif task in {"ucf101_tfvd", "k600_tfvd"} and not arguments.build_only:
                print(_tfvd_result_table(result))
            else:
                print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_task_cli(task: str, argv: Sequence[str] | None = None) -> int:
    from vrae.training.common.distributed import shutdown_distributed

    try:
        return _run_task_cli(task, argv)
    finally:
        shutdown_distributed()


__all__ = [
    "DEFAULT_GLOBAL_BATCH_SIZE",
    "CHECKPOINT_SWEEP_TASKS",
    "EvaluationConfigError",
    "EvaluationProtocol",
    "EvaluationRunPlan",
    "FeatureMatrix",
    "PROTOCOLS",
    "build_argument_parser",
    "build_evaluation_plan",
    "discover_checkpoint_sweep",
    "load_feature_matrix",
    "protocol_for",
    "run_evaluation",
    "run_checkpoint_sweep",
    "run_task_cli",
    "validate_video_batch",
]
