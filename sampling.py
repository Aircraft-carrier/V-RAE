#!/usr/bin/env python3
"""Reconstruct the three sample videos with a released V-RAE checkpoint."""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SAMPLES = tuple(PROJECT_ROOT / "assets" / f"sample{index}.mp4" for index in range(1, 4))
VARIANTS = {
    "dino": ("dinov3", PROJECT_ROOT / "ckpts/vrae/vrae_dinov3.pt"),
    "siglip": ("siglip2", PROJECT_ROOT / "ckpts/vrae/vrae_siglip2.pt"),
    "vjepa": ("vjepa2_1", PROJECT_ROOT / "ckpts/vrae/vrae_vjepa2.1.pt"),
    "eupe": ("eupe", PROJECT_ROOT / "ckpts/vrae/vrae_eupe.pt"),
}

SUPPORTED_NUM_FRAMES = (16, 20)
IMAGE_SIZE = (256, 256)
HEADER_HEIGHT = 40
HEADER_FONT_SIZE = 20
LABELS = ("Original Video", "Reconstructed Video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct assets/sample1.mp4 through sample3.mp4 on cuda:0."
    )
    parser.add_argument(
        "variant",
        choices=tuple(VARIANTS),
        help="V-RAE checkpoint to use",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        choices=SUPPORTED_NUM_FRAMES,
        default=None,
    )
    return parser.parse_args()


def as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def prepare_model_config(
    raw_config: Mapping[str, Any],
    expected_encoder: str,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(raw_config))
    has_model_section = "model" in config
    model_config = dict(
        as_mapping(config["model"] if has_model_section else config, "model config")
    )
    encoder_config = as_mapping(model_config.get("encoder"), "encoder config")
    encoder_name = str(encoder_config.get("name", ""))
    if encoder_name != expected_encoder:
        raise ValueError(f"checkpoint encoder is {encoder_name!r}; expected {expected_encoder!r}")

    decoder_config = dict(as_mapping(model_config.get("decoder"), "decoder config"))
    decoder_config["init"] = "scratch"
    decoder_config.pop("checkpoint", None)
    parameters = decoder_config.get("parameters")
    if isinstance(parameters, Mapping):
        parameters = dict(parameters)
        parameters["attention_backend"] = "sdpa"
        decoder_config["parameters"] = parameters
    else:
        decoder_config["attention_backend"] = "sdpa"
    model_config["decoder"] = decoder_config

    if has_model_section:
        config["model"] = model_config
    else:
        config.update(model_config)
    return config


def load_ema_weights(model: nn.Module, payload: Mapping[str, Any]) -> None:
    ema = as_mapping(payload.get("ema"), "checkpoint EMA state")
    shadow = as_mapping(ema.get("shadow"), "checkpoint EMA weights")
    trainable = nn.ModuleDict(model.trainable_groups())
    state = trainable.state_dict()
    expected = {name for name, value in state.items() if torch.is_floating_point(value)}
    if set(shadow) != expected:
        missing = sorted(expected.difference(shadow))
        unexpected = sorted(set(shadow).difference(expected))
        raise ValueError(
            "checkpoint EMA weights do not match the model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, value in shadow.items():
        state[name].copy_(torch.as_tensor(value).to(state[name]))


def load_model(variant: str, device: torch.device) -> nn.Module:
    from vrae.checkpoint import load_checkpoint
    from vrae.models.autoencoder import VRAE
    from vrae.paths import ProjectPaths

    expected_encoder, checkpoint_path = VARIANTS[variant]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = load_checkpoint(checkpoint_path, map_location="cpu", mmap=True)
    raw_config = as_mapping(payload.get("resolved_config"), "checkpoint model config")
    config = prepare_model_config(raw_config, expected_encoder)
    model = VRAE.from_config(
        config,
        project_paths=ProjectPaths(project_root=PROJECT_ROOT),
    )
    model.load_state_dict(payload["model"], strict=True)
    load_ema_weights(model, payload)

    del payload
    gc.collect()
    return model.requires_grad_(False).eval().to(device)


def load_sample(path: Path, num_frames: int | None = None) -> tuple[torch.Tensor, float]:
    from vrae.data import VideoReader, resize_center_crop, uint8_to_float

    reader = VideoReader(path, backend="auto", num_threads=1, seek_mode="exact")
    available_frames = len(reader)
    if num_frames is None:
        if available_frames not in SUPPORTED_NUM_FRAMES:
            supported = " or ".join(str(value) for value in SUPPORTED_NUM_FRAMES)
            raise ValueError(
                f"{path} contains {available_frames} frames; expected {supported}"
            )
        num_frames = available_frames
    elif available_frames < num_frames:
        raise ValueError(
            f"{path} contains {available_frames} frames; "
            f"cannot reconstruct {num_frames} frames"
        )

    frames = reader.get_frames(range(num_frames))
    video = resize_center_crop(
        uint8_to_float(frames),
        IMAGE_SIZE,
        mode="bilinear",
        antialias=True,
        crop_rounding="round",
    ).clamp_(0.0, 1.0)
    fps = float(reader.metadata.fps) if reader.metadata.fps else 30.0
    return video.contiguous(), fps


@torch.inference_mode()
def reconstruct(model: nn.Module, video: torch.Tensor, device: torch.device) -> torch.Tensor:
    expected_tail = (3, *IMAGE_SIZE)
    if video.ndim != 4 or tuple(video.shape[1:]) != expected_tail:
        raise ValueError(
            f"sample must have shape [T, {expected_tail[0]}, "
            f"{expected_tail[1]}, {expected_tail[2]}], got {tuple(video.shape)}"
        )
    if video.shape[0] not in SUPPORTED_NUM_FRAMES:
        supported = " or ".join(str(value) for value in SUPPORTED_NUM_FRAMES)
        raise ValueError(f"sample must contain {supported} frames, got {video.shape[0]}")

    model_input = video.unsqueeze(0).to(device=device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reconstruction = model(model_input)["recon"]
    if reconstruction.shape != model_input.shape:
        raise RuntimeError(
            "reconstruction shape does not match the input: "
            f"{tuple(reconstruction.shape)} != {tuple(model_input.shape)}"
        )
    return reconstruction[0].float().clamp_(0.0, 1.0).cpu()


def make_comparison(original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    from PIL import Image, ImageDraw, ImageFont

    if original.shape != reconstruction.shape:
        raise ValueError("original and reconstructed videos must have the same shape")

    video = torch.cat((original, reconstruction), dim=-1)
    width = int(original.shape[-1])
    header_image = Image.new("RGB", (width * 2, HEADER_HEIGHT), color=(18, 18, 18))
    draw = ImageDraw.Draw(header_image)
    font = ImageFont.load_default(size=HEADER_FONT_SIZE)
    for column, label in enumerate(LABELS):
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        x = column * width + (width - text_width) // 2 - left
        y = (HEADER_HEIGHT - text_height) // 2 - top
        draw.text((x, y), label, font=font, fill="white")

    header_array = np.asarray(header_image, dtype=np.uint8).copy()
    header = torch.from_numpy(header_array).permute(2, 0, 1).float().div_(255.0)
    header = header.unsqueeze(0).expand(video.shape[0], -1, -1, -1)
    return torch.cat((header, video), dim=-2).contiguous()


def write_video(path: Path, video: torch.Tensor, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError("saving MP4 files requires imageio and imageio-ffmpeg") from error

    frames = (
        video.clamp(0.0, 1.0)
        .mul(255.0)
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
            fps=fps,
            codec="libx264",
            quality=9,
            macro_block_size=1,
            pixelformat="yuv420p",
            output_params=["-movflags", "+faststart"],
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    missing = [path for path in SAMPLES if not path.is_file()]
    if missing:
        paths = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"sample video(s) not found:\n{paths}")
    if not torch.cuda.is_available():
        raise RuntimeError("sampling requires CUDA")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint_path = VARIANTS[args.variant][1]
    print(f"Loading {checkpoint_path} on {device}", flush=True)
    model = load_model(args.variant, device)

    output_dir = PROJECT_ROOT / "outputs" / args.variant
    for index, input_path in enumerate(SAMPLES, start=1):
        output_path = output_dir / f"{input_path.stem}-comparison.mp4"
        original, fps = load_sample(input_path, args.num_frames)
        print(
            f"[{index}/{len(SAMPLES)}] Reconstructing {input_path.name} "
            f"({original.shape[0]} frames)",
            flush=True,
        )
        reconstruction = reconstruct(model, original, device)
        write_video(output_path, make_comparison(original, reconstruction), fps)
        print(f"[{index}/{len(SAMPLES)}] Saved {output_path}", flush=True)
        del original, reconstruction
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
