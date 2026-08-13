from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vrae.models.encoders.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    EncoderAdapter,
    EncoderSpec,
    load_torch_mapping,
    normalize_state_dict,
    resolve_checkpoint_path,
    resolve_source_path,
    validate_encoder_config,
)
from vrae.registry import ENCODERS

VJEPA2_1_SPEC = EncoderSpec(
    name="vjepa2_1",
    variant="vit_l16",
    layers=(11, 13, 15, 17, 19, 21, 23),
    fusion="mean_plus_final_spatial_mean",
    hidden_size=1024,
    num_blocks=24,
    patch_size=16,
    encoder_tubelet_size=2,
    pixel_normalization="imagenet",
)


@contextmanager
def _local_source_import_path(source_dir: Path) -> Iterator[None]:
    source_text = str(source_dir)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(source_text)


def _import_local_vision_transformer(source_dir: Path) -> Any:
    required = (
        source_dir / "app" / "vjepa_2_1" / "models" / "vision_transformer.py",
        source_dir / "app" / "vjepa_2_1" / "models" / "utils" / "modules.py",
        source_dir / "src" / "masks" / "utils.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete V-JEPA2.1 local source; missing={missing}")

    module_name = "app.vjepa_2_1.models.vision_transformer"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        module_file = Path(str(getattr(loaded, "__file__", ""))).resolve()
        try:
            module_file.relative_to(source_dir)
        except ValueError as error:
            raise RuntimeError(
                "V-JEPA2.1 module is already loaded from a different local checkout: "
                f"loaded={module_file} requested={source_dir}"
            ) from error
        return loaded

    with _local_source_import_path(source_dir):
        module = importlib.import_module(module_name)
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        module_file.relative_to(source_dir)
    except ValueError as error:
        raise RuntimeError(
            f"Imported V-JEPA2.1 from {module_file}, expected source under {source_dir}"
        ) from error
    return module


def load_local_vjepa2_1(*, source_dir: Path, checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V-JEPA2.1 checkpoint is not a file: {checkpoint_path}")
    with _local_source_import_path(source_dir):
        vision_transformer = _import_local_vision_transformer(source_dir)
        backbone = vision_transformer.vit_large(
            patch_size=16,
            img_size=(384, 384),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
    checkpoint = load_torch_mapping(checkpoint_path)
    raw_encoder = checkpoint.get("ema_encoder")
    if not isinstance(raw_encoder, Mapping):
        raise KeyError(
            f"V-JEPA2.1 checkpoint is missing mapping key 'ema_encoder': {checkpoint_path}"
        )
    state = normalize_state_dict(raw_encoder)
    result = backbone.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Strict V-JEPA2.1 load returned incompatibilities: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    return backbone


@ENCODERS.decorator("vjepa2_1")
class VJEPA21Adapter(EncoderAdapter):
    SPEC = VJEPA2_1_SPEC

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        paths: Any | None = None,
        source_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        if backbone is None:
            validate_encoder_config(config, self.SPEC)
            checkpoint = resolve_checkpoint_path(
                config, paths=paths, checkpoint_path=checkpoint_path
            )
            source = resolve_source_path("vjepa2_1", paths=paths, source_dir=source_dir)
            backbone = load_local_vjepa2_1(source_dir=source, checkpoint_path=checkpoint)
        super().__init__(
            config=config,
            backbone=backbone,
            normalization_mean=IMAGENET_MEAN,
            normalization_std=IMAGENET_STD,
        )

    def _validate_backbone(self) -> None:
        super()._validate_backbone()
        if len(getattr(self.backbone, "blocks", ())) != self.SPEC.num_blocks:
            raise ValueError("V-JEPA2.1 ViT-L backbone must contain exactly 24 blocks")
        if not callable(getattr(self.backbone, "patch_embed", None)):
            raise TypeError("V-JEPA2.1 backbone must expose patch_embed")
        tubelet_size = getattr(self.backbone, "tubelet_size", None)
        if tubelet_size is not None and int(tubelet_size) != self.SPEC.encoder_tubelet_size:
            raise ValueError("V-JEPA2.1 backbone must use tubelet size 2")
        norms = getattr(self.backbone, "norms_block", None)
        if not isinstance(norms, (nn.ModuleList, list, tuple)) or not norms:
            raise TypeError("V-JEPA2.1 backbone must expose official norms_block")
        official_layers = getattr(self.backbone, "out_layers_distillation", None)
        if official_layers is not None and list(official_layers) != [5, 11, 17, 23]:
            raise ValueError("V-JEPA2.1 backbone has unexpected official output-norm taps")

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        video = self._normalize_video(video)
        return video.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _block_hidden(output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError(f"Unexpected V-JEPA2.1 block output type: {type(output).__name__}")
        return output

    def _encode_preprocessed(
        self, video: torch.Tensor, *, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        batch, channels, time, height, width = (int(value) for value in video.shape)
        time_tokens = time // self.encoder_tubelet_size
        grid_h, grid_w = grid_size
        spatial_tokens = grid_h * grid_w
        expected_tokens = time_tokens * spatial_tokens

        hidden = self.backbone.patch_embed(video)
        if not torch.is_tensor(hidden) or tuple(hidden.shape) != (
            batch,
            expected_tokens,
            self.hidden_size,
        ):
            actual = tuple(hidden.shape) if torch.is_tensor(hidden) else type(hidden).__name__
            raise RuntimeError(
                "Unexpected V-JEPA2.1 patch embedding shape: "
                f"got={actual} expected={(batch, expected_tokens, self.hidden_size)}"
            )
        if bool(getattr(self.backbone, "modality_embedding", False)):
            embedding = self.backbone.video_mod_embed.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + embedding

        selected: list[torch.Tensor] = []
        selected_indices = set(self.SPEC.layers)
        final_norm = self.backbone.norms_block[-1]
        for index, block in enumerate(self.backbone.blocks):
            hidden = self._block_hidden(
                block(
                    hidden,
                    mask=None,
                    T=time_tokens,
                    H_patches=grid_h,
                    W_patches=grid_w,
                    return_attn=False,
                    mode="video",
                )
            )
            if index in selected_indices:
                selected.append(final_norm(hidden))
        if len(selected) != len(self.SPEC.layers):
            raise RuntimeError(
                f"Collected {len(selected)}/{len(self.SPEC.layers)} V-JEPA2.1 K7 outputs"
            )

        fused = torch.stack(selected, dim=0).mean(dim=0)
        final_grid = selected[-1].reshape(batch, time_tokens, spatial_tokens, self.hidden_size)
        final_spatial_mean = final_grid.mean(dim=2, keepdim=True)
        fused = fused.reshape(batch, time_tokens, spatial_tokens, self.hidden_size)
        fused = fused + final_spatial_mean
        return (
            fused.reshape(batch, time_tokens, grid_h, grid_w, self.hidden_size)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
