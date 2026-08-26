from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vrae.models.decoder import VRAEDecoder
from vrae.models.pooling import TemporalAttentionPool
from vrae.registry import DECODERS, ENCODERS, MODELS, POOLERS, register_builtin_models

LOGGER = logging.getLogger(__name__)


def _image_decoder_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Image decoder checkpoint must contain a mapping: {path}")
    for key in ("state_dict", "model", "decoder"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    state: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "decoder.")
    for raw_key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        state[key] = value
    if not state:
        raise TypeError(f"Image decoder checkpoint has no tensor state dict: {path}")
    return state


@MODELS.decorator("vrae")
class VRAE(nn.Module):
    temporal_compression_ratio = 4

    def __init__(
        self, encoder: nn.Module, temporal_pool: TemporalAttentionPool, decoder: VRAEDecoder
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal_pool = temporal_pool
        self.decoder = decoder
        self.multiview_camera_keys: list[dict[str, Any]] = []
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.temporal_pool.requires_grad_(True)
        self.decoder.requires_grad_(True)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> VRAE:
        register_builtin_models()
        model_config = config.get("model", config)
        encoder_config = dict(model_config["encoder"])
        data_config = config.get("data", {})
        if isinstance(data_config, Mapping) and "image_size" in data_config:
            encoder_config["runtime_image_size"] = data_config["image_size"]
        encoder_kwargs = dict(kwargs)
        project_paths = encoder_kwargs.pop("project_paths", None)
        if project_paths is not None:
            if "paths" in encoder_kwargs:
                raise TypeError("Pass either project_paths or paths, not both")
            encoder_kwargs["paths"] = project_paths
        encoder = ENCODERS.build(encoder_config, **encoder_kwargs)
        metadata = encoder.metadata()
        pool = POOLERS.build(
            {**model_config["pooling"], "dim": int(metadata["hidden_size"])},
            dim=int(metadata["hidden_size"]),
            group_size=int(model_config["pooling"]["group_size"]),
        )
        compression = int(metadata["encoder_tubelet_size"]) * pool.group_size
        if compression != cls.temporal_compression_ratio:
            raise ValueError(
                f"Encoder tubelet and pooling group must compress time by 4, got {compression}"
            )
        decoder_value = dict(model_config["decoder"])
        multiview = model_config.get("multiview", {})
        if isinstance(multiview, Mapping):
            for key in ("multiview_enabled", "num_views", "num_streams", "use_view_embedding", "use_view_attention"):
                source = "enabled" if key == "multiview_enabled" else key
                if source in multiview and key not in decoder_value:
                    decoder_value[key] = multiview[source]
        decoder_value.setdefault("name", "vrae_decoder")
        init_mode = str(decoder_value.pop("init", "scratch"))
        init_checkpoint = decoder_value.pop("checkpoint", None)
        decoder_config = decoder_value.pop("parameters", decoder_value)
        if isinstance(multiview, Mapping):
            decoder_config = dict(decoder_config)
            for key in ("multiview_enabled", "num_views", "num_streams", "use_view_embedding", "use_view_attention"):
                source = "enabled" if key == "multiview_enabled" else key
                if source in multiview and key not in decoder_config:
                    decoder_config[key] = multiview[source]
        decoder = DECODERS.build(
            {"name": "vrae_decoder", **decoder_config}, input_dim=int(metadata["hidden_size"])
        )
        if decoder.config.tubelet_size != cls.temporal_compression_ratio:
            raise ValueError("V-RAE decoder tubelet_size must be 4")
        if init_mode == "raev2_image":
            if project_paths is None:
                raise ValueError("decoder.init=raev2_image requires project_paths")
            if init_checkpoint is None:
                raise ValueError("decoder.init=raev2_image requires decoder.checkpoint")
            checkpoint_path = project_paths.checkpoint(init_checkpoint, require_exists=True)
            report = decoder.load_image_decoder_weights(_image_decoder_state(checkpoint_path))
            missing = [key for key in report["missing"] if "view_embedding" not in key]
            if missing or report["unexpected"]:
                raise RuntimeError(
                    "Incomplete RAEv2 image decoder initialization from "
                    f"{checkpoint_path}: missing={missing} "
                    f"unexpected={report['unexpected']}"
                )
            decoder.initialization_report = report
            LOGGER.info("RAEv2 image decoder initialization from %s: %s", checkpoint_path, report)
        elif init_mode != "scratch":
            raise ValueError(f"Unknown decoder initialization mode: {init_mode}")
        elif init_checkpoint is not None:
            raise ValueError("decoder.checkpoint is only valid with decoder.init=raev2_image")
        instance = cls(encoder, pool, decoder)
        if isinstance(data_config, Mapping) and isinstance(data_config.get("camera_keys"), list):
            instance.multiview_camera_keys = [dict(item) if isinstance(item, Mapping) else {"key": str(item)} for item in data_config["camera_keys"]]
        return instance

    def train(self, mode: bool = True) -> VRAE:
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        self._validate_video(video)
        return self.encoder(video)

    def encode(self, video: torch.Tensor, stream_ids: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(self.decoder, "multiview_enabled", False):
            if video.ndim != 6:
                raise ValueError(f"Expected multiview video [B,T,V,3,H,W], got {tuple(video.shape)}")
            batch, time, views, channels, height, width = video.shape
            if channels != 3:
                raise ValueError("Multiview video must have RGB channels")
            flat = video.permute(0, 2, 1, 3, 4, 5).reshape(batch * views, time, channels, height, width)
            latent = self.temporal_pool(self.encode_frames(flat))
            _, chunks, latent_channels, latent_height, latent_width = latent.shape
            return latent.reshape(batch, views, chunks, latent_channels, latent_height, latent_width).permute(0, 2, 1, 3, 4, 5).contiguous()
        features = self.encode_frames(video)
        return self.temporal_pool(features)

    def decode(self, clean_latents: torch.Tensor, stream_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(clean_latents, stream_ids=stream_ids)

    def forward(self, video: torch.Tensor, stream_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        clean_latents = self.encode(video, stream_ids=stream_ids)
        return {"recon": self.decode(clean_latents, stream_ids=stream_ids), "latents": clean_latents}

    def trainable_groups(self) -> dict[str, nn.Module]:
        return {"temporal_pool": self.temporal_pool, "decoder": self.decoder}

    def metadata(self) -> dict[str, Any]:
        encoder_metadata = self.encoder.metadata()
        decoder = self.decoder.config
        return {
            **encoder_metadata,
            "pool_group": self.temporal_pool.group_size,
            "final_norm_affine": False,
            "decoder_input_dim": decoder.input_dim,
            "decoder_hidden_size": decoder.hidden_size,
            "decoder_depth": decoder.depth,
            "decoder_num_heads": decoder.num_heads,
            "decoder_mlp_ratio": decoder.mlp_ratio,
            "decoder_patch_size": decoder.patch_size,
            "decoder_tubelet": decoder.tubelet_size,
            "decoder_image_size": list(decoder.image_size),
            "decoder_num_channels": decoder.num_channels,
            "decoder_layer_norm_eps": decoder.layer_norm_eps,
            "decoder_attention_dropout": decoder.attention_dropout,
            "decoder_attention_mode": decoder.attention_mode,
            "decoder_attention_backend": decoder.attention_backend,
            "decoder_rope_theta": decoder.rope_theta,
            "decoder_spatial_position_kind": "parameter",
            "decoder_spatial_position_trainable_during_stage1": True,
            "decoder_spatial_position_resize": "bicubic",
            "decoder_execution": self.decoder.execution_metadata(),
            "temporal_compression_ratio": self.temporal_compression_ratio,
            "multiview_enabled": bool(getattr(self.decoder.config, "multiview_enabled", False)),
            "num_views": int(getattr(self.decoder.config, "num_views", 1)),
            "num_streams": int(getattr(self.decoder.config, "num_streams", 1)),
            "camera_keys": list(self.multiview_camera_keys),
        }

    @staticmethod
    def _validate_video(video: torch.Tensor) -> None:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Expected RGB video [B,T,3,H,W], got {tuple(video.shape)}")
        if video.shape[1] % 4:
            raise ValueError(
                "Input frame count must be divisible by four; padding/truncation is forbidden"
            )
