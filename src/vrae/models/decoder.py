from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.rope3d import apply_3d_rope_cache, build_3d_rope_cache
from vrae.registry import DECODERS

LOGGER = logging.getLogger(__name__)
ATTENTION_BACKENDS = {"sdpa", "fa3", "fa3_fwd", "fa4_cute", "auto"}
_FA3_MAX_HEAD_DIM = 96


@dataclass(frozen=True)
class DecoderConfig:
    input_dim: int = 1024
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    tubelet_size: int = 4
    image_size: tuple[int, int] = (256, 256)
    num_channels: int = 3
    layer_norm_eps: float = 1.0e-6
    attention_dropout: float = 0.0
    attention_mode: str = "chunk_causal"
    attention_backend: str = "auto"
    rope_theta: float = 10_000.0
    gradient_checkpointing: bool = False
    multiview_enabled: bool = False
    num_views: int = 1
    num_streams: int = 1
    use_view_embedding: bool = True
    use_view_attention: bool = True

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, input_dim: int | None = None
    ) -> DecoderConfig:
        data = dict(value)
        data.pop("name", None)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown decoder configuration fields: {unknown}")
        if input_dim is not None:
            data["input_dim"] = input_dim
        if "image_size" in data:
            data["image_size"] = tuple(int(part) for part in data["image_size"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if self.attention_mode not in {"chunk_causal", "full"}:
            raise ValueError("attention_mode must be chunk_causal or full")
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if len(self.image_size) != 2 or any(size % self.patch_size for size in self.image_size):
            raise ValueError("Both image dimensions must be divisible by patch_size")
        if self.num_views <= 0 or self.num_streams <= 0 or self.num_views > self.num_streams:
            raise ValueError("num_views must be positive and no greater than num_streams")


def _fa3_training() -> Callable[..., Any]:
    try:
        return importlib.import_module("flash_attn_interface").flash_attn_func
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Training-capable FA3 is unavailable") from error


def _fa3_forward_only() -> Callable[..., Any]:
    try:
        return importlib.import_module("fa3_fwd_interface").flash_attn_func
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Forward-only FA3 is unavailable") from error


def _fa4() -> Callable[..., Any]:
    try:
        return importlib.import_module("flash_attn.cute.interface").flash_attn_func
    except (ImportError, AttributeError) as error:
        raise RuntimeError("FA4 CuTe is unavailable") from error


def resolve_attention_backend(requested: str, tensor: torch.Tensor, *, training: bool) -> str:
    requested = requested.lower()
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(f"Unknown attention backend: {requested}")
    if requested == "sdpa":
        return requested
    if requested == "fa3":
        _fa3_training()
        return requested
    if requested == "fa3_fwd":
        if training:
            raise RuntimeError("A forward-only FA3 kernel cannot be used for training")
        _fa3_forward_only()
        return requested
    if requested == "fa4_cute":
        _fa4()
        return requested
    if tensor.is_cuda and tensor.dtype in {torch.float16, torch.bfloat16}:
        try:
            _fa4()
            return "fa4_cute"
        except RuntimeError:
            pass
        if tensor.dtype == torch.bfloat16 and int(tensor.shape[-1]) <= _FA3_MAX_HEAD_DIM:
            try:
                _fa3_training()
                return "fa3"
            except RuntimeError:
                pass
    return "sdpa"


def _call_flash(
    function: Callable[..., Any], query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()
    try:
        output = function(query, key, value, causal=False)
    except TypeError:
        output = function(query, key, value)
    if isinstance(output, tuple):
        output = output[0]
    return output.permute(0, 2, 1, 3).contiguous()


def _flash_attention(
    backend: str, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    if query.dtype not in {torch.float16, torch.bfloat16} or not query.is_cuda:
        raise ValueError(f"{backend} requires CUDA fp16/bf16 tensors")
    if backend == "fa3":
        function = _fa3_training()
        return _call_flash(function, query, key, value)
    if backend == "fa3_fwd":
        function = _fa3_forward_only()
        return _call_flash(function, query, key, value)
    function = _fa4()
    head_dim = query.shape[-1]
    if head_dim == 72 and any(item.requires_grad for item in (query, key, value)):
        padded_dim = 96
        # FlashAttention derives its scale from D96; compensate Q to preserve D72 logits.
        scale = math.sqrt(padded_dim / head_dim)
        query = F.pad(query, (0, padded_dim - head_dim)) * scale
        key = F.pad(key, (0, padded_dim - head_dim))
        value = F.pad(value, (0, padded_dim - head_dim))
        return _call_flash(function, query, key, value)[..., :head_dim]
    return _call_flash(function, query, key, value)


def chunk_prefix_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_chunks: int,
    tokens_per_chunk: int,
    backend: str,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    expected = num_chunks * tokens_per_chunk
    if any(item.ndim != 4 or item.shape[-2] != expected for item in (query, key, value)):
        raise ValueError(f"Q/K/V must be [B,heads,{expected},head_dim]")
    batch, heads, _, head_dim = query.shape
    q = query.reshape(batch, heads, num_chunks, tokens_per_chunk, head_dim)
    k = key.reshape(batch, heads, num_chunks, tokens_per_chunk, head_dim)
    v = value.reshape(batch, heads, num_chunks, tokens_per_chunk, head_dim)
    outputs = []
    for chunk_index in range(num_chunks):
        q_chunk = q[:, :, chunk_index]
        k_prefix = k[:, :, : chunk_index + 1].reshape(
            batch, heads, (chunk_index + 1) * tokens_per_chunk, head_dim
        )
        v_prefix = v[:, :, : chunk_index + 1].reshape_as(k_prefix)
        if backend == "sdpa":
            output = F.scaled_dot_product_attention(
                q_chunk,
                k_prefix,
                v_prefix,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            if dropout_p:
                raise ValueError(f"{backend} does not support attention dropout here")
            output = _flash_attention(backend, q_chunk, k_prefix, v_prefix)
        outputs.append(output)
    return torch.cat(outputs, dim=2)


def _sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Position embedding dimension must be even")
    frequency = 1.0 / (10_000 ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
    phase = positions.float().reshape(-1, 1) * frequency.reshape(1, -1)
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def spatial_sincos(dim: int, height: int, width: int, *, device: torch.device) -> torch.Tensor:
    if dim % 4:
        raise ValueError("Decoder hidden size must be divisible by four for 2D sin/cos positions")
    y, x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    # RAEv2 constructs ``meshgrid(grid_w, grid_h)`` and stores the x channels
    # before the y channels.  Keep that ordering so scratch initialization is
    # numerically identical to the image decoder source.
    return torch.cat((_sincos_1d(dim // 2, x), _sincos_1d(dim // 2, y)), dim=-1).reshape(
        1, height * width, dim
    )


class DecoderAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self._reported_backend: str | None = None
        self._reported_dtype: str | None = None
        self.last_padding_route = "none"

    def forward(
        self,
        x: torch.Tensor,
        *,
        chunks: int,
        height: int,
        width: int,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
        views: int = 1,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        heads = self.config.num_heads
        head_dim = channels // heads
        qkv = self.qkv(x).reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        self._reported_dtype = str(query.dtype).removeprefix("torch.")
        query = apply_3d_rope_cache(query, rope_cosine, rope_sine)
        key = apply_3d_rope_cache(key, rope_cosine, rope_sine)
        backend = resolve_attention_backend(
            self.config.attention_backend, query, training=self.training
        )
        self.last_padding_route = (
            "d72_to_d96_q_scale"
            if backend == "fa4_cute"
            and head_dim == 72
            and any(item.requires_grad for item in (query, key, value))
            else "none"
        )
        if backend != self._reported_backend:
            LOGGER.info("VRAEDecoder attention backend: %s", backend)
            self._reported_backend = backend
        dropout = self.config.attention_dropout if self.training else 0.0
        if self.config.attention_mode == "chunk_causal":
            result = chunk_prefix_attention(
                query,
                key,
                value,
                num_chunks=chunks,
                tokens_per_chunk=views * height * width,
                backend=backend,
                dropout_p=dropout,
            )
        elif backend == "sdpa":
            result = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, is_causal=False
            )
        else:
            if dropout:
                raise ValueError(f"{backend} does not support attention dropout here")
            result = _flash_attention(backend, query, key, value)
        result = result.transpose(1, 2).reshape(batch, tokens, channels)
        return self.out(result)


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DecoderAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        intermediate = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, config.hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        chunks: int,
        height: int,
        width: int,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
        views: int = 1,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.norm1(x),
            chunks=chunks,
            height=height,
            width=width,
            views=views,
            rope_cosine=rope_cosine,
            rope_sine=rope_sine,
        )
        return x + self.mlp(self.norm2(x))


@DECODERS.decorator("vrae_decoder")
class VRAEDecoder(nn.Module):
    def __init__(
        self,
        config: DecoderConfig | Mapping[str, Any],
        *,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, DecoderConfig):
            config = DecoderConfig.from_mapping(config, input_dim=input_dim)
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_size)
        nominal_height = config.image_size[0] // config.patch_size
        nominal_width = config.image_size[1] // config.patch_size
        self._nominal_grid = (nominal_height, nominal_width)
        nominal_position = spatial_sincos(
            config.hidden_size, nominal_height, nominal_width, device=torch.device("cpu")
        )
        # The public RAEv2 decoder declares this as a Parameter with gradients
        # initially disabled, then enables the whole decoder for reconstruction
        # training. It must remain a Parameter so that transition is possible.
        self.decoder_spatial_pos_embed = nn.Parameter(nominal_position, requires_grad=False)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        output_dim = config.tubelet_size * config.patch_size**2 * config.num_channels
        self.prediction = nn.Linear(config.hidden_size, output_dim)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.multiview_enabled = bool(config.multiview_enabled)
        self.view_embedding = nn.Embedding(config.num_streams, config.hidden_size)
        nn.init.zeros_(self.view_embedding.weight)
        self._compiled_forward: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.register_buffer("_rope_cosine", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sine", torch.empty(0), persistent=False)
        self._rope_cache_key: tuple[int, int, int, torch.device, torch.dtype] | None = None

    def enable_compile(self, **kwargs: Any) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable")
        self._compiled_forward = torch.compile(self._forward_impl, **kwargs)

    @torch.no_grad()
    def prime_rope_cache(
        self,
        *,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if int(num_frames) <= 0 or int(num_frames) % self.config.tubelet_size:
            raise ValueError("num_frames must be a positive multiple of decoder tubelet_size")
        chunks = int(num_frames) // self.config.tubelet_size
        height, width = self._nominal_grid
        reference = torch.empty((), device=device, dtype=dtype)
        self._rope_cache(chunks, height, width, reference)

    def _position(self, height: int, width: int, x: torch.Tensor) -> torch.Tensor:
        if self._nominal_grid == (height, width):
            position = self.decoder_spatial_pos_embed
        else:
            nominal_height, nominal_width = self._nominal_grid
            position = self.decoder_spatial_pos_embed.reshape(
                1, nominal_height, nominal_width, self.config.hidden_size
            ).permute(0, 3, 1, 2)
            position = F.interpolate(
                position.float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).to(dtype=self.decoder_spatial_pos_embed.dtype)
            position = position.permute(0, 2, 3, 1).reshape(
                1, height * width, self.config.hidden_size
            )
        return position.to(device=x.device, dtype=x.dtype)

    def _rope_cache(
        self,
        chunks: int,
        height: int,
        width: int,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (chunks, height, width, hidden.device, hidden.dtype)
        if self._rope_cache_key != key:
            self._rope_cosine, self._rope_sine = build_3d_rope_cache(
                chunks,
                height,
                width,
                self.config.hidden_size // self.config.num_heads,
                device=hidden.device,
                dtype=hidden.dtype,
                theta=self.config.rope_theta,
            )
            self._rope_cache_key = key
        return self._rope_cosine, self._rope_sine

    def depatchify(
        self, tokens: torch.Tensor, *, chunks: int, height: int, width: int
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        patch = self.config.patch_size
        tubelet = self.config.tubelet_size
        channels = self.config.num_channels
        tokens = tokens.reshape(batch, chunks, height, width, tubelet, patch, patch, channels)
        tokens = tokens.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
        return tokens.reshape(batch, chunks * tubelet, channels, height * patch, width * patch)

    def _forward_impl(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Expected [B,Tlatent,C,H,W], got {tuple(latents.shape)}")
        batch, chunks, channels, height, width = latents.shape
        if channels != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} latent channels, got {channels}")
        hidden = latents.permute(0, 1, 3, 4, 2).reshape(batch, chunks * height * width, channels)
        hidden = self.input_projection(hidden)
        position = self._position(height, width, hidden)
        hidden = hidden.reshape(batch, chunks, height * width, -1)
        hidden = (hidden + position[:, None]).reshape(batch, chunks * height * width, -1)
        rope_cosine, rope_sine = self._rope_cache(chunks, height, width, hidden)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    block,
                    hidden,
                    chunks,
                    height,
                    width,
                    rope_cosine,
                    rope_sine,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden,
                    chunks,
                    height,
                    width,
                    rope_cosine,
                    rope_sine,
                )
        predicted = self.prediction(self.norm(hidden))
        return self.depatchify(predicted, chunks=chunks, height=height, width=width)

    def _forward_multiview(self, latents: torch.Tensor, stream_ids: torch.Tensor | None) -> torch.Tensor:
        if latents.ndim != 6:
            raise ValueError(f"Expected [B,Tlatent,V,C,H,W], got {tuple(latents.shape)}")
        batch, chunks, views, channels, height, width = latents.shape
        if views != self.config.num_views:
            raise ValueError(f"Expected {self.config.num_views} views, got {views}")
        if channels != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} latent channels, got {channels}")
        if stream_ids is None:
            stream_ids = torch.arange(views, device=latents.device).expand(batch, views)
        if stream_ids.shape != (batch, views):
            raise ValueError(f"stream_ids must have shape [{batch},{views}], got {tuple(stream_ids.shape)}")
        if stream_ids.min() < 0 or stream_ids.max() >= self.config.num_streams:
            raise ValueError("stream_ids are outside the configured num_streams")
        if not self.config.use_view_attention:
            flattened = latents.permute(0, 2, 1, 3, 4, 5).reshape(batch * views, chunks, channels, height, width)
            output = self._forward_impl(flattened)
            return output.reshape(batch, views, output.shape[1], output.shape[2], output.shape[3], output.shape[4]).permute(0, 2, 1, 3, 4, 5).contiguous()
        hidden = latents.permute(0, 1, 2, 4, 5, 3).reshape(batch, chunks * views * height * width, channels)
        hidden = self.input_projection(hidden)
        position = self._position(height, width, hidden).reshape(1, 1, height * width, -1)
        hidden = hidden.reshape(batch, chunks, views, height * width, -1)
        hidden = hidden + position[:, None, None]
        if self.config.use_view_embedding:
            hidden = hidden + self.view_embedding(stream_ids).to(hidden.dtype)[:, None, :, None]
        hidden = hidden.reshape(batch, chunks * views * height * width, -1)
        rope_cosine, rope_sine = self._rope_cache(chunks, height, width, hidden[:, : chunks * height * width])
        rope_cosine = rope_cosine.repeat_interleave(views, dim=0)
        rope_sine = rope_sine.repeat_interleave(views, dim=0)
        for block in self.blocks:
            hidden = block(hidden, chunks, height, width, rope_cosine, rope_sine, views)
        predicted = self.prediction(self.norm(hidden)).reshape(batch, chunks, views, height * width, -1)
        predicted = predicted.permute(0, 2, 1, 3, 4).reshape(batch * views, chunks * height * width, -1)
        output = self.depatchify(predicted, chunks=chunks, height=height, width=width)
        return output.reshape(batch, views, output.shape[1], output.shape[2], output.shape[3], output.shape[4]).permute(0, 2, 1, 3, 4, 5).contiguous()

    def forward(self, latents: torch.Tensor, *, stream_ids: torch.Tensor | None = None) -> torch.Tensor:
        return (
            self._compiled_forward(latents)
            if self._compiled_forward is not None
            else self._forward_multiview(latents, stream_ids) if self.multiview_enabled else self._forward_impl(latents)
        )

    def execution_metadata(self) -> dict[str, Any]:
        resolved = {
            block.attention._reported_backend
            for block in self.blocks
            if block.attention._reported_backend is not None
        }
        dtypes = {
            block.attention._reported_dtype
            for block in self.blocks
            if block.attention._reported_dtype is not None
        }
        routes = {block.attention.last_padding_route for block in self.blocks}
        return {
            "attention_backend_requested": self.config.attention_backend,
            "attention_backend_resolved": (
                next(iter(resolved)) if len(resolved) == 1 else sorted(resolved) or "unresolved"
            ),
            "attention_dtype": (
                next(iter(dtypes)) if len(dtypes) == 1 else sorted(dtypes) or "unresolved"
            ),
            "attention_head_dim": self.config.hidden_size // self.config.num_heads,
            "attention_padding_route": (next(iter(routes)) if len(routes) == 1 else sorted(routes)),
            "rope_cache_scope": "decoder_instance",
        }

    def load_image_decoder_weights(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, list[str]]:
        own = self.state_dict()
        loadable: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()
        inflated: list[str] = []
        skipped: list[str] = []
        position = state_dict.get("decoder_pos_embed")
        if (
            torch.is_tensor(position)
            and position.ndim == 3
            and position.shape[1] >= own["decoder_spatial_pos_embed"].shape[1] + 1
        ):
            spatial_tokens = own["decoder_spatial_pos_embed"].shape[1]
            spatial = position[:, 1 : 1 + spatial_tokens]
            if spatial.shape == own["decoder_spatial_pos_embed"].shape:
                loadable["decoder_spatial_pos_embed"] = spatial
                consumed.add("decoder_pos_embed")
                inflated.append("decoder_pos_embed -> decoder_spatial_pos_embed")
        block_aliases = {
            "layernorm_before.weight": "norm1.weight",
            "layernorm_before.bias": "norm1.bias",
            "layernorm_after.weight": "norm2.weight",
            "layernorm_after.bias": "norm2.bias",
            "attention.output.dense.weight": "attention.out.weight",
            "attention.output.dense.bias": "attention.out.bias",
            "intermediate.dense.weight": "mlp.0.weight",
            "intermediate.dense.bias": "mlp.0.bias",
            "output.dense.weight": "mlp.2.weight",
            "output.dense.bias": "mlp.2.bias",
        }
        for index in range(self.config.depth):
            source_prefix = f"decoder_layers.{index}."
            target_prefix = f"blocks.{index}."
            for source_suffix, target_suffix in block_aliases.items():
                source_key = source_prefix + source_suffix
                target_key = target_prefix + target_suffix
                value = state_dict.get(source_key)
                if torch.is_tensor(value) and value.shape == own[target_key].shape:
                    loadable[target_key] = value
                    consumed.add(source_key)
            qkv_weight_keys = tuple(
                source_prefix + f"attention.attention.{name}.weight"
                for name in ("query", "key", "value")
            )
            qkv_bias_keys = tuple(
                source_prefix + f"attention.attention.{name}.bias"
                for name in ("query", "key", "value")
            )
            if all(torch.is_tensor(state_dict.get(key)) for key in qkv_weight_keys):
                qkv_weight = torch.cat([state_dict[key] for key in qkv_weight_keys], dim=0)
                target_key = target_prefix + "attention.qkv.weight"
                if qkv_weight.shape == own[target_key].shape:
                    loadable[target_key] = qkv_weight
                    consumed.update(qkv_weight_keys)
            if all(torch.is_tensor(state_dict.get(key)) for key in qkv_bias_keys):
                qkv_bias = torch.cat([state_dict[key] for key in qkv_bias_keys], dim=0)
                target_key = target_prefix + "attention.qkv.bias"
                if qkv_bias.shape == own[target_key].shape:
                    loadable[target_key] = qkv_bias
                    consumed.update(qkv_bias_keys)
        aliases = {
            "decoder_embed.weight": "input_projection.weight",
            "decoder_embed.bias": "input_projection.bias",
            "decoder_norm.weight": "norm.weight",
            "decoder_norm.bias": "norm.bias",
            "decoder_pred.weight": "prediction.weight",
            "decoder_pred.bias": "prediction.bias",
        }
        for source_key, value in state_dict.items():
            if source_key in consumed:
                continue
            if source_key in {"decoder_pos_embed", "trainable_cls_token"}:
                skipped.append(source_key)
                continue
            target_key = aliases.get(source_key, source_key)
            if target_key not in own:
                skipped.append(source_key)
                continue
            target = own[target_key]
            if value.shape == target.shape:
                loadable[target_key] = value
                consumed.add(source_key)
            elif (
                target_key in {"prediction.weight", "prediction.bias"}
                and target.shape[0] == value.shape[0] * self.config.tubelet_size
            ):
                loadable[target_key] = value.repeat(
                    self.config.tubelet_size, *([1] * (value.ndim - 1))
                )
                consumed.add(source_key)
                inflated.append(source_key)
            else:
                skipped.append(source_key)
        missing, unexpected = self.load_state_dict(loadable, strict=False)
        return {
            "loaded": sorted(loadable),
            "inflated": sorted(inflated),
            "skipped": sorted(set(skipped)),
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
        }
