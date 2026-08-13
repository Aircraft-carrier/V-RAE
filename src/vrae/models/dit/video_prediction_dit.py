from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrae.models.dit.blocks import (
    AdaLNZeroDDTBlock,
    DDTDecoderBlock,
    DDTFinalLayer,
    FramePatchEmbed,
    GaussianFourierTimeEmbedding,
    as_pair,
    embed_video_frames,
    unpatchify_video_tokens,
    video_tokens_to_grid,
)
from vrae.models.dit.conditioning import ContextLatentConditionAdapter
from vrae.models.dit.video_dit import _architecture_pair, _factory_parameters
from vrae.models.rope3d import build_3d_positions
from vrae.registry import DIT_MODELS


def _prediction_grid_size(
    grid_size: int | Sequence[int],
    *,
    input_hw: Sequence[int] | None,
    input_size: int | Sequence[int] | None,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    result = as_pair(grid_size, "grid_size")
    for name, value in (("input_hw", input_hw), ("input_size", input_size)):
        if value is None:
            continue
        alias = as_pair(value, name)
        if result != (27, 48) and result != alias:
            raise ValueError(f"grid_size and {name} disagree")
        result = alias
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("height and width must be specified together")
        explicit = as_pair((height, width), "height/width")
        if result != (27, 48) and result != explicit:
            raise ValueError("grid_size and height/width disagree")
        result = explicit
    return result


class VRAEVideoPredictionDiT(nn.Module):
    """Cityscapes context-conditioned DDT that returns future chunks only."""

    def __init__(
        self,
        *,
        grid_size: int | Sequence[int] = (27, 48),
        in_channels: int = 1024,
        patch_size: int | Sequence[int] = 1,
        hidden_size: int | Sequence[int] = (1536, 2048),
        depth: int | Sequence[int] = (28, 2),
        num_heads: int | Sequence[int] = (24, 16),
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        context_dropout: float = 0.1,
        context_chunks: int = 3,
        future_chunks: int = 3,
        rope_theta: float = 10_000.0,
        base_model_depth: int = 8,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
        gradient_checkpointing: bool = False,
        input_dim: int | None = None,
        input_hw: Sequence[int] | None = None,
        input_size: int | Sequence[int] | None = None,
        height: int | None = None,
        width: int | None = None,
        encoder_depth: int | None = None,
        num_context_chunks: int | None = None,
        num_future_chunks: int | None = None,
        num_frames: int | None = None,
        use_gradient_checkpointing: bool | None = None,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            if in_channels != 1024 and int(input_dim) != int(in_channels):
                raise ValueError("input_dim and in_channels disagree")
            in_channels = int(input_dim)
        if encoder_depth is not None:
            if base_model_depth != 8 and int(encoder_depth) != int(base_model_depth):
                raise ValueError("encoder_depth and base_model_depth disagree")
            base_model_depth = int(encoder_depth)
        if num_context_chunks is not None:
            if context_chunks != 3 and int(num_context_chunks) != int(context_chunks):
                raise ValueError("num_context_chunks and context_chunks disagree")
            context_chunks = int(num_context_chunks)
        if num_future_chunks is not None:
            if future_chunks != 3 and int(num_future_chunks) != int(future_chunks):
                raise ValueError("num_future_chunks and future_chunks disagree")
            future_chunks = int(num_future_chunks)
        if use_gradient_checkpointing is not None:
            gradient_checkpointing = bool(use_gradient_checkpointing)

        self.context_chunks = int(context_chunks)
        self.future_chunks = int(future_chunks)
        if self.context_chunks <= 0 or self.future_chunks <= 0:
            raise ValueError("context_chunks and future_chunks must be positive")
        if num_frames is not None and int(num_frames) != self.total_chunks:
            raise ValueError("num_frames must equal context_chunks + future_chunks")
        self.grid_size = _prediction_grid_size(
            grid_size,
            input_hw=input_hw,
            input_size=input_size,
            height=height,
            width=width,
        )
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.patch_size = as_pair(patch_size, "patch_size")
        if any(size % patch for size, patch in zip(self.grid_size, self.patch_size, strict=True)):
            raise ValueError("grid_size must be divisible by patch_size")
        self.patch_grid = tuple(
            size // patch for size, patch in zip(self.grid_size, self.patch_size, strict=True)
        )
        self.tokens_per_chunk = self.grid_size[0] * self.grid_size[1]
        self.patches_per_chunk = self.patch_grid[0] * self.patch_grid[1]

        enc_dim, dec_dim = _architecture_pair(hidden_size, name="hidden_size")
        enc_depth, dec_depth = _architecture_pair(depth, name="depth", scalar_second=2)
        enc_heads, dec_heads = _architecture_pair(num_heads, name="num_heads")
        if enc_dim % enc_heads or dec_dim % dec_heads:
            raise ValueError("each hidden size must be divisible by its number of heads")
        self.enc_hidden_size = enc_dim
        self.dec_hidden_size = dec_dim
        self.num_enc_blocks = enc_depth
        self.num_dec_blocks = dec_depth
        self.base_model_depth = int(base_model_depth)
        if not 1 <= self.base_model_depth <= self.num_enc_blocks:
            raise ValueError(
                f"base_model_depth must be in [1,{self.num_enc_blocks}], "
                f"got {self.base_model_depth}"
            )
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.encoder_embed = FramePatchEmbed(self.in_channels, enc_dim, self.patch_size)
        self.decoder_embed = FramePatchEmbed(self.in_channels, dec_dim, self.patch_size)
        self.encoder_blocks = nn.ModuleList(
            [
                AdaLNZeroDDTBlock(
                    enc_dim,
                    enc_heads,
                    mlp_ratio,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(enc_depth)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DDTDecoderBlock(
                    dec_dim,
                    dec_heads,
                    mlp_ratio,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(dec_depth)
            ]
        )
        self.time_embedder = GaussianFourierTimeEmbedding(
            enc_dim,
            embedding_size=int(time_embed_dim),
        )
        self.condition_adapter = ContextLatentConditionAdapter(
            self.in_channels,
            enc_dim,
            dropout_prob=context_dropout,
        )
        self.encoder_to_decoder: nn.Module = (
            nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        )
        self.final_layer = DDTFinalLayer(dec_dim, self.patch_size, self.in_channels)
        self.base_final_layer = DDTFinalLayer(enc_dim, self.patch_size, self.in_channels)

        encoder_positions = build_3d_positions(
            self.total_chunks,
            self.patch_grid[0],
            self.patch_grid[1],
        )
        decoder_positions = build_3d_positions(
            self.future_chunks,
            self.patch_grid[0],
            self.patch_grid[1],
        )
        decoder_positions[:, 0] += self.context_chunks
        self.register_buffer("_encoder_positions", encoder_positions, persistent=False)
        self.register_buffer("_decoder_positions", decoder_positions, persistent=False)
        self._initialize_weights()

    @property
    def total_chunks(self) -> int:
        return self.context_chunks + self.future_chunks

    @property
    def encoder_sequence_length(self) -> int:
        return self.total_chunks * self.patches_per_chunk

    @property
    def future_sequence_length(self) -> int:
        return self.future_chunks * self.patches_per_chunk

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _initialize_weights(self) -> None:
        for embedder in (self.encoder_embed, self.decoder_embed):
            weight = embedder.proj.weight.data
            nn.init.xavier_uniform_(weight.reshape(weight.shape[0], -1))
            if embedder.proj.bias is not None:
                nn.init.zeros_(embedder.proj.bias)
        for layer in (self.condition_adapter.projection[0], self.condition_adapter.projection[2]):
            nn.init.normal_(layer.weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

    def _run_block(
        self,
        block: AdaLNZeroDDTBlock,
        value: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda current, cond: block(current, cond, positions),
                value,
                condition,
                use_reentrant=False,
            )
        return block(value, condition, positions)

    def _unpatchify_future(self, value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        return unpatchify_video_tokens(
            value,
            batch_size=batch_size,
            chunks=self.future_chunks,
            patch_grid=self.patch_grid,
            patch_size=self.patch_size,
            channels=self.in_channels,
        )

    def forward(
        self,
        noisy_future: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        context_latents: torch.Tensor | None = None,
        context_drop_mask: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
        condition_generator: torch.Generator | None = None,
        return_base: bool = False,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        if context is None:
            context = context_latents
        elif context_latents is not None:
            raise ValueError("pass context as either context or context_latents, not both")
        if context is None:
            raise ValueError("clean context latents are required")
        if context_drop_mask is not None and condition_drop_mask is not None:
            raise ValueError("pass only one of context_drop_mask or condition_drop_mask")
        drop_mask = context_drop_mask if context_drop_mask is not None else condition_drop_mask
        future_grid, batch = video_tokens_to_grid(
            noisy_future,
            chunks=self.future_chunks,
            grid_size=self.grid_size,
            channels=self.in_channels,
            name="noisy_future",
        )
        video_tokens_to_grid(
            context,
            chunks=self.context_chunks,
            grid_size=self.grid_size,
            channels=self.in_channels,
            name="context",
        )
        if context.shape[0] != batch:
            raise ValueError("context and noisy_future batch sizes must match")
        if context.device != noisy_future.device:
            raise ValueError("context and noisy_future must be on the same device")
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim != 1 or time.shape[0] != batch:
            raise ValueError(f"time must have shape [{batch}], got {tuple(time.shape)}")

        dropped_context, context_embedding, _ = self.condition_adapter.prepare(
            context,
            drop_mask=drop_mask,
            generator=condition_generator,
        )
        combined = torch.cat((dropped_context, noisy_future), dim=1)
        combined_grid, _ = video_tokens_to_grid(
            combined,
            chunks=self.total_chunks,
            grid_size=self.grid_size,
            channels=self.in_channels,
            name="context_and_future",
        )
        time_embedding = self.time_embedder(time.to(noisy_future.device))
        encoder_condition = time_embedding + context_embedding.to(time_embedding.dtype)
        sequence = embed_video_frames(
            self.encoder_embed,
            combined_grid,
            batch_size=batch,
            chunks=self.total_chunks,
        )
        base_sequence: torch.Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            sequence = self._run_block(
                block,
                sequence,
                encoder_condition,
                self._encoder_positions,
            )
            if index == self.base_model_depth:
                base_sequence = sequence
        if base_sequence is None:
            raise RuntimeError("base encoder activation was not captured")

        context_length = self.context_chunks * self.patches_per_chunk
        future_sequence = sequence[:, context_length:]
        decoder_condition = self.encoder_to_decoder(F.silu(time_embedding + future_sequence))
        decoder_sequence = embed_video_frames(
            self.decoder_embed,
            future_grid,
            batch_size=batch,
            chunks=self.future_chunks,
        )
        for block in self.decoder_blocks:
            decoder_sequence = self._run_block(
                block,
                decoder_sequence,
                decoder_condition,
                self._decoder_positions,
            )
        full = self._unpatchify_future(
            self.final_layer(decoder_sequence, decoder_condition),
            batch_size=batch,
        )
        base_future = base_sequence[:, context_length:]
        base_condition = F.silu(time_embedding + base_future)
        base = self._unpatchify_future(
            self.base_final_layer(base_condition, base_condition),
            batch_size=batch,
        )
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        result = (full, base) if return_base else full
        if return_intermediate:
            return result, base_future
        return result


@DIT_MODELS.decorator("vrae_video_prediction_dit")
def build_vrae_video_prediction_dit(
    *,
    config: Mapping[str, Any],
    **overrides: Any,
) -> VRAEVideoPredictionDiT:
    return VRAEVideoPredictionDiT(**_factory_parameters(config, overrides))


__all__ = ["VRAEVideoPredictionDiT", "build_vrae_video_prediction_dit"]
