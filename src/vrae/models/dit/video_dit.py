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
    VideoDiTRoPECache,
    as_pair,
    embed_video_frames,
    unpatchify_video_tokens,
    video_tokens_to_grid,
)
from vrae.models.dit.conditioning import LabelConditionAdapter
from vrae.models.rope3d import (
    build_3d_positions,
    build_video_dit_3d_rope_cache_from_positions,
)
from vrae.registry import DIT_MODELS


def _architecture_pair(
    value: int | Sequence[int],
    *,
    name: str,
    scalar_second: int | None = None,
) -> tuple[int, int]:
    if isinstance(value, int):
        if scalar_second is None:
            second = int(value)
        else:
            second = int(scalar_second)
        result = (int(value), second)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a length-2 sequence")
        result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def _resolve_grid_size(
    grid_size: int | Sequence[int],
    *,
    input_size: int | Sequence[int] | None,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    result = as_pair(grid_size, "grid_size")
    if input_size is not None:
        alias = as_pair(input_size, "input_size")
        if (
            result != (16, 16)
            and result != alias
        ):
            raise ValueError("grid_size and input_size disagree")
        result = alias
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("height and width must be specified together")
        explicit = (int(height), int(width))
        if (
            result != (16, 16)
            and result != explicit
        ):
            raise ValueError("grid_size and height/width disagree")
        result = as_pair(explicit, "height/width")
    return result

# Shape notation used below:
#   B = batch size
#   T = number of temporal chunks (self.num_chunks)
#   V = number of views (self.num_views)
#   N = number of spatial tokens per chunk
#   C = input channel count (self.in_channels)
#   P = number of patches per chunk
#   E = encoder hidden size
#   D = decoder hidden size
#   L = sequence length: T*P for single-view, T*V*P for multiview
#
# The diffusion or flow-matching transport supplies the noisy latent:
#   single-view: noisy [B,T,N,C]
#   multiview:   noisy [B,T,V,N,C]
#
# Overall data flow:
#
#   noisy
#      │
#      ├──► encoder patch embedding
#      │         │
#      │         ▼
#      │      Encoder blocks
#      │         │
#      │         ├──► base_sequence
#      │         │       └──► base_final_layer ──► base
#      │         │
#      │         └──► encoder final sequence
#      │                    │
#      │                    └──► encoder_to_decoder
#      │                                  │
#      └──► decoder patch embedding       ▼
#                    │              decoder_condition
#                    ▼                    │
#                Decoder blocks ◄─────────┘
#                    │
#                    ▼
#                final_layer ──► full
#
#   Possible outputs:
#   full
#   (full, base)
#   (full, base_sequence)
#   ((full, base), base_sequence)
class VRAEVideoDiT(nn.Module):
    """Class-conditional VideoDiT over frozen V-RAE latent tokens.

    Single-view inputs and outputs use ``[B,T,H*W,C]``. Multiview inputs and
    outputs use ``[B,T,V,H*W,C]``.
    """

    def __init__(
        self,
        *,
        num_chunks: int = 4,
        grid_size: int | Sequence[int] = (16, 16),
        in_channels: int = 1024,
        patch_size: int | Sequence[int] = 1,
        hidden_size: int | Sequence[int] = (1536, 2048),
        depth: int | Sequence[int] = (28, 2),
        num_heads: int | Sequence[int] = (24, 16),
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        num_classes: int = 40,
        class_dropout: float = 0.1,
        rope_theta: float = 10_000.0,
        base_model_depth: int = 8,
        attention_dropout: float = 0.0,
        attention_backend: str = "auto",
        gradient_checkpointing: bool = False,
        input_dim: int | None = None,
        num_frames: int | None = None,
        input_size: int | Sequence[int] | None = None,
        height: int | None = None,
        width: int | None = None,
        encoder_depth: int | None = None,
        condition_dropout: float | None = None,
        use_gradient_checkpointing: bool | None = None,
        multiview_enabled: bool = False,
        num_views: int = 1,
        num_streams: int = 1,
        use_view_embedding: bool = True,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            if (
                in_channels != 1024
                and int(input_dim) != int(in_channels)
            ):
                raise ValueError("input_dim and in_channels disagree")
            in_channels = int(input_dim)
        if num_frames is not None:
            if (
                num_chunks != 4
                and int(num_frames) != int(num_chunks)
            ):
                raise ValueError("num_frames and num_chunks disagree")
            num_chunks = int(num_frames)
        if encoder_depth is not None:
            if (
                base_model_depth != 8
                and int(encoder_depth) != int(base_model_depth)
            ):
                raise ValueError("encoder_depth and base_model_depth disagree")
            base_model_depth = int(encoder_depth)
        if condition_dropout is not None:
            if (
                class_dropout != 0.1
                and float(condition_dropout) != float(class_dropout)
            ):
                raise ValueError("condition_dropout and class_dropout disagree")
            class_dropout = float(condition_dropout)
        if use_gradient_checkpointing is not None:
            gradient_checkpointing = bool(use_gradient_checkpointing)

        self.num_chunks = int(num_chunks)
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        self.grid_size = _resolve_grid_size(
            grid_size,
            input_size=input_size,
            height=height,
            width=width,
        )
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.patch_size = as_pair(patch_size, "patch_size")
        if any(
            size % patch
            for size, patch in zip(self.grid_size, self.patch_size, strict=True)
        ):
            raise ValueError("grid_size must be divisible by patch_size")
        self.patch_grid = tuple(
            size // patch
            for size, patch in zip(self.grid_size, self.patch_size, strict=True)
        )
        self.tokens_per_chunk = self.grid_size[0] * self.grid_size[1]
        self.patches_per_chunk = self.patch_grid[0] * self.patch_grid[1]

        enc_dim, dec_dim = _architecture_pair(hidden_size, name="hidden_size")
        enc_depth, dec_depth = _architecture_pair(depth, name="depth", scalar_second=2)
        enc_heads, dec_heads = _architecture_pair(num_heads, name="num_heads")
        if enc_dim % enc_heads or dec_dim % dec_heads:
            raise ValueError(
                "each hidden size must be divisible by its number of heads"
            )
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
        self.num_classes = int(num_classes)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.multiview_enabled = bool(multiview_enabled)
        self.num_views = int(num_views)
        self.num_streams = int(num_streams)
        self.use_view_embedding = bool(use_view_embedding)
        if self.num_views <= 0 or self.num_streams < self.num_views:
            raise ValueError(
                "num_views must be positive and no greater than num_streams"
            )

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
        self.condition_adapter = LabelConditionAdapter(
            enc_dim,
            self.num_classes,
            dropout_prob=class_dropout,
        )
        if enc_dim != dec_dim:
            self.encoder_to_decoder: nn.Module = nn.Linear(enc_dim, dec_dim)
        else:
            self.encoder_to_decoder = nn.Identity()
        self.final_layer = DDTFinalLayer(
            dec_dim,
            self.patch_size,
            self.in_channels,
        )
        self.base_final_layer = DDTFinalLayer(
            enc_dim,
            self.patch_size,
            self.in_channels,
        )
        self.encoder_view_embedding = nn.Embedding(self.num_streams, enc_dim)
        self.decoder_view_embedding = nn.Embedding(self.num_streams, dec_dim)
        nn.init.zeros_(self.encoder_view_embedding.weight)
        nn.init.zeros_(self.decoder_view_embedding.weight)

        positions = build_3d_positions(
            self.num_chunks,
            self.patch_grid[0],
            self.patch_grid[1],
        )
        self.register_buffer("_encoder_positions", positions, persistent=False)
        self.register_buffer("_decoder_positions", positions.clone(), persistent=False)
        self._encoder_rope_cache: tuple[object, ...] | None = None
        self._decoder_rope_cache: tuple[object, ...] | None = None
        self._initialize_weights()

    @property
    def num_visual_tokens(self) -> int:
        return self.num_chunks * self.patches_per_chunk

    @property
    def s_embedder(self) -> FramePatchEmbed:
        return self.encoder_embed

    @property
    def x_embedder(self) -> FramePatchEmbed:
        return self.decoder_embed

    @property
    def t_embedder(self) -> GaussianFourierTimeEmbedding:
        return self.time_embedder

    @property
    def ctx_embedder(self) -> LabelConditionAdapter:
        return self.condition_adapter

    @property
    def s_projector(self) -> nn.Module:
        return self.encoder_to_decoder

    @property
    def blocks(self) -> tuple[AdaLNZeroDDTBlock, ...]:
        return (
            tuple(self.encoder_blocks)
            + tuple(self.decoder_blocks)
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _initialize_weights(self) -> None:
        for embedder in (self.encoder_embed, self.decoder_embed):
            weight = embedder.proj.weight.data
            nn.init.xavier_uniform_(weight.reshape(weight.shape[0], -1))
            if embedder.proj.bias is not None:
                nn.init.zeros_(embedder.proj.bias)
        nn.init.normal_(self.condition_adapter.embedding.weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

    def _run_conditioned_block(
        self,
        block: AdaLNZeroDDTBlock,
        value: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        rope_cache: VideoDiTRoPECache,
    ) -> torch.Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                lambda current, cond: block(
                    current,
                    cond,
                    positions,
                    rope_cache=rope_cache,
                ),
                value,
                condition,
                use_reentrant=False,
            )
        return block(
            value,
            condition,
            positions,
            rope_cache=rope_cache,
        )

    def _cached_rope(
        self,
        *,
        scope: str,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        block: AdaLNZeroDDTBlock,
    ) -> VideoDiTRoPECache:
        if positions.device != hidden.device:
            raise ValueError(
                "VideoDiT positions and hidden states must be on the same device"
            )
        attribute = f"_{scope}_rope_cache"
        cached = getattr(self, attribute)
        head_dim = block.attn.head_dim
        theta = block.attn.rope_theta
        if (
            cached is None
            or cached[0] is not positions
            or cached[1] != hidden.dtype
            or cached[2] != head_dim
            or cached[3] != theta
        ):
            cosine, sine = build_video_dit_3d_rope_cache_from_positions(
                positions,
                head_dim,
                theta=theta,
                dtype=hidden.dtype,
            )
            cached = (positions, hidden.dtype, head_dim, theta, cosine, sine)
            setattr(self, attribute, cached)
        return cached[4], cached[5]

    def _unpatchify(self, value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        return unpatchify_video_tokens(
            value,
            batch_size=batch_size,
            chunks=self.num_chunks,
            patch_grid=self.patch_grid,
            patch_size=self.patch_size,
            channels=self.in_channels,
        )

    def forward(
        self,
        noisy: torch.Tensor,
        time: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
        class_drop_mask: torch.Tensor | None = None,
        condition_generator: torch.Generator | None = None,
        stream_ids: torch.Tensor | None = None,
        return_base: bool = False,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        if labels is None:
            labels = context
        elif context is not None:
            raise ValueError("pass class labels as either labels or context, not both")
        if labels is None:
            raise ValueError("class labels are required")
        if condition_drop_mask is not None and class_drop_mask is not None:
            raise ValueError("pass only one of condition_drop_mask or class_drop_mask")
        if class_drop_mask is not None:
            drop_mask = class_drop_mask
        else:
            drop_mask = condition_drop_mask
        # Shape notation:
        #   B = batch size
        #   T = self.num_chunks (temporal chunks)
        #   V = self.num_views (views)
        #   H,W = self.grid_size
        #   C = self.in_channels
        #   N = H*W (input spatial tokens)
        multiview = self.multiview_enabled
        if multiview:
            # Multiview inputs follow the [B,T,V,N,C] layout.
            if (
                noisy.ndim != 5
                or tuple(noisy.shape[1:3]) != (self.num_chunks, self.num_views)
            ):
                raise ValueError(
                    f"noisy must have shape [B,{self.num_chunks},{self.num_views},N,C]"
                )

            batch = noisy.shape[0]
            expected_tokens = self.grid_size[0] * self.grid_size[1]

            # Check the spatial token count N and channel count C.
            if tuple(noisy.shape[3:]) != (
                expected_tokens,
                self.in_channels,
            ):
                raise ValueError("noisy token grid has incorrect patch shape")

            # Treat each view as an independent sample for the frame patch embedder.
            #
            # Intended shape change:
            #   [B,T,V,N,C] -> [B*V,T,N,C]
            #
            # The input is [B,T,V,N,C]. Move V before T before merging B and V;
            # a direct reshape would interleave temporal chunks and views.
            grid = (
                noisy.permute(0, 2, 1, 3, 4)
                .contiguous()
                .reshape(
                    batch * self.num_views,
                    self.num_chunks,
                    expected_tokens,
                    self.in_channels,
                )
            )

            # Restore the flattened N=H*W tokens to a 2D spatial grid:
            #   [B*V,T,N,C] -> [B*V,T,H,W,C]
            grid = grid.reshape(
                batch * self.num_views,
                self.num_chunks,
                self.grid_size[0],
                self.grid_size[1],
                self.in_channels,
            )

            # Move channels to the standard PyTorch image-tensor position:
            #   [B*V,T,H,W,C] -> [B*V,T,C,H,W]
            #
            # Then merge temporal chunks into the batch dimension:
            #   [B*V,T,C,H,W] -> [B*V*T,C,H,W]
            grid = (
                grid.permute(0, 1, 4, 2, 3)
                .reshape(
                    batch * self.num_views * self.num_chunks,
                    self.in_channels,
                    *self.grid_size,
                )
            )
        else:
            # The helper validates and converts the single-view layout:
            #   noisy [B,T,H*W,C]
            #       -> grid [B*T,C,H,W]
            grid, batch = video_tokens_to_grid(
                noisy,
                chunks=self.num_chunks,
                grid_size=self.grid_size,
                channels=self.in_channels,
                name="noisy",
            )
        # `time` may be a scalar or one value per batch element.
        # A scalar with shape [] is expanded to [B].
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim != 1 or time.shape[0] != batch:
            raise ValueError(f"time must have shape [{batch}], got {tuple(time.shape)}")
        if labels.shape != (batch,):
            raise ValueError(
                f"labels must have shape [{batch}], got {tuple(labels.shape)}"
            )
        if multiview:
            # If stream IDs are omitted, use 0,1,...,V-1 by default.
            if stream_ids is None:
                stream_ids = torch.arange(
                    self.num_views,
                    device=noisy.device,
                ).expand(batch, self.num_views)
            if stream_ids.shape != (batch, self.num_views):
                raise ValueError(
                    f"stream_ids must have shape [{batch},{self.num_views}]"
                )
            # Stream IDs must be valid embedding indices.
            if stream_ids.min() < 0 or stream_ids.max() >= self.num_streams:
                raise ValueError("stream_ids are outside num_streams")
        # Move time to the input device. The embedding has shape [B,E], where
        # E is self.enc_hidden_size.
        time_embedding = self.time_embedder(time.to(noisy.device))
        # `prepare` applies label dropout and returns the label embedding.
        # The other two return values are not needed here.
        label_embedding, _, _ = self.condition_adapter.prepare(
            labels,
            drop_mask=drop_mask,
            generator=condition_generator,
        )
        # Match the label embedding dtype to the time embedding before adding:
        #   time_embedding  [B,E]
        #   label_embedding [B,E]
        #   encoder_condition [B,E]
        encoder_condition = time_embedding + label_embedding.to(
            time_embedding.dtype
        )
        # Convert the frame grid into a token sequence with the encoder embedder.
        #
        # Single-view:
        #   grid [B*T,C,H,W]
        #   -> sequence [B,T*P,E]
        #
        # Multiview:
        #   grid [B*V*T,C,H,W]
        #   -> sequence [B*V,T*P,E]
        #
        # P = patches per temporal chunk; E = encoder hidden size.
        if multiview:
            embed_batch_size = batch * self.num_views
        else:
            embed_batch_size = batch
        sequence = embed_video_frames(
            self.encoder_embed,
            grid,
            batch_size=embed_batch_size,
            chunks=self.num_chunks,
        )
        if multiview:
            patches = sequence.shape[1] // self.num_chunks

            # Split out the view and temporal-chunk dimensions:
            #   [B*V,T*P,E]
            #   -> [B,V,T,P,E]
            sequence = sequence.reshape(
                batch,
                self.num_views,
                self.num_chunks,
                patches,
                sequence.shape[-1],
            )

            # Reorder to [B,T,V,P,E], grouping views within each chunk.
            sequence = sequence.permute(0, 2, 1, 3, 4)

            if self.use_view_embedding:
                ids = stream_ids

                # self.encoder_view_embedding(ids):
                #   ids [B,V] -> embedding [B,V,E]
                #
                # After adding singleton dimensions:
                #   [B,V,E] -> [B,1,V,1,E]
                #
                # This broadcasts over every token in [B,T,V,P,E].
                sequence = sequence + (
                    self.encoder_view_embedding(ids)
                    .to(sequence.dtype)[:, None, :, None]
                )

            # Transformers consume a two-dimensional sequence:
            #   [B,T,V,P,E] -> [B,T*V*P,E]
            #
            # For example, T=2, V=2, P=3 gives a sequence length of 12.
            sequence = sequence.reshape(
                batch,
                -1,
                sequence.shape[-1],
            )

            # Build 3D position coordinates for each token. The base positions
            # have shape [T,P,3] and become [T,V,P,3] after view expansion.
            # Flattening gives [T*V*P,3] in the same order as the sequence.
            encoder_positions = (
                self._encoder_positions
                .reshape(self.num_chunks, patches, 3)[:, None]
                .expand(-1, self.num_views, -1, -1)
                .reshape(-1, 3)
            )

        else:
            # In single-view mode, the number of positions is T*P.
            encoder_positions = self._encoder_positions

        encoder_rope_cache = self._cached_rope(
            scope="encoder",
            positions=encoder_positions,
            hidden=sequence,
            block=self.encoder_blocks[0],
        )
        # Save the encoder activation at base_model_depth for the base branch.
        base_sequence: torch.Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            # The sequence remains [B,L,E], where L is T*P in single-view mode
            # and T*V*P in multiview mode.
            sequence = self._run_conditioned_block(
                block,
                sequence,
                encoder_condition,
                encoder_positions,
                encoder_rope_cache,
            )
            if index == self.base_model_depth:
                base_sequence = sequence
        if base_sequence is None:
            raise RuntimeError("base encoder activation was not captured")

        # `sequence` is now the final encoder output [B,L,E]. The time embedding
        # [B,E] broadcasts across the token dimension:
        #
        #   [B,E] + [B,L,E] -> [B,L,E]
        #
        # Apply SiLU, then project from E to D to form the decoder condition.
        decoder_condition = self.encoder_to_decoder(
            F.silu(time_embedding + sequence)
        )
        # decoder_condition: [B,L,D], where D is the decoder hidden size.

        # The decoder uses its own patch embedder on the same grid rather than
        # reusing the encoder tokens.
        decoder_sequence = embed_video_frames(
            self.decoder_embed,
            grid,
            batch_size=embed_batch_size,
            chunks=self.num_chunks,
        )

        if multiview:
            patches = decoder_sequence.shape[1] // self.num_chunks

            # [B*V,T*P,D] -> [B,V,T,P,D]
            decoder_sequence = decoder_sequence.reshape(
                batch,
                self.num_views,
                self.num_chunks,
                patches,
                decoder_sequence.shape[-1],
            )

            # Reorder to [B,T,V,P,D].
            decoder_sequence = decoder_sequence.permute(0, 2, 1, 3, 4)

            if self.use_view_embedding:
                ids = stream_ids

                # [B,V,D] -> [B,1,V,1,D], then broadcast over chunks and patches.
                decoder_sequence = decoder_sequence + (
                    self.decoder_view_embedding(ids)
                    .to(decoder_sequence.dtype)[:, None, :, None]
                )

            # [B,T,V,P,D] -> [B,T*V*P,D]
            decoder_sequence = decoder_sequence.reshape(
                batch,
                -1,
                decoder_sequence.shape[-1],
            )

            # The decoder uses its own position-encoding cache.
            decoder_positions = (
                self._decoder_positions
                .reshape(self.num_chunks, patches, 3)[:, None]
                .expand(-1, self.num_views, -1, -1)
                .reshape(-1, 3)
            )

        else:
            decoder_positions = self._decoder_positions

        decoder_rope_cache = self._cached_rope(
            scope="decoder",
            positions=decoder_positions,
            hidden=decoder_sequence,
            block=self.decoder_blocks[0],
        )
        for block in self.decoder_blocks:
            decoder_sequence = self._run_conditioned_block(
                block,
                decoder_sequence,
                decoder_condition,
                decoder_positions,
                decoder_rope_cache,
            )
        full_tokens = self.final_layer(decoder_sequence, decoder_condition)
        base_condition = F.silu(time_embedding + base_sequence)
        base_tokens = self.base_final_layer(base_condition, base_condition)
        if multiview:

            def restore(tokens: torch.Tensor) -> torch.Tensor:
                tokens = tokens.reshape(
                    batch,
                    self.num_chunks,
                    self.num_views,
                    self.patches_per_chunk,
                    -1,
                )
                tokens = tokens.permute(
                    0,
                    2,
                    1,
                    3,
                    4,
                ).reshape(
                    batch * self.num_views,
                    self.num_chunks * self.patches_per_chunk,
                    -1,
                )
                restored = self._unpatchify(tokens, batch_size=batch * self.num_views)
                restored = restored.reshape(
                    batch,
                    self.num_views,
                    self.num_chunks,
                    self.patches_per_chunk,
                    -1,
                )
                return restored.permute(
                    0,
                    2,
                    1,
                    3,
                    4,
                ).contiguous()
            full = restore(full_tokens)
            base = restore(base_tokens)
        else:
            full = self._unpatchify(full_tokens, batch_size=batch)
            base = self._unpatchify(base_tokens, batch_size=batch)
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if return_base:
            result = (full, base)
        else:
            result = full
        if return_intermediate:
            return result, base_sequence
        return result


def _factory_parameters(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(config)
    values.pop("name", None)
    nested = values.pop("parameters", values.pop("params", {}))
    if nested:
        if not isinstance(nested, Mapping):
            raise TypeError("DiT parameters must be a mapping")
        parameters = dict(nested)
        parameters.update(values)
    else:
        parameters = values
    parameters.update(overrides)
    return parameters


@DIT_MODELS.decorator("vrae_video_dit")
def build_vrae_video_dit(
    *,
    config: Mapping[str, Any],
    **overrides: Any,
) -> VRAEVideoDiT:
    parameters = _factory_parameters(config, overrides)
    return VRAEVideoDiT(**parameters)


__all__ = ["VRAEVideoDiT", "build_vrae_video_dit"]
