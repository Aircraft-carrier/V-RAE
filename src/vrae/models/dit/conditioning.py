from __future__ import annotations

import torch
from torch import nn


def sample_condition_dropout(
    batch_size: int,
    probability: float,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError("condition dropout probability must be in [0,1]")
    if probability == 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if probability == 1:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return torch.rand(batch_size, device=device, generator=generator) < probability


def _resolve_drop_mask(
    *,
    batch_size: int,
    device: torch.device,
    probability: float,
    training: bool,
    drop_mask: torch.Tensor | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if drop_mask is not None:
        mask = torch.as_tensor(drop_mask, device=device, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != batch_size:
            raise ValueError(f"drop_mask must have shape [{batch_size}], got {tuple(mask.shape)}")
        return mask
    if not training:
        return torch.zeros(batch_size, device=device, dtype=torch.bool)
    return sample_condition_dropout(
        batch_size,
        probability,
        device=device,
        generator=generator,
    )


class LabelConditionAdapter(nn.Module):
    """Class embedding with a single learned classifier-free null class."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive")
        if not 0 <= float(dropout_prob) <= 1:
            raise ValueError("dropout_prob must be in [0,1]")
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.null_label = self.num_classes
        self.dropout_prob = float(dropout_prob)
        self.embedding = nn.Embedding(self.num_classes + 1, self.hidden_size)

    @property
    def embedding_table(self) -> nn.Embedding:
        return self.embedding

    def prepare(
        self,
        labels: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}")
        if labels.dtype == torch.bool or labels.is_floating_point():
            raise TypeError("labels must use an integer dtype")
        labels = labels.to(device=self.embedding.weight.device, dtype=torch.long)
        if bool(((labels < 0) | (labels >= self.num_classes)).any()):
            raise ValueError(f"labels must be in [0,{self.num_classes - 1}]")
        mask = _resolve_drop_mask(
            batch_size=labels.shape[0],
            device=labels.device,
            probability=self.dropout_prob,
            training=self.training,
            drop_mask=drop_mask,
            generator=generator,
        )
        effective_labels = torch.where(
            mask,
            torch.full_like(labels, self.null_label),
            labels,
        )
        embedding = self.embedding(effective_labels).unsqueeze(1)
        return embedding, effective_labels, mask

    def forward(
        self,
        labels: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        embedding, _, _ = self.prepare(labels, drop_mask=drop_mask, generator=generator)
        return embedding


class ContextLatentConditionAdapter(nn.Module):
    """Pool clean context latents after optional per-sample null replacement."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(hidden_size) <= 0:
            raise ValueError("in_channels and hidden_size must be positive")
        if not 0 <= float(dropout_prob) <= 1:
            raise ValueError("dropout_prob must be in [0,1]")
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.dropout_prob = float(dropout_prob)
        self.null_context = nn.Parameter(torch.zeros(1, 1, 1, self.in_channels))
        self.projection = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def _validate(self, context: torch.Tensor) -> None:
        if context.ndim != 4 or context.shape[-1] != self.in_channels:
            raise ValueError(
                "context must have shape [B,chunks,tokens,"
                f"{self.in_channels}], got {tuple(context.shape)}"
            )
        if not context.is_floating_point():
            raise TypeError("context must be floating point")

    def apply_dropout(
        self,
        context: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(context)
        mask = _resolve_drop_mask(
            batch_size=context.shape[0],
            device=context.device,
            probability=self.dropout_prob,
            training=self.training,
            drop_mask=drop_mask,
            generator=generator,
        )
        null = self.null_context.to(device=context.device, dtype=context.dtype).expand_as(context)
        dropped = torch.where(mask[:, None, None, None], null, context)
        return dropped, mask

    def prepare(
        self,
        context: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dropped, mask = self.apply_dropout(
            context,
            drop_mask=drop_mask,
            generator=generator,
        )
        pooled = dropped.mean(dim=(1, 2))
        parameter = self.projection[0].weight
        pooled = pooled.to(device=parameter.device, dtype=parameter.dtype)
        embedding = self.projection(pooled).unsqueeze(1)
        return dropped, embedding, mask

    def forward(
        self,
        context: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        _, embedding, _ = self.prepare(
            context,
            drop_mask=drop_mask,
            generator=generator,
        )
        return embedding


__all__ = [
    "ContextLatentConditionAdapter",
    "LabelConditionAdapter",
    "sample_condition_dropout",
]
