"""Cityscapes 12-frame context to 12-frame future prediction."""

from vrae.training.cityscapes_video_pred.data import (
    CITYSCAPES_IMAGE_SIZE,
    CONTEXT_RELATIVE_INDICES,
    FUTURE_RELATIVE_INDICES,
    CityscapesSequenceDataset,
)
from vrae.training.cityscapes_video_pred.latent_cache import (
    LatentCacheDataset,
    LatentCacheWriter,
    encode_context_future_separately,
)

__all__ = [
    "CITYSCAPES_IMAGE_SIZE",
    "CONTEXT_RELATIVE_INDICES",
    "CityscapesSequenceDataset",
    "FUTURE_RELATIVE_INDICES",
    "LatentCacheDataset",
    "LatentCacheWriter",
    "encode_context_future_separately",
]
