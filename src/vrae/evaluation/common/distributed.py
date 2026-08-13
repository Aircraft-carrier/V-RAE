from __future__ import annotations

import torch

from vrae.training.common.distributed import all_gather_variable, is_distributed


def gather_indexed_features(
    sample_ids: torch.Tensor, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_ids.ndim != 1 or features.ndim != 2 or sample_ids.shape[0] != features.shape[0]:
        raise ValueError("Expected sample IDs [N] and features [N,D]")
    ids = torch.cat(all_gather_variable(sample_ids))
    values = torch.cat(all_gather_variable(features))
    order = torch.argsort(ids)
    ids = ids[order]
    values = values[order]
    if ids.numel() > 1 and torch.any(ids[1:] == ids[:-1]):
        raise ValueError("Distributed feature gather produced duplicate sample IDs")
    return ids, values


def all_reduce_sums(values: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return values
    result = values.clone()
    torch.distributed.all_reduce(result)
    return result
