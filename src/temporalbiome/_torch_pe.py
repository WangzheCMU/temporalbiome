from typing import Optional

import torch
from torch import Tensor, nn

from ._types import (
    DISEASE_PHASE_STATES,
    HIDDEN_DIMENSION,
    TREATMENT_PHASE_STATES,
)


def torch_continuous_time_encoding(
    timestamps: Tensor,
    dimension: int,
    base_period: float = 10_000.0,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError("encoding dimension must be even")
    times = timestamps.unsqueeze(-1)
    half_dimension = dimension // 2
    pair_index = torch.arange(half_dimension, dtype=times.dtype, device=times.device)
    angular_frequency = torch.pow(
        torch.tensor(base_period, dtype=times.dtype, device=times.device),
        -2.0 * pair_index / dimension,
    )
    angles = times * angular_frequency
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    out = torch.empty(*times.shape[:-1], dimension, dtype=times.dtype, device=times.device)
    out[..., 0::2] = sin_part
    out[..., 1::2] = cos_part
    return out


class TorchCompositePositionalEncoding(nn.Module):
    def __init__(
        self,
        dimension: int = HIDDEN_DIMENSION,
        treatment_states: int = TREATMENT_PHASE_STATES,
        disease_states: int = DISEASE_PHASE_STATES,
        base_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("encoding dimension must be even")
        self.dimension = dimension
        self.base_period = base_period
        self.treatment_embedding = nn.Embedding(treatment_states, dimension)
        self.disease_embedding = nn.Embedding(disease_states, dimension)
        self._reset_phase_embeddings()

    def _reset_phase_embeddings(self) -> None:
        nn.init.orthogonal_(self.treatment_embedding.weight, gain=1.0)
        nn.init.orthogonal_(self.disease_embedding.weight, gain=1.0)

    def forward(
        self,
        timestamps: Tensor,
        treatment_indices: Tensor,
        disease_indices: Tensor,
    ) -> Tensor:
        if not (timestamps.shape == treatment_indices.shape == disease_indices.shape):
            raise ValueError("timestamps, treatment, and disease indices must align")
        if treatment_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("treatment indices must be integer tensors")
        if disease_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("disease indices must be integer tensors")
        time_component = torch_continuous_time_encoding(
            timestamps.to(self.treatment_embedding.weight.dtype),
            dimension=self.dimension,
            base_period=self.base_period,
        )
        treatment_component = self.treatment_embedding(treatment_indices)
        disease_component = self.disease_embedding(disease_indices)
        return time_component + treatment_component + disease_component


__all__ = [
    "torch_continuous_time_encoding",
    "TorchCompositePositionalEncoding",
]
