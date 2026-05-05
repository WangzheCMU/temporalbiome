from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn

from ._types import DROPOUT_PROBABILITY, HIDDEN_DIMENSION


class RiskScoreOutput(NamedTuple):
    raw_logit: Tensor
    calibrated_probability: Tensor
    pooling_weights: Tensor
    pooled_representation: Tensor


class AttentionWeightedTemporalPooling(nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.score_projection = nn.Linear(hidden_dimension, 1)
        nn.init.zeros_(self.score_projection.bias)

    def forward(
        self,
        states: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        scores = self.score_projection(states).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        finite_mask = torch.isfinite(weights)
        weights = torch.where(finite_mask, weights, torch.zeros_like(weights))
        pooled = torch.einsum("btd,bt->bd", states, weights)
        return pooled, weights


class RealTimeRiskScoreGenerator(nn.Module):
    def __init__(
        self,
        fused_dimension: int = 2 * HIDDEN_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        dropout: float = DROPOUT_PROBABILITY,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.pooling = AttentionWeightedTemporalPooling(hidden_dimension=fused_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(fused_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        self.register_buffer("temperature", torch.tensor(float(initial_temperature)))

    def set_temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("temperature must be positive")
        with torch.no_grad():
            self.temperature.fill_(float(value))

    def forward(
        self,
        fused_states: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> RiskScoreOutput:
        if fused_states.dim() != 3:
            raise ValueError("fused_states must be (batch, time, fused_dimension)")
        pooled, weights = self.pooling(fused_states, padding_mask=padding_mask)
        raw_logit = self.feed_forward(pooled).squeeze(-1)
        if self.training:
            calibrated_probability = torch.sigmoid(raw_logit)
        else:
            calibrated_probability = torch.sigmoid(raw_logit / self.temperature)
        return RiskScoreOutput(
            raw_logit=raw_logit,
            calibrated_probability=calibrated_probability,
            pooling_weights=weights,
            pooled_representation=pooled,
        )


__all__ = [
    "RiskScoreOutput",
    "AttentionWeightedTemporalPooling",
    "RealTimeRiskScoreGenerator",
]
