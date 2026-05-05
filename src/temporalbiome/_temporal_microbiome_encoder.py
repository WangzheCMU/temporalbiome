from typing import Optional

import torch
from torch import Tensor, nn

from ._torch_pe import TorchCompositePositionalEncoding
from ._types import (
    ATTENTION_HEADS,
    DISEASE_PHASE_STATES,
    DROPOUT_PROBABILITY,
    FEED_FORWARD_DIMENSION,
    HIDDEN_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    TRANSFORMER_LAYERS,
    TREATMENT_PHASE_STATES,
)


class TemporalMicrobiomeEncoder(nn.Module):
    def __init__(
        self,
        microbiome_dimension: int = MICROBIOME_PANEL_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        num_layers: int = TRANSFORMER_LAYERS,
        num_attention_heads: int = ATTENTION_HEADS,
        feed_forward_dimension: int = FEED_FORWARD_DIMENSION,
        dropout: float = DROPOUT_PROBABILITY,
        treatment_states: int = TREATMENT_PHASE_STATES,
        disease_states: int = DISEASE_PHASE_STATES,
    ) -> None:
        super().__init__()
        if hidden_dimension % num_attention_heads != 0:
            raise ValueError("hidden_dimension must be divisible by num_attention_heads")
        self.input_projection = nn.Linear(microbiome_dimension, hidden_dimension)
        self.positional_encoding = TorchCompositePositionalEncoding(
            dimension=hidden_dimension,
            treatment_states=treatment_states,
            disease_states=disease_states,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=num_attention_heads,
            dim_feedforward=feed_forward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dimension),
        )
        self._initialise_projection_weights()

    def _initialise_projection_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(
        self,
        microbiome_features: Tensor,
        timestamps: Tensor,
        treatment_indices: Tensor,
        disease_indices: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if microbiome_features.dim() != 3:
            raise ValueError("microbiome_features must be (batch, time, microbiome_dimension)")
        projected = self.input_projection(microbiome_features)
        positional = self.positional_encoding(timestamps, treatment_indices, disease_indices)
        embedded = projected + positional
        contextual = self.encoder_stack(embedded, src_key_padding_mask=padding_mask)
        return contextual


__all__ = [
    "TemporalMicrobiomeEncoder",
]
