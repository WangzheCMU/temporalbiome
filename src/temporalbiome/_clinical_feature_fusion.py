from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn

from ._torch_pe import TorchCompositePositionalEncoding
from ._types import (
    ATTENTION_HEADS,
    CLINICAL_PANEL_DIMENSION,
    DISEASE_PHASE_STATES,
    DROPOUT_PROBABILITY,
    FEED_FORWARD_DIMENSION,
    HIDDEN_DIMENSION,
    TREATMENT_PHASE_STATES,
)


class FusionOutput(NamedTuple):
    fused: Tensor
    microbiome_to_clinical: Tensor
    clinical_to_microbiome: Tensor
    gating: Tensor


class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        clinical_dimension: int = CLINICAL_PANEL_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        num_layers: int = 2,
        num_attention_heads: int = ATTENTION_HEADS,
        feed_forward_dimension: int = FEED_FORWARD_DIMENSION,
        dropout: float = DROPOUT_PROBABILITY,
        treatment_states: int = TREATMENT_PHASE_STATES,
        disease_states: int = DISEASE_PHASE_STATES,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(clinical_dimension, hidden_dimension)
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
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(
        self,
        clinical_features: Tensor,
        timestamps: Tensor,
        treatment_indices: Tensor,
        disease_indices: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if clinical_features.dim() != 3:
            raise ValueError("clinical_features must be (batch, time, clinical_dimension)")
        projected = self.input_projection(clinical_features)
        positional = self.positional_encoding(timestamps, treatment_indices, disease_indices)
        embedded = projected + positional
        return self.encoder_stack(embedded, src_key_padding_mask=padding_mask)


class _DirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dimension: int,
        num_attention_heads: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dimension,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dimension, feed_forward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dimension, hidden_dimension),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dimension)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        first_residual = self.attention_norm(query + self.attention_dropout(attended))
        feed_forward_output = self.feed_forward(first_residual)
        return self.feed_forward_norm(first_residual + self.feed_forward_dropout(feed_forward_output))


class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dimension: int = HIDDEN_DIMENSION,
        num_attention_heads: int = ATTENTION_HEADS,
        feed_forward_dimension: int = FEED_FORWARD_DIMENSION,
        dropout: float = DROPOUT_PROBABILITY,
    ) -> None:
        super().__init__()
        self.microbiome_to_clinical = _DirectionalCrossAttention(
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.clinical_to_microbiome = _DirectionalCrossAttention(
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.gate_projection = nn.Linear(2 * hidden_dimension, 2 * hidden_dimension)
        nn.init.zeros_(self.gate_projection.bias)

    def forward(
        self,
        microbiome_states: Tensor,
        clinical_states: Tensor,
        microbiome_padding_mask: Optional[Tensor] = None,
        clinical_padding_mask: Optional[Tensor] = None,
    ) -> FusionOutput:
        if microbiome_states.shape[0] != clinical_states.shape[0]:
            raise ValueError("microbiome and clinical batches must align")
        microbiome_to_clinical = self.microbiome_to_clinical(
            query=microbiome_states,
            key_value=clinical_states,
            key_padding_mask=clinical_padding_mask,
        )
        clinical_to_microbiome = self.clinical_to_microbiome(
            query=clinical_states,
            key_value=microbiome_states,
            key_padding_mask=microbiome_padding_mask,
        )
        microbiome_length = microbiome_to_clinical.shape[1]
        clinical_length = clinical_to_microbiome.shape[1]
        target_length = min(microbiome_length, clinical_length)
        microbiome_aligned = microbiome_to_clinical[:, :target_length]
        clinical_aligned = clinical_to_microbiome[:, :target_length]
        concatenated = torch.cat([microbiome_aligned, clinical_aligned], dim=-1)
        gating = torch.sigmoid(self.gate_projection(concatenated))
        fused = gating * concatenated
        return FusionOutput(
            fused=fused,
            microbiome_to_clinical=microbiome_aligned,
            clinical_to_microbiome=clinical_aligned,
            gating=gating,
        )


class ClinicalFeatureFusion(nn.Module):
    def __init__(
        self,
        clinical_dimension: int = CLINICAL_PANEL_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        clinical_layers: int = 2,
        num_attention_heads: int = ATTENTION_HEADS,
        feed_forward_dimension: int = FEED_FORWARD_DIMENSION,
        dropout: float = DROPOUT_PROBABILITY,
        treatment_states: int = TREATMENT_PHASE_STATES,
        disease_states: int = DISEASE_PHASE_STATES,
    ) -> None:
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(
            clinical_dimension=clinical_dimension,
            hidden_dimension=hidden_dimension,
            num_layers=clinical_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
            treatment_states=treatment_states,
            disease_states=disease_states,
        )
        self.cross_attention = BidirectionalCrossAttention(
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )

    def forward(
        self,
        microbiome_states: Tensor,
        clinical_features: Tensor,
        clinical_timestamps: Tensor,
        clinical_treatment_indices: Tensor,
        clinical_disease_indices: Tensor,
        microbiome_padding_mask: Optional[Tensor] = None,
        clinical_padding_mask: Optional[Tensor] = None,
    ) -> FusionOutput:
        clinical_states = self.clinical_encoder(
            clinical_features=clinical_features,
            timestamps=clinical_timestamps,
            treatment_indices=clinical_treatment_indices,
            disease_indices=clinical_disease_indices,
            padding_mask=clinical_padding_mask,
        )
        return self.cross_attention(
            microbiome_states=microbiome_states,
            clinical_states=clinical_states,
            microbiome_padding_mask=microbiome_padding_mask,
            clinical_padding_mask=clinical_padding_mask,
        )


__all__ = [
    "FusionOutput",
    "ClinicalEncoder",
    "BidirectionalCrossAttention",
    "ClinicalFeatureFusion",
]
