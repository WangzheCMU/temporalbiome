from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn

from ._clinical_feature_fusion import ClinicalFeatureFusion, FusionOutput
from ._risk_score_generator import RealTimeRiskScoreGenerator, RiskScoreOutput
from ._temporal_microbiome_encoder import TemporalMicrobiomeEncoder
from ._types import (
    ATTENTION_HEADS,
    CLINICAL_PANEL_DIMENSION,
    DISEASE_PHASE_STATES,
    DROPOUT_PROBABILITY,
    FEED_FORWARD_DIMENSION,
    HIDDEN_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    TRANSFORMER_LAYERS,
    TREATMENT_PHASE_STATES,
)


class MicrobiomeRiskScoreOutput(NamedTuple):
    raw_logit: Tensor
    calibrated_probability: Tensor
    fusion: FusionOutput
    risk_head: RiskScoreOutput
    microbiome_states: Tensor


class MicrobiomeRiskScoreNetwork(nn.Module):
    def __init__(
        self,
        microbiome_dimension: int = MICROBIOME_PANEL_DIMENSION,
        clinical_dimension: int = CLINICAL_PANEL_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        microbiome_layers: int = TRANSFORMER_LAYERS,
        clinical_layers: int = 2,
        num_attention_heads: int = ATTENTION_HEADS,
        feed_forward_dimension: int = FEED_FORWARD_DIMENSION,
        dropout: float = DROPOUT_PROBABILITY,
        treatment_states: int = TREATMENT_PHASE_STATES,
        disease_states: int = DISEASE_PHASE_STATES,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temporal_microbiome_encoder = TemporalMicrobiomeEncoder(
            microbiome_dimension=microbiome_dimension,
            hidden_dimension=hidden_dimension,
            num_layers=microbiome_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
            treatment_states=treatment_states,
            disease_states=disease_states,
        )
        self.clinical_feature_fusion = ClinicalFeatureFusion(
            clinical_dimension=clinical_dimension,
            hidden_dimension=hidden_dimension,
            clinical_layers=clinical_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
            treatment_states=treatment_states,
            disease_states=disease_states,
        )
        self.risk_score_generator = RealTimeRiskScoreGenerator(
            fused_dimension=2 * hidden_dimension,
            hidden_dimension=hidden_dimension,
            dropout=dropout,
            initial_temperature=initial_temperature,
        )

    def forward(
        self,
        microbiome_features: Tensor,
        microbiome_timestamps: Tensor,
        microbiome_treatment_indices: Tensor,
        microbiome_disease_indices: Tensor,
        clinical_features: Tensor,
        clinical_timestamps: Tensor,
        clinical_treatment_indices: Tensor,
        clinical_disease_indices: Tensor,
        microbiome_padding_mask: Optional[Tensor] = None,
        clinical_padding_mask: Optional[Tensor] = None,
    ) -> MicrobiomeRiskScoreOutput:
        microbiome_states = self.temporal_microbiome_encoder(
            microbiome_features=microbiome_features,
            timestamps=microbiome_timestamps,
            treatment_indices=microbiome_treatment_indices,
            disease_indices=microbiome_disease_indices,
            padding_mask=microbiome_padding_mask,
        )
        fusion = self.clinical_feature_fusion(
            microbiome_states=microbiome_states,
            clinical_features=clinical_features,
            clinical_timestamps=clinical_timestamps,
            clinical_treatment_indices=clinical_treatment_indices,
            clinical_disease_indices=clinical_disease_indices,
            microbiome_padding_mask=microbiome_padding_mask,
            clinical_padding_mask=clinical_padding_mask,
        )
        fused_padding_mask = self._derive_fused_mask(
            microbiome_padding_mask=microbiome_padding_mask,
            clinical_padding_mask=clinical_padding_mask,
            fused_length=fusion.fused.shape[1],
        )
        risk_output = self.risk_score_generator(
            fused_states=fusion.fused,
            padding_mask=fused_padding_mask,
        )
        return MicrobiomeRiskScoreOutput(
            raw_logit=risk_output.raw_logit,
            calibrated_probability=risk_output.calibrated_probability,
            fusion=fusion,
            risk_head=risk_output,
            microbiome_states=microbiome_states,
        )

    @staticmethod
    def _derive_fused_mask(
        microbiome_padding_mask: Optional[Tensor],
        clinical_padding_mask: Optional[Tensor],
        fused_length: int,
    ) -> Optional[Tensor]:
        if microbiome_padding_mask is None and clinical_padding_mask is None:
            return None
        if microbiome_padding_mask is not None:
            microbiome_aligned = microbiome_padding_mask[:, :fused_length]
        else:
            microbiome_aligned = None
        if clinical_padding_mask is not None:
            clinical_aligned = clinical_padding_mask[:, :fused_length]
        else:
            clinical_aligned = None
        if microbiome_aligned is None:
            return clinical_aligned
        if clinical_aligned is None:
            return microbiome_aligned
        return microbiome_aligned & clinical_aligned

    def parameter_summary(self) -> dict[str, int]:
        return {
            name: int(sum(parameter.numel() for parameter in module.parameters()))
            for name, module in self.named_children()
        }

    def total_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


__all__ = [
    "MicrobiomeRiskScoreOutput",
    "MicrobiomeRiskScoreNetwork",
]
