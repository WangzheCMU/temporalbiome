import pytest

torch = pytest.importorskip("torch")

from temporalbiome import (
    CLINICAL_PANEL_DIMENSION,
    HIDDEN_DIMENSION,
    BidirectionalCrossAttention,
    ClinicalEncoder,
    ClinicalFeatureFusion,
)


def test_clinical_encoder_emits_hidden_dimension_states() -> None:
    encoder = ClinicalEncoder().eval()
    batch_size, timepoints = 2, 12
    clinical_features = torch.randn(batch_size, timepoints, CLINICAL_PANEL_DIMENSION)
    timestamps = torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1)
    treatment = torch.zeros(batch_size, timepoints, dtype=torch.long)
    disease = torch.zeros(batch_size, timepoints, dtype=torch.long)
    with torch.no_grad():
        contextual = encoder(clinical_features, timestamps, treatment, disease)
    assert contextual.shape == (batch_size, timepoints, HIDDEN_DIMENSION)


def test_bidirectional_cross_attention_returns_fused_double_dimension() -> None:
    fusion_module = BidirectionalCrossAttention().eval()
    batch_size = 2
    micro_length = 10
    clinical_length = 8
    microbiome_states = torch.randn(batch_size, micro_length, HIDDEN_DIMENSION)
    clinical_states = torch.randn(batch_size, clinical_length, HIDDEN_DIMENSION)
    with torch.no_grad():
        fusion_output = fusion_module(microbiome_states, clinical_states)
    expected_length = min(micro_length, clinical_length)
    assert fusion_output.fused.shape == (batch_size, expected_length, 2 * HIDDEN_DIMENSION)
    assert fusion_output.gating.shape == fusion_output.fused.shape
    assert fusion_output.microbiome_to_clinical.shape == (batch_size, expected_length, HIDDEN_DIMENSION)
    assert fusion_output.clinical_to_microbiome.shape == (batch_size, expected_length, HIDDEN_DIMENSION)


def test_full_clinical_feature_fusion_pipeline_shapes() -> None:
    fusion = ClinicalFeatureFusion().eval()
    batch_size = 3
    micro_length = 12
    clinical_length = 12
    microbiome_states = torch.randn(batch_size, micro_length, HIDDEN_DIMENSION)
    clinical_features = torch.randn(batch_size, clinical_length, CLINICAL_PANEL_DIMENSION)
    timestamps = torch.arange(clinical_length, dtype=torch.float32).expand(batch_size, -1)
    treatment = torch.zeros(batch_size, clinical_length, dtype=torch.long)
    disease = torch.zeros(batch_size, clinical_length, dtype=torch.long)
    with torch.no_grad():
        out = fusion(
            microbiome_states=microbiome_states,
            clinical_features=clinical_features,
            clinical_timestamps=timestamps,
            clinical_treatment_indices=treatment,
            clinical_disease_indices=disease,
        )
    assert out.fused.shape == (batch_size, micro_length, 2 * HIDDEN_DIMENSION)
    assert torch.all(out.gating >= 0.0)
    assert torch.all(out.gating <= 1.0)


def test_cff_handles_misaligned_lengths_via_truncation() -> None:
    fusion_module = BidirectionalCrossAttention().eval()
    microbiome_states = torch.randn(1, 6, HIDDEN_DIMENSION)
    clinical_states = torch.randn(1, 14, HIDDEN_DIMENSION)
    with torch.no_grad():
        fusion_output = fusion_module(microbiome_states, clinical_states)
    assert fusion_output.fused.shape[1] == 6
