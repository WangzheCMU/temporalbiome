import pytest

torch = pytest.importorskip("torch")

from temporalbiome import (
    CLINICAL_PANEL_DIMENSION,
    HIDDEN_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    MicrobiomeRiskScoreNetwork,
)


def _synthetic_inputs(batch_size: int, timepoints: int) -> dict:
    return {
        "microbiome_features": torch.randn(batch_size, timepoints, MICROBIOME_PANEL_DIMENSION),
        "microbiome_timestamps": torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1),
        "microbiome_treatment_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "microbiome_disease_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "clinical_features": torch.randn(batch_size, timepoints, CLINICAL_PANEL_DIMENSION),
        "clinical_timestamps": torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1),
        "clinical_treatment_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "clinical_disease_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
    }


@pytest.mark.parametrize("batch_size,timepoints", [(1, 1), (2, 12), (4, 7)])
def test_mrs_full_forward_returns_per_sample_probability(batch_size: int, timepoints: int) -> None:
    network = MicrobiomeRiskScoreNetwork().eval()
    inputs = _synthetic_inputs(batch_size, timepoints)
    with torch.no_grad():
        output = network(**inputs)
    assert output.raw_logit.shape == (batch_size,)
    assert output.calibrated_probability.shape == (batch_size,)
    assert output.microbiome_states.shape == (batch_size, timepoints, HIDDEN_DIMENSION)
    assert output.fusion.fused.shape == (batch_size, timepoints, 2 * HIDDEN_DIMENSION)


def test_mrs_handles_padding_in_both_modalities() -> None:
    network = MicrobiomeRiskScoreNetwork().eval()
    batch_size, timepoints = 2, 16
    inputs = _synthetic_inputs(batch_size, timepoints)
    microbiome_padding = torch.zeros(batch_size, timepoints, dtype=torch.bool)
    microbiome_padding[:, 12:] = True
    clinical_padding = torch.zeros(batch_size, timepoints, dtype=torch.bool)
    clinical_padding[:, 14:] = True
    inputs["microbiome_padding_mask"] = microbiome_padding
    inputs["clinical_padding_mask"] = clinical_padding
    with torch.no_grad():
        output = network(**inputs)
    assert torch.isfinite(output.raw_logit).all()
    assert torch.all(output.calibrated_probability >= 0.0)
    assert torch.all(output.calibrated_probability <= 1.0)


def test_parameter_summary_contains_three_subsystems() -> None:
    network = MicrobiomeRiskScoreNetwork()
    summary = network.parameter_summary()
    assert set(summary.keys()) == {
        "temporal_microbiome_encoder",
        "clinical_feature_fusion",
        "risk_score_generator",
    }
    assert all(value > 0 for value in summary.values())
    assert sum(summary.values()) == network.total_parameter_count()
