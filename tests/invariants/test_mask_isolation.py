import pytest

torch = pytest.importorskip("torch")

from temporalbiome import (
    AttentionWeightedTemporalPooling,
    HIDDEN_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    TemporalMicrobiomeEncoder,
)


def test_attention_pooling_invariant_under_changes_to_padded_positions() -> None:
    pooling = AttentionWeightedTemporalPooling(HIDDEN_DIMENSION).eval()
    batch_size, timepoints = 2, 12
    states = torch.randn(batch_size, timepoints, HIDDEN_DIMENSION)
    padding_mask = torch.zeros(batch_size, timepoints, dtype=torch.bool)
    padding_mask[:, 8:] = True
    pooled_first, weights_first = pooling(states, padding_mask=padding_mask)
    perturbed = states.clone()
    perturbed[:, 8:] = perturbed[:, 8:] + 100.0 * torch.randn_like(perturbed[:, 8:])
    pooled_second, weights_second = pooling(perturbed, padding_mask=padding_mask)
    assert torch.allclose(pooled_first, pooled_second, atol=1e-6)
    assert torch.allclose(weights_first, weights_second, atol=1e-7)


def test_temporal_encoder_outputs_in_unmasked_region_unchanged_when_padded_changes() -> None:
    torch.manual_seed(0)
    encoder = TemporalMicrobiomeEncoder().eval()
    batch_size, timepoints = 1, 12
    base_features = torch.randn(batch_size, timepoints, MICROBIOME_PANEL_DIMENSION)
    timestamps = torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1)
    treatment = torch.zeros(batch_size, timepoints, dtype=torch.long)
    disease = torch.zeros(batch_size, timepoints, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, timepoints, dtype=torch.bool)
    padding_mask[:, 8:] = True
    with torch.no_grad():
        baseline_states = encoder(base_features, timestamps, treatment, disease, padding_mask=padding_mask)
    perturbed = base_features.clone()
    perturbed[:, 8:] = perturbed[:, 8:] + 50.0 * torch.randn_like(perturbed[:, 8:])
    with torch.no_grad():
        perturbed_states = encoder(perturbed, timestamps, treatment, disease, padding_mask=padding_mask)
    assert torch.allclose(baseline_states[:, :8], perturbed_states[:, :8], atol=1e-5)


def test_temporal_encoder_deterministic_in_eval_mode() -> None:
    torch.manual_seed(13)
    encoder = TemporalMicrobiomeEncoder().eval()
    batch_size, timepoints = 2, 6
    features = torch.randn(batch_size, timepoints, MICROBIOME_PANEL_DIMENSION)
    timestamps = torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1)
    treatment = torch.zeros(batch_size, timepoints, dtype=torch.long)
    disease = torch.zeros(batch_size, timepoints, dtype=torch.long)
    with torch.no_grad():
        first = encoder(features, timestamps, treatment, disease)
        second = encoder(features, timestamps, treatment, disease)
    assert torch.equal(first, second)
