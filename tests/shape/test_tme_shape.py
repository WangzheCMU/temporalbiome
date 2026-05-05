import pytest

torch = pytest.importorskip("torch")

import numpy as np

from temporalbiome import (
    HIDDEN_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    TREATMENT_PHASE_STATES,
    TemporalMicrobiomeEncoder,
)


@pytest.mark.parametrize("batch_size,timepoints", [(1, 1), (2, 12), (4, 7), (8, 24)])
def test_tme_output_shape_matches_hidden_dimension(batch_size: int, timepoints: int) -> None:
    encoder = TemporalMicrobiomeEncoder().eval()
    microbiome_features = torch.randn(batch_size, timepoints, MICROBIOME_PANEL_DIMENSION)
    timestamps = torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1)
    treatment = torch.zeros(batch_size, timepoints, dtype=torch.long)
    disease = torch.zeros(batch_size, timepoints, dtype=torch.long)
    with torch.no_grad():
        contextual = encoder(microbiome_features, timestamps, treatment, disease)
    assert contextual.shape == (batch_size, timepoints, HIDDEN_DIMENSION)


def test_tme_accepts_padding_mask() -> None:
    encoder = TemporalMicrobiomeEncoder().eval()
    batch_size, timepoints = 3, 10
    microbiome_features = torch.randn(batch_size, timepoints, MICROBIOME_PANEL_DIMENSION)
    timestamps = torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1)
    treatment = torch.full((batch_size, timepoints), fill_value=4, dtype=torch.long)
    disease = torch.full((batch_size, timepoints), fill_value=1, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, timepoints, dtype=torch.bool)
    padding_mask[:, 7:] = True
    with torch.no_grad():
        contextual = encoder(
            microbiome_features=microbiome_features,
            timestamps=timestamps,
            treatment_indices=treatment,
            disease_indices=disease,
            padding_mask=padding_mask,
        )
    assert contextual.shape == (batch_size, timepoints, HIDDEN_DIMENSION)
    assert torch.isfinite(contextual[:, :7]).all()


def test_tme_rejects_treatment_indices_out_of_range() -> None:
    encoder = TemporalMicrobiomeEncoder().eval()
    microbiome_features = torch.randn(1, 4, MICROBIOME_PANEL_DIMENSION)
    timestamps = torch.zeros(1, 4)
    treatment = torch.tensor([[0, 1, TREATMENT_PHASE_STATES, 0]], dtype=torch.long)
    disease = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises((IndexError, RuntimeError)):
        encoder(microbiome_features, timestamps, treatment, disease)
