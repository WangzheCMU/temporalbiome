import numpy as np
import pytest

from temporalbiome import (
    DISEASE_PHASE_STATES,
    HIDDEN_DIMENSION,
    TREATMENT_PHASE_STATES,
    composite_positional_encoding,
    continuous_time_sinusoidal_encoding,
    disease_phase_embedding_table,
    treatment_phase_embedding_table,
)


@pytest.mark.parametrize("dimension", [64, 128, HIDDEN_DIMENSION, 512])
def test_continuous_time_sinusoidal_shape(dimension: int) -> None:
    timestamps = np.linspace(0.0, 72.0, num=24)
    encoding = continuous_time_sinusoidal_encoding(timestamps, dimension=dimension)
    assert encoding.shape == (24, dimension)
    assert np.all(np.isfinite(encoding))


def test_treatment_phase_table_dimensions() -> None:
    table = treatment_phase_embedding_table()
    assert table.shape == (TREATMENT_PHASE_STATES, HIDDEN_DIMENSION)
    norms = np.linalg.norm(table, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_disease_phase_table_dimensions() -> None:
    table = disease_phase_embedding_table()
    assert table.shape == (DISEASE_PHASE_STATES, HIDDEN_DIMENSION)


def test_composite_encoding_shape_matches_default_dimension() -> None:
    n_timepoints = 12
    timestamps = np.array([6.0 * t for t in range(n_timepoints)], dtype=np.float64)
    treatment_indices = np.full(n_timepoints, fill_value=5, dtype=np.int64)
    disease_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int64)
    encoding = composite_positional_encoding(timestamps, treatment_indices, disease_indices)
    assert encoding.shape == (n_timepoints, HIDDEN_DIMENSION)


@pytest.mark.parametrize("invalid_index", [-1, TREATMENT_PHASE_STATES])
def test_composite_encoding_rejects_out_of_range_treatment_index(invalid_index: int) -> None:
    timestamps = np.array([0.0, 6.0])
    treatment_indices = np.array([0, invalid_index])
    disease_indices = np.array([0, 0])
    with pytest.raises(ValueError):
        composite_positional_encoding(timestamps, treatment_indices, disease_indices)


def test_three_components_summed_match_individual_lookup() -> None:
    timestamps = np.array([0.0], dtype=np.float64)
    treatment = np.array([3], dtype=np.int64)
    disease = np.array([1], dtype=np.int64)
    encoding = composite_positional_encoding(timestamps, treatment, disease)
    individual_time = continuous_time_sinusoidal_encoding(timestamps)
    individual_treatment = treatment_phase_embedding_table()[3]
    individual_disease = disease_phase_embedding_table()[1]
    expected = individual_time[0] + individual_treatment + individual_disease
    assert np.allclose(encoding[0], expected, atol=1e-12)
