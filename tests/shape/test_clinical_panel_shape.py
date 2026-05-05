import numpy as np
import pytest

from temporalbiome import (
    CLINICAL_FEATURE_NAMES,
    CLINICAL_PANEL_DIMENSION,
    MICROBIOME_BIN_HOURS,
    aggregate_to_six_hour_panel,
    feature_index_lookup,
    impute_clinical_panel,
    six_hour_bin_edges,
)


def test_clinical_feature_names_total_eighty_seven() -> None:
    assert len(CLINICAL_FEATURE_NAMES) == CLINICAL_PANEL_DIMENSION


def test_clinical_feature_names_are_unique() -> None:
    assert len(set(CLINICAL_FEATURE_NAMES)) == len(CLINICAL_FEATURE_NAMES)


def test_feature_index_lookup_round_trips() -> None:
    lookup = feature_index_lookup()
    for index, name in enumerate(CLINICAL_FEATURE_NAMES):
        assert lookup[name] == index


@pytest.mark.parametrize(
    "hour_start,hour_end,expected_bins",
    [
        (0.0, 24.0, 4),
        (0.0, 72.0, 12),
        (12.0, 84.0, 12),
        (0.0, 6.0, 1),
    ],
)
def test_six_hour_bin_edges_count(hour_start: float, hour_end: float, expected_bins: int) -> None:
    edges = six_hour_bin_edges(hour_start, hour_end)
    assert edges.shape[0] == expected_bins + 1


def test_aggregate_panel_shape_independent_of_irregular_sampling() -> None:
    generator = np.random.default_rng(0)
    timestamps = np.sort(generator.uniform(low=0.0, high=72.0, size=200))
    feature_matrix = generator.normal(size=(200, len(CLINICAL_FEATURE_NAMES))).astype(np.float64)
    panel = aggregate_to_six_hour_panel(timestamps, feature_matrix, hour_start=0.0, hour_end=72.0)
    assert panel.shape == (12, len(CLINICAL_FEATURE_NAMES))


def test_impute_panel_preserves_shape() -> None:
    panel = np.full((12, len(CLINICAL_FEATURE_NAMES)), np.nan, dtype=np.float64)
    panel[0] = 1.0
    panel[3] = 2.0
    panel[7] = 0.5
    imputed = impute_clinical_panel(panel, forward_fill_window_bins=2)
    assert imputed.shape == panel.shape
    assert not np.any(np.isnan(imputed))


def test_aggregate_rejects_misaligned_inputs() -> None:
    timestamps = np.array([0.0, 6.0, 12.0])
    feature_matrix = np.zeros((4, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        aggregate_to_six_hour_panel(timestamps, feature_matrix, hour_start=0.0, hour_end=24.0)


def test_bin_edges_match_microbiome_bin_constant() -> None:
    edges = six_hour_bin_edges(0.0, 24.0)
    spacings = np.diff(edges)
    assert np.allclose(spacings, MICROBIOME_BIN_HOURS)
