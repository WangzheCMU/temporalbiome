import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from temporalbiome import (
    alpha_diversity_panel,
    observed_richness,
    pielou_evenness,
    shannon_entropy,
)


def _simplex(dimension: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=dimension,
        elements=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
    ).map(lambda raw: raw / raw.sum())


@given(simplex=_simplex(dimension=12))
@settings(max_examples=200, deadline=None)
def test_shannon_entropy_is_nonnegative(simplex: np.ndarray) -> None:
    value = float(shannon_entropy(simplex)[0])
    assert value >= 0.0
    assert value <= np.log(simplex.shape[0]) + 1e-9


@given(simplex=_simplex(dimension=20))
@settings(max_examples=200, deadline=None)
def test_pielou_evenness_lies_in_unit_interval(simplex: np.ndarray) -> None:
    evenness = float(pielou_evenness(simplex)[0])
    assert 0.0 <= evenness <= 1.0


@given(simplex=_simplex(dimension=15))
@settings(max_examples=200, deadline=None)
def test_observed_richness_does_not_exceed_alphabet(simplex: np.ndarray) -> None:
    richness = int(observed_richness(simplex)[0])
    assert richness <= simplex.shape[0]
    assert richness > 0


@given(simplex=_simplex(dimension=8))
@settings(max_examples=100, deadline=None)
def test_alpha_panel_aligns_with_individual_metrics(simplex: np.ndarray) -> None:
    panel = alpha_diversity_panel(simplex)
    expected_entropy = float(shannon_entropy(simplex)[0])
    expected_richness = float(observed_richness(simplex)[0])
    expected_evenness = float(pielou_evenness(simplex)[0])
    assert panel.shape == (1, 3)
    assert np.isclose(float(panel[0, 0]), expected_entropy)
    assert np.isclose(float(panel[0, 1]), expected_richness)
    assert np.isclose(float(panel[0, 2]), expected_evenness)


def test_uniform_simplex_attains_log_alphabet_entropy() -> None:
    dimension = 16
    simplex = np.full((1, dimension), 1.0 / dimension, dtype=np.float64)
    value = float(shannon_entropy(simplex)[0])
    assert np.isclose(value, np.log(dimension))


def test_deterministic_simplex_has_zero_entropy() -> None:
    simplex = np.zeros((1, 8), dtype=np.float64)
    simplex[0, 3] = 1.0
    value = float(shannon_entropy(simplex)[0])
    assert np.isclose(value, 0.0, atol=1e-9)
