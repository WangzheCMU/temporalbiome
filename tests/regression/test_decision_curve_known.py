import numpy as np
import pytest

from temporalbiome import (
    decision_curve,
    number_needed_to_screen,
    number_needed_to_treat,
)


def test_perfect_classifier_dominates_treat_all_at_low_thresholds() -> None:
    probabilities = np.array([0.95, 0.98, 0.05, 0.02, 0.99, 0.01, 0.96, 0.03])
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int64)
    curve = decision_curve(probabilities, labels, threshold_grid=np.linspace(0.05, 0.50, 10))
    assert np.all(curve.net_benefit_model >= curve.net_benefit_treat_all - 1e-9)
    assert np.all(curve.net_benefit_treat_none == 0.0)


def test_useless_classifier_yields_zero_net_benefit_at_prevalence_threshold() -> None:
    n = 200
    probabilities = np.full(n, 0.5)
    labels = np.zeros(n, dtype=np.int64)
    labels[: n // 4] = 1
    grid = np.array([0.50])
    curve = decision_curve(probabilities, labels, threshold_grid=grid)
    flagged = (probabilities >= 0.50).sum()
    true_positive = int(((probabilities >= 0.50) & (labels == 1)).sum())
    false_positive = flagged - true_positive
    expected = (true_positive / n) - (false_positive / n) * (0.50 / 0.50)
    assert curve.net_benefit_model[0] == pytest.approx(expected, abs=1e-12)


def test_number_needed_to_screen_matches_definition() -> None:
    probabilities = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    threshold = 0.5
    flagged_positive = (probabilities >= threshold) & (labels == 1)
    flagged = (probabilities >= threshold).sum()
    expected = float(flagged) / float(flagged_positive.sum())
    assert number_needed_to_screen(probabilities, labels, threshold) == pytest.approx(
        expected, abs=1e-12
    )


def test_number_needed_to_treat_with_full_efficacy_equals_inverse_risk() -> None:
    probabilities = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    threshold = 0.5
    flagged_total = (probabilities >= threshold).sum()
    true_positive = int(((probabilities >= threshold) & (labels == 1)).sum())
    risk = true_positive / flagged_total
    expected = 1.0 / risk
    assert number_needed_to_treat(
        probabilities, labels, threshold, intervention_efficacy=1.0
    ) == pytest.approx(expected, abs=1e-12)


def test_number_needed_to_treat_scales_inversely_with_efficacy() -> None:
    probabilities = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0, 0, 0], dtype=np.int64)
    threshold = 0.5
    full = number_needed_to_treat(probabilities, labels, threshold, intervention_efficacy=1.0)
    half = number_needed_to_treat(probabilities, labels, threshold, intervention_efficacy=0.5)
    assert half == pytest.approx(2.0 * full, abs=1e-12)


def test_decision_curve_threshold_grid_propagates_unchanged() -> None:
    grid = np.array([0.10, 0.30, 0.50])
    probabilities = np.array([0.6, 0.4, 0.2, 0.8, 0.05])
    labels = np.array([1, 0, 0, 1, 0], dtype=np.int64)
    curve = decision_curve(probabilities, labels, threshold_grid=grid)
    assert np.array_equal(curve.thresholds, grid)
    assert curve.net_benefit_model.shape == grid.shape
    assert curve.net_benefit_treat_all.shape == grid.shape
