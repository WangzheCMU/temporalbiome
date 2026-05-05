import numpy as np
import pytest

from temporalbiome import bca_interval, patient_clustered_bootstrap


def _mean_statistic_factory(values: np.ndarray):
    sorted_values = np.asarray(values, dtype=np.float64)

    def statistic(selected_ids: np.ndarray) -> float:
        return float(sorted_values[np.asarray(selected_ids, dtype=np.int64)].mean())

    return statistic


def test_bca_interval_contains_point_estimate_for_normal_sample() -> None:
    generator = np.random.default_rng(42)
    sample = generator.normal(loc=3.5, scale=1.0, size=80).astype(np.float64)
    result = patient_clustered_bootstrap(
        statistic=_mean_statistic_factory(sample),
        patient_ids=list(range(sample.shape[0])),
        n_resamples=500,
        confidence=0.95,
        seed=2026,
        method="bca",
    )
    assert result.lower <= result.point_estimate <= result.upper


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_bca_is_within_percentile_envelope_for_symmetric_data(seed: int) -> None:
    generator = np.random.default_rng(seed)
    sample = generator.uniform(low=-1.0, high=1.0, size=120).astype(np.float64)
    statistic = _mean_statistic_factory(sample)
    bca_result = patient_clustered_bootstrap(
        statistic=statistic,
        patient_ids=list(range(sample.shape[0])),
        n_resamples=400,
        confidence=0.95,
        seed=seed,
        method="bca",
    )
    percentile_result = patient_clustered_bootstrap(
        statistic=statistic,
        patient_ids=list(range(sample.shape[0])),
        n_resamples=400,
        confidence=0.95,
        seed=seed,
        method="percentile",
    )
    assert abs(bca_result.lower - percentile_result.lower) < 0.20
    assert abs(bca_result.upper - percentile_result.upper) < 0.20


def test_bca_is_deterministic_for_fixed_seed() -> None:
    sample = np.linspace(-1.0, 1.0, num=100, dtype=np.float64)
    statistic = _mean_statistic_factory(sample)
    first = patient_clustered_bootstrap(
        statistic=statistic,
        patient_ids=list(range(sample.shape[0])),
        n_resamples=300,
        confidence=0.90,
        seed=999,
        method="bca",
    )
    second = patient_clustered_bootstrap(
        statistic=statistic,
        patient_ids=list(range(sample.shape[0])),
        n_resamples=300,
        confidence=0.90,
        seed=999,
        method="bca",
    )
    assert first.point_estimate == pytest.approx(second.point_estimate, abs=1e-12)
    assert first.lower == pytest.approx(second.lower, abs=1e-12)
    assert first.upper == pytest.approx(second.upper, abs=1e-12)
    assert np.array_equal(first.replicate_values, second.replicate_values)


def test_bca_lower_below_upper() -> None:
    sample = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    statistic = _mean_statistic_factory(sample)
    result = patient_clustered_bootstrap(
        statistic=statistic,
        patient_ids=list(range(sample.shape[0])),
        n_resamples=200,
        confidence=0.95,
        seed=11,
        method="bca",
    )
    assert result.lower < result.upper


def test_bca_interval_helper_matches_inline_invocation() -> None:
    sample = np.linspace(0.0, 5.0, num=50, dtype=np.float64)
    statistic = _mean_statistic_factory(sample)
    unique_ids = np.arange(sample.shape[0], dtype=np.int64)
    generator = np.random.default_rng(7)
    replicates = np.array([
        statistic(generator.choice(unique_ids, size=unique_ids.size, replace=True))
        for _ in range(250)
    ])
    point = float(statistic(unique_ids))
    lower, upper = bca_interval(replicates, point, statistic, unique_ids, confidence=0.90)
    assert lower < point < upper
