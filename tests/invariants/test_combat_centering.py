import numpy as np
import pytest

from temporalbiome import composition_aware_zscore, harmonise_across_centres


@pytest.mark.parametrize("seed", [0, 13, 256])
def test_composition_zscore_yields_zero_mean_unit_variance(seed: int) -> None:
    generator = np.random.default_rng(seed)
    raw = generator.dirichlet(alpha=np.ones(20), size=80).astype(np.float64)
    standardised = composition_aware_zscore(raw)
    feature_means = standardised.mean(axis=0)
    feature_std = standardised.std(axis=0, ddof=0)
    assert np.allclose(feature_means, 0.0, atol=1e-9)
    assert np.allclose(feature_std, 1.0, atol=1e-9)


def test_harmonisation_removes_known_centre_shift() -> None:
    generator = np.random.default_rng(7)
    n_per_centre = 60
    n_features = 12
    centre_codes = np.concatenate([np.zeros(n_per_centre), np.ones(n_per_centre), 2 * np.ones(n_per_centre)]).astype(np.int64)
    base_signal = generator.normal(size=(3 * n_per_centre, n_features))
    centre_offsets = np.array([0.0, 2.5, -1.7])
    shifted = base_signal + centre_offsets[centre_codes].reshape(-1, 1)
    harmonised = harmonise_across_centres(shifted, centre_codes)
    for centre_value in (0, 1, 2):
        mask = centre_codes == centre_value
        local_mean = harmonised[mask].mean(axis=0)
        assert np.max(np.abs(local_mean - harmonised.mean(axis=0))) < 1e-6


def test_harmonisation_preserves_biological_signal() -> None:
    generator = np.random.default_rng(101)
    n_per_centre = 50
    n_features = 8
    centre_codes = np.concatenate([np.zeros(n_per_centre), np.ones(n_per_centre)]).astype(np.int64)
    outcomes = generator.integers(low=0, high=2, size=2 * n_per_centre).astype(np.int64)
    biological_effect = np.zeros((2 * n_per_centre, n_features))
    biological_effect[:, 0] = (outcomes == 1).astype(np.float64) * 1.2
    background = generator.normal(size=(2 * n_per_centre, n_features), scale=0.3)
    shift = (centre_codes == 1).astype(np.float64).reshape(-1, 1) * 1.0
    raw = biological_effect + background + shift
    harmonised = harmonise_across_centres(raw, centre_codes, biological_outcomes=outcomes)
    delta_in_first_feature = (
        harmonised[outcomes == 1, 0].mean() - harmonised[outcomes == 0, 0].mean()
    )
    assert delta_in_first_feature > 0.5
