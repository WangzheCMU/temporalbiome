import numpy as np
import pytest

from temporalbiome import auroc, paired_delong


@pytest.mark.parametrize("seed", [0, 17, 91, 314, 2026])
def test_paired_delong_against_self_yields_zero_z_and_unit_p(seed: int) -> None:
    generator = np.random.default_rng(seed)
    n = 128
    labels = generator.integers(low=0, high=2, size=n).astype(np.int64)
    if labels.sum() == 0:
        labels[0] = 1
    if labels.sum() == labels.shape[0]:
        labels[0] = 0
    scores = generator.normal(loc=labels.astype(float), scale=0.5).astype(np.float64)
    result = paired_delong(scores, scores.copy(), labels)
    assert result.delta == 0.0
    assert result.z_statistic == 0.0
    assert result.p_value_unadjusted == pytest.approx(1.0, abs=1e-12)
    assert result.p_value_bonferroni == pytest.approx(1.0, abs=1e-12)


def test_auroc_is_invariant_under_strictly_monotone_transformation() -> None:
    generator = np.random.default_rng(42)
    n = 200
    labels = generator.integers(low=0, high=2, size=n).astype(np.int64)
    if labels.sum() == 0:
        labels[0] = 1
    scores_a = generator.normal(loc=labels.astype(float), scale=0.7).astype(np.float64)
    scores_b = np.exp(scores_a)
    assert pytest.approx(auroc(scores_a, labels), abs=1e-12) == auroc(scores_b, labels)


def test_paired_delong_signals_known_difference() -> None:
    generator = np.random.default_rng(11)
    n = 600
    labels = (generator.uniform(size=n) < 0.4).astype(np.int64)
    strong = generator.normal(loc=labels.astype(float) * 1.5, scale=1.0).astype(np.float64)
    weak = generator.normal(loc=labels.astype(float) * 0.3, scale=1.0).astype(np.float64)
    result = paired_delong(strong, weak, labels)
    assert result.delta > 0.10
    assert result.p_value_bonferroni < 0.05
