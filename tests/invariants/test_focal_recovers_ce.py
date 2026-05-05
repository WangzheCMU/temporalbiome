import numpy as np
import pytest

from temporalbiome import focal_cross_entropy, focal_cross_entropy_gradient


def _stable_log_sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    negative = ~positive
    out[positive] = -np.log1p(np.exp(-z[positive]))
    out[negative] = z[negative] - np.log1p(np.exp(z[negative]))
    return out


def _reference_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    log_p1 = _stable_log_sigmoid(logits)
    log_p0 = _stable_log_sigmoid(-logits)
    contributions = np.where(labels == 1, log_p1, log_p0)
    return float(-0.5 * contributions.mean())


def _reference_cross_entropy_gradient(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    sign = np.where(labels == 1, 1.0, -1.0)
    return -0.5 * sign * (1.0 - np.where(labels == 1, sigmoid, 1.0 - sigmoid)) / float(logits.shape[0])


@pytest.mark.parametrize("seed", [0, 1, 7, 23, 91])
def test_focal_with_gamma_zero_and_alpha_half_recovers_cross_entropy(seed: int) -> None:
    generator = np.random.default_rng(seed)
    logits = generator.normal(size=128).astype(np.float64)
    labels = generator.integers(low=0, high=2, size=128).astype(np.int64)
    focal_value = focal_cross_entropy(logits, labels, alpha=0.5, gamma=0.0)
    expected = _reference_cross_entropy(logits, labels)
    assert isinstance(focal_value, float)
    assert abs(focal_value - expected) < 1e-12


@pytest.mark.parametrize("seed", [0, 11, 42, 314, 1729])
def test_focal_gradient_matches_reference_when_gamma_zero(seed: int) -> None:
    generator = np.random.default_rng(seed)
    logits = generator.normal(size=64).astype(np.float64)
    labels = generator.integers(low=0, high=2, size=64).astype(np.int64)
    grad_focal = focal_cross_entropy_gradient(logits, labels, alpha=0.5, gamma=0.0, reduction="mean")
    grad_reference = _reference_cross_entropy_gradient(logits, labels)
    assert np.allclose(grad_focal, grad_reference, atol=1e-12)


def test_focal_modulator_emphasises_hard_examples() -> None:
    easy_logit = np.array([10.0])
    hard_logit = np.array([0.0])
    label = np.array([1])
    easy_value = focal_cross_entropy(easy_logit, label, alpha=0.5, gamma=2.0)
    hard_value = focal_cross_entropy(hard_logit, label, alpha=0.5, gamma=2.0)
    easy_value_no_focal = focal_cross_entropy(easy_logit, label, alpha=0.5, gamma=0.0)
    hard_value_no_focal = focal_cross_entropy(hard_logit, label, alpha=0.5, gamma=0.0)
    relative_easy = easy_value / max(easy_value_no_focal, 1e-12)
    relative_hard = hard_value / max(hard_value_no_focal, 1e-12)
    assert relative_easy < relative_hard


def test_focal_loss_is_nonnegative_for_random_inputs() -> None:
    generator = np.random.default_rng(2026)
    for trial in range(40):
        logits = generator.normal(size=32).astype(np.float64)
        labels = generator.integers(low=0, high=2, size=32).astype(np.int64)
        value = focal_cross_entropy(logits, labels)
        assert value >= 0.0
