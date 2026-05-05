import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from temporalbiome import apply_temperature


def _logits(size: int, bound: float = 3.0) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=size,
        elements=st.floats(min_value=-bound, max_value=bound, allow_nan=False, allow_infinity=False),
    )


def _well_separated_logits(size: int = 64, low: float = -3.0, high: float = 3.0) -> st.SearchStrategy[np.ndarray]:
    base = np.linspace(low, high, num=size, dtype=np.float64)
    return st.permutations(list(range(size))).map(
        lambda permutation: base[np.asarray(permutation, dtype=np.int64)]
    )


@given(
    logits=_well_separated_logits(size=64, low=-3.0, high=3.0),
    temperature=st.floats(min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=150, deadline=None)
def test_temperature_scaling_preserves_rank(logits: np.ndarray, temperature: float) -> None:
    baseline = apply_temperature(logits, temperature=1.0)
    scaled = apply_temperature(logits, temperature=temperature)
    base_rank = np.argsort(baseline, kind="mergesort")
    scaled_rank = np.argsort(scaled, kind="mergesort")
    assert np.array_equal(base_rank, scaled_rank)


@given(logits=_logits(48, bound=8.0))
@settings(max_examples=100, deadline=None)
def test_temperature_one_recovers_sigmoid(logits: np.ndarray) -> None:
    expected = 1.0 / (1.0 + np.exp(-logits))
    out = apply_temperature(logits, temperature=1.0)
    assert np.allclose(out, expected, atol=1e-12)


@given(logits=_logits(40, bound=4.0))
@settings(max_examples=100, deadline=None)
def test_higher_temperature_compresses_probabilities_toward_half(logits: np.ndarray) -> None:
    cool_temperature = apply_temperature(logits, temperature=0.5)
    hot_temperature = apply_temperature(logits, temperature=4.0)
    cool_spread = float(np.abs(cool_temperature - 0.5).max())
    hot_spread = float(np.abs(hot_temperature - 0.5).max())
    assert hot_spread <= cool_spread + 1e-12
