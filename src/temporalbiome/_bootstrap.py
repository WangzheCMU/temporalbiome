from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ._types import BootstrapResult, IncompatibleScoreVectorsError


def cohens_h(proportion_a: float, proportion_b: float) -> float:
    if not (0.0 <= proportion_a <= 1.0 and 0.0 <= proportion_b <= 1.0):
        raise ValueError("proportions must lie in [0, 1]")
    phi_a = 2.0 * np.arcsin(np.sqrt(proportion_a))
    phi_b = 2.0 * np.arcsin(np.sqrt(proportion_b))
    return float(phi_a - phi_b)


def patient_clustered_bootstrap(
    statistic: Callable[[NDArray[np.int64]], float],
    patient_ids: Sequence[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    method: str = "bca",
) -> BootstrapResult:
    ids = np.asarray(patient_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise IncompatibleScoreVectorsError("patient_ids must be nonempty")
    unique_ids = np.unique(ids)
    if unique_ids.size < 2:
        point_value = float(statistic(unique_ids))
        return BootstrapResult(
            point_estimate=point_value,
            lower=point_value,
            upper=point_value,
            replicate_values=np.full(n_resamples, point_value, dtype=np.float64),
        )
    point_estimate = float(statistic(unique_ids))
    generator = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled_ids = generator.choice(unique_ids, size=unique_ids.size, replace=True)
        replicates[index] = float(statistic(sampled_ids))
    if method == "percentile":
        lower, upper = _percentile_interval(replicates, confidence)
    elif method == "bca":
        lower, upper = _bca_interval(replicates, point_estimate, statistic, unique_ids, confidence)
    else:
        raise ValueError("method must be one of 'percentile', 'bca'")
    return BootstrapResult(
        point_estimate=point_estimate,
        lower=float(lower),
        upper=float(upper),
        replicate_values=replicates,
    )


def _percentile_interval(
    replicates: NDArray[np.float64],
    confidence: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower = float(np.quantile(replicates, alpha / 2.0))
    upper = float(np.quantile(replicates, 1.0 - alpha / 2.0))
    return lower, upper


def _bca_interval(
    replicates: NDArray[np.float64],
    point_estimate: float,
    statistic: Callable[[NDArray[np.int64]], float],
    unique_ids: NDArray[np.int64],
    confidence: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    if replicates.size == 0:
        return point_estimate, point_estimate
    proportion_below = float(np.mean(replicates < point_estimate))
    proportion_below = min(max(proportion_below, 1.0e-10), 1.0 - 1.0e-10)
    z0 = float(norm.ppf(proportion_below))
    jackknife = np.empty(unique_ids.size, dtype=np.float64)
    for index in range(unique_ids.size):
        leave_one = np.delete(unique_ids, index)
        jackknife[index] = float(statistic(leave_one))
    jackknife_mean = float(jackknife.mean())
    deviations = jackknife_mean - jackknife
    numerator = float((deviations ** 3).sum())
    denominator = 6.0 * float(((deviations ** 2).sum()) ** 1.5 + 1.0e-12)
    acceleration = numerator / denominator
    z_lower = norm.ppf(alpha / 2.0)
    z_upper = norm.ppf(1.0 - alpha / 2.0)
    adjusted_lower = z0 + (z0 + z_lower) / (1.0 - acceleration * (z0 + z_lower))
    adjusted_upper = z0 + (z0 + z_upper) / (1.0 - acceleration * (z0 + z_upper))
    quantile_lower = float(norm.cdf(adjusted_lower))
    quantile_upper = float(norm.cdf(adjusted_upper))
    quantile_lower = min(max(quantile_lower, 0.0), 1.0)
    quantile_upper = min(max(quantile_upper, 0.0), 1.0)
    if quantile_lower >= quantile_upper:
        return _percentile_interval(replicates, confidence)
    lower = float(np.quantile(replicates, quantile_lower))
    upper = float(np.quantile(replicates, quantile_upper))
    return lower, upper


def bca_interval(
    replicates: NDArray[np.float64],
    point_estimate: float,
    statistic: Callable[[NDArray[np.int64]], float],
    unique_ids: NDArray[np.int64],
    confidence: float = 0.95,
) -> tuple[float, float]:
    return _bca_interval(replicates, point_estimate, statistic, unique_ids, confidence)


__all__ = [
    "cohens_h",
    "patient_clustered_bootstrap",
    "bca_interval",
]
