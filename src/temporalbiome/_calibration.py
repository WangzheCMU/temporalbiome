from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import (
    CalibrationConvergenceError,
    CalibrationFit,
    PlattFit,
    ReliabilityBin,
)


def _stable_sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0
    negative_mask = ~positive_mask
    out[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_value = np.exp(logits[negative_mask])
    out[negative_mask] = exp_value / (1.0 + exp_value)
    return out


def _stable_log_sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0
    negative_mask = ~positive_mask
    out[positive_mask] = -np.log1p(np.exp(-logits[positive_mask]))
    out[negative_mask] = logits[negative_mask] - np.log1p(np.exp(logits[negative_mask]))
    return out


def fit_temperature(
    validation_logits: NDArray[np.float64],
    validation_labels: NDArray[np.int64],
    initial_temperature: float = 1.0,
    max_iterations: int = 64,
    tolerance: float = 1.0e-9,
) -> CalibrationFit:
    z = np.asarray(validation_logits, dtype=np.float64).reshape(-1)
    y = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
    if z.shape != y.shape:
        raise ValueError("logits and labels must align in shape")
    if z.size == 0:
        raise CalibrationConvergenceError("validation set is empty")
    temperature = float(initial_temperature)
    if temperature <= 0.0:
        temperature = 1.0
    last_log_likelihood = -np.inf
    iterations_used = 0
    for iteration in range(1, max_iterations + 1):
        scaled = z / temperature
        log_p1 = _stable_log_sigmoid(scaled)
        log_p0 = _stable_log_sigmoid(-scaled)
        log_likelihood = float(np.where(y == 1, log_p1, log_p0).sum())
        sigmoid_value = _stable_sigmoid(scaled)
        residual = sigmoid_value - y.astype(np.float64)
        gradient = -float((residual * z).sum()) / (temperature ** 2)
        weights = sigmoid_value * (1.0 - sigmoid_value)
        hessian_curvature = float((weights * (z ** 2)).sum()) / (temperature ** 4)
        hessian_curvature += 2.0 * float((residual * z).sum()) / (temperature ** 3)
        hessian_curvature = max(hessian_curvature, 1.0e-9)
        step = gradient / hessian_curvature
        new_temperature = temperature - step
        if not np.isfinite(new_temperature) or new_temperature <= 1.0e-3:
            new_temperature = temperature * 0.5
        if abs(new_temperature - temperature) < tolerance:
            temperature = new_temperature
            iterations_used = iteration
            last_log_likelihood = log_likelihood
            break
        temperature = new_temperature
        last_log_likelihood = log_likelihood
        iterations_used = iteration
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationConvergenceError("temperature did not converge to a positive value")
    return CalibrationFit(
        temperature=float(temperature),
        log_likelihood=float(last_log_likelihood),
        iterations=int(iterations_used),
    )


def apply_temperature(logits: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    return _stable_sigmoid(z)


def fit_platt(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
    iterations: int = 80,
    tolerance: float = 1.0e-10,
) -> PlattFit:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if z.shape != y.shape:
        raise ValueError("logits and labels must align in shape")
    n_positive = float((y == 1).sum())
    n_negative = float((y == 0).sum())
    if n_positive == 0.0 or n_negative == 0.0:
        return PlattFit(slope=1.0, intercept=0.0)
    smoothed = np.where(
        y == 1,
        (n_positive + 1.0) / (n_positive + 2.0),
        1.0 / (n_negative + 2.0),
    )
    slope = 1.0
    intercept = 0.0
    for _ in range(iterations):
        scaled = slope * z + intercept
        sigmoid = _stable_sigmoid(scaled)
        residual = sigmoid - smoothed
        weights = sigmoid * (1.0 - sigmoid)
        gradient_slope = float((residual * z).sum())
        gradient_intercept = float(residual.sum())
        hessian_aa = float((weights * z * z).sum()) + 1.0e-9
        hessian_ab = float((weights * z).sum())
        hessian_bb = float(weights.sum()) + 1.0e-9
        determinant = hessian_aa * hessian_bb - hessian_ab * hessian_ab
        if abs(determinant) < 1.0e-18:
            break
        delta_slope = (hessian_bb * gradient_slope - hessian_ab * gradient_intercept) / determinant
        delta_intercept = (-hessian_ab * gradient_slope + hessian_aa * gradient_intercept) / determinant
        slope -= delta_slope
        intercept -= delta_intercept
        if abs(delta_slope) + abs(delta_intercept) < tolerance:
            break
    return PlattFit(slope=float(slope), intercept=float(intercept))


def apply_platt(logits: NDArray[np.float64], fit: PlattFit) -> NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64)
    return _stable_sigmoid(fit.slope * z + fit.intercept)


def brier_score(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must align")
    return float(((p - y) ** 2).mean())


def reliability_bins(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_bins: int = 10,
) -> Sequence[ReliabilityBin]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must align")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for index in range(n_bins):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(p[mask].mean())
        fraction_positive = float(y[mask].mean())
        out.append(
            ReliabilityBin(
                lower=low,
                upper=high,
                mean_predicted=mean_predicted,
                fraction_positive=fraction_positive,
                count=count,
            )
        )
    return tuple(out)


def expected_calibration_error(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_bins: int = 10,
) -> float:
    bins = reliability_bins(probabilities, labels, n_bins=n_bins)
    if not bins:
        return 0.0
    total = float(sum(b.count for b in bins))
    return float(
        sum(b.count / total * abs(b.fraction_positive - b.mean_predicted) for b in bins)
    )


def calibration_slope(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
    iterations: int = 50,
    tolerance: float = 1.0e-9,
) -> float:
    fit = fit_platt(logits, labels, iterations=iterations, tolerance=tolerance)
    return fit.slope


__all__ = [
    "fit_temperature",
    "apply_temperature",
    "fit_platt",
    "apply_platt",
    "brier_score",
    "reliability_bins",
    "expected_calibration_error",
    "calibration_slope",
]
