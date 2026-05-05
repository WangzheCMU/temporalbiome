from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import DecisionCurve


def _confusion(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold: float,
) -> tuple[int, int, int, int]:
    predicted_positive = probabilities >= threshold
    actual_positive = labels == 1
    true_positive = int(np.sum(predicted_positive & actual_positive))
    false_positive = int(np.sum(predicted_positive & ~actual_positive))
    false_negative = int(np.sum(~predicted_positive & actual_positive))
    true_negative = int(np.sum(~predicted_positive & ~actual_positive))
    return true_positive, false_positive, false_negative, true_negative


def decision_curve(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold_grid: Sequence[float] = tuple(np.linspace(0.05, 0.50, 10)),
) -> DecisionCurve:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must align")
    grid = np.asarray(threshold_grid, dtype=np.float64).reshape(-1)
    n = p.size
    prevalence = float(y.mean())
    net_benefit_model = np.empty(grid.shape[0], dtype=np.float64)
    net_benefit_treat_all = np.empty(grid.shape[0], dtype=np.float64)
    net_benefit_treat_none = np.zeros(grid.shape[0], dtype=np.float64)
    for index, threshold in enumerate(grid):
        true_positive, false_positive, _, _ = _confusion(p, y, float(threshold))
        weight = float(threshold) / max(1.0 - float(threshold), 1.0e-12)
        net_benefit_model[index] = (true_positive / n) - (false_positive / n) * weight
        net_benefit_treat_all[index] = prevalence - (1.0 - prevalence) * weight
    return DecisionCurve(
        thresholds=grid,
        net_benefit_model=net_benefit_model,
        net_benefit_treat_all=net_benefit_treat_all,
        net_benefit_treat_none=net_benefit_treat_none,
    )


def number_needed_to_screen(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold: float,
) -> float:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must align")
    true_positive, false_positive, _, _ = _confusion(p, y, threshold)
    flagged = true_positive + false_positive
    if flagged == 0:
        return float("inf")
    return float(flagged) / max(true_positive, 1)


def number_needed_to_treat(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold: float,
    intervention_efficacy: float,
) -> float:
    if not (0.0 < intervention_efficacy <= 1.0):
        raise ValueError("intervention_efficacy must lie in (0, 1]")
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must align")
    true_positive, _, _, _ = _confusion(p, y, threshold)
    flagged = (p >= threshold).sum()
    if flagged == 0 or true_positive == 0:
        return float("inf")
    risk_in_flagged = true_positive / float(flagged)
    risk_reduction = intervention_efficacy * risk_in_flagged
    if risk_reduction == 0.0:
        return float("inf")
    return 1.0 / risk_reduction


__all__ = [
    "decision_curve",
    "number_needed_to_screen",
    "number_needed_to_treat",
]
