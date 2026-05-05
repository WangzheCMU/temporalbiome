from typing import Union

import numpy as np
from numpy.typing import NDArray

from ._types import FOCAL_ALPHA, FOCAL_GAMMA


def _stable_sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    positive_mask = logits >= 0
    negative_mask = ~positive_mask
    out = np.empty_like(logits, dtype=np.float64)
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


def focal_cross_entropy(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    reduction: str = "mean",
) -> Union[float, NDArray[np.float64]]:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if z.shape != y.shape:
        raise ValueError("logits and labels must align in shape")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary {0, 1}")
    log_p1 = _stable_log_sigmoid(z)
    log_p0 = _stable_log_sigmoid(-z)
    p1 = np.exp(log_p1)
    p0 = np.exp(log_p0)
    pt = np.where(y == 1, p1, p0)
    log_pt = np.where(y == 1, log_p1, log_p0)
    weight = np.where(y == 1, alpha, 1.0 - alpha)
    focal_modulator = np.power(np.clip(1.0 - pt, 0.0, 1.0), gamma)
    per_sample = -weight * focal_modulator * log_pt
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return float(per_sample.mean())
    if reduction == "sum":
        return float(per_sample.sum())
    raise ValueError("reduction must be one of 'none', 'mean', 'sum'")


def focal_cross_entropy_gradient(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    reduction: str = "mean",
) -> NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if z.shape != y.shape:
        raise ValueError("logits and labels must align in shape")
    p1 = _stable_sigmoid(z)
    p0 = 1.0 - p1
    pt = np.where(y == 1, p1, p0)
    log_pt = np.where(y == 1, _stable_log_sigmoid(z), _stable_log_sigmoid(-z))
    weight = np.where(y == 1, alpha, 1.0 - alpha)
    sign = np.where(y == 1, 1.0, -1.0)
    one_minus_pt = np.clip(1.0 - pt, 0.0, 1.0)
    if gamma == 0.0:
        modulator = np.ones_like(z)
        modulator_derivative = np.zeros_like(z)
    else:
        modulator = np.power(one_minus_pt, gamma)
        modulator_derivative = -gamma * np.power(one_minus_pt, max(gamma - 1.0, 0.0))
    grad_per_sample = -weight * sign * (
        modulator_derivative * log_pt * pt * (1.0 - pt) + modulator * (1.0 - pt)
    )
    if reduction == "none":
        return grad_per_sample
    if reduction == "mean":
        return grad_per_sample / float(z.shape[0])
    if reduction == "sum":
        return grad_per_sample
    raise ValueError("reduction must be one of 'none', 'mean', 'sum'")


__all__ = [
    "focal_cross_entropy",
    "focal_cross_entropy_gradient",
]
