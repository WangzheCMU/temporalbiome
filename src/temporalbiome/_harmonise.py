from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ._calibration import fit_platt
from ._types import EmptyAbundanceError


def composition_aware_zscore(
    abundances: NDArray[np.float64],
    pseudocount: float = 1.0e-6,
) -> NDArray[np.float64]:
    matrix = np.asarray(abundances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise EmptyAbundanceError("abundance matrix must be 2d and nonempty")
    log_clr_input = np.log(np.clip(matrix, pseudocount, None))
    geometric_mean = log_clr_input.mean(axis=1, keepdims=True)
    centred_log_ratio = log_clr_input - geometric_mean
    feature_means = centred_log_ratio.mean(axis=0, keepdims=True)
    feature_std = centred_log_ratio.std(axis=0, keepdims=True, ddof=0)
    feature_std = np.where(feature_std < 1.0e-12, 1.0, feature_std)
    return (centred_log_ratio - feature_means) / feature_std


def harmonise_across_centres(
    features: NDArray[np.float64],
    centre_codes: Sequence[int],
    biological_outcomes: Optional[Sequence[int]] = None,
    pooled_variance_floor: float = 1.0e-8,
) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise EmptyAbundanceError("feature matrix must be 2d and nonempty")
    centres = np.asarray(centre_codes, dtype=np.int64)
    if centres.shape[0] != matrix.shape[0]:
        raise EmptyAbundanceError("centre codes must align with feature rows")
    unique_centres = np.unique(centres)
    if biological_outcomes is None:
        biological_design = np.zeros((matrix.shape[0], 0), dtype=np.float64)
    else:
        outcomes = np.asarray(biological_outcomes, dtype=np.float64).reshape(-1, 1)
        if outcomes.shape[0] != matrix.shape[0]:
            raise EmptyAbundanceError("biological outcomes must align with rows")
        biological_design = outcomes - outcomes.mean(axis=0, keepdims=True)

    grand_mean = matrix.mean(axis=0, keepdims=True)
    residuals = matrix - grand_mean
    if biological_design.size > 0:
        gram = biological_design.T @ biological_design
        gram_inv = np.linalg.pinv(gram + 1.0e-9 * np.eye(gram.shape[0]))
        coefficients = gram_inv @ biological_design.T @ residuals
        residuals = residuals - biological_design @ coefficients
        biology_signal = biological_design @ coefficients
    else:
        biology_signal = np.zeros_like(residuals)

    pooled_variance = (residuals ** 2).mean(axis=0, keepdims=True)
    pooled_variance = np.maximum(pooled_variance, pooled_variance_floor)

    adjusted = residuals.copy()
    for centre_value in unique_centres:
        mask = centres == centre_value
        size = int(mask.sum())
        if size <= 1:
            continue
        local_mean = residuals[mask].mean(axis=0, keepdims=True)
        local_variance = residuals[mask].var(axis=0, keepdims=True, ddof=0)
        local_variance = np.maximum(local_variance, pooled_variance_floor)
        scaling = np.sqrt(pooled_variance / local_variance)
        adjusted[mask] = (residuals[mask] - local_mean) * scaling

    return adjusted + biology_signal + grand_mean


def per_centre_platt_correction(
    raw_logits: NDArray[np.float64],
    centre_codes: Sequence[int],
    held_out_logits: NDArray[np.float64],
    held_out_labels: NDArray[np.int64],
    held_out_centres: Sequence[int],
) -> NDArray[np.float64]:
    raw = np.asarray(raw_logits, dtype=np.float64)
    centres = np.asarray(centre_codes, dtype=np.int64)
    if raw.shape[0] != centres.shape[0]:
        raise EmptyAbundanceError("logits and centre codes must align")
    held = np.asarray(held_out_logits, dtype=np.float64)
    held_labels = np.asarray(held_out_labels, dtype=np.int64)
    held_centres = np.asarray(held_out_centres, dtype=np.int64)
    adjusted = raw.copy()
    for centre_value in np.unique(centres):
        mask_held = held_centres == centre_value
        if mask_held.sum() < 4:
            continue
        fit = fit_platt(held[mask_held], held_labels[mask_held])
        mask_target = centres == centre_value
        adjusted[mask_target] = fit.slope * raw[mask_target] + fit.intercept
    return adjusted


__all__ = [
    "composition_aware_zscore",
    "harmonise_across_centres",
    "per_centre_platt_correction",
]
