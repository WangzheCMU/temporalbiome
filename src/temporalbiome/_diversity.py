import numpy as np
from numpy.typing import NDArray

from ._types import EmptyAbundanceError


def _ensure_two_dimensional(abundances: NDArray[np.float64]) -> NDArray[np.float64]:
    if abundances.ndim == 1:
        return abundances.reshape(1, -1)
    if abundances.ndim != 2:
        raise EmptyAbundanceError("abundances must be a 1d or 2d array")
    return abundances


def _validate_simplex(abundances: NDArray[np.float64], tolerance: float = 1.0e-6) -> None:
    if abundances.size == 0:
        raise EmptyAbundanceError("empty abundance array")
    row_sum = abundances.sum(axis=1)
    if np.any(np.abs(row_sum - 1.0) > tolerance):
        raise EmptyAbundanceError("abundance rows must sum to one within tolerance")
    if np.any(abundances < -tolerance):
        raise EmptyAbundanceError("abundance values must be nonnegative")


def shannon_entropy(abundances: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _ensure_two_dimensional(np.asarray(abundances, dtype=np.float64))
    _validate_simplex(matrix)
    safe = np.clip(matrix, 1.0e-12, 1.0)
    contributions = np.where(matrix > 0.0, -safe * np.log(safe), 0.0)
    return contributions.sum(axis=1)


def observed_richness(abundances: NDArray[np.float64], detection: float = 0.0) -> NDArray[np.int64]:
    matrix = _ensure_two_dimensional(np.asarray(abundances, dtype=np.float64))
    return (matrix > detection).sum(axis=1).astype(np.int64)


def pielou_evenness(abundances: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _ensure_two_dimensional(np.asarray(abundances, dtype=np.float64))
    _validate_simplex(matrix)
    entropy = shannon_entropy(matrix)
    richness = observed_richness(matrix)
    denominator = np.where(richness > 1, np.log(np.maximum(richness, 2)), 1.0)
    evenness = np.where(richness > 1, entropy / denominator, 0.0)
    return np.clip(evenness, 0.0, 1.0)


def taxon_ratio(
    abundances: NDArray[np.float64],
    numerator_index: int,
    denominator_index: int,
    floor: float = 1.0e-9,
) -> NDArray[np.float64]:
    matrix = _ensure_two_dimensional(np.asarray(abundances, dtype=np.float64))
    numerator = np.maximum(matrix[:, numerator_index], 0.0)
    denominator = np.maximum(matrix[:, denominator_index], 0.0)
    return numerator / np.maximum(denominator, floor)


def alpha_diversity_panel(
    abundances: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _ensure_two_dimensional(np.asarray(abundances, dtype=np.float64))
    _validate_simplex(matrix)
    entropy = shannon_entropy(matrix)
    richness = observed_richness(matrix).astype(np.float64)
    evenness = pielou_evenness(matrix)
    return np.stack([entropy, richness, evenness], axis=1)


__all__ = [
    "shannon_entropy",
    "observed_richness",
    "pielou_evenness",
    "taxon_ratio",
    "alpha_diversity_panel",
]
