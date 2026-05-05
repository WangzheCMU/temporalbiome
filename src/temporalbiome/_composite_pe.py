from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ._types import (
    DISEASE_PHASE_STATES,
    DiseasePhase,
    HIDDEN_DIMENSION,
    TREATMENT_PHASE_STATES,
)


def continuous_time_sinusoidal_encoding(
    timestamps: Sequence[float],
    dimension: int = HIDDEN_DIMENSION,
    base_period: float = 10_000.0,
) -> NDArray[np.float64]:
    if dimension % 2 != 0:
        raise ValueError("encoding dimension must be even")
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1, 1)
    half_dimension = dimension // 2
    pair_index = np.arange(half_dimension, dtype=np.float64).reshape(1, -1)
    angular_frequency = np.power(base_period, -2.0 * pair_index / dimension)
    angles = times * angular_frequency
    encoding = np.empty((times.shape[0], dimension), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding


def _orthogonal_initialiser(rows: int, columns: int, seed: int) -> NDArray[np.float64]:
    generator = np.random.default_rng(seed)
    candidate = generator.standard_normal(size=(max(rows, columns), max(rows, columns)))
    q, r = np.linalg.qr(candidate)
    diag = np.sign(np.diag(r))
    diag[diag == 0.0] = 1.0
    q = q * diag
    return q[:rows, :columns]


def treatment_phase_embedding_table(
    states: int = TREATMENT_PHASE_STATES,
    dimension: int = HIDDEN_DIMENSION,
    seed: int = 0,
) -> NDArray[np.float64]:
    return _orthogonal_initialiser(states, dimension, seed)


def disease_phase_embedding_table(
    states: int = DISEASE_PHASE_STATES,
    dimension: int = HIDDEN_DIMENSION,
    seed: int = 1,
) -> NDArray[np.float64]:
    return _orthogonal_initialiser(states, dimension, seed)


def lookup_phase_embedding(
    indices: Sequence[int],
    table: NDArray[np.float64],
) -> NDArray[np.float64]:
    state_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(state_indices < 0) or np.any(state_indices >= table.shape[0]):
        raise ValueError("phase indices out of valid range")
    return table[state_indices]


def composite_positional_encoding(
    timestamps: Sequence[float],
    treatment_phase_indices: Sequence[int],
    disease_phase_indices: Union[Sequence[int], Sequence[DiseasePhase]],
    dimension: int = HIDDEN_DIMENSION,
    treatment_table: Optional[NDArray[np.float64]] = None,
    disease_table: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    treatment_indices = np.asarray(treatment_phase_indices, dtype=np.int64).reshape(-1)
    disease_indices = np.asarray(
        [int(value) for value in disease_phase_indices], dtype=np.int64
    ).reshape(-1)
    if not (times.shape == treatment_indices.shape == disease_indices.shape):
        raise ValueError("timestamps, treatment, and disease indices must align")
    time_component = continuous_time_sinusoidal_encoding(times, dimension=dimension)
    treatment_table = (
        treatment_phase_embedding_table(dimension=dimension)
        if treatment_table is None
        else treatment_table
    )
    disease_table = (
        disease_phase_embedding_table(dimension=dimension)
        if disease_table is None
        else disease_table
    )
    treatment_component = lookup_phase_embedding(treatment_indices, treatment_table)
    disease_component = lookup_phase_embedding(disease_indices, disease_table)
    return time_component + treatment_component + disease_component


__all__ = [
    "continuous_time_sinusoidal_encoding",
    "treatment_phase_embedding_table",
    "disease_phase_embedding_table",
    "lookup_phase_embedding",
    "composite_positional_encoding",
]
