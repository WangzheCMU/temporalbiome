import numpy as np
from numpy.typing import NDArray

from ._types import (
    RISK_TIER_HIGH_CEILING,
    RISK_TIER_LOW_CEILING,
    RISK_TIER_MODERATE_CEILING,
    RiskTier,
)


def classify_risk_tier(
    probability: float,
    low_ceiling: float = RISK_TIER_LOW_CEILING,
    moderate_ceiling: float = RISK_TIER_MODERATE_CEILING,
    high_ceiling: float = RISK_TIER_HIGH_CEILING,
) -> RiskTier:
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must lie in [0, 1]")
    if not (0.0 < low_ceiling < moderate_ceiling < high_ceiling < 1.0):
        raise ValueError("tier ceilings must be strictly ordered within (0, 1)")
    if probability < low_ceiling:
        return RiskTier.LOW
    if probability < moderate_ceiling:
        return RiskTier.MODERATE
    if probability < high_ceiling:
        return RiskTier.HIGH
    return RiskTier.CRITICAL


def vectorised_risk_tier(
    probabilities: NDArray[np.float64],
    low_ceiling: float = RISK_TIER_LOW_CEILING,
    moderate_ceiling: float = RISK_TIER_MODERATE_CEILING,
    high_ceiling: float = RISK_TIER_HIGH_CEILING,
) -> NDArray[np.int64]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    out = np.full(p.shape[0], RiskTier.CRITICAL, dtype=np.int64)
    out[p < high_ceiling] = RiskTier.HIGH
    out[p < moderate_ceiling] = RiskTier.MODERATE
    out[p < low_ceiling] = RiskTier.LOW
    return out


def lead_time_threshold_crossing(
    risk_trajectory: NDArray[np.float64],
    timestamps_hours: NDArray[np.float64],
    threshold: float,
    onset_time_hours: float,
) -> float:
    trajectory = np.asarray(risk_trajectory, dtype=np.float64).reshape(-1)
    times = np.asarray(timestamps_hours, dtype=np.float64).reshape(-1)
    if trajectory.shape != times.shape:
        raise ValueError("risk_trajectory and timestamps_hours must align")
    above_threshold_indices = np.where(trajectory >= threshold)[0]
    if above_threshold_indices.size == 0:
        return 0.0
    first_crossing_time = float(times[above_threshold_indices[0]])
    lead = float(onset_time_hours) - first_crossing_time
    return max(0.0, lead)


__all__ = [
    "classify_risk_tier",
    "vectorised_risk_tier",
    "lead_time_threshold_crossing",
]
