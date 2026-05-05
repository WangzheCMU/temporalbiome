from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import (
    MatchedPair,
    SIMULATED_INTEGRATION_MINIMUM_OVERLAP_HOURS,
    SIMULATED_INTEGRATION_MINIMUM_TIMEPOINTS,
    SIMULATED_INTEGRATION_SOFA_TOLERANCE,
    TreatmentModality,
)


def irregular_intervals(timestamps_hours: Sequence[float]) -> NDArray[np.float64]:
    times = np.asarray(timestamps_hours, dtype=np.float64).reshape(-1)
    if times.size <= 1:
        return np.zeros(0, dtype=np.float64)
    sorted_times = np.sort(times)
    return np.diff(sorted_times)


def overlap_window_hours(
    microbiome_times: Sequence[float],
    clinical_times: Sequence[float],
) -> float:
    micro = np.asarray(microbiome_times, dtype=np.float64).reshape(-1)
    clin = np.asarray(clinical_times, dtype=np.float64).reshape(-1)
    if micro.size == 0 or clin.size == 0:
        return 0.0
    overlap_low = max(float(micro.min()), float(clin.min()))
    overlap_high = min(float(micro.max()), float(clin.max()))
    return max(0.0, overlap_high - overlap_low)


def match_simulated_integration_pair(
    microbiome_patient_id: int,
    microbiome_times: Sequence[float],
    microbiome_sofa: float,
    microbiome_treatment: TreatmentModality,
    clinical_patient_id: int,
    clinical_times: Sequence[float],
    clinical_sofa: float,
    clinical_treatment: TreatmentModality,
    sofa_tolerance: int = SIMULATED_INTEGRATION_SOFA_TOLERANCE,
    minimum_overlap_hours: int = SIMULATED_INTEGRATION_MINIMUM_OVERLAP_HOURS,
    minimum_timepoints: int = SIMULATED_INTEGRATION_MINIMUM_TIMEPOINTS,
) -> Optional[MatchedPair]:
    if int(microbiome_treatment) != int(clinical_treatment):
        return None
    sofa_distance = abs(float(microbiome_sofa) - float(clinical_sofa))
    if sofa_distance > sofa_tolerance:
        return None
    micro_array = np.asarray(microbiome_times, dtype=np.float64).reshape(-1)
    if micro_array.size < minimum_timepoints:
        return None
    overlap = overlap_window_hours(microbiome_times, clinical_times)
    if overlap < minimum_overlap_hours:
        return None
    return MatchedPair(
        microbiome_patient_id=int(microbiome_patient_id),
        clinical_patient_id=int(clinical_patient_id),
        overlap_hours=float(overlap),
        sofa_distance=float(sofa_distance),
        treatment_modality=microbiome_treatment,
    )


def best_first_simulated_integration(
    microbiome_records: Iterable[tuple[int, Sequence[float], float, TreatmentModality]],
    clinical_records: Iterable[tuple[int, Sequence[float], float, TreatmentModality]],
    sofa_tolerance: int = SIMULATED_INTEGRATION_SOFA_TOLERANCE,
    minimum_overlap_hours: int = SIMULATED_INTEGRATION_MINIMUM_OVERLAP_HOURS,
    minimum_timepoints: int = SIMULATED_INTEGRATION_MINIMUM_TIMEPOINTS,
) -> tuple[MatchedPair, ...]:
    micro_list = list(microbiome_records)
    clinical_list = list(clinical_records)
    used_clinical: set[int] = set()
    used_microbiome: set[int] = set()
    candidates: list[MatchedPair] = []
    for micro_id, micro_times, micro_sofa, micro_treatment in micro_list:
        for clinical_id, clinical_times, clinical_sofa, clinical_treatment in clinical_list:
            pair = match_simulated_integration_pair(
                microbiome_patient_id=micro_id,
                microbiome_times=micro_times,
                microbiome_sofa=micro_sofa,
                microbiome_treatment=micro_treatment,
                clinical_patient_id=clinical_id,
                clinical_times=clinical_times,
                clinical_sofa=clinical_sofa,
                clinical_treatment=clinical_treatment,
                sofa_tolerance=sofa_tolerance,
                minimum_overlap_hours=minimum_overlap_hours,
                minimum_timepoints=minimum_timepoints,
            )
            if pair is not None:
                candidates.append(pair)
    candidates.sort(key=lambda pair: (-pair.overlap_hours, pair.sofa_distance))
    selected: list[MatchedPair] = []
    for pair in candidates:
        if pair.microbiome_patient_id in used_microbiome:
            continue
        if pair.clinical_patient_id in used_clinical:
            continue
        used_microbiome.add(pair.microbiome_patient_id)
        used_clinical.add(pair.clinical_patient_id)
        selected.append(pair)
    return tuple(selected)


__all__ = [
    "irregular_intervals",
    "overlap_window_hours",
    "match_simulated_integration_pair",
    "best_first_simulated_integration",
]
