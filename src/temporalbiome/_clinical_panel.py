from typing import Final, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import CLINICAL_PANEL_DIMENSION, MICROBIOME_BIN_HOURS


VITAL_SIGN_FEATURES: Final[tuple[str, ...]] = (
    "heart_rate",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "mean_arterial_pressure",
    "respiratory_rate",
    "temperature",
    "oxygen_saturation",
)

LABORATORY_FEATURES: Final[tuple[str, ...]] = (
    "white_blood_cell_count",
    "neutrophil_count",
    "lymphocyte_count",
    "monocyte_count",
    "platelet_count",
    "hemoglobin",
    "hematocrit",
    "c_reactive_protein",
    "procalcitonin",
    "lactate",
    "creatinine",
    "blood_urea_nitrogen",
    "albumin",
    "total_bilirubin",
    "alanine_aminotransferase",
    "aspartate_aminotransferase",
    "alkaline_phosphatase",
    "international_normalized_ratio",
    "prothrombin_time",
    "activated_partial_thromboplastin_time",
    "potassium",
    "sodium",
    "chloride",
    "bicarbonate",
    "glucose",
    "magnesium",
    "phosphate",
    "calcium",
    "anion_gap",
    "ph_arterial",
    "partial_pressure_oxygen",
    "partial_pressure_carbon_dioxide",
    "base_excess",
    "fraction_inspired_oxygen",
)

DRUG_CLASS_FLAGS: Final[tuple[str, ...]] = (
    "antibiotic_beta_lactam",
    "antibiotic_glycopeptide",
    "antibiotic_fluoroquinolone",
    "antibiotic_carbapenem",
    "antibiotic_aminoglycoside",
    "antibiotic_metronidazole",
    "antibiotic_macrolide",
    "antibiotic_sulfonamide",
    "antifungal",
    "antiviral",
    "chemotherapy_fluoropyrimidine",
    "chemotherapy_oxaliplatin",
    "chemotherapy_irinotecan",
    "chemotherapy_taxane",
    "chemotherapy_anthracycline",
    "immunosuppressant_corticosteroid",
    "immunosuppressant_calcineurin_inhibitor",
    "immunosuppressant_antimetabolite",
    "vasopressor_norepinephrine",
    "vasopressor_phenylephrine",
    "vasopressor_vasopressin",
    "vasopressor_dopamine",
)

PROCEDURE_FLAGS: Final[tuple[str, ...]] = (
    "central_line_present",
    "mechanical_ventilation",
    "renal_replacement_therapy",
    "surgical_drain_present",
    "urinary_catheter_present",
    "endoscopy_within_seven_days",
    "colorectal_resection_within_thirty_days",
    "stoma_present",
    "graft_versus_host_prophylaxis",
    "myeloablative_conditioning",
)

CLINICAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    VITAL_SIGN_FEATURES
    + LABORATORY_FEATURES
    + DRUG_CLASS_FLAGS
    + PROCEDURE_FLAGS
    + (
        "chemotherapy_cycle_number",
        "days_since_surgery",
        "days_since_last_antibiotic",
        "days_since_admission",
        "absolute_neutrophil_count_low_flag",
        "neutropenia_grade",
        "graft_versus_host_disease_grade",
        "patient_age_years",
        "body_mass_index",
        "sex_female_flag",
        "sex_male_flag",
        "sequential_organ_failure_assessment_score",
        "acute_physiology_and_chronic_health_evaluation_score",
        "national_early_warning_score",
    )
)


def _validate_panel_dimension() -> None:
    expected = CLINICAL_PANEL_DIMENSION
    actual = len(CLINICAL_FEATURE_NAMES)
    if actual != expected:
        raise AssertionError(
            f"clinical panel size mismatch: expected {expected}, got {actual}"
        )


_validate_panel_dimension()


def six_hour_bin_edges(
    hour_start: float,
    hour_end: float,
    bin_hours: int = MICROBIOME_BIN_HOURS,
) -> NDArray[np.float64]:
    if hour_end <= hour_start:
        raise ValueError("hour_end must exceed hour_start")
    n_bins = int(np.ceil((hour_end - hour_start) / bin_hours))
    return hour_start + np.arange(n_bins + 1, dtype=np.float64) * bin_hours


def aggregate_to_six_hour_panel(
    timestamps_hours: NDArray[np.float64],
    feature_matrix: NDArray[np.float64],
    hour_start: float,
    hour_end: float,
    bin_hours: int = MICROBIOME_BIN_HOURS,
) -> NDArray[np.float64]:
    times = np.asarray(timestamps_hours, dtype=np.float64).reshape(-1)
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != times.shape[0]:
        raise ValueError("feature_matrix must be (n_samples, n_features) aligned with timestamps")
    edges = six_hour_bin_edges(hour_start, hour_end, bin_hours=bin_hours)
    n_bins = edges.shape[0] - 1
    n_features = matrix.shape[1]
    panel = np.full((n_bins, n_features), np.nan, dtype=np.float64)
    for index in range(n_bins):
        low = edges[index]
        high = edges[index + 1]
        mask = (times >= low) & (times < high)
        if mask.any():
            panel[index] = matrix[mask].mean(axis=0)
    return panel


def impute_clinical_panel(
    panel: NDArray[np.float64],
    forward_fill_window_bins: int = 1,
) -> NDArray[np.float64]:
    matrix = np.asarray(panel, dtype=np.float64).copy()
    if matrix.ndim != 2:
        raise ValueError("panel must be (n_bins, n_features)")
    n_bins, n_features = matrix.shape
    for feature_index in range(n_features):
        last_value: Optional[float] = None
        last_filled_index = -10_000
        for bin_index in range(n_bins):
            value = matrix[bin_index, feature_index]
            if not np.isnan(value):
                last_value = float(value)
                last_filled_index = bin_index
            elif last_value is not None and (bin_index - last_filled_index) <= forward_fill_window_bins:
                matrix[bin_index, feature_index] = last_value
    column_medians = np.nanmedian(matrix, axis=0)
    column_medians = np.where(np.isnan(column_medians), 0.0, column_medians)
    fill_indices = np.isnan(matrix)
    matrix = np.where(fill_indices, column_medians, matrix)
    return matrix


def feature_index_lookup(panel_names: Sequence[str] = CLINICAL_FEATURE_NAMES) -> Mapping[str, int]:
    return {name: index for index, name in enumerate(panel_names)}


__all__ = [
    "VITAL_SIGN_FEATURES",
    "LABORATORY_FEATURES",
    "DRUG_CLASS_FLAGS",
    "PROCEDURE_FLAGS",
    "CLINICAL_FEATURE_NAMES",
    "six_hour_bin_edges",
    "aggregate_to_six_hour_panel",
    "impute_clinical_panel",
    "feature_index_lookup",
]
