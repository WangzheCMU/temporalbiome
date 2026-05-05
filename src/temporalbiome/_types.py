from enum import IntEnum
from typing import Final, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray


HIDDEN_DIMENSION: Final[int] = 256
TRANSFORMER_LAYERS: Final[int] = 4
ATTENTION_HEADS: Final[int] = 8
FEED_FORWARD_DIMENSION: Final[int] = 1024
DROPOUT_PROBABILITY: Final[float] = 0.20

TREATMENT_PHASE_STATES: Final[int] = 32
DISEASE_PHASE_STATES: Final[int] = 4

CLINICAL_PANEL_DIMENSION: Final[int] = 87
MICROBIOME_PANEL_DIMENSION: Final[int] = 274
MICROBIOME_BIN_HOURS: Final[int] = 6
TEMPORAL_WINDOW_HOURS: Final[int] = 72

MINIMUM_READS_PER_SAMPLE: Final[int] = 1000

FOCAL_ALPHA: Final[float] = 0.75
FOCAL_GAMMA: Final[float] = 2.0

DEFAULT_LEARNING_RATE: Final[float] = 1.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-2
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_PATIENCE_EPOCHS: Final[int] = 15
DEFAULT_MAX_EPOCHS: Final[int] = 100

RISK_TIER_LOW_CEILING: Final[float] = 0.20
RISK_TIER_MODERATE_CEILING: Final[float] = 0.50
RISK_TIER_HIGH_CEILING: Final[float] = 0.80

SIMULATED_INTEGRATION_SOFA_TOLERANCE: Final[int] = 2
SIMULATED_INTEGRATION_MINIMUM_OVERLAP_HOURS: Final[int] = 72
SIMULATED_INTEGRATION_MINIMUM_TIMEPOINTS: Final[int] = 3

BONFERRONI_BASELINE_COUNT: Final[int] = 19

PAPER_RANDOM_SEEDS: Final[tuple[int, ...]] = (
    42, 123, 256, 384, 512, 768, 1024, 1357, 1729, 2048,
    2718, 3141, 3776, 4096, 5040,
)

SPECIES_RESOLVED_GENERA: Final[frozenset[str]] = frozenset({
    "Enterococcus",
    "Klebsiella",
    "Escherichia",
})


class RiskTier(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


class DiseasePhase(IntEnum):
    PRE_TREATMENT = 0
    ACTIVE_TREATMENT = 1
    RECOVERY = 2
    MAINTENANCE = 3


class TreatmentModality(IntEnum):
    CHEMOTHERAPY_ONLY = 0
    SURGERY_ONLY = 1
    RADIATION_ONLY = 2
    COMBINED = 3


class CohortSplit(NamedTuple):
    train_patient_ids: NDArray[np.int64]
    validation_patient_ids: NDArray[np.int64]
    test_patient_ids: NDArray[np.int64]
    label_balance: tuple[float, float, float]


class BootstrapResult(NamedTuple):
    point_estimate: float
    lower: float
    upper: float
    replicate_values: NDArray[np.float64]


class CalibrationFit(NamedTuple):
    temperature: float
    log_likelihood: float
    iterations: int


class PlattFit(NamedTuple):
    slope: float
    intercept: float


class DeLongResult(NamedTuple):
    auroc_a: float
    auroc_b: float
    delta: float
    z_statistic: float
    p_value_unadjusted: float
    p_value_bonferroni: float


class DecisionCurve(NamedTuple):
    thresholds: NDArray[np.float64]
    net_benefit_model: NDArray[np.float64]
    net_benefit_treat_all: NDArray[np.float64]
    net_benefit_treat_none: NDArray[np.float64]


class ReliabilityBin(NamedTuple):
    lower: float
    upper: float
    mean_predicted: float
    fraction_positive: float
    count: int


class MatchedPair(NamedTuple):
    microbiome_patient_id: int
    clinical_patient_id: int
    overlap_hours: float
    sofa_distance: float
    treatment_modality: TreatmentModality


class AbundanceTable(NamedTuple):
    abundances: NDArray[np.float64]
    taxa: tuple[str, ...]


class RiskScorer(Protocol):
    def __call__(self, microbiome: NDArray[np.float64], clinical: NDArray[np.float64]) -> float: ...


class EmptyAbundanceError(ValueError):
    pass


class InsufficientReadsError(ValueError):
    pass


class CohortLeakageError(RuntimeError):
    pass


class CalibrationConvergenceError(RuntimeError):
    pass


class IncompatibleScoreVectorsError(ValueError):
    pass


__all__ = [
    "HIDDEN_DIMENSION",
    "TRANSFORMER_LAYERS",
    "ATTENTION_HEADS",
    "FEED_FORWARD_DIMENSION",
    "DROPOUT_PROBABILITY",
    "TREATMENT_PHASE_STATES",
    "DISEASE_PHASE_STATES",
    "CLINICAL_PANEL_DIMENSION",
    "MICROBIOME_PANEL_DIMENSION",
    "MICROBIOME_BIN_HOURS",
    "TEMPORAL_WINDOW_HOURS",
    "MINIMUM_READS_PER_SAMPLE",
    "FOCAL_ALPHA",
    "FOCAL_GAMMA",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PATIENCE_EPOCHS",
    "DEFAULT_MAX_EPOCHS",
    "RISK_TIER_LOW_CEILING",
    "RISK_TIER_MODERATE_CEILING",
    "RISK_TIER_HIGH_CEILING",
    "SIMULATED_INTEGRATION_SOFA_TOLERANCE",
    "SIMULATED_INTEGRATION_MINIMUM_OVERLAP_HOURS",
    "SIMULATED_INTEGRATION_MINIMUM_TIMEPOINTS",
    "BONFERRONI_BASELINE_COUNT",
    "PAPER_RANDOM_SEEDS",
    "SPECIES_RESOLVED_GENERA",
    "RiskTier",
    "DiseasePhase",
    "TreatmentModality",
    "CohortSplit",
    "BootstrapResult",
    "CalibrationFit",
    "PlattFit",
    "DeLongResult",
    "DecisionCurve",
    "ReliabilityBin",
    "MatchedPair",
    "AbundanceTable",
    "RiskScorer",
    "EmptyAbundanceError",
    "InsufficientReadsError",
    "CohortLeakageError",
    "CalibrationConvergenceError",
    "IncompatibleScoreVectorsError",
]
