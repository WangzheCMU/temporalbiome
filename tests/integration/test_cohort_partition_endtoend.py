import numpy as np
import pytest

from temporalbiome import (
    CohortLeakageError,
    assert_split_disjoint,
    stratified_patient_disjoint_split,
)


def _generate_cohort(n_patients: int, prevalence: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    patient_ids = np.arange(1, n_patients + 1, dtype=np.int64)
    labels = (generator.uniform(size=n_patients) < prevalence).astype(np.int64)
    if labels.sum() == 0:
        labels[0] = 1
    elif labels.sum() == n_patients:
        labels[0] = 0
    return patient_ids, labels


@pytest.mark.parametrize("prevalence", [0.08, 0.12, 0.15])
def test_split_proportions_close_to_six_two_two(prevalence: float) -> None:
    n_patients = 1000
    patient_ids, labels = _generate_cohort(n_patients, prevalence, seed=42)
    split = stratified_patient_disjoint_split(patient_ids, labels, seed=42)
    assert_split_disjoint(split)
    n_train = split.train_patient_ids.shape[0]
    n_validation = split.validation_patient_ids.shape[0]
    n_test = split.test_patient_ids.shape[0]
    assert n_train + n_validation + n_test == n_patients
    assert abs(n_train / n_patients - 0.60) < 0.02
    assert abs(n_validation / n_patients - 0.20) < 0.02
    assert abs(n_test / n_patients - 0.20) < 0.02


def test_label_balance_drift_below_half_a_percentage_point() -> None:
    n_patients = 5000
    prevalence = 0.12
    patient_ids, labels = _generate_cohort(n_patients, prevalence, seed=7)
    split = stratified_patient_disjoint_split(patient_ids, labels, seed=7)
    train_balance, validation_balance, test_balance = split.label_balance
    assert abs(train_balance - prevalence) < 0.005
    assert abs(validation_balance - prevalence) < 0.005
    assert abs(test_balance - prevalence) < 0.005


def test_split_is_patient_disjoint_after_repeated_rows() -> None:
    rows_per_patient = 8
    n_patients = 200
    patient_ids = np.repeat(np.arange(1, n_patients + 1, dtype=np.int64), rows_per_patient)
    label_for_each_patient = (np.arange(n_patients) % 3 == 0).astype(np.int64)
    labels = np.repeat(label_for_each_patient, rows_per_patient)
    split = stratified_patient_disjoint_split(patient_ids, labels, seed=2026)
    assert_split_disjoint(split)
    train_set = set(split.train_patient_ids.tolist())
    validation_set = set(split.validation_patient_ids.tolist())
    test_set = set(split.test_patient_ids.tolist())
    assert train_set | validation_set | test_set == set(range(1, n_patients + 1))


def test_inconsistent_labels_per_patient_are_rejected() -> None:
    patient_ids = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
    labels = np.array([0, 1, 0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(CohortLeakageError):
        stratified_patient_disjoint_split(patient_ids, labels, seed=0)


def test_split_is_deterministic_across_runs() -> None:
    n_patients = 700
    patient_ids, labels = _generate_cohort(n_patients, prevalence=0.10, seed=314)
    first = stratified_patient_disjoint_split(patient_ids, labels, seed=314)
    second = stratified_patient_disjoint_split(patient_ids, labels, seed=314)
    assert np.array_equal(first.train_patient_ids, second.train_patient_ids)
    assert np.array_equal(first.validation_patient_ids, second.validation_patient_ids)
    assert np.array_equal(first.test_patient_ids, second.test_patient_ids)


def test_paper_msk_cohort_size_replicates_eight_four_seven_partition() -> None:
    n_patients = 847
    prevalence = 0.12
    patient_ids, labels = _generate_cohort(n_patients, prevalence, seed=42)
    split = stratified_patient_disjoint_split(patient_ids, labels, seed=42)
    assert_split_disjoint(split)
    total = (
        split.train_patient_ids.shape[0]
        + split.validation_patient_ids.shape[0]
        + split.test_patient_ids.shape[0]
    )
    assert total == n_patients
