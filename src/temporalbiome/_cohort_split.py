from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import CohortLeakageError, CohortSplit


def _validate_proportions(train: float, validation: float, test: float) -> None:
    total = train + validation + test
    if not np.isclose(total, 1.0, atol=1.0e-9):
        raise ValueError("train, validation, and test proportions must sum to 1.0")
    if min(train, validation, test) <= 0.0:
        raise ValueError("each split proportion must be strictly positive")


def stratified_patient_disjoint_split(
    patient_ids: Sequence[int],
    outcome_labels: Sequence[int],
    train_proportion: float = 0.60,
    validation_proportion: float = 0.20,
    test_proportion: float = 0.20,
    seed: int = 0,
) -> CohortSplit:
    _validate_proportions(train_proportion, validation_proportion, test_proportion)
    ids = np.asarray(patient_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(outcome_labels, dtype=np.int64).reshape(-1)
    if ids.shape != labels.shape:
        raise CohortLeakageError("patient_ids and outcome_labels must align")
    if ids.size == 0:
        raise CohortLeakageError("inputs must be nonempty")
    unique_ids, first_occurrence_indices = np.unique(ids, return_index=True)
    patient_labels = labels[first_occurrence_indices]
    for unique_id in unique_ids:
        rows = labels[ids == unique_id]
        if not np.all(rows == rows[0]):
            raise CohortLeakageError("patient outcome labels must be consistent across rows")
    generator = np.random.default_rng(seed)
    train_ids: list[int] = []
    validation_ids: list[int] = []
    test_ids: list[int] = []
    for class_value in np.unique(patient_labels):
        class_mask = patient_labels == class_value
        class_patients = unique_ids[class_mask]
        permuted = generator.permutation(class_patients)
        n = permuted.shape[0]
        n_train = int(round(train_proportion * n))
        n_validation = int(round(validation_proportion * n))
        n_test = n - n_train - n_validation
        if n_test < 0:
            n_validation += n_test
            n_test = 0
        train_ids.extend(int(value) for value in permuted[:n_train])
        validation_ids.extend(int(value) for value in permuted[n_train:n_train + n_validation])
        test_ids.extend(int(value) for value in permuted[n_train + n_validation:n_train + n_validation + n_test])
    train_array = np.asarray(sorted(train_ids), dtype=np.int64)
    validation_array = np.asarray(sorted(validation_ids), dtype=np.int64)
    test_array = np.asarray(sorted(test_ids), dtype=np.int64)
    train_set = set(train_array.tolist())
    validation_set = set(validation_array.tolist())
    test_set = set(test_array.tolist())
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise CohortLeakageError("split must be patient-disjoint")
    label_for = {int(unique): int(class_label) for unique, class_label in zip(unique_ids, patient_labels)}
    train_balance = float(np.mean([label_for[i] for i in train_array]) if train_array.size else 0.0)
    validation_balance = float(np.mean([label_for[i] for i in validation_array]) if validation_array.size else 0.0)
    test_balance = float(np.mean([label_for[i] for i in test_array]) if test_array.size else 0.0)
    return CohortSplit(
        train_patient_ids=train_array,
        validation_patient_ids=validation_array,
        test_patient_ids=test_array,
        label_balance=(train_balance, validation_balance, test_balance),
    )


def assert_split_disjoint(split: CohortSplit) -> None:
    train_set = set(split.train_patient_ids.tolist())
    validation_set = set(split.validation_patient_ids.tolist())
    test_set = set(split.test_patient_ids.tolist())
    if train_set & validation_set:
        raise CohortLeakageError("train and validation overlap")
    if train_set & test_set:
        raise CohortLeakageError("train and test overlap")
    if validation_set & test_set:
        raise CohortLeakageError("validation and test overlap")


__all__ = [
    "stratified_patient_disjoint_split",
    "assert_split_disjoint",
]
