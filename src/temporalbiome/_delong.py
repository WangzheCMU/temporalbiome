from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ._types import (
    BONFERRONI_BASELINE_COUNT,
    DeLongResult,
    IncompatibleScoreVectorsError,
)


def _midrank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    sorted_indices = np.argsort(values, kind="mergesort")
    sorted_values = values[sorted_indices]
    n = values.shape[0]
    midranks = np.empty(n, dtype=np.float64)
    cursor = 0
    while cursor < n:
        runner = cursor
        while runner < n and sorted_values[runner] == sorted_values[cursor]:
            runner += 1
        average = 0.5 * (cursor + runner + 1)
        for index in range(cursor, runner):
            midranks[sorted_indices[index]] = average
        cursor = runner
    return midranks


def _structural_components(
    positive_scores: NDArray[np.float64],
    negative_scores: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    m = positive_scores.shape[0]
    n = negative_scores.shape[0]
    combined = np.concatenate([positive_scores, negative_scores])
    combined_rank = _midrank(combined)
    positive_self_rank = _midrank(positive_scores)
    negative_self_rank = _midrank(negative_scores)
    v10 = (combined_rank[:m] - positive_self_rank) / float(n)
    v01 = 1.0 - (combined_rank[m:] - negative_self_rank) / float(m)
    auroc = float(v10.mean())
    return auroc, v10, v01


def auroc(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if s.shape != y.shape:
        raise IncompatibleScoreVectorsError("scores and labels must align")
    positives = s[y == 1]
    negatives = s[y == 0]
    if positives.size == 0 or negatives.size == 0:
        raise IncompatibleScoreVectorsError("auroc requires at least one positive and one negative case")
    auroc_value, _, _ = _structural_components(positives, negatives)
    return auroc_value


def paired_delong(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_comparisons_for_bonferroni: int = BONFERRONI_BASELINE_COUNT,
) -> DeLongResult:
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (a.shape == b.shape == y.shape):
        raise IncompatibleScoreVectorsError("score vectors and labels must align")
    if a.size == 0:
        raise IncompatibleScoreVectorsError("inputs must be nonempty")
    positive_mask = y == 1
    negative_mask = y == 0
    pos_a = a[positive_mask]
    neg_a = a[negative_mask]
    pos_b = b[positive_mask]
    neg_b = b[negative_mask]
    if pos_a.size == 0 or neg_a.size == 0:
        raise IncompatibleScoreVectorsError("paired DeLong requires both classes")
    auroc_a, v10_a, v01_a = _structural_components(pos_a, neg_a)
    auroc_b, v10_b, v01_b = _structural_components(pos_b, neg_b)
    m = pos_a.shape[0]
    n = neg_a.shape[0]
    if m < 2 or n < 2:
        return DeLongResult(
            auroc_a=auroc_a,
            auroc_b=auroc_b,
            delta=auroc_a - auroc_b,
            z_statistic=0.0,
            p_value_unadjusted=1.0,
            p_value_bonferroni=1.0,
        )
    s10_aa = float(((v10_a - auroc_a) ** 2).sum()) / (m - 1)
    s10_bb = float(((v10_b - auroc_b) ** 2).sum()) / (m - 1)
    s10_ab = float(((v10_a - auroc_a) * (v10_b - auroc_b)).sum()) / (m - 1)
    s01_aa = float(((v01_a - auroc_a) ** 2).sum()) / (n - 1)
    s01_bb = float(((v01_b - auroc_b) ** 2).sum()) / (n - 1)
    s01_ab = float(((v01_a - auroc_a) * (v01_b - auroc_b)).sum()) / (n - 1)
    var_a = s10_aa / m + s01_aa / n
    var_b = s10_bb / m + s01_bb / n
    cov_ab = s10_ab / m + s01_ab / n
    delta = auroc_a - auroc_b
    variance_of_delta = var_a + var_b - 2.0 * cov_ab
    if variance_of_delta <= 0.0:
        z = 0.0
        p_unadjusted = 1.0
    else:
        z = delta / np.sqrt(variance_of_delta)
        p_unadjusted = 2.0 * float(norm.sf(abs(z)))
    p_bonferroni = float(min(1.0, p_unadjusted * max(1, n_comparisons_for_bonferroni)))
    return DeLongResult(
        auroc_a=auroc_a,
        auroc_b=auroc_b,
        delta=delta,
        z_statistic=float(z),
        p_value_unadjusted=p_unadjusted,
        p_value_bonferroni=p_bonferroni,
    )


def bonferroni_adjust(
    raw_p_values: Sequence[float],
    family_size: Optional[int] = None,
) -> NDArray[np.float64]:
    raw = np.asarray(raw_p_values, dtype=np.float64).reshape(-1)
    factor = float(family_size if family_size is not None else raw.shape[0])
    return np.clip(raw * factor, 0.0, 1.0)


__all__ = [
    "auroc",
    "paired_delong",
    "bonferroni_adjust",
]
