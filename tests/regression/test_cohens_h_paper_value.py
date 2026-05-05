import math

import numpy as np
import pytest

from temporalbiome import cohens_h


def test_cohens_h_zero_when_proportions_match() -> None:
    for proportion in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert cohens_h(proportion, proportion) == pytest.approx(0.0, abs=1e-12)


def test_cohens_h_unit_extreme_equals_pi() -> None:
    assert cohens_h(1.0, 0.0) == pytest.approx(math.pi, abs=1e-12)


def test_cohens_h_half_to_zero_equals_pi_over_two() -> None:
    assert cohens_h(0.5, 0.0) == pytest.approx(math.pi / 2.0, abs=1e-12)


def test_cohens_h_paper_quoted_auroc_pair_matches_canonical_formula() -> None:
    proportion_mrs = 0.921
    proportion_xgboost = 0.854
    expected = (
        2.0 * math.asin(math.sqrt(proportion_mrs))
        - 2.0 * math.asin(math.sqrt(proportion_xgboost))
    )
    assert cohens_h(proportion_mrs, proportion_xgboost) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(0.21432, abs=5e-4)


def test_cohens_h_signs_with_direction() -> None:
    positive = cohens_h(0.7, 0.5)
    negative = cohens_h(0.5, 0.7)
    assert positive > 0.0
    assert negative < 0.0
    assert positive == pytest.approx(-negative, abs=1e-12)


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01, 1.5, -1.0])
def test_cohens_h_rejects_out_of_range_proportions(invalid_value: float) -> None:
    with pytest.raises(ValueError):
        cohens_h(invalid_value, 0.5)
