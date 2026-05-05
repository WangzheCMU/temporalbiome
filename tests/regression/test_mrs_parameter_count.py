import pytest

torch = pytest.importorskip("torch")

from temporalbiome import MicrobiomeRiskScoreNetwork


def test_default_parameter_count_lies_within_paper_envelope() -> None:
    network = MicrobiomeRiskScoreNetwork()
    total = network.total_parameter_count()
    assert 4_000_000 <= total <= 12_000_000


def test_subsystem_breakdown_records_three_named_children() -> None:
    network = MicrobiomeRiskScoreNetwork()
    breakdown = network.parameter_summary()
    assert breakdown["temporal_microbiome_encoder"] > breakdown["risk_score_generator"]
    assert breakdown["clinical_feature_fusion"] > breakdown["risk_score_generator"]


def test_parameter_count_scales_predictably_with_layer_depth() -> None:
    shallow = MicrobiomeRiskScoreNetwork(microbiome_layers=1, clinical_layers=1).total_parameter_count()
    deep = MicrobiomeRiskScoreNetwork(microbiome_layers=8, clinical_layers=4).total_parameter_count()
    assert deep > shallow
    assert deep - shallow > 2_000_000
