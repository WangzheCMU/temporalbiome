import pytest

torch = pytest.importorskip("torch")

from temporalbiome import (
    HIDDEN_DIMENSION,
    AttentionWeightedTemporalPooling,
    RealTimeRiskScoreGenerator,
)


def test_attention_pooling_output_dimensions() -> None:
    fused_dimension = 2 * HIDDEN_DIMENSION
    pooling = AttentionWeightedTemporalPooling(fused_dimension).eval()
    batch_size, timepoints = 4, 9
    states = torch.randn(batch_size, timepoints, fused_dimension)
    pooled, weights = pooling(states)
    assert pooled.shape == (batch_size, fused_dimension)
    assert weights.shape == (batch_size, timepoints)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_attention_pooling_zero_weights_on_padded_positions() -> None:
    pooling = AttentionWeightedTemporalPooling(HIDDEN_DIMENSION).eval()
    batch_size, timepoints = 2, 8
    states = torch.randn(batch_size, timepoints, HIDDEN_DIMENSION)
    padding_mask = torch.zeros(batch_size, timepoints, dtype=torch.bool)
    padding_mask[:, 5:] = True
    _, weights = pooling(states, padding_mask=padding_mask)
    assert torch.allclose(weights[:, 5:], torch.zeros(batch_size, timepoints - 5), atol=1e-7)


def test_risk_score_generator_returns_scalar_per_sample() -> None:
    rsg = RealTimeRiskScoreGenerator().eval()
    batch_size, timepoints = 3, 10
    fused_states = torch.randn(batch_size, timepoints, 2 * HIDDEN_DIMENSION)
    output = rsg(fused_states)
    assert output.raw_logit.shape == (batch_size,)
    assert output.calibrated_probability.shape == (batch_size,)
    assert torch.all(output.calibrated_probability >= 0.0)
    assert torch.all(output.calibrated_probability <= 1.0)
    assert output.pooling_weights.shape == (batch_size, timepoints)
    assert output.pooled_representation.shape == (batch_size, 2 * HIDDEN_DIMENSION)


def test_risk_score_generator_temperature_only_affects_eval_mode() -> None:
    rsg = RealTimeRiskScoreGenerator(initial_temperature=2.0)
    fused_states = torch.randn(2, 4, 2 * HIDDEN_DIMENSION)
    rsg.train()
    train_output = rsg(fused_states)
    rsg.eval()
    eval_output = rsg(fused_states)
    expected_train = torch.sigmoid(train_output.raw_logit)
    expected_eval = torch.sigmoid(eval_output.raw_logit / 2.0)
    assert torch.allclose(train_output.calibrated_probability, expected_train, atol=1e-6)
    assert torch.allclose(eval_output.calibrated_probability, expected_eval, atol=1e-6)


def test_set_temperature_rejects_nonpositive_values() -> None:
    rsg = RealTimeRiskScoreGenerator()
    with pytest.raises(ValueError):
        rsg.set_temperature(0.0)
    with pytest.raises(ValueError):
        rsg.set_temperature(-1.0)
