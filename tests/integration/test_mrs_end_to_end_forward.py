import pytest

torch = pytest.importorskip("torch")

import numpy as np

from temporalbiome import (
    CLINICAL_PANEL_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    MicrobiomeRiskScoreNetwork,
    PAPER_RANDOM_SEEDS,
    classify_risk_tier,
    focal_cross_entropy,
)


def _build_inputs(batch_size: int, timepoints: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "microbiome_features": torch.randn(
            batch_size, timepoints, MICROBIOME_PANEL_DIMENSION, generator=generator
        ),
        "microbiome_timestamps": torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1),
        "microbiome_treatment_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "microbiome_disease_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "clinical_features": torch.randn(
            batch_size, timepoints, CLINICAL_PANEL_DIMENSION, generator=generator
        ),
        "clinical_timestamps": torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1),
        "clinical_treatment_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "clinical_disease_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
    }


def test_full_pipeline_forward_loss_backward_then_classify_tiers() -> None:
    torch.manual_seed(PAPER_RANDOM_SEEDS[0])
    network = MicrobiomeRiskScoreNetwork(microbiome_layers=2, clinical_layers=1, dropout=0.0)
    network.train()
    inputs = _build_inputs(batch_size=4, timepoints=8, seed=PAPER_RANDOM_SEEDS[0])
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    output = network(**inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.raw_logit, labels.to(output.raw_logit.dtype)
    )
    loss.backward()
    assert torch.isfinite(loss).item()
    network.eval()
    with torch.no_grad():
        evaluation = network(**inputs)
    probabilities = evaluation.calibrated_probability.cpu().numpy()
    tiers = [classify_risk_tier(float(p)).name for p in probabilities]
    assert all(tier in {"LOW", "MODERATE", "HIGH", "CRITICAL"} for tier in tiers)


def test_focal_loss_consumes_network_logits() -> None:
    torch.manual_seed(PAPER_RANDOM_SEEDS[1])
    network = MicrobiomeRiskScoreNetwork(microbiome_layers=1, clinical_layers=1, dropout=0.0).eval()
    inputs = _build_inputs(batch_size=6, timepoints=5, seed=PAPER_RANDOM_SEEDS[1])
    labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    with torch.no_grad():
        output = network(**inputs)
    focal_value = focal_cross_entropy(output.raw_logit.cpu().numpy(), labels)
    assert focal_value >= 0.0


def test_temperature_scaling_on_eval_mode_yields_distinct_probabilities() -> None:
    torch.manual_seed(PAPER_RANDOM_SEEDS[2])
    network = MicrobiomeRiskScoreNetwork(initial_temperature=1.0).eval()
    inputs = _build_inputs(batch_size=2, timepoints=6, seed=PAPER_RANDOM_SEEDS[2])
    with torch.no_grad():
        cool_output = network(**inputs)
    network.risk_score_generator.set_temperature(2.5)
    with torch.no_grad():
        warm_output = network(**inputs)
    assert torch.allclose(cool_output.raw_logit, warm_output.raw_logit, atol=1e-7)
    assert not torch.allclose(
        cool_output.calibrated_probability,
        warm_output.calibrated_probability,
        atol=1e-3,
    )
