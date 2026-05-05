import pytest

torch = pytest.importorskip("torch")

from temporalbiome import (
    CLINICAL_PANEL_DIMENSION,
    MICROBIOME_PANEL_DIMENSION,
    MicrobiomeRiskScoreNetwork,
    focal_cross_entropy,
)


def _build_tiny_batch(batch_size: int, timepoints: int, seed: int = 0) -> tuple[dict, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        "microbiome_features": torch.randn(batch_size, timepoints, MICROBIOME_PANEL_DIMENSION, generator=generator),
        "microbiome_timestamps": torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1),
        "microbiome_treatment_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "microbiome_disease_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "clinical_features": torch.randn(batch_size, timepoints, CLINICAL_PANEL_DIMENSION, generator=generator),
        "clinical_timestamps": torch.arange(timepoints, dtype=torch.float32).expand(batch_size, -1),
        "clinical_treatment_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
        "clinical_disease_indices": torch.zeros(batch_size, timepoints, dtype=torch.long),
    }
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long)[:batch_size]
    return inputs, labels


def _binary_cross_entropy_with_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))


@pytest.mark.parametrize("seed", [0, 7])
def test_mrs_drives_loss_below_tenth_of_initial_on_eight_sample_batch(seed: int) -> None:
    torch.manual_seed(seed)
    network = MicrobiomeRiskScoreNetwork(
        microbiome_layers=2,
        clinical_layers=1,
        feed_forward_dimension=512,
        dropout=0.0,
    )
    network.train()
    batch_size = 8
    timepoints = 6
    inputs, labels = _build_tiny_batch(batch_size, timepoints, seed=seed)
    optimiser = torch.optim.AdamW(network.parameters(), lr=2.0e-3, weight_decay=0.0)
    output = network(**inputs)
    initial_loss = _binary_cross_entropy_with_logits(output.raw_logit, labels).detach().item()
    final_loss = initial_loss
    for _ in range(120):
        optimiser.zero_grad(set_to_none=True)
        output = network(**inputs)
        loss = _binary_cross_entropy_with_logits(output.raw_logit, labels)
        loss.backward()
        optimiser.step()
        final_loss = loss.detach().item()
    assert final_loss < initial_loss * 0.10


def test_mrs_training_drives_predictions_to_correct_side_of_half() -> None:
    torch.manual_seed(123)
    network = MicrobiomeRiskScoreNetwork(
        microbiome_layers=2,
        clinical_layers=1,
        feed_forward_dimension=512,
        dropout=0.0,
    )
    network.train()
    batch_size = 4
    timepoints = 5
    inputs, labels = _build_tiny_batch(batch_size, timepoints, seed=42)
    optimiser = torch.optim.AdamW(network.parameters(), lr=3.0e-3, weight_decay=0.0)
    for _ in range(160):
        optimiser.zero_grad(set_to_none=True)
        output = network(**inputs)
        torch_loss = _binary_cross_entropy_with_logits(output.raw_logit, labels)
        torch_loss.backward()
        optimiser.step()
    with torch.no_grad():
        diagnostic_focal = focal_cross_entropy(
            network(**inputs).raw_logit.detach().cpu().numpy(),
            labels.cpu().numpy(),
        )
    assert diagnostic_focal >= 0.0
    network.eval()
    with torch.no_grad():
        final = network(**inputs)
    predictions = (final.raw_logit > 0).long()
    assert torch.equal(predictions, labels)
