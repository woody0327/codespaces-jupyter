"""Unit tests for the CIFAR-10 image classifier model."""

import torch

from image_classifier import Net


def test_net_outputs_logits_with_expected_shape() -> None:
    """The model should emit class logits for each image in the batch."""
    batch_size = 4
    dummy_batch = torch.randn(batch_size, 3, 32, 32)
    model = Net()

    logits = model(dummy_batch)

    assert logits.shape == (batch_size, 10)
