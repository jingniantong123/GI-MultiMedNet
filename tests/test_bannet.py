"""
test_bannet.py
--------------
Unit tests for BANNet model.

These tests validate:
- Model forward pass
- Output shape correctness
- Basic inference functionality
"""

import torch
from src.bannet.models import BANNet


def test_bannet_forward():
    """
    Ensure model forward pass works with dummy data.
    """

    model = BANNet(num_classes=2)

    batch_size = 4
    dummy_image = torch.randn(batch_size, 1, 224, 224)
    dummy_biomech = torch.randn(batch_size, 50, 32)
    dummy_physio = torch.randn(batch_size, 32)
    dummy_perf = torch.randn(batch_size, 32)

    out = model(dummy_image, dummy_biomech, dummy_physio, dummy_perf)

    # Check logits shape
    assert "logits" in out, "BANNet output must contain 'logits'"
    logits = out["logits"]
    assert logits.shape == (batch_size, 2), "Logits must match (B, num_classes)"


def test_bannet_device_cpu():
    """
    Check if model can run on CPU without issues.
    """
    model = BANNet(num_classes=2)
    model.to("cpu")

    dummy_image = torch.randn(1, 1, 224, 224)
    dummy_biomech = torch.randn(1, 50, 32)
    dummy_physio = torch.randn(1, 32)
    dummy_perf = torch.randn(1, 32)

    out = model(dummy_image, dummy_biomech, dummy_physio, dummy_perf)
    assert out["logits"].shape == (1, 2)
