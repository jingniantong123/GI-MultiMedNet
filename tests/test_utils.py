"""
test_utils.py
-------------
Unit tests for utils:
- data_loader.py
- metrics.py
- visualization.py (basic smoke tests)
"""

import os
import numpy as np
import torch

from src.utils.data_loader import MultimodalDataset
from src.utils.metrics import classification_scores, regression_scores
from src.utils.visualization import plot_learning_curve


# -----------------------------------------------------------
# Data Loader Tests
# -----------------------------------------------------------
def test_data_loader_structure(tmp_path):
    """
    Create a fake sample folder and verify loading.
    """
    sample = tmp_path / "sample_001"
    sample.mkdir()

    # Fake data files
    np.save(sample / "biomech.npy", np.random.randn(50, 32))
    np.save(sample / "physio.npy", np.random.randn(32))
    np.save(sample / "performance.npy", np.random.randn(32))
    (sample / "label.txt").write_text("1")

    # Fake image
    img = (np.random.rand(224, 224) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img).save(sample / "image.png")

    dataset = MultimodalDataset(root_dir=str(tmp_path))
    sample_data = dataset[0]

    assert "image" in sample_data
    assert "biomech" in sample_data
    assert "physio" in sample_data
    assert "performance" in sample_data
    assert "label" in sample_data


# -----------------------------------------------------------
# Metrics Tests
# -----------------------------------------------------------
def test_classification_metrics():
    y_true = [0, 1, 1]
    y_pred = [0, 1, 0]

    scores = classification_scores(y_true, y_pred)

    assert "accuracy" in scores
    assert "precision" in scores
    assert "confusion_matrix" in scores
    assert scores["accuracy"] >= 0


def test_regression_metrics():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 2.5]

    scores = regression_scores(y_true, y_pred)
    assert "MAE" in scores
    assert "RMSE" in scores


# -----------------------------------------------------------
# Visualization Smoke Test
# -----------------------------------------------------------
def test_plot_learning_curve_smoke():
    """
    Smoke test: ensure function executes without crashing.
    """
    plot_learning_curve([1,2,3,4], title="Test Curve")
