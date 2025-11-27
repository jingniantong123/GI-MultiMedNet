"""
visualization.py
----------------
Visualization utilities for BANNet and ARPOS.

Supports:
- Medical imaging visualization
- Multimodal feature plotting
- RL learning curves
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


# -----------------------------------------------------------
# Medical Image Visualization
# -----------------------------------------------------------
def show_medical_image(image_tensor, title="Medical Image"):
    """
    Display a single image tensor (C, H, W).
    """
    if image_tensor.ndim == 3:
        image = image_tensor.squeeze(0).numpy()
    else:
        raise ValueError("Expected image tensor of shape (C, H, W)")

    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# -----------------------------------------------------------
# RL Learning Curve
# -----------------------------------------------------------
def plot_learning_curve(reward_history, title="Learning Curve"):
    """
    Plot the reward progression over episodes.
    """
    plt.plot(reward_history)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------
# Feature Embedding Visualization
# -----------------------------------------------------------
def plot_feature_distribution(features, title="Feature Distribution"):
    """
    Visualize distribution of fused features from BANNet.

    features: (N, D) numpy or tensor
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.boxplot(features)
    plt.title(title)
    plt.xlabel("Feature Dimension")
    plt.ylabel("Value")
    plt.show()
