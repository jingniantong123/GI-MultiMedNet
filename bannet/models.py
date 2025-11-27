"""
models.py
---------
Main model architecture for BANNet (Biomechanical-Aware Neural Network).

This module defines:
- Spatial encoder for medical imaging
- Temporal encoder for biomechanical and physiological signals
- Fusion network for multimodal integration
- BANNet model for GI injury classification or regression

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Image Encoder (e.g., medical imaging: MRI, ultrasound)
# -----------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, img_channels=1, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------------------------------------
# Biomechanical / Physiological Sequence Encoder (RNN)
# -----------------------------------------------------------
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, feature_dim=128):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, seq):
        # seq: (B, T, F)
        outputs, _ = self.rnn(seq)
        final = outputs[:, -1, :]  # last timestep
        return self.fc(final)


# -----------------------------------------------------------
# Multimodal Fusion Network
# -----------------------------------------------------------
class FusionNetwork(nn.Module):
    def __init__(self, img_dim=128, seq_dim=128, fused_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + seq_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU()
        )

    def forward(self, img_feat, seq_feat):
        x = torch.cat([img_feat, seq_feat], dim=1)
        return self.fc(x)


# -----------------------------------------------------------
# Full BANNet Model
# -----------------------------------------------------------
class BANNet(nn.Module):
    def __init__(self, num_classes=2, img_channels=1, seq_input_dim=32):
        super().__init__()
        self.image_encoder = ImageEncoder(img_channels=img_channels)
        self.sequence_encoder = SequenceEncoder(input_dim=seq_input_dim)

        self.fusion = FusionNetwork()

        self.output_head = nn.Linear(256 // 2, num_classes)

    def forward(self, image, sequence):
        img_feat = self.image_encoder(image)
        seq_feat = self.sequence_encoder(sequence)

        fused = self.fusion(img_feat, seq_feat)

        return self.output_head(fused)
