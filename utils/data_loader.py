"""
data_loader.py
--------------
Multimodal data loading utilities for BANNet and ARPOS.

This includes:
- Medical imaging preprocessing
- Biomechanical & physiological sequence loading
- Performance metric loading
- Unified multimodal batch construction

Customize paths and formats to match your actual dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MultimodalDataset(Dataset):
    """
    Generic multimodal dataset for:
        - Medical imaging (JPEG/PNG/DICOM converted to PNG)
        - Biomechanical sequences (CSV or numpy)
        - Physiological signals
        - Performance metrics
        - Labels (classification / regression)

    Expected directory structure:

        dataset/
            sample_001/
                image.png
                biomech.npy
                physio.npy
                performance.npy
                label.txt

    Modify according to your real data.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = sorted(os.listdir(root_dir))
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def load_image(self, img_path):
        img = Image.open(img_path).convert("L")
        return self.transform(img)

    def load_array(self, path):
        return torch.tensor(np.load(path), dtype=torch.float32)

    def load_label(self, path):
        with open(path, "r") as f:
            return torch.tensor(int(f.read().strip()), dtype=torch.long)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])

        image = self.load_image(os.path.join(sample_dir, "image.png"))
        biomech = self.load_array(os.path.join(sample_dir, "biomech.npy"))
        physio = self.load_array(os.path.join(sample_dir, "physio.npy"))
        performance = self.load_array(os.path.join(sample_dir, "performance.npy"))
        label = self.load_label(os.path.join(sample_dir, "label.txt"))

        return {
            "image": image,
            "biomech": biomech,
            "physio": physio,
            "performance": performance,
            "label": label
        }
