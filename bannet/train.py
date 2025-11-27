"""
train.py
--------
Training script for BANNet.

This script covers:
- Dataset loading (placeholder)
- Model initialization
- Training loop
- Evaluation loop
- Checkpoint saving

Update dataset code to match your actual multimodal dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models import BANNet


# -----------------------------------------------------------
# Placeholder dataset (replace with your real dataset)
# -----------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, length=100):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Fake image: (1, 64, 64)
        image = torch.randn(1, 64, 64)

        # Fake sequence: (T=50, F=32)
        sequence = torch.randn(50, 32)

        # Fake label
        label = torch.randint(0, 2, (1,)).item()

        return image, sequence, label


# -----------------------------------------------------------
# Training Function
# -----------------------------------------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for image, seq, label in dataloader:
        image = image.to(device)
        seq = seq.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(image, seq)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for image, seq, label in dataloader:
            image = image.to(device)
            seq = seq.to(device)
            label = label.to(device)

            pred = model(image, seq)
            loss = criterion(pred, label)
            total_loss += loss.item()

            predicted = pred.argmax(dim=1)
            correct += (predicted == label).sum().item()

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


# -----------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset & DataLoader
    train_ds = MultimodalDataset(length=200)
    val_ds = MultimodalDataset(length=50)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # Model
    model = BANNet(num_classes=2).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), "bannet_checkpoint.pth")
    print("Model saved as bannet_checkpoint.pth")


if __name__ == "__main__":
    main()
