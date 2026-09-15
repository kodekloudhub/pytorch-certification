"""Demonstrate evaluation mode and device-aware batched inference."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(32, 4),
).to(device)

features = torch.randn(128, 16)
targets = torch.randint(0, 4, (128,))
test_loader = DataLoader(TensorDataset(features, targets), batch_size=32)

model.eval()
predictions = []
correct = 0

with torch.inference_mode():
    for batch_features, batch_targets in test_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        batch_predictions = model(batch_features).argmax(dim=1)
        predictions.append(batch_predictions.cpu())
        correct += (batch_predictions == batch_targets).sum().item()

predictions = torch.cat(predictions)
print("device:", device)
print("predictions:", predictions.shape)
print(f"accuracy: {100.0 * correct / len(test_loader.dataset):.2f}%")
