"""
Calculate breast cancer classification accuracy with device-aware batch inference.

NOTE: Move both features and labels to the selected device before comparing predictions with labels.
"""

import torch

from pre import test_loader
from question_1 import device, model


model.eval()
correct = 0
total = 0

with torch.____():
    for features, labels in test_loader:
        features = features.____(device)
        labels = labels.____(device)
        predictions = model(features).argmax(dim=1)
        correct += (predictions == labels).____().item()
        total += labels.____(0)

print(f"Accuracy: {100.0 * correct / total:.2f}%")
