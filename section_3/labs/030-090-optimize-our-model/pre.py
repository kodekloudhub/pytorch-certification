"""Shared deterministic tensors and model factory for the optimization lab."""

import torch
from torch import nn


SEED = 42

# Multiclass classification: raw logits and integer class indices.
multiclass_logits = torch.tensor([
    [2.0, 0.5, -0.5],
    [0.1, 1.7, 0.2],
    [-0.3, 0.4, 1.9],
])
class_targets = torch.tensor([0, 1, 2], dtype=torch.long)

# Binary classification: logits and floating-point targets with matching shapes.
binary_logits = torch.tensor([[1.4], [-0.8], [0.2], [1.1]])
binary_targets = torch.tensor([[1.0], [0.0], [0.0], [1.0]])

# Regression: continuous predictions and targets with matching shapes.
regression_predictions = torch.tensor([[2.5], [3.5], [4.0]])
regression_targets = torch.tensor([[3.0], [3.0], [5.0]])

# A small deterministic regression problem for optimizer exercises.
regression_inputs = torch.linspace(-1, 1, 64).unsqueeze(1)
regression_labels = 3 * regression_inputs + 0.5


def create_regression_model():
    """Return the same initial model whenever an optimizer is compared."""
    torch.manual_seed(SEED)
    return nn.Linear(1, 1)
