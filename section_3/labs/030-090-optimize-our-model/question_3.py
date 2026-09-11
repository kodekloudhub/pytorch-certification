"""
Create an MSELoss function and use it to calculate the loss for the provided regression predictions and targets. Print the loss.

NOTE: MSELoss is commonly used for regression. Predictions and targets should have matching shapes to avoid unintended broadcasting.
"""
from torch import nn
from pre import regression_predictions, regression_targets


criterion = nn.____()
loss = ____(____, ____)
print(f"MSELoss: {loss.item():.4f}")
