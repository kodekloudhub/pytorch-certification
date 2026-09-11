"""
Create a BCEWithLogitsLoss function and use it to calculate the loss for the provided binary logits and targets. Print the loss.

NOTE: BCEWithLogitsLoss expects logits and floating-point targets with matching shapes. It already includes the sigmoid operation.
"""
from torch import nn
from pre import binary_logits, binary_targets


criterion = nn.____()
loss = ____(____, ____)
print(f"BCEWithLogitsLoss: {loss.item():.4f}")
