"""
Create a CrossEntropyLoss function and use it to calculate the loss for the provided multiclass logits and class targets. Print the loss.

NOTE: CrossEntropyLoss expects raw logits shaped as [batch, classes] and integer class targets. Do not apply softmax first.
"""
from torch import nn
from pre import class_targets, multiclass_logits


criterion = nn.____()
loss = ____(____, ____)
print(f"CrossEntropyLoss: {loss.item():.4f}")
