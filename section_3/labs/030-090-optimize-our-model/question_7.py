"""
Complete one optimization step for the provided regression problem using MSELoss and Adam. Print the loss and whether the model weights changed.

NOTE: Perform the step in this order: clear gradients, run the forward pass, calculate loss, call backward, and update the parameters.
"""
import torch
from torch import nn, optim
from pre import create_regression_model, regression_inputs, regression_labels


model = create_regression_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
weights_before = model.weight.detach().clone()

optimizer.____()
predictions = ____(____)
loss = ____(____, ____)
loss.____()
optimizer.____()

print(f"Loss: {loss.item():.4f}")
print(f"Weights changed: {not torch.equal(weights_before, model.weight.detach())}")
