"""
Call backward twice without clearing the gradient and print the gradient after each call.

NOTE: PyTorch accumulates gradients in the grad attribute by default.
"""
import torch

x = torch.tensor(2.0, requires_grad=True)
for step in range(2):
    loss = x**2
    loss.____()
    print(f"after backward {step + 1}:", x.____)
