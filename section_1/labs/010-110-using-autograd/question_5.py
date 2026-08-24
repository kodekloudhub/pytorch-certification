"""
Clear the tensor's gradient between backward passes so the gradient does not accumulate.

NOTE: Set the existing grad tensor to zero after each training step.
"""
import torch

x = torch.tensor(2.0, requires_grad=True)
for step in range(2):
    loss = x**2
    loss.backward()
    print(f"step {step + 1}:", x.grad)
    x.____.____()
