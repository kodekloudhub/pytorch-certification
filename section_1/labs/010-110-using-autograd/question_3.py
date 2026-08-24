"""
Inspect the computation graph nodes for y and z before calling backward(), then print the gradient dz/dx.

NOTE: The grad_fn attribute identifies the operation that created a non-leaf tensor.
"""
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * x
z = y + 3
print("y node:", y.____)
print("z node:", z.____)
z.backward()
print("dz/dx:", x.grad)
