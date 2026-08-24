"""
Build the calculation y = sum(x**2), run backpropagation, and print the gradient dy/dx.

NOTE: Square the tensor with torch.pow, reduce it to a scalar, and then call backward().
"""
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.____(x, ____).____()
y.____()
print(x.____)
