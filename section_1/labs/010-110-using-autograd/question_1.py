"""
Create a floating-point tensor that tracks operations for automatic differentiation. Print its requires_grad and is_leaf attributes.

NOTE: Enable gradient tracking when the tensor is created.
"""
import torch

x = torch.tensor([2.0, 3.0], ____=____)
print(x.requires_grad, x.is_leaf)
