"""
Concatenate the two tensors on dimension 0, then stack the same tensors on a new dimension 0. Print both shapes.

NOTE: Concatenation joins an existing dimension while stacking creates a new dimension.
"""
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
concatenated = torch.____((____, ____), dim=____)
stacked = torch.____((____, ____), dim=____)
print(concatenated.shape, stacked.shape)
