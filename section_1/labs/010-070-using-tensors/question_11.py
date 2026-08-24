"""
Add the bias tensor to every row of the scores tensor using broadcasting and print the result.

NOTE: The trailing dimension of each tensor must be compatible for broadcasting.
"""
import torch

scores = torch.ones((2, 3))
bias = torch.tensor([0.1, 0.2, 0.3])
adjusted_scores = ____ + ____
print(adjusted_scores)
