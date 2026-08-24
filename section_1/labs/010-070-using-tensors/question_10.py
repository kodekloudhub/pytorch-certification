"""
Reshape the tensor to 2 rows without changing its values and print the tensor and its shape.

NOTE: Let PyTorch infer the number of columns from the number of values.
"""
import torch

tensor = torch.arange(12)
reshaped = tensor.____(____, ____)
print(reshaped, reshaped.shape)
