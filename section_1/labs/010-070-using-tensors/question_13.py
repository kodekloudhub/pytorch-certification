"""
Convert the integer tensor to the float32 data type without modifying the original tensor. Print both data types.

NOTE: Use the tensor method that can change a tensor's data type or device.
"""
import torch

integer_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
float_tensor = integer_tensor.____(dtype=____)
print(integer_tensor.dtype, float_tensor.dtype)
