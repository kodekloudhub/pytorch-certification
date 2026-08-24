"""
Select the first CUDA device when available, then MPS when available, and otherwise CPU. Move the tensor to the selected device and print its device type and index.

NOTE: Use cuda:0 for the first CUDA device. An MPS or CPU device does not require an index.
"""
import torch

if torch.____.____():
    device = torch.device("____:____")
elif torch.backends.____.____():
    device = torch.device("____")
else:
    device = torch.device("____")

tensor = torch.arange(5).____(device)
print(tensor.device.type, tensor.device.index)
