"""
Select CUDA, MPS, or CPU in that order. Make the right tensor compatible with the left tensor's data type and device, then add the tensors and print the result on the CPU.

NOTE: Inspect the left tensor's device and data type when converting the right tensor. Use cpu() before printing an accelerator result for CPU-side use.
"""
import torch

if torch.cuda.____():
    device = torch.device("cuda:0")
elif torch.backends.mps.____():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

left = torch.ones(3, dtype=torch.float32, device=device)
right = torch.tensor([1, 2, 3], dtype=torch.int64)
right = right.____(device=left.____, dtype=left.____)
result = left + right
print(result.____())
