"""
Print whether CUDA is available, the number of CUDA devices, whether MPS support is built into PyTorch, and whether MPS is available.

NOTE: Only request the CUDA device count when CUDA is available. For MPS, use both is_built() and is_available().
"""
import torch

cuda_available = torch.cuda.____()
cuda_device_count = torch.cuda.____() if cuda_available else 0
mps_built = torch.backends.mps.____()
mps_available = torch.backends.mps.____()

print(cuda_available, cuda_device_count, mps_built, mps_available)
