"""
Configure the DataLoader to use no worker subprocesses, enable pinned memory only when CUDA is available, and discard the incomplete final batch. Print the number of batches.

NOTE: Use num_workers, pin_memory, and drop_last to configure these behaviors.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.arange(22))
loader = DataLoader(
    dataset,
    batch_size=5,
    num_workers=____,
    pin_memory=____.cuda.is_available(),
    drop_last=____,
)
print("batches:", len(loader))
