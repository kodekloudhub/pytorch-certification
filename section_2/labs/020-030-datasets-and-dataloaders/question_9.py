"""
Create a shuffled DataLoader whose ordering is reproducible and print its first batch.

NOTE: Create a torch.Generator, seed it with 42, and pass it to the DataLoader.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.arange(20))
generator = torch.____().____(42)
loader = DataLoader(dataset, batch_size=5, shuffle=____, generator=____)
print(next(iter(loader))[0])
