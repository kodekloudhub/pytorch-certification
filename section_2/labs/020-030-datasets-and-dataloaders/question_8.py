"""
Create a DataLoader that produces four batches of eight samples. Print the number of batches and the shapes of the feature and label tensors.

NOTE: Configure the batch_size argument and use the shape attribute for both tensors.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.rand(32, 3, 16, 16), torch.randint(0, 2, (32,)))
loader = DataLoader(dataset, ____=____, shuffle=False)
features, labels = next(iter(loader))
print(len(loader), features.____, labels.____)
