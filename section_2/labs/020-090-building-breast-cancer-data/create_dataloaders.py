"""
Create PyTorch DataLoaders called train_loader, val_loader and test_loader from their corresponding datasets.

This will define how the data is passed to the model during training and is the last step before you train the model. 

The train_loader should take in the train_dataset with a batch size of 64 and should shuffle. 

The val_loader and test_loader should use a batch size of 32 and should not shuffle.

NOTE: Seed a torch.Generator with 42 for reproducible training shuffles. Keep num_workers at zero for this lab and enable pin_memory only when CUDA is available.
"""
import torch
from create_datasets import train_dataset, val_dataset, test_dataset
# Import DataLoader
from ____ import ____

generator = torch.____().____(42)

# Create the training DataLoader
____ = ____(
    ____,
    ____=____,
    shuffle=____,
    num_workers=____,
    pin_memory=torch.cuda.is_available(),
    generator=____,
)

# Create the Validation DataLoader
____ = ____(____, batch_size=____, ____=____, num_workers=0)

# Create the Testing DataLoader
____ = ____(____, batch_size=____, shuffle=____, num_workers=0)
