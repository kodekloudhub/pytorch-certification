"""
Split the initial_dataset into training, validation and testing datasets using the PyTorch random split function.

Call the training data train_dataset, validation data val_dataset and the testing data test_dataset.

NOTE: Use 70% for training, 20% for validation and 10% for testing. Seed a torch.Generator with 42 and pass it to the split function so the split is reproducible.
"""
import torch
from initial_dataset import initial_dataset
# Import the random split function 
from ____ import ____

# Define size of Training data
train_size = int(____ * len(initial_dataset))
# Define size of Validation data 
val_size = int(____ * len(initial_dataset))
# Finally define the rest as test 
test_size = len(initial_dataset) - train_size - val_size

# Seed a dedicated generator so reruns preserve the same subset membership.
generator = torch.____().____(42)

# Randomly split the data into training, validation, and testing subsets.
____, ____, ____ = ____(
    initial_dataset,
    [train_size, val_size, test_size],
    generator=____,
)
