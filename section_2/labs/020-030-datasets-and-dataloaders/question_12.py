"""
Create training and validation DataLoaders using the conventional shuffle behavior for each dataset split.

NOTE: Training data is normally shuffled, while validation data should preserve deterministic ordering.
"""
from question_6 import cd_dataset
from torch.utils.data import DataLoader

train_loader = DataLoader(cd_dataset, batch_size=4, shuffle=____)
validation_loader = DataLoader(cd_dataset, batch_size=4, shuffle=____)
print(train_loader.batch_size, validation_loader.batch_size)
