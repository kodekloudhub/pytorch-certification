"""
Create an ExponentialLR scheduler that multiplies the breast cancer classifier's learning rate by 0.9 after each epoch.

NOTE: The gamma argument is the multiplicative decay factor.
"""

import torch

from pre import create_optimizer


optimizer = create_optimizer()
scheduler = torch.optim.lr_scheduler.____(optimizer, gamma=____)
print(scheduler)
