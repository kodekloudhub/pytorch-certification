"""
Create a ReduceLROnPlateau scheduler for the breast cancer classifier that halves the learning rate after one epoch without validation-loss improvement.

NOTE: Monitor for a minimum, use factor=0.5, and set patience=1.
"""

import torch

from pre import create_optimizer


optimizer = create_optimizer()
scheduler = torch.optim.lr_scheduler.____(
    optimizer,
    mode=____,
    factor=____,
    patience=____,
)
print(scheduler)
