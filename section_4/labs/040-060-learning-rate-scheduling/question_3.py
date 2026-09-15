"""
Create a CosineAnnealingLR scheduler for the breast cancer classifier that follows a 10-epoch cosine schedule and has a minimum learning rate of 0.001.

NOTE: Set T_max to 10 and eta_min to 0.001.
"""

import torch

from pre import create_optimizer


optimizer = create_optimizer()
scheduler = torch.optim.lr_scheduler.____(optimizer, T_max=____, eta_min=____)
print(scheduler)
