"""
Create a StepLR scheduler that reduces the breast cancer classifier's learning rate by half every 3 epochs.

NOTE: Use step_size=3 and gamma=0.5.
"""

import torch

from pre import create_optimizer


optimizer = create_optimizer()
scheduler = torch.optim.lr_scheduler.____(optimizer, step_size=____, gamma=____)
print(scheduler)
