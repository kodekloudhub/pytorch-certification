"""
Select CUDA when it is available, otherwise select MPS when it is available, and otherwise select CPU. Then move the breast cancer classifier to the selected device.

NOTE: The model and every input batch must use the same device.
"""

import torch

from pre import model


if torch.cuda.____():
    device = torch.device("____")
elif torch.backends.mps.____():
    device = torch.device("____")
else:
    device = torch.device("____")
model = model.____(device)
print("device:", device)
