"""Restore the breast-cancer classifier for scheduler exercises."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models


EPOCHS = 10
VALIDATION_LOSSES = [1.0, 0.8, 0.7, 0.69, 0.69, 0.70, 0.68, 0.68, 0.67, 0.67]
LAB_DIR = Path(__file__).resolve().parent
COURSE_ROOT = LAB_DIR.parents[2]
local_checkpoint = LAB_DIR / "mobilenet_checkpoint.tar"
checkpoint_path = (
    local_checkpoint
    if local_checkpoint.exists()
    else COURSE_ROOT
    / "section_3/labs/030-150-evaluate-our-models/mobilenet_checkpoint.tar"
)

model = models.mobilenet_v3_large(weights=None)
model.classifier[-1] = nn.Linear(1280, 2)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])


def create_optimizer():
    return torch.optim.SGD(model.classifier[-1].parameters(), lr=0.1)
