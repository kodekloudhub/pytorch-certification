"""
Define `train_transform`, `val_transform`, and `test_transform` pipelines.

For the `train_transform` create the following pipeline in order:
Resize of 128 x 128 pixels
Set to Grayscale
30 degrees random rotation
50% chance of a random horizontal flip
converted to a tensor
normalized mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) for 3 channels

For the `val_transform` and `test_transform` create the following pipeline:
Resize of 128 x 128 pixels
Set to Grayscale
converted to a tensor
normalized mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) for 3 channels

NOTE: Use the torchvision transforms V2 API. Training transformations include augmentation, while validation and testing transformations should be deterministic.
"""
import torch
# Import transforms version 2
from ____ import ____

# Train Pipeline
____ = ____.____([
    ____.____((____, ____)),
    ____.____(degrees=____),
    ____.____(p=____),
    ____.____(), 
    ____.____(torch.float32, ____),
    ____.____(mean=[____, ____, ____], 
              std=[____, ____, ____])
])

# Validation Pipeline. Hint: Copy
____ = ____.____([
    ____.____((____, ____)),
    ____.____(), 
    ____.____(torch.float32, ____),
    ____.____(mean=[____, ____, ____], 
              std=[____, ____, ____])
])

# Testing uses deterministic evaluation transformations, just like validation.
____ = ____.____([
    ____.____((____, ____)),
    ____.____(),
    ____.____(torch.float32, ____),
    ____.____(mean=[____, ____, ____],
              std=[____, ____, ____])
])
