"""
Run batch inference over the breast cancer test loader and store every prediction on the CPU.

NOTE: Move each input batch to the model device, and move predictions to CPU before retaining them in a list.
"""

import torch

from pre import test_loader
from question_1 import device, model


model.eval()
all_predictions = []

with torch.inference_mode():
    for features, labels in ____:
        features = features.____(device)
        predictions = model(features).argmax(dim=1)
        all_predictions.append(predictions.____())

all_predictions = torch.____(all_predictions)
print(all_predictions.shape)
