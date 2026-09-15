"""
Set the breast cancer classifier to evaluation mode and generate predictions for one image batch using both torch.no_grad() and torch.inference_mode().

NOTE: model.eval() changes evaluation-sensitive layers; inference_mode() disables gradient tracking and additional autograd bookkeeping.
"""

import torch

from pre import test_loader
from question_1 import device, model


model.____()
features, labels = next(iter(test_loader))
features = features.____(device)

with torch.____():
    no_grad_predictions = model(features).argmax(dim=1)

with torch.____():
    predictions = model(features).____(dim=1)

print("Predictions match:", torch.equal(no_grad_predictions, predictions))
print(predictions)
