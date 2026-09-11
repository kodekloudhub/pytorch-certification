"""
Create a regression model and configure an AdamW optimizer using the model parameters, a learning rate of 0.05, and weight decay of 0.01. Print the optimizer.

NOTE: AdamW decouples weight decay from its adaptive gradient update. Use the weight_decay argument.
"""
from torch import optim
from pre import create_regression_model


model = create_regression_model()
optimizer = optim.____(
    ____.____(),
    lr=____,
    weight_decay=____,
)
print(optimizer)
