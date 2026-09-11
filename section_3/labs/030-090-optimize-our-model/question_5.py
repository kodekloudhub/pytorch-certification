"""
Create a regression model and configure an Adam optimizer using the model parameters and a learning rate of 0.05. Print the optimizer.

NOTE: Import Adam from torch.optim and pass model.parameters() to the optimizer.
"""
from torch import optim
from pre import create_regression_model


model = create_regression_model()
optimizer = optim.____(____.____(), lr=____)
print(optimizer)
