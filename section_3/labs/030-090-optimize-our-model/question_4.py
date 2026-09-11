"""
Create a regression model and configure an SGD optimizer using the model parameters, a learning rate of 0.1, and momentum of 0.9. Print the optimizer.

NOTE: Import SGD from torch.optim and pass model.parameters() to the optimizer.
"""
from torch import optim
from pre import create_regression_model


model = create_regression_model()
optimizer = optim.____(____.____(), lr=____, momentum=____)
print(optimizer)
