"""
Complete one training step using the optimizer to clear gradients, calculate gradients, and update the weight.

NOTE: Use the order zero_grad(), forward calculation, backward(), and step().
"""
import torch

weight = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([weight], lr=0.1)
target = torch.tensor(4.0)

optimizer.____()
prediction = weight * 2
loss = (prediction - target) ** 2
loss.____()
print("gradient:", weight.grad)
optimizer.____()
print("updated weight:", weight.item())
