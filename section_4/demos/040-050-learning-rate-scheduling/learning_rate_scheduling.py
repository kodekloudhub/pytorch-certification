"""Visualize four common PyTorch learning-rate scheduling strategies."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


EPOCHS = 12
VALIDATION_LOSSES = [1.0, 0.8, 0.7, 0.69, 0.69, 0.70, 0.68, 0.68, 0.68, 0.67, 0.67, 0.67]


def make_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.SGD([parameter], lr=0.1)


def collect(name):
    optimizer = make_optimizer()
    schedulers = {
        "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5),
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85),
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=0.001
        ),
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        ),
    }
    scheduler = schedulers[name]
    rates = []
    for epoch in range(EPOCHS):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if name == "ReduceLROnPlateau":
            scheduler.step(VALIDATION_LOSSES[epoch])
        else:
            scheduler.step()
    return rates


histories = {
    name: collect(name)
    for name in ("StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau")
}

for name, rates in histories.items():
    formatted = ", ".join(f"{rate:.4f}" for rate in rates)
    print(f"{name:22}: {formatted}")
    plt.plot(range(1, EPOCHS + 1), rates, marker="o", label=name)

plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("PyTorch learning-rate schedulers")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("scheduler_comparison.png", dpi=150)
print("saved scheduler_comparison.png")
