"""
Step a ReduceLROnPlateau scheduler with each validation loss and save the optimizer and scheduler state dictionaries in a checkpoint.

NOTE: A metric-based scheduler receives validation loss. Save its state so training can resume with the same scheduling history.
"""

import torch

from pre import VALIDATION_LOSSES, create_optimizer, model


optimizer = create_optimizer()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
)

for validation_loss in VALIDATION_LOSSES:
    optimizer.step()
    scheduler.____(____)

checkpoint = {
    "model_state_dict": model.____(),
    "optimizer_state_dict": optimizer.____(),
    "scheduler_state_dict": scheduler.____(),
}
torch.save(checkpoint, "scheduler_checkpoint.tar")
print("final learning rate:", optimizer.param_groups[0]["lr"])
