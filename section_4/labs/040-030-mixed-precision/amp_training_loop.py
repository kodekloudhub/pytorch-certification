"""
Continue training the breast cancer classifier for four batches using automatic mixed precision. Move each image batch and label batch to the selected device, run the forward pass with autocast, scale the loss before backward(), and update the optimizer through the gradient scaler.

NOTE: GradScaler should be enabled for CUDA float16 training. CPU autocast uses bfloat16 and does not require gradient scaling.
"""

import torch

from pre import (
    autocast_dtype,
    criterion,
    device,
    model,
    optimizer,
    train_loader,
)


scaler = torch.amp.____("cuda", enabled=device.type == "cuda")

model.____()
for batch_index, (images, labels) in enumerate(train_loader):
    images = images.____(device)
    labels = labels.____(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.____(device_type=device.type, dtype=autocast_dtype):
        logits = model(images)
        loss = criterion(logits, labels)

    scaler.____(loss).backward()
    scaler.____(optimizer)
    scaler.____()

    if batch_index == 3:
        break

print("final loss:", loss.item())
