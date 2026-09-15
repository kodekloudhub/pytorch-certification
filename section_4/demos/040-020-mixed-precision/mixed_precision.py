"""Automatic mixed precision training with a safe CPU fallback."""

from time import perf_counter

import torch
from torch import nn


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
if hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
else:
    # Compatibility for environments older than the pinned course version.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
features = torch.randn(512, 128, device=device)
targets = torch.randint(0, 10, (512,), device=device)


def amp_step():
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
        logits = model(features)
        loss = loss_fn(logits, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


start = perf_counter()
loss = amp_step()
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = perf_counter() - start

print("device:", device)
print("autocast dtype:", autocast_dtype)
print("GradScaler enabled:", scaler.is_enabled())
print("loss:", loss.item())
print(f"one AMP step: {elapsed:.6f} seconds")
print("Benchmark after warmup; one step is for API demonstration only.")
