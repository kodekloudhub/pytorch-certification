# Demo: Inference Best Practices

Use `inference-best-practices.ipynb` for the course demo. It separates
`model.eval()` from gradient-disabling contexts, compares `torch.no_grad()` and
`torch.inference_mode()`, and demonstrates device-aware single-sample and batch
inference. The companion Python file is a standalone reference.
