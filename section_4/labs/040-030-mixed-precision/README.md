# Lab: Mixed Precision Training

Continue training the breast-cancer MobileNetV3 classifier created in Section
3, now using automatic mixed precision. `pre.py` restores the fine-tuned model
checkpoint and creates a DataLoader from the breast-cancer training images.

Complete `amp_training_loop.py` to move each batch to the model device, keep
the forward pass and loss calculation inside autocast, scale the loss before
`backward()`, and update the optimizer through the scaler. The exercise runs
four batches with a small batch size so it remains practical in the lab server.

## Environment setup

The KodeKloud lab environment runs `setup.sh` to install the pinned course
requirements, download `pre.py` and the exercise, restore the Section 3
MobileNetV3 checkpoint, and extract the breast-cancer training images.

For a manual environment, run:

```bash
bash section_4/labs/040-030-mixed-precision/setup.sh
```
