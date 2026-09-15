# Lab: Inference Best Practices

Evaluate the fine-tuned breast-cancer MobileNetV3 classifier from Section 3.
The exercises select a device, switch the restored model to evaluation mode,
compare gradient-disabling contexts, process real image batches, calculate test
accuracy, and return stored predictions to CPU.

Complete each `____` placeholder in order. The setup downloads the Section 3
checkpoint, test annotations, and shared breast-cancer image dataset.

## Environment setup

The KodeKloud lab environment runs `setup.sh` to install the pinned course
requirements, download the four questions and model checkpoint, and extract the
breast-cancer test images.

For a manual environment, run:

```bash
bash section_4/labs/040-090-inference-best-practices/setup.sh
```
