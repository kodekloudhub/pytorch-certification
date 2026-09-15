# Lab: Learning Rate Scheduling

Continue optimizing the fine-tuned breast-cancer MobileNetV3 classifier from
Section 3. Configure and compare fixed-step, exponential, cosine, and
validation-metric-driven schedules, then preserve the model, optimizer, and
scheduler state in a training checkpoint.

Complete each `____` placeholder in order. `pre.py` supplies a fresh optimizer
for each exercise so one scheduler does not change another exercise's results.

## Environment setup

The KodeKloud lab environment runs `setup.sh` to install the pinned course
requirements, download the five questions, and restore the Section 3
MobileNetV3 checkpoint.

For a manual environment, run:

```bash
bash section_4/labs/040-060-learning-rate-scheduling/setup.sh
```
