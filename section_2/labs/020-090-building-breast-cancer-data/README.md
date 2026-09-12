# Lab: Building Breast Cancer Data

Build annotation-backed datasets, create a seeded 70/20/10 split, write the
three subset annotation files, apply training versus evaluation transforms, and
configure train/validation/test DataLoaders. The same seed should reproduce the
same subset membership on every run.

The setup script downloads all seven exercise files and extracts the shared
breast-cancer image dataset used by this and later labs.

## Environment setup

```bash
git clone https://github.com/kodekloudhub/pytorch-certification.git
```

```bash
bash pytorch-certification/section_2/labs/020-090-building-breast-cancer-data/setup.sh
```
