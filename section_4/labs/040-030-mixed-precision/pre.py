"""Restore the breast-cancer classifier and prepare its training data."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2


LAB_DIR = Path(__file__).resolve().parent
COURSE_ROOT = LAB_DIR.parents[2]


def asset_path(local_name, course_path):
    local_path = LAB_DIR / local_name
    return local_path if local_path.exists() else COURSE_ROOT / course_path


class BreastCancerDataset(Dataset):
    def __init__(self, annotations_file, image_root, transform):
        self.annotations = pd.read_csv(annotations_file)
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_encoding = {"malignant": 0, "benign": 1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = self.image_root / self.annotations.iloc[index, 0]
        image = self.transform(Image.open(image_path).convert("RGB"))
        label = self.label_encoding[self.annotations.iloc[index, 1]]
        return image, label


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
image_root = LAB_DIR if (LAB_DIR / "data").exists() else COURSE_ROOT
training_csv = asset_path(
    "training_data.csv",
    "section_3/labs/030-120-transfer-learning/training_data.csv",
)
checkpoint_path = asset_path(
    "mobilenet_checkpoint.tar",
    "section_3/labs/030-150-evaluate-our-models/mobilenet_checkpoint.tar",
)

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
train_dataset = BreastCancerDataset(training_csv, image_root, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = models.mobilenet_v3_large(weights=None)
model.classifier[-1] = nn.Linear(1280, 2)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
