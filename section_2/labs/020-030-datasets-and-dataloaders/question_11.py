"""
Resize every image to 64 x 64 pixels before loading a batch. Print the image and label tensor shapes.

NOTE: DataLoader collation requires the images in a batch to have compatible shapes. Add Resize to the transformation pipeline.
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.____((64, 64)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder("images", transform=____)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(loader))
print(images.shape, labels.shape)
